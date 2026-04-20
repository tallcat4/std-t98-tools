#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import select
import sys
import time

from core.pipeline.multi_channel_dashboard import ChannelContext, print_dashboard
from core.pipeline.runtime_status import StatusPublisher
from core.protocol.dewhiten import dewhiten
from core.protocol.frame_layout import parse_frame_symbols
from core.protocol.pich import decode_pich
from core.protocol.rich import decode_rich
from core.protocol.sacch import decode_sacch
from core.protocol.tch import block_strings_to_payloads_3600, split_traffic_blocks
from ipc.message_schema import STATUS_SOURCE_PROTOCOL, VOICE_FORMAT_RAW_3600, FramePacket, VoiceBurstPacket
from ipc.transport.uds_seqpacket import (
    UdsSeqpacketClient,
    UdsSeqpacketServer,
    resolve_frame_socket_path,
    resolve_voice_socket_path,
)


CHANNEL_COUNT = 30
CHANNEL_CLOSE_TIMEOUT = 0.5
VOICE_KEY_ID = 0
DEBUG_REPORT_INTERVAL_SEC = 1.0
DEBUG_ACTIVE_WINDOW_SEC = 2.0


@dataclass
class ProtocolDebugStats:
    frames_received: int = 0
    rich_decode_failures: int = 0
    sync_bursts: int = 0
    traffic_bursts: int = 0
    voice_sent: int = 0
    voice_send_blocked: int = 0
    sacch_crc_ok: int = 0
    sacch_crc_fail: int = 0
    last_frame_at: float = 0.0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the split multi-channel protocol service.")
    parser.add_argument("--headless", action="store_true", help="Disable the local dashboard UI.")
    parser.add_argument("--status-socket", help="Optional UDS socket path used to publish status events.")
    return parser.parse_args(argv)


def _publish_channel_state(status_publisher, channel_id, channel_context):
    if status_publisher is None:
        return
    status_publisher.publish(
        channel_id=channel_id,
        payload_dict={
            "event": "channel_state",
            "rx_status": channel_context.rx_status,
            "protocol_status": channel_context.audio_status,
            "csm": channel_context.csm,
            "sacch": channel_context.sacch,
        },
    )


def _format_protocol_debug(debug_stats):
    return (
        f"frm={debug_stats.frames_received} sync={debug_stats.sync_bursts} "
        f"traf={debug_stats.traffic_bursts} ipc={debug_stats.voice_sent}/{debug_stats.voice_send_blocked} "
        f"rich_fail={debug_stats.rich_decode_failures} sacch={debug_stats.sacch_crc_ok}/{debug_stats.sacch_crc_fail}"
    )


def _format_protocol_service_summary(debug_by_channel, current_time):
    totals = ProtocolDebugStats()
    active_channels = 0

    for debug_stats in debug_by_channel.values():
        totals.frames_received += debug_stats.frames_received
        totals.rich_decode_failures += debug_stats.rich_decode_failures
        totals.sync_bursts += debug_stats.sync_bursts
        totals.traffic_bursts += debug_stats.traffic_bursts
        totals.voice_sent += debug_stats.voice_sent
        totals.voice_send_blocked += debug_stats.voice_send_blocked
        totals.sacch_crc_ok += debug_stats.sacch_crc_ok
        totals.sacch_crc_fail += debug_stats.sacch_crc_fail
        if debug_stats.last_frame_at and (current_time - debug_stats.last_frame_at) <= DEBUG_ACTIVE_WINDOW_SEC:
            active_channels += 1

    return (
        f"frames={totals.frames_received} sync={totals.sync_bursts} traffic={totals.traffic_bursts} "
        f"ipc={totals.voice_sent}/{totals.voice_send_blocked} rich_fail={totals.rich_decode_failures} "
        f"sacch={totals.sacch_crc_ok}/{totals.sacch_crc_fail} active={active_channels}"
    )


def _maybe_publish_protocol_metrics(status_publisher, debug_by_channel, next_report_at, current_time):
    if current_time < next_report_at:
        return next_report_at

    status_publisher.publish(
        payload_dict={
            "event": "service_metrics",
            "summary": _format_protocol_service_summary(debug_by_channel, current_time),
        }
    )

    for channel_id, debug_stats in debug_by_channel.items():
        if debug_stats.frames_received <= 0:
            continue
        if (current_time - debug_stats.last_frame_at) > DEBUG_ACTIVE_WINDOW_SEC and debug_stats.voice_send_blocked <= 0:
            continue
        status_publisher.publish(
            channel_id=channel_id,
            payload_dict={
                "event": "channel_state",
                "protocol_debug": _format_protocol_debug(debug_stats),
            },
        )

    return current_time + DEBUG_REPORT_INTERVAL_SEC


def main(argv=None):
    args = _parse_args(argv)
    frame_socket_path = resolve_frame_socket_path(channel_count=CHANNEL_COUNT)
    voice_socket_path = resolve_voice_socket_path(channel_count=CHANNEL_COUNT)
    status_publisher = StatusPublisher(socket_path=args.status_socket, source=STATUS_SOURCE_PROTOCOL)

    frame_client = UdsSeqpacketClient(socket_path=frame_socket_path)
    voice_server = UdsSeqpacketServer(socket_path=voice_socket_path)
    poller = select.poll()
    poller.register(frame_client.fileno(), select.POLLIN)

    channels = {}
    debug_by_channel = {}
    voice_burst_index = 0
    next_debug_report_at = time.time() + DEBUG_REPORT_INTERVAL_SEC

    if not args.headless:
        print(f"Protocol service listening to frames on {frame_socket_path}")
        print(f"Protocol service publishing voice bursts on {voice_socket_path}\n")
        print("\n" * 15, end="")
        printed_lines = print_dashboard(channels, 0, title="STD-T98 Multi Protocol Service")
    else:
        printed_lines = 0

    status_publisher.publish(
        payload_dict={
            "event": "service_started",
            "service": "protocol",
            "frame_socket": frame_socket_path,
            "voice_socket": voice_socket_path,
        }
    )

    try:
        while True:
            events = dict(poller.poll(100))
            current_time = time.time()
            dashboard_changed = False

            if frame_client.fileno() in events:
                while True:
                    payload = frame_client.try_recv()
                    if not payload:
                        break

                    packet = FramePacket.decode(payload)
                    channel_id = packet.channel_id
                    channel_changed = False

                    if channel_id not in channels:
                        channels[channel_id] = ChannelContext()
                        channel_changed = True

                    ctx_ch = channels[channel_id]
                    debug_stats = debug_by_channel.setdefault(channel_id, ProtocolDebugStats())
                    debug_stats.frames_received += 1
                    debug_stats.last_frame_at = current_time

                    if ctx_ch.rx_status != "OPEN":
                        ctx_ch.rx_status = "OPEN"
                        channel_changed = True

                    ctx_ch.last_update = current_time

                    fields = parse_frame_symbols(dewhiten(packet.symbols))

                    try:
                        rich_data = decode_rich(fields["RICH"])
                    except Exception:
                        debug_stats.rich_decode_failures += 1
                        continue

                    if rich_data["F"] == 0:
                        debug_stats.sync_bursts += 1
                        pich_data = decode_pich(fields["TCH1"])
                        if pich_data.get("CRC_OK") and ctx_ch.csm != pich_data.get("CSM"):
                            ctx_ch.csm = pich_data.get("CSM")
                            channel_changed = True

                        if ctx_ch.audio_status != "Sync Burst":
                            ctx_ch.audio_status = "Sync Burst"
                            channel_changed = True

                        dashboard_changed = dashboard_changed or channel_changed
                        if channel_changed:
                            _publish_channel_state(status_publisher, channel_id, ctx_ch)
                        continue

                    if rich_data["F"] != 1:
                        continue

                    debug_stats.traffic_bursts += 1
                    sacch_data = decode_sacch(fields["SACCH"])
                    if sacch_data.get("CRC_OK"):
                        debug_stats.sacch_crc_ok += 1
                        if ctx_ch.sacch != sacch_data:
                            ctx_ch.sacch = sacch_data
                            channel_changed = True
                    else:
                        debug_stats.sacch_crc_fail += 1

                    blocks_payload = b"".join(block_strings_to_payloads_3600(split_traffic_blocks(fields)))
                    voice_packet = VoiceBurstPacket(
                        sequence=packet.sequence,
                        channel_id=channel_id,
                        call_stat=sacch_data.get("CallStat", 0) if sacch_data.get("CRC_OK") else 0,
                        key_id=VOICE_KEY_ID,
                        burst_index=voice_burst_index,
                        payload_format=VOICE_FORMAT_RAW_3600,
                        payload=blocks_payload,
                    )
                    voice_burst_index += 1

                    if voice_server.send(voice_packet.encode()):
                        debug_stats.voice_sent += 1
                        next_audio_status = "Traffic->IPC"
                    else:
                        debug_stats.voice_send_blocked += 1
                        next_audio_status = "Traffic (no client)"

                    if ctx_ch.audio_status != next_audio_status:
                        ctx_ch.audio_status = next_audio_status
                        channel_changed = True

                    dashboard_changed = dashboard_changed or channel_changed
                    if channel_changed:
                        _publish_channel_state(status_publisher, channel_id, ctx_ch)
            else:
                for channel_id, ctx_ch in channels.items():
                    if ctx_ch.rx_status == "OPEN" and (current_time - ctx_ch.last_update) > CHANNEL_CLOSE_TIMEOUT:
                        ctx_ch.rx_status = "CLOSE"
                        ctx_ch.audio_status = "Idle"
                        dashboard_changed = True
                        _publish_channel_state(status_publisher, channel_id, ctx_ch)

            next_debug_report_at = _maybe_publish_protocol_metrics(
                status_publisher,
                debug_by_channel,
                next_debug_report_at,
                current_time,
            )

            if dashboard_changed and not args.headless:
                printed_lines = print_dashboard(channels, printed_lines, title="STD-T98 Multi Protocol Service")

    except KeyboardInterrupt:
        if not args.headless:
            sys.stdout.write("\n")
            print("Protocol service stopped by user.")
    finally:
        status_publisher.close()
        frame_client.close()
        voice_server.close()


if __name__ == "__main__":
    main()