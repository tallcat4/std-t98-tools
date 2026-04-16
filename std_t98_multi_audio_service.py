#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import deque
from dataclasses import dataclass, field
import importlib
import select
import sys
import time

from core.audio.ambe_adapter import create_decoder, decode_2450_payloads_to_pcm, payloads_3600_to_2450
from core.crypto.pn_sequence import generate_pn_sequence_196
from core.crypto.secret_voice import descramble_burst
from core.pipeline.runtime_status import StatusPublisher
from ipc.message_schema import (
    SECRET_RESULT_CURRENT_KEY,
    SECRET_RESULT_FULL_SEARCH,
    SECRET_RESULT_GLOBAL_CACHE,
    SECRET_RESULT_NONE,
    STATUS_SOURCE_AUDIO,
    VOICE_BURST_BLOCK_BYTES_AMBE_2450,
    VOICE_BURST_BLOCK_BYTES_RAW_3600,
    VOICE_FORMAT_RAW_3600,
    SecretCrackRequestPacket,
    SecretCrackResultPacket,
    VoiceBurstPacket,
)
from ipc.transport.uds_seqpacket import (
    UdsSeqpacketClient,
    resolve_secret_request_socket_path,
    resolve_secret_result_socket_path,
    resolve_voice_socket_path,
)


sd = importlib.import_module("sounddevice")


CHANNEL_COUNT = 30
DECRYPTION_KEY = 0
SAMPLE_RATE = 48000
UPSAMPLE_FACTOR = 6
CALL_STAT_SECRET = 1
SECRET_MIN_WINDOW_BURSTS = 5
SECRET_MAX_WINDOW_BURSTS = 10
SECRET_RECHECK_INTERVAL_BURSTS = 10

SECRET_RESULT_LABELS = {
    SECRET_RESULT_NONE: "Search Miss",
    SECRET_RESULT_CURRENT_KEY: "Current Key Verified",
    SECRET_RESULT_GLOBAL_CACHE: "Global Cache Hit",
    SECRET_RESULT_FULL_SEARCH: "Full Search Hit",
}


@dataclass
class SecretChannelState:
    session_id: int = 0
    secret_active: bool = False
    active_key: int = 0
    pending_request: bool = False
    last_request_burst_index: int = -SECRET_RECHECK_INTERVAL_BURSTS
    burst_window: deque[bytes] = field(default_factory=lambda: deque(maxlen=SECRET_MAX_WINDOW_BURSTS))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the split multi-channel audio service.")
    parser.add_argument("--headless", action="store_true", help="Suppress local startup logs.")
    parser.add_argument("--status-socket", help="Optional UDS socket path used to publish status events.")
    parser.add_argument("--device", default="pipewire", help="Audio output device name passed to sounddevice.")
    return parser.parse_args(argv)


def _split_payload_blocks(payload: bytes, block_size: int):
    return [payload[index : index + block_size] for index in range(0, len(payload), block_size)]


def _ensure_key_sequence(key_sequences, key):
    if key not in key_sequences:
        key_sequences[key] = generate_pn_sequence_196(key)
    return key_sequences[key]


def _secret_burst_payload(payloads_2450):
    return b"".join(payloads_2450)


def _start_secret_session(secret_state):
    secret_state.session_id += 1
    secret_state.secret_active = True
    secret_state.pending_request = False
    secret_state.last_request_burst_index = -SECRET_RECHECK_INTERVAL_BURSTS
    secret_state.burst_window.clear()


def _stop_secret_session(secret_state):
    secret_state.secret_active = False
    secret_state.pending_request = False
    secret_state.last_request_burst_index = -SECRET_RECHECK_INTERVAL_BURSTS
    secret_state.burst_window.clear()


def _maybe_send_secret_request(packet, secret_state, request_client, request_sequence):
    if secret_state.pending_request or len(secret_state.burst_window) < SECRET_MIN_WINDOW_BURSTS:
        return request_sequence, False

    if secret_state.last_request_burst_index < 0:
        window_bursts = list(secret_state.burst_window)[-SECRET_MIN_WINDOW_BURSTS:]
    else:
        burst_gap = packet.burst_index - secret_state.last_request_burst_index
        if burst_gap < SECRET_RECHECK_INTERVAL_BURSTS or len(secret_state.burst_window) < SECRET_MAX_WINDOW_BURSTS:
            return request_sequence, False
        window_bursts = list(secret_state.burst_window)[-SECRET_MAX_WINDOW_BURSTS:]

    request_packet = SecretCrackRequestPacket(
        sequence=request_sequence,
        channel_id=packet.channel_id,
        session_id=secret_state.session_id,
        current_key=secret_state.active_key,
        burst_count=len(window_bursts),
        payload=b"".join(window_bursts),
    )
    request_client.send(request_packet.encode())
    secret_state.pending_request = True
    secret_state.last_request_burst_index = packet.burst_index
    return request_sequence + 1, True


def main(argv=None):
    args = _parse_args(argv)
    voice_socket_path = resolve_voice_socket_path(channel_count=CHANNEL_COUNT)
    secret_request_socket_path = resolve_secret_request_socket_path(channel_count=CHANNEL_COUNT)
    secret_result_socket_path = resolve_secret_result_socket_path(channel_count=CHANNEL_COUNT)
    voice_client = UdsSeqpacketClient(socket_path=voice_socket_path)
    secret_request_client = UdsSeqpacketClient(socket_path=secret_request_socket_path)
    secret_result_client = UdsSeqpacketClient(socket_path=secret_result_socket_path)
    poller = select.poll()
    poller.register(voice_client.fileno(), select.POLLIN)
    poller.register(secret_result_client.fileno(), select.POLLIN)
    key_sequences = {DECRYPTION_KEY: generate_pn_sequence_196(DECRYPTION_KEY)}
    decoders = {}
    secret_states = {}
    last_playing_publish = {}
    secret_request_sequence = 0
    status_publisher = StatusPublisher(socket_path=args.status_socket, source=STATUS_SOURCE_AUDIO)

    if not args.headless:
        print(f"Audio service waiting for voice bursts on {voice_socket_path}")
        print(f"Decryption Key (LFSR Init) set to: {DECRYPTION_KEY}")
        print(f"Secret request socket: {secret_request_socket_path}")
        print(f"Secret result socket: {secret_result_socket_path}")

    status_publisher.publish(
        payload_dict={
            "event": "service_started",
            "service": "audio",
            "voice_socket": voice_socket_path,
            "secret_request_socket": secret_request_socket_path,
            "secret_result_socket": secret_result_socket_path,
            "device": args.device,
        }
    )

    try:
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", latency="low", device=args.device) as audio_stream:
            while True:
                events = dict(poller.poll(100))
                if not events:
                    continue

                if secret_result_client.fileno() in events:
                    result_packet = SecretCrackResultPacket.decode(secret_result_client.recv())
                    secret_state = secret_states.get(result_packet.channel_id)
                    if secret_state is not None and secret_state.session_id == result_packet.session_id:
                        secret_state.pending_request = False
                        if result_packet.resolved_key > 0:
                            secret_state.active_key = result_packet.resolved_key
                        status_publisher.publish(
                            channel_id=result_packet.channel_id,
                            payload_dict={
                                "event": "channel_state",
                                "audio_status": "Playing",
                                "secret_status": SECRET_RESULT_LABELS[result_packet.result_source],
                                "secret_key": secret_state.active_key,
                            },
                        )

                if voice_client.fileno() not in events:
                    continue

                packet = VoiceBurstPacket.decode(voice_client.recv())
                secret_state = secret_states.setdefault(packet.channel_id, SecretChannelState(active_key=DECRYPTION_KEY))

                if packet.channel_id not in decoders:
                    decoders[packet.channel_id] = create_decoder()
                    if not args.headless:
                        print(f"Opened audio decoder for channel {packet.channel_id:02d}")
                    status_publisher.publish(
                        channel_id=packet.channel_id,
                        payload_dict={
                            "event": "channel_state",
                            "audio_status": "Decoder Ready",
                        },
                    )

                decoder = decoders[packet.channel_id]
                if packet.payload_format == VOICE_FORMAT_RAW_3600:
                    payloads_3600 = _split_payload_blocks(packet.payload, VOICE_BURST_BLOCK_BYTES_RAW_3600)
                    payloads_2450 = payloads_3600_to_2450(payloads_3600)
                else:
                    payloads_2450 = _split_payload_blocks(packet.payload, VOICE_BURST_BLOCK_BYTES_AMBE_2450)

                if packet.call_stat == CALL_STAT_SECRET:
                    if not secret_state.secret_active:
                        _start_secret_session(secret_state)
                        status_publisher.publish(
                            channel_id=packet.channel_id,
                            payload_dict={
                                "event": "channel_state",
                                "audio_status": "Playing",
                                "secret_status": "Collecting",
                                "secret_key": secret_state.active_key,
                            },
                        )

                    secret_state.burst_window.append(_secret_burst_payload(payloads_2450))
                    secret_request_sequence, request_sent = _maybe_send_secret_request(
                        packet,
                        secret_state,
                        secret_request_client,
                        secret_request_sequence,
                    )
                    if request_sent:
                        status_publisher.publish(
                            channel_id=packet.channel_id,
                            payload_dict={
                                "event": "channel_state",
                                "audio_status": "Playing",
                                "secret_status": "Search Pending",
                                "secret_key": secret_state.active_key,
                            },
                        )
                elif secret_state.secret_active:
                    _stop_secret_session(secret_state)
                    status_publisher.publish(
                        channel_id=packet.channel_id,
                        payload_dict={
                            "event": "channel_state",
                            "audio_status": "Playing",
                            "secret_status": "Idle",
                            "secret_key": secret_state.active_key,
                        },
                    )

                decryption_key = secret_state.active_key if packet.call_stat == CALL_STAT_SECRET else DECRYPTION_KEY
                descrambled_payloads = descramble_burst(payloads_2450, _ensure_key_sequence(key_sequences, decryption_key))
                pcm_out = decode_2450_payloads_to_pcm(
                    descrambled_payloads,
                    decoder,
                    upsample_factor=UPSAMPLE_FACTOR,
                )
                if pcm_out:
                    audio_stream.write(pcm_out)
                    current_time = time.time()
                    last_publish_time = last_playing_publish.get(packet.channel_id, 0.0)
                    if current_time - last_publish_time >= 1.0:
                        status_publisher.publish(
                            channel_id=packet.channel_id,
                            payload_dict={
                                "event": "channel_state",
                                "audio_status": "Playing",
                            },
                        )
                        last_playing_publish[packet.channel_id] = current_time

    except KeyboardInterrupt:
        if not args.headless:
            sys.stdout.write("\n")
            print("Audio service stopped by user.")
    finally:
        status_publisher.close()
        voice_client.close()
        secret_request_client.close()
        secret_result_client.close()


if __name__ == "__main__":
    main()