#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import deque
from dataclasses import dataclass, field
import importlib
import os
import select
import sys
import threading
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
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"
AUDIO_BYTES_PER_FRAME = 2
AUDIO_RECOVERY_DELAY_SEC = 0.5
AUDIO_UNDERFLOW_RECOVERY_THRESHOLD = 8
AUDIO_QUEUE_MAX_SECONDS = 0.5
AUDIO_QUEUE_MAX_BYTES = int(SAMPLE_RATE * AUDIO_BYTES_PER_FRAME * AUDIO_QUEUE_MAX_SECONDS)
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


def _parse_audio_latency(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in ("low", "high"):
        return lowered

    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid audio latency value: {value}") from exc


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
    parser.add_argument(
        "--latency",
        default=_parse_audio_latency(os.environ.get("STD_T98_AUDIO_LATENCY")),
        type=_parse_audio_latency,
        help="Optional sounddevice latency. Use 'high', 'low', or seconds. Default uses PortAudio's device default.",
    )
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
    if not request_client.try_send(request_packet.encode()):
        return request_sequence, False
    secret_state.pending_request = True
    secret_state.last_request_burst_index = packet.burst_index
    return request_sequence + 1, True


def _audio_stream_kwargs(device, latency):
    kwargs = {
        "samplerate": SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "dtype": AUDIO_DTYPE,
        "device": device,
    }
    if latency is not None:
        kwargs["latency"] = latency
    return kwargs


def _open_audio_stream(device, latency):
    audio_stream = sd.RawOutputStream(**_audio_stream_kwargs(device, latency))
    audio_stream.start()
    return audio_stream


def _close_audio_stream(audio_stream):
    if audio_stream is None:
        return

    try:
        audio_stream.close()
    except Exception:
        pass


def _open_audio_stream_with_retry(device, latency, headless):
    waiting_reported = False
    while True:
        try:
            return _open_audio_stream(device, latency)
        except sd.PortAudioError as exc:
            if not headless:
                if not waiting_reported:
                    print("Audio output unavailable; retrying until the device becomes ready...", file=sys.stderr)
                    waiting_reported = True
                print(f"Audio stream open failed: {exc}", file=sys.stderr)
            time.sleep(AUDIO_RECOVERY_DELAY_SEC)


def _trim_audio_backlog(audio_queue, queued_bytes, max_bytes):
    while audio_queue and queued_bytes > max_bytes:
        queued_bytes -= len(audio_queue.popleft())
    return queued_bytes


class AudioOutputWorker:
    def __init__(self, device, latency, headless):
        self.device = device
        self.latency = latency
        self.headless = headless
        self.audio_stream = None
        self.closed = False
        self.pending_pcm = deque()
        self.pending_bytes = 0
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._run, name="std-t98-audio-output", daemon=True)

    def start(self):
        self.thread.start()

    def enqueue(self, pcm_out):
        if not pcm_out:
            return False

        if len(pcm_out) > AUDIO_QUEUE_MAX_BYTES:
            pcm_out = pcm_out[-AUDIO_QUEUE_MAX_BYTES:]

        with self.condition:
            if self.closed:
                return False
            self.pending_pcm.append(pcm_out)
            self.pending_bytes += len(pcm_out)
            self.pending_bytes = _trim_audio_backlog(self.pending_pcm, self.pending_bytes, AUDIO_QUEUE_MAX_BYTES)
            self.condition.notify()
        return True

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

        self.thread.join(timeout=1.0)
        _close_audio_stream(self.audio_stream)
        self.audio_stream = None

    def _next_chunk(self):
        with self.condition:
            while not self.pending_pcm and not self.closed:
                self.condition.wait(timeout=0.1)

            if not self.pending_pcm:
                return None

            pcm_out = self.pending_pcm.popleft()
            self.pending_bytes -= len(pcm_out)
            return pcm_out

    def _run(self):
        output_underflow_count = 0

        while True:
            pcm_out = self._next_chunk()
            if pcm_out is None:
                if self.closed:
                    break
                continue

            if self.audio_stream is None:
                self.audio_stream = _open_audio_stream_with_retry(self.device, self.latency, self.headless)

            try:
                underflowed = self.audio_stream.write(pcm_out)
            except sd.PortAudioError as exc:
                if not self.headless:
                    print(f"Audio write failed: {exc}", file=sys.stderr)
                _close_audio_stream(self.audio_stream)
                self.audio_stream = None
                output_underflow_count = 0
                time.sleep(AUDIO_RECOVERY_DELAY_SEC)
                continue

            if underflowed:
                output_underflow_count += 1
                if output_underflow_count >= AUDIO_UNDERFLOW_RECOVERY_THRESHOLD:
                    if not self.headless:
                        print("Audio output kept underflowing; reopening the audio stream.", file=sys.stderr)
                    _close_audio_stream(self.audio_stream)
                    self.audio_stream = None
                    output_underflow_count = 0
                continue

            output_underflow_count = 0

        _close_audio_stream(self.audio_stream)
        self.audio_stream = None


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
    audio_output = AudioOutputWorker(args.device, args.latency, args.headless)

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
            "latency": args.latency if args.latency is not None else "default",
        }
    )

    try:
        audio_output.start()

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
            if not pcm_out:
                continue

            audio_output.enqueue(pcm_out)

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
        audio_output.close()
        status_publisher.close()
        voice_client.close()
        secret_request_client.close()
        secret_result_client.close()


if __name__ == "__main__":
    main()