#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import sys
import time

from core.audio.ambe_adapter import create_decoder, decode_2450_payloads_to_pcm, payloads_3600_to_2450
from core.crypto.pn_sequence import generate_pn_sequence_196
from core.crypto.secret_voice import descramble_burst
from core.pipeline.runtime_status import StatusPublisher
from ipc.message_schema import (
    STATUS_SOURCE_AUDIO,
    VOICE_BURST_BLOCK_BYTES_AMBE_2450,
    VOICE_BURST_BLOCK_BYTES_RAW_3600,
    VOICE_FORMAT_RAW_3600,
    VoiceBurstPacket,
)
from ipc.transport.uds_seqpacket import UdsSeqpacketClient, resolve_voice_socket_path


sd = importlib.import_module("sounddevice")


CHANNEL_COUNT = 30
DECRYPTION_KEY = 0
SAMPLE_RATE = 48000
UPSAMPLE_FACTOR = 6


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the split multi-channel audio service.")
    parser.add_argument("--headless", action="store_true", help="Suppress local startup logs.")
    parser.add_argument("--status-socket", help="Optional UDS socket path used to publish status events.")
    parser.add_argument("--device", default="pipewire", help="Audio output device name passed to sounddevice.")
    return parser.parse_args(argv)


def _split_payload_blocks(payload: bytes, block_size: int):
    return [payload[index : index + block_size] for index in range(0, len(payload), block_size)]


def main(argv=None):
    args = _parse_args(argv)
    voice_socket_path = resolve_voice_socket_path(channel_count=CHANNEL_COUNT)
    client = UdsSeqpacketClient(socket_path=voice_socket_path)
    key_196 = generate_pn_sequence_196(DECRYPTION_KEY)
    decoders = {}
    last_playing_publish = {}
    status_publisher = StatusPublisher(socket_path=args.status_socket, source=STATUS_SOURCE_AUDIO)

    if not args.headless:
        print(f"Audio service waiting for voice bursts on {voice_socket_path}")
        print(f"Decryption Key (LFSR Init) set to: {DECRYPTION_KEY}")

    status_publisher.publish(
        payload_dict={
            "event": "service_started",
            "service": "audio",
            "voice_socket": voice_socket_path,
            "device": args.device,
        }
    )

    try:
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", latency="low", device=args.device) as audio_stream:
            while True:
                packet = VoiceBurstPacket.decode(client.recv())

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

                descrambled_payloads = descramble_burst(payloads_2450, key_196)
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
        client.close()


if __name__ == "__main__":
    main()