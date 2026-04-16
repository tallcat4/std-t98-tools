#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

from core.pipeline.runtime_status import StatusPublisher
from core.secret.cracker import SecretCracker
from ipc.message_schema import (
    SECRET_RESULT_CURRENT_KEY,
    SECRET_RESULT_FULL_SEARCH,
    SECRET_RESULT_GLOBAL_CACHE,
    SECRET_RESULT_NONE,
    STATUS_SOURCE_SECRET,
    SecretCrackRequestPacket,
    SecretCrackResultPacket,
)
from ipc.transport.uds_seqpacket import (
    UdsSeqpacketReceiver,
    UdsSeqpacketServer,
    resolve_secret_request_socket_path,
    resolve_secret_result_socket_path,
)


CHANNEL_COUNT = 30
RESULT_LABELS = {
    SECRET_RESULT_NONE: "Search Miss",
    SECRET_RESULT_CURRENT_KEY: "Current Key Verified",
    SECRET_RESULT_GLOBAL_CACHE: "Global Cache Hit",
    SECRET_RESULT_FULL_SEARCH: "Full Search Hit",
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the split multi-channel secret cracking service.")
    parser.add_argument("--headless", action="store_true", help="Suppress local startup logs.")
    parser.add_argument("--status-socket", help="Optional UDS socket path used to publish status events.")
    parser.add_argument("--models-dir", help="Path to the local secret voice model directory.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    models_dir = Path(args.models_dir) if args.models_dir else repo_root / "models" / "secret_voice"
    ffnn_model_path = models_dir / "ambe2_ffnn.safetensors"
    hybrid_model_path = models_dir / "ambe2_hybrid.safetensors"
    request_socket_path = resolve_secret_request_socket_path(channel_count=CHANNEL_COUNT)
    result_socket_path = resolve_secret_result_socket_path(channel_count=CHANNEL_COUNT)
    request_receiver = UdsSeqpacketReceiver(request_socket_path)
    result_server = UdsSeqpacketServer(result_socket_path)
    status_publisher = StatusPublisher(socket_path=args.status_socket, source=STATUS_SOURCE_SECRET)
    cracker = SecretCracker(ffnn_model_path=ffnn_model_path, hybrid_model_path=hybrid_model_path)

    if not args.headless:
        print(f"Secret service listening for crack requests on {request_socket_path}")
        print(f"Secret service publishing results on {result_socket_path}")
        print(f"Secret models loaded from {models_dir}")

    status_publisher.publish(
        payload_dict={
            "event": "service_started",
            "service": "secret",
            "request_socket": request_socket_path,
            "result_socket": result_socket_path,
            "models_dir": str(models_dir),
        }
    )

    try:
        while True:
            payload = request_receiver.recv(timeout_ms=100)
            if payload is None:
                continue

            request = SecretCrackRequestPacket.decode(payload)
            resolution = cracker.resolve_key(
                current_key=request.current_key,
                burst_payload=request.payload,
                burst_count=request.burst_count,
            )
            result_packet = SecretCrackResultPacket(
                sequence=request.sequence,
                channel_id=request.channel_id,
                session_id=request.session_id,
                resolved_key=resolution.resolved_key,
                result_source=resolution.result_source,
            )
            result_server.send(result_packet.encode())

            status_publisher.publish(
                channel_id=request.channel_id,
                payload_dict={
                    "event": "channel_state",
                    "secret_status": RESULT_LABELS[resolution.result_source],
                    "secret_key": resolution.resolved_key or request.current_key,
                    "secret_cache_keys": list(resolution.cache_keys[:5]),
                },
            )
    except KeyboardInterrupt:
        if not args.headless:
            sys.stdout.write("\n")
            print("Secret service stopped by user.")
    finally:
        status_publisher.close()
        request_receiver.close()
        result_server.close()


if __name__ == "__main__":
    main()