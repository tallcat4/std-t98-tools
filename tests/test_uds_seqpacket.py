import numpy as np
import pytest

from ipc.message_schema import FRAME_FLAG_SYNC_DETECTED, FramePacket
from ipc.transport.uds_seqpacket import (
    DEFAULT_MULTI_FRAME_SOCKET_PATH,
    DEFAULT_MULTI_SECRET_REQUEST_SOCKET_PATH,
    DEFAULT_MULTI_SECRET_RESULT_SOCKET_PATH,
    DEFAULT_MULTI_STATUS_SOCKET_PATH,
    DEFAULT_MULTI_VOICE_SOCKET_PATH,
    UdsSeqpacketClient,
    UdsSeqpacketReceiver,
    UdsSeqpacketServer,
    resolve_frame_socket_path,
    resolve_secret_request_socket_path,
    resolve_secret_result_socket_path,
    resolve_status_socket_path,
    resolve_voice_socket_path,
)


def test_socket_path_resolvers_use_multi_defaults_and_custom_overrides():
    assert resolve_frame_socket_path(channel_count=1) == DEFAULT_MULTI_FRAME_SOCKET_PATH
    assert resolve_frame_socket_path(channel_count=30) == DEFAULT_MULTI_FRAME_SOCKET_PATH
    assert resolve_frame_socket_path(socket_path="/tmp/custom_frame.sock") == "/tmp/custom_frame.sock"
    assert resolve_voice_socket_path(channel_count=1) == DEFAULT_MULTI_VOICE_SOCKET_PATH
    assert resolve_voice_socket_path(channel_count=30) == DEFAULT_MULTI_VOICE_SOCKET_PATH
    assert resolve_voice_socket_path(socket_path="/tmp/custom_voice.sock") == "/tmp/custom_voice.sock"
    assert resolve_status_socket_path(channel_count=1) == DEFAULT_MULTI_STATUS_SOCKET_PATH
    assert resolve_status_socket_path(channel_count=30) == DEFAULT_MULTI_STATUS_SOCKET_PATH
    assert resolve_status_socket_path(socket_path="/tmp/custom_status.sock") == "/tmp/custom_status.sock"
    assert resolve_secret_request_socket_path(channel_count=1) == DEFAULT_MULTI_SECRET_REQUEST_SOCKET_PATH
    assert resolve_secret_request_socket_path(channel_count=30) == DEFAULT_MULTI_SECRET_REQUEST_SOCKET_PATH
    assert resolve_secret_request_socket_path(socket_path="/tmp/custom_secret_request.sock") == "/tmp/custom_secret_request.sock"
    assert resolve_secret_result_socket_path(channel_count=1) == DEFAULT_MULTI_SECRET_RESULT_SOCKET_PATH
    assert resolve_secret_result_socket_path(channel_count=30) == DEFAULT_MULTI_SECRET_RESULT_SOCKET_PATH
    assert resolve_secret_result_socket_path(socket_path="/tmp/custom_secret_result.sock") == "/tmp/custom_secret_result.sock"


def test_uds_seqpacket_roundtrip(tmp_path):
    socket_path = tmp_path / "frames.sock"
    server = UdsSeqpacketServer(str(socket_path))
    client = UdsSeqpacketClient(str(socket_path), retry_interval=0.01, connect_timeout=1.0)
    packet = FramePacket(
        sequence=23,
        monotonic_ns=987654321,
        channel_id=5,
        flags=FRAME_FLAG_SYNC_DETECTED,
        symbols=np.array(([-3, -1, 1, 3] * 48)[:192], dtype=np.int8),
    )

    try:
        assert server.send(packet.encode()) is True

        decoded = FramePacket.decode(client.recv())

        assert decoded.sequence == 23
        assert decoded.monotonic_ns == 987654321
        assert decoded.channel_id == 5
        assert decoded.flags == FRAME_FLAG_SYNC_DETECTED
        assert np.array_equal(decoded.symbols, packet.symbols)
    finally:
        client.close()
        server.close()

    assert not socket_path.exists()


def test_uds_seqpacket_client_times_out_when_socket_is_missing(tmp_path):
    missing_socket = tmp_path / "missing.sock"

    with pytest.raises(TimeoutError, match=str(missing_socket)):
        UdsSeqpacketClient(str(missing_socket), retry_interval=0.01, connect_timeout=0.05)


def test_uds_seqpacket_receiver_accepts_multiple_clients(tmp_path):
    socket_path = tmp_path / "status.sock"
    receiver = UdsSeqpacketReceiver(str(socket_path))
    client_a = UdsSeqpacketClient(str(socket_path), retry_interval=0.01, connect_timeout=1.0)
    client_b = UdsSeqpacketClient(str(socket_path), retry_interval=0.01, connect_timeout=1.0)

    try:
        client_a.send(b"first")
        client_b.send(b"second")

        received = {receiver.recv(timeout_ms=1000), receiver.recv(timeout_ms=1000)}
        assert received == {b"first", b"second"}
    finally:
        client_a.close()
        client_b.close()
        receiver.close()