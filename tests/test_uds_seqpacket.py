import errno
import importlib
import os
import socket

import numpy as np
import pytest

import ipc.transport.uds_seqpacket as uds

from ipc.message_schema import FRAME_FLAG_SYNC_DETECTED, FramePacket
from ipc.transport.uds_seqpacket import (
    DEFAULT_MULTI_CONTROL_SOCKET_PATH,
    DEFAULT_MULTI_FRAME_SOCKET_PATH,
    DEFAULT_MULTI_SECRET_REQUEST_SOCKET_PATH,
    DEFAULT_MULTI_SECRET_RESULT_SOCKET_PATH,
    DEFAULT_MULTI_STATUS_SOCKET_PATH,
    DEFAULT_MULTI_VOICE_SOCKET_PATH,
    UdsSeqpacketClient,
    UdsSeqpacketReceiver,
    UdsSeqpacketServer,
    resolve_control_socket_path,
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
    assert resolve_control_socket_path(channel_count=1) == DEFAULT_MULTI_CONTROL_SOCKET_PATH
    assert resolve_control_socket_path(channel_count=30) == DEFAULT_MULTI_CONTROL_SOCKET_PATH
    assert resolve_control_socket_path(socket_path="/tmp/custom_control.sock") == "/tmp/custom_control.sock"


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


def test_uds_seqpacket_server_preserves_client_on_backpressure(monkeypatch):
    class FakeClient:
        def send(self, payload):
            del payload
            raise BlockingIOError(errno.EAGAIN, "socket buffer full")

    server = UdsSeqpacketServer.__new__(UdsSeqpacketServer)
    server.client = FakeClient()
    server._accept_client = lambda: True
    drop_calls = []

    def fake_drop_client():
        drop_calls.append(True)

    monkeypatch.setattr(server, "_drop_client", fake_drop_client)

    assert server.send(b"payload") is False
    assert drop_calls == []


def test_uds_seqpacket_client_try_send_returns_false_on_backpressure():
    class FakeSocket:
        def send(self, payload, flags=0):
            del payload, flags
            raise BlockingIOError(errno.EAGAIN, "socket buffer full")

    client = UdsSeqpacketClient.__new__(UdsSeqpacketClient)
    client.socket = FakeSocket()

    assert client.try_send(b"payload") is False


def test_uds_seqpacket_client_try_send_uses_nonblocking_flag():
    calls = []

    class FakeSocket:
        def send(self, payload, flags=0):
            calls.append((payload, flags))
            return len(payload)

    client = UdsSeqpacketClient.__new__(UdsSeqpacketClient)
    client.socket = FakeSocket()

    assert client.try_send(b"payload") is True
    assert calls == [(b"payload", getattr(socket, "MSG_DONTWAIT", 0))]

# --- default socket directory ------------------------------------------------
#
# The defaults are module-level constants, so each case reloads the module with
# the environment it wants instead of trying to mutate them in place.

_SOCKET_ENV_VARS = (
    "STD_T98_RUNTIME_DIR",
    "XDG_RUNTIME_DIR",
    "STD_T98_MULTI_FRAME_SOCKET",
)


def _frame_path_with_env(monkeypatch, **env):
    for name in _SOCKET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reloaded = importlib.reload(uds)
    return reloaded.DEFAULT_MULTI_FRAME_SOCKET_PATH


@pytest.fixture(autouse=True)
def _restore_uds_module():
    # Leave the module matching the real environment for other tests.
    yield
    importlib.reload(uds)


def test_sockets_live_under_xdg_runtime_dir(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir()

    path = _frame_path_with_env(monkeypatch, XDG_RUNTIME_DIR=str(runtime_dir))

    assert path == str(runtime_dir / "std-t98" / "std_t98_multi_frame.sock")


def test_sockets_fall_back_to_tmp_without_xdg_runtime_dir(monkeypatch):
    path = _frame_path_with_env(monkeypatch)

    assert path == "/tmp/std_t98_multi_frame.sock"


def test_unusable_xdg_runtime_dir_falls_back_to_tmp(monkeypatch, tmp_path):
    path = _frame_path_with_env(
        monkeypatch, XDG_RUNTIME_DIR=str(tmp_path / "does-not-exist")
    )

    assert path == "/tmp/std_t98_multi_frame.sock"


def test_runtime_dir_override_wins_over_xdg(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir()
    override = tmp_path / "elsewhere"

    path = _frame_path_with_env(
        monkeypatch,
        XDG_RUNTIME_DIR=str(runtime_dir),
        STD_T98_RUNTIME_DIR=str(override),
    )

    assert path == str(override / "std_t98_multi_frame.sock")


def test_per_socket_env_var_wins_over_runtime_dir(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir()

    path = _frame_path_with_env(
        monkeypatch,
        XDG_RUNTIME_DIR=str(runtime_dir),
        STD_T98_MULTI_FRAME_SOCKET="/tmp/explicit_frame.sock",
    )

    assert path == "/tmp/explicit_frame.sock"


def test_overlong_runtime_dir_falls_back_to_tmp(monkeypatch, tmp_path):
    # sun_path is 108 bytes; a deep runtime dir must not produce a path that
    # only fails later at bind().
    too_long = "/" + "x" * 120

    path = _frame_path_with_env(monkeypatch, STD_T98_RUNTIME_DIR=too_long)

    assert path == "/tmp/std_t98_multi_frame.sock"


def test_server_creates_missing_socket_directory(tmp_path):
    # The runtime dir has no std-t98 subdirectory until a server binds.
    socket_path = tmp_path / "std-t98" / "frame.sock"
    assert not socket_path.parent.exists()

    server = uds.UdsSeqpacketServer(str(socket_path))
    try:
        assert socket_path.exists()
        assert os.stat(socket_path.parent).st_mode & 0o777 == 0o700

        client = uds.UdsSeqpacketClient(str(socket_path), connect_timeout=2.0)
        try:
            # send() accepts the pending connection on its first call, so it
            # can legitimately report False once before the client attaches.
            assert any(server.send(b"hello") for _ in range(10))
            assert client.recv() == b"hello"
        finally:
            client.close()
    finally:
        server.close()
