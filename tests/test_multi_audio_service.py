from collections import deque
from types import SimpleNamespace

from ipc.message_schema import SECRET_BURST_BYTES_AMBE_2450
from std_t98_multi_audio_service import (
    AUDIO_QUEUE_MAX_BYTES,
    SECRET_MIN_WINDOW_BURSTS,
    SECRET_RECHECK_INTERVAL_BURSTS,
    SecretChannelState,
    _maybe_send_secret_request,
    _parse_audio_latency,
    _trim_audio_backlog,
)


def test_maybe_send_secret_request_does_not_block_channel_when_send_is_busy():
    class FakeRequestClient:
        def __init__(self):
            self.payloads = []

        def try_send(self, payload):
            self.payloads.append(payload)
            return False

    secret_state = SecretChannelState(active_key=123)
    secret_state.session_id = 7
    secret_state.burst_window.extend([b"\x00" * SECRET_BURST_BYTES_AMBE_2450] * SECRET_MIN_WINDOW_BURSTS)
    request_client = FakeRequestClient()

    sequence, request_sent = _maybe_send_secret_request(
        packet=SimpleNamespace(channel_id=12, burst_index=34),
        secret_state=secret_state,
        request_client=request_client,
        request_sequence=99,
    )

    assert sequence == 99
    assert request_sent is False
    assert len(request_client.payloads) == 1
    assert secret_state.pending_request is False
    assert secret_state.last_request_burst_index == -SECRET_RECHECK_INTERVAL_BURSTS


def test_parse_audio_latency_accepts_named_and_numeric_values():
    assert _parse_audio_latency(None) is None
    assert _parse_audio_latency("") is None
    assert _parse_audio_latency("high") == "high"
    assert _parse_audio_latency("LOW") == "low"
    assert _parse_audio_latency("0.25") == 0.25


def test_trim_audio_backlog_keeps_newest_audio_within_limit():
    audio_queue = deque([b"oldest", b"older", b"newest"])
    queued_bytes = sum(len(chunk) for chunk in audio_queue)

    remaining = _trim_audio_backlog(audio_queue, queued_bytes, len(b"older") + len(b"newest"))

    assert list(audio_queue) == [b"older", b"newest"]
    assert remaining == len(b"older") + len(b"newest")


def test_trim_audio_backlog_clears_large_queue_until_under_limit():
    audio_queue = deque([b"a" * (AUDIO_QUEUE_MAX_BYTES // 2), b"b" * (AUDIO_QUEUE_MAX_BYTES // 2), b"c" * 8])
    queued_bytes = sum(len(chunk) for chunk in audio_queue)

    remaining = _trim_audio_backlog(audio_queue, queued_bytes, AUDIO_QUEUE_MAX_BYTES)

    assert remaining <= AUDIO_QUEUE_MAX_BYTES
    assert audio_queue[-1] == b"c" * 8