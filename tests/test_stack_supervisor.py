from pathlib import Path

import core.pipeline.stack_supervisor as supervisor_mod
from core.pipeline.stack_supervisor import StackSupervisor, StatusAggregator
from core.pipeline.multi_stack_dashboard import ProcessView
from ipc.message_schema import (
    ControlSquelchPacket,
    STATUS_SOURCE_AUDIO,
    STATUS_SOURCE_PROTOCOL,
    STATUS_SOURCE_RF,
    StatusPacket,
)


def _fake_resolve(monkeypatch, service_python="/fake/env/python", backend_python="/fake/sys/python"):
    def resolver(override, candidate_paths, import_checks, role_name):
        if role_name == "RF backend":
            return Path(backend_python)
        return Path(service_python)

    monkeypatch.setattr(supervisor_mod, "_resolve_python", resolver)
    monkeypatch.setattr(supervisor_mod, "resolve_status_socket_path", lambda channel_count: "/tmp/status.sock")
    monkeypatch.setattr(supervisor_mod, "resolve_control_socket_path", lambda: "/tmp/control.sock")


def test_resolve_builds_specs_and_process_views(monkeypatch):
    _fake_resolve(monkeypatch)
    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"))
    specs = sup.resolve()

    assert [s.name for s in specs] == ["protocol", "secret", "audio", "backend"]
    assert [v.name for v in sup.process_views] == ["protocol", "secret", "audio", "backend"]
    assert specs[0].args == ("--headless", "--status-socket", "/tmp/status.sock")
    assert specs[3].args == ("--control-socket", "/tmp/control.sock")
    assert str(specs[3].python_executable) == "/fake/sys/python"
    assert sup.mode_label == "full-stack"
    assert sup.control_socket_path == "/tmp/control.sock"


def test_resolve_services_only_skips_backend(monkeypatch):
    _fake_resolve(monkeypatch)
    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"), services_only=True)
    specs = sup.resolve()

    assert [s.name for s in specs] == ["protocol", "secret", "audio"]
    assert sup.backend_python is None
    assert sup.mode_label == "services-only"
    # No managed backend, so there is nothing to send live control to.
    assert sup.control_socket_path is None


def test_dry_run_lines_include_backend_args(monkeypatch):
    _fake_resolve(monkeypatch)
    sup = StackSupervisor(
        repo_root=Path("/tmp/std-t98-tools"),
        backend_args=["--replay", "capture.cf32"],
    )
    lines = sup.dry_run_lines()

    assert any(line.startswith("backend:") and "--replay capture.cf32" in line for line in lines)
    assert len(lines) == 4


def test_status_aggregator_folds_channel_and_service_state():
    aggregator = StatusAggregator()
    aggregator.set_process_views([
        ProcessView(name="protocol", python_executable="/x", script_name="protocol.py"),
    ])

    channel_packet = StatusPacket.from_dict(
        sequence=0,
        monotonic_ns=0,
        source=STATUS_SOURCE_PROTOCOL,
        channel_id=5,
        payload_dict={"event": "channel_state", "rx_status": "OPEN", "protocol_status": "Traffic->IPC"},
    )
    assert aggregator.apply_packet(channel_packet) is True
    assert aggregator.channels[5].rx_status == "OPEN"

    service_packet = StatusPacket.from_dict(
        sequence=1,
        monotonic_ns=0,
        source=STATUS_SOURCE_PROTOCOL,
        channel_id=0,
        payload_dict={"event": "service_metrics", "summary": "frames=10 active=1"},
    )
    assert aggregator.apply_packet(service_packet) is True
    assert aggregator.process_views[0].detail == "frames=10 active=1"

    # Audio status on the same channel updates without touching protocol fields.
    audio_packet = StatusPacket.from_dict(
        sequence=2,
        monotonic_ns=0,
        source=STATUS_SOURCE_AUDIO,
        channel_id=5,
        payload_dict={"event": "channel_state", "audio_status": "Playing"},
    )
    assert aggregator.apply_packet(audio_packet) is True
    assert aggregator.channels[5].audio_status == "Playing"
    assert aggregator.channels[5].rx_status == "OPEN"


class _FakeProcess:
    def __init__(self, pid=123):
        self.pid = pid

    def poll(self):
        return None  # still alive


class _FakeStatusReceiver:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    def recv(self, timeout_ms=None):
        return self._payloads.pop(0) if self._payloads else None


def test_start_marks_all_processes_starting_not_running(monkeypatch):
    _fake_resolve(monkeypatch)
    monkeypatch.setattr(supervisor_mod, "_spawn_process", lambda spec, **kw: (_FakeProcess(), None))
    monkeypatch.setattr(supervisor_mod, "UdsSeqpacketReceiver", lambda path: _FakeStatusReceiver([]))
    monkeypatch.setattr(supervisor_mod, "UdsSeqpacketServer", lambda path: object())

    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"))
    sup.start()

    # Alive is not the same as ready: a process (esp. the backend, still
    # opening a possibly slow SDR) starts silent, not RUNNING.
    assert [v.state for v in sup.process_views] == ["STARTING"] * 4


def test_poll_promotes_only_the_process_that_actually_reported_in(monkeypatch):
    _fake_resolve(monkeypatch)
    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"))
    sup.resolve()
    sup._started = True
    sup._processes = [(view.name, _FakeProcess()) for view in sup.process_views]
    for view in sup.process_views:
        view.state = "STARTING"

    packet = StatusPacket.from_dict(
        sequence=0,
        monotonic_ns=0,
        source=STATUS_SOURCE_RF,
        channel_id=0,
        payload_dict={"event": "service_metrics", "summary": "sync=0 ipc=0/0"},
    )
    sup._status_receiver = _FakeStatusReceiver([packet.encode()])

    changed = sup.poll(timeout_ms=0)

    assert changed is True
    by_name = {v.name: v.state for v in sup.process_views}
    assert by_name["backend"] == "RUNNING"
    # No packet arrived from these -- a live-but-silent backend must not drag
    # the others (or itself, before its first report) along with it.
    assert by_name["protocol"] == "STARTING"
    assert by_name["secret"] == "STARTING"
    assert by_name["audio"] == "STARTING"


def test_set_squelch_without_control_server_returns_false():
    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"))
    assert sup.set_squelch(-40.0) is False


def test_set_squelch_sends_encoded_packet_to_control_server():
    sent = []

    class FakeControlServer:
        def send(self, payload):
            sent.append(payload)
            return True

    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"))
    sup._control_server = FakeControlServer()

    assert sup.set_squelch(-42.5) is True
    assert sent == [ControlSquelchPacket(threshold_db=-42.5).encode()]


def test_health_payload_folds_into_process_view():
    aggregator = StatusAggregator()
    aggregator.set_process_views([
        ProcessView(name="backend", python_executable="/x", script_name="backend.py"),
    ])

    healthy = StatusPacket.from_dict(
        sequence=0, monotonic_ns=0, source=STATUS_SOURCE_RF, channel_id=0,
        payload_dict={"event": "health", "ok": True, "summary": "OK"},
    )
    assert aggregator.apply_packet(healthy) is True
    backend_view = aggregator.process_views[0]
    assert backend_view.health_ok is True
    assert backend_view.health == "OK"

    degraded = StatusPacket.from_dict(
        sequence=1, monotonic_ns=0, source=STATUS_SOURCE_RF, channel_id=0,
        payload_dict={"event": "health", "ok": False, "summary": "drift: antenna"},
    )
    assert aggregator.apply_packet(degraded) is True
    assert backend_view.health_ok is False
    assert backend_view.health == "drift: antenna"

    # Re-applying the same payload is a no-op (nothing changed).
    assert aggregator.apply_packet(degraded) is False


def test_health_payload_does_not_touch_service_metrics_detail():
    aggregator = StatusAggregator()
    aggregator.set_process_views([
        ProcessView(name="backend", python_executable="/x", script_name="backend.py"),
    ])

    metrics = StatusPacket.from_dict(
        sequence=0, monotonic_ns=0, source=STATUS_SOURCE_RF, channel_id=0,
        payload_dict={"event": "service_metrics", "summary": "sync=1 ipc=1/0"},
    )
    aggregator.apply_packet(metrics)

    health = StatusPacket.from_dict(
        sequence=1, monotonic_ns=0, source=STATUS_SOURCE_RF, channel_id=0,
        payload_dict={"event": "health", "ok": True, "summary": "OK"},
    )
    aggregator.apply_packet(health)

    backend_view = aggregator.process_views[0]
    assert backend_view.detail == "sync=1 ipc=1/0"
    assert backend_view.health == "OK"
