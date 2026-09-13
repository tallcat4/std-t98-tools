from pathlib import Path

import core.pipeline.stack_supervisor as supervisor_mod
from core.pipeline.stack_supervisor import StackSupervisor, StatusAggregator
from core.pipeline.multi_stack_dashboard import ProcessView
from ipc.message_schema import STATUS_SOURCE_PROTOCOL, STATUS_SOURCE_AUDIO, StatusPacket


def _fake_resolve(monkeypatch, service_python="/fake/env/python", backend_python="/fake/sys/python"):
    def resolver(override, candidate_paths, import_checks, role_name):
        if role_name == "RF backend":
            return Path(backend_python)
        return Path(service_python)

    monkeypatch.setattr(supervisor_mod, "_resolve_python", resolver)
    monkeypatch.setattr(supervisor_mod, "resolve_status_socket_path", lambda channel_count: "/tmp/status.sock")


def test_resolve_builds_specs_and_process_views(monkeypatch):
    _fake_resolve(monkeypatch)
    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"))
    specs = sup.resolve()

    assert [s.name for s in specs] == ["protocol", "secret", "audio", "backend"]
    assert [v.name for v in sup.process_views] == ["protocol", "secret", "audio", "backend"]
    assert specs[0].args == ("--headless", "--status-socket", "/tmp/status.sock")
    assert str(specs[3].python_executable) == "/fake/sys/python"
    assert sup.mode_label == "full-stack"


def test_resolve_services_only_skips_backend(monkeypatch):
    _fake_resolve(monkeypatch)
    sup = StackSupervisor(repo_root=Path("/tmp/std-t98-tools"), services_only=True)
    specs = sup.resolve()

    assert [s.name for s in specs] == ["protocol", "secret", "audio"]
    assert sup.backend_python is None
    assert sup.mode_label == "services-only"


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
