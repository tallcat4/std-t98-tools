import signal
from pathlib import Path
from types import SimpleNamespace

from std_t98_multi_service_launcher import (
    ProcessSpec,
    _apply_status_payload,
    _spawn_process,
    _terminate_processes,
    build_process_specs,
)


def test_build_process_specs_for_full_stack():
    repo_root = Path("/tmp/std-t98-tools")
    service_python = Path("/tmp/std-t98-tools/env/bin/python")
    backend_python = Path("/usr/bin/python")
    status_socket_path = "/tmp/std_t98_multi_status.sock"

    process_specs = build_process_specs(
        repo_root=repo_root,
        service_python=service_python,
        status_socket_path=status_socket_path,
        backend_python=backend_python,
        include_backend=True,
    )

    assert [process_spec.name for process_spec in process_specs] == ["protocol", "secret", "audio", "backend"]
    assert process_specs[0].python_executable == service_python
    assert process_specs[0].args == ("--headless", "--status-socket", status_socket_path)
    assert process_specs[1].script_path == repo_root / "std_t98_multi_secret_service.py"
    assert process_specs[1].args == ("--headless", "--status-socket", status_socket_path)
    assert process_specs[2].script_path == repo_root / "std_t98_multi_audio_service.py"
    assert process_specs[2].args == ("--headless", "--status-socket", status_socket_path)
    assert process_specs[3].python_executable == backend_python
    assert process_specs[3].script_path == repo_root / "std_t98_30ch_multi_rf_backend.py"


def test_build_process_specs_for_services_only():
    repo_root = Path("/tmp/std-t98-tools")
    service_python = Path("/tmp/std-t98-tools/env/bin/python")
    status_socket_path = "/tmp/std_t98_multi_status.sock"

    process_specs = build_process_specs(
        repo_root=repo_root,
        service_python=service_python,
        status_socket_path=status_socket_path,
        include_backend=False,
    )

    assert [process_spec.name for process_spec in process_specs] == ["protocol", "secret", "audio"]
    assert all(process_spec.python_executable == service_python for process_spec in process_specs)
    assert all(process_spec.args == ("--headless", "--status-socket", status_socket_path) for process_spec in process_specs)


def test_apply_status_payload_resets_stale_audio_state_on_close():
    channels = {}

    changed = _apply_status_payload(
        channels,
        source=1,
        channel_id=4,
        payload_dict={
            "event": "channel_state",
            "rx_status": "OPEN",
            "protocol_status": "Traffic->IPC",
            "csm": "200446991",
            "sacch": {"CallStat": 0, "UserCode": 1, "MakerCode": 2},
        },
    )
    assert changed is True

    changed = _apply_status_payload(
        channels,
        source=2,
        channel_id=4,
        payload_dict={
            "event": "channel_state",
            "audio_status": "Playing",
        },
    )
    assert changed is True
    assert channels[4].audio_status == "Playing"

    changed = _apply_status_payload(
        channels,
        source=1,
        channel_id=4,
        payload_dict={
            "event": "channel_state",
            "rx_status": "CLOSE",
            "protocol_status": "Idle",
        },
    )
    assert changed is True
    assert channels[4].audio_status == "Idle"


def test_spawn_process_can_suppress_start_announcement(monkeypatch, capsys):
    popen_calls = []

    def fake_popen(command, cwd, env, stdout, stderr):
        popen_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr("std_t98_multi_service_launcher.subprocess.Popen", fake_popen)

    process = _spawn_process(
        ProcessSpec(
            name="protocol",
            python_executable=Path("/tmp/service-python"),
            script_path=Path("/tmp/std-t98-tools/std_t98_multi_protocol_service.py"),
            args=("--headless",),
        ),
        announce_start=False,
    )

    captured = capsys.readouterr()

    assert process.pid == 4321
    assert captured.out == ""
    assert len(popen_calls) == 1


def test_terminate_processes_prefers_sigint_for_graceful_shutdown():
    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.signals = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            self.signals.append(sig)
            if sig == signal.SIGINT:
                self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def terminate(self):
            self.terminate_calls += 1
            self.returncode = 0

        def kill(self):
            self.kill_calls += 1
            self.returncode = -9

    process = FakeProcess()

    _terminate_processes([process])

    assert process.signals == [signal.SIGINT]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_apply_status_payload_updates_secret_fields():
    channels = {}

    changed = _apply_status_payload(
        channels,
        source=2,
        channel_id=7,
        payload_dict={
            "event": "channel_state",
            "audio_status": "Playing",
            "secret_status": "Search Pending",
            "secret_key": 0,
        },
    )

    assert changed is True
    assert channels[7].audio_status == "Playing"
    assert channels[7].secret_status == "Search Pending"
    assert channels[7].secret_key == 0

    changed = _apply_status_payload(
        channels,
        source=4,
        channel_id=7,
        payload_dict={
            "event": "channel_state",
            "secret_status": "Global Cache Hit",
            "secret_key": 42,
            "secret_cache_keys": [42, 77],
        },
    )

    assert changed is True
    assert channels[7].secret_status == "Global Cache Hit"
    assert channels[7].secret_key == 42
    assert channels[7].secret_cache_keys == (42, 77)