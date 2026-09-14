#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable supervisor for the split STD-T98 multi-channel stack.

This holds everything the terminal launcher and the desktop GUI share: locating
the two Python interpreters, building the process specs, spawning and tearing
down the children, and folding the status socket into the ChannelView /
ProcessView state both front-ends render. The front-ends differ only in how they
draw that state and drive the poll loop.
"""

import os
import signal
import shutil
import subprocess
import tempfile
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core.pipeline.multi_stack_dashboard import ChannelView, ProcessView
from ipc.message_schema import (
    STATUS_SOURCE_AUDIO,
    STATUS_SOURCE_PROTOCOL,
    STATUS_SOURCE_RF,
    STATUS_SOURCE_SECRET,
    ControlSquelchPacket,
    StatusPacket,
)
from ipc.transport.uds_seqpacket import (
    UdsSeqpacketReceiver,
    UdsSeqpacketServer,
    resolve_control_socket_path,
    resolve_status_socket_path,
)


BACKEND_IMPORT_CHECKS = ("from gnuradio import gr", "from gnuradio import uhd")
SERVICE_IMPORT_CHECKS = (
    "import pyambelib",
    "import sounddevice",
    "import torch",
    "from safetensors.torch import load_file",
)
IMPORT_CHECK_CODE = (
    "import sys\n"
    "for statement in sys.argv[1:]:\n"
    "    try:\n"
    "        exec(statement, {})\n"
    "    except Exception:\n"
    "        sys.exit(1)\n"
    "sys.exit(0)\n"
)
IMPORT_CHECK_TIMEOUT_SEC = 5.0
SOURCE_PROCESS_NAMES = {
    STATUS_SOURCE_PROTOCOL: "protocol",
    STATUS_SOURCE_AUDIO: "audio",
    STATUS_SOURCE_SECRET: "secret",
    STATUS_SOURCE_RF: "backend",
}


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    python_executable: Path
    script_path: Path
    args: tuple[str, ...] = ()


def _terminate_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)

    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _existing_candidates(candidate_paths):
    resolved = []
    seen = set()
    for candidate in candidate_paths:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            resolved.append(path)
    return resolved


def _python_supports_import_checks(python_executable, import_checks):
    for import_check in import_checks:
        try:
            result = subprocess.run(
                [str(python_executable), "-c", IMPORT_CHECK_CODE, import_check],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=IMPORT_CHECK_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return False

        if result.returncode != 0:
            return False

    return True


def _resolve_python(override, candidate_paths, import_checks, role_name):
    candidates = _existing_candidates(([override] if override else []) + list(candidate_paths))
    for candidate in candidates:
        if _python_supports_import_checks(candidate, import_checks):
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates) or "<none>"
    raise RuntimeError(
        f"Could not find a Python interpreter for {role_name} with required imports {import_checks}. "
        f"Searched: {searched}"
    )


def build_process_specs(repo_root, service_python, status_socket_path=None, backend_python=None, include_backend=True, backend_args=(), control_socket_path=None):
    service_python = Path(service_python)
    process_specs = [
        ProcessSpec(
            "protocol",
            service_python,
            repo_root / "std_t98_multi_protocol_service.py",
            args=("--headless", "--status-socket", status_socket_path) if status_socket_path else ("--headless",),
        ),
        ProcessSpec(
            "secret",
            service_python,
            repo_root / "std_t98_multi_secret_service.py",
            args=("--headless", "--status-socket", status_socket_path) if status_socket_path else ("--headless",),
        ),
        ProcessSpec(
            "audio",
            service_python,
            repo_root / "std_t98_multi_audio_service.py",
            args=("--headless", "--status-socket", status_socket_path) if status_socket_path else ("--headless",),
        ),
    ]

    if include_backend:
        if backend_python is None:
            raise ValueError("backend_python is required when include_backend is True")
        control_args = ("--control-socket", control_socket_path) if control_socket_path else ()
        process_specs.append(ProcessSpec(
            "backend", Path(backend_python),
            repo_root / "std_t98_30ch_multi_rf_backend.py",
            args=(*backend_args, *control_args),
        ))

    return process_specs


def _spawn_process(process_spec, passthrough_output=False, announce_start=False):
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    log_file = None
    if passthrough_output:
        stdout = None
        stderr = None
    else:
        # Not DEVNULL: keep the output so a non-zero exit can be explained.
        log_file = tempfile.NamedTemporaryFile(
            mode="w+", prefix=f"std-t98-{process_spec.name}-", suffix=".log")
        stdout = log_file
        stderr = subprocess.STDOUT
    process = subprocess.Popen(
        [str(process_spec.python_executable), str(process_spec.script_path), *process_spec.args],
        cwd=process_spec.script_path.parent,
        env=child_env,
        stdout=stdout,
        stderr=stderr,
    )
    if announce_start:
        print(
            f"Started {process_spec.name} with {process_spec.python_executable} "
            f"(pid {process.pid})"
        )
    return process, log_file


def _tail_log(log_file, max_lines=8):
    """Last few non-empty lines of a captured child log, for a crash message."""
    if log_file is None:
        return ""
    try:
        log_file.flush()
        log_file.seek(0)
        lines = [line.rstrip() for line in log_file if line.strip()]
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def _apply_status_payload(channels, source, channel_id, payload_dict):
    if payload_dict.get("event") != "channel_state":
        return False

    channel = channels.setdefault(channel_id, ChannelView(channel_id=channel_id))
    changed = False

    if source == STATUS_SOURCE_PROTOCOL:
        for field_name, payload_key in (
            ("rx_status", "rx_status"),
            ("protocol_status", "protocol_status"),
            ("csm", "csm"),
            ("sacch", "sacch"),
            ("protocol_debug", "protocol_debug"),
        ):
            new_value = payload_dict.get(payload_key, getattr(channel, field_name))
            if getattr(channel, field_name) != new_value:
                setattr(channel, field_name, new_value)
                changed = True

        if channel.rx_status == "CLOSE" or channel.protocol_status in ("Idle", "Sync Burst"):
            if channel.audio_status != "Idle":
                channel.audio_status = "Idle"
                changed = True
            if channel.secret_status != "Idle":
                channel.secret_status = "Idle"
                changed = True
    elif source == STATUS_SOURCE_AUDIO:
        new_audio_status = payload_dict.get("audio_status", channel.audio_status)
        if channel.audio_status != new_audio_status:
            channel.audio_status = new_audio_status
            changed = True

        new_audio_debug = payload_dict.get("audio_debug", channel.audio_debug)
        if channel.audio_debug != new_audio_debug:
            channel.audio_debug = new_audio_debug
            changed = True

        new_secret_status = payload_dict.get("secret_status", channel.secret_status)
        if channel.secret_status != new_secret_status:
            channel.secret_status = new_secret_status
            changed = True

        new_secret_key = payload_dict.get("secret_key", channel.secret_key)
        if channel.secret_key != new_secret_key:
            channel.secret_key = new_secret_key
            changed = True
    elif source == STATUS_SOURCE_SECRET:
        new_secret_status = payload_dict.get("secret_status", channel.secret_status)
        if channel.secret_status != new_secret_status:
            channel.secret_status = new_secret_status
            changed = True

        new_secret_key = payload_dict.get("secret_key", channel.secret_key)
        if channel.secret_key != new_secret_key:
            channel.secret_key = new_secret_key
            changed = True

        new_secret_cache_keys = tuple(payload_dict.get("secret_cache_keys", channel.secret_cache_keys))
        if channel.secret_cache_keys != new_secret_cache_keys:
            channel.secret_cache_keys = new_secret_cache_keys
            changed = True
    elif source == STATUS_SOURCE_RF:
        new_rf_debug = payload_dict.get("rf_debug", channel.rf_debug)
        if channel.rf_debug != new_rf_debug:
            channel.rf_debug = new_rf_debug
            changed = True

    if changed:
        channel.last_update = time.time()
    return changed


def _apply_service_payload(process_views_by_name, source, payload_dict):
    if payload_dict.get("event") != "service_metrics":
        return False

    process_name = SOURCE_PROCESS_NAMES.get(source)
    if process_name is None:
        return False

    process_view = process_views_by_name.get(process_name)
    if process_view is None:
        return False

    summary = payload_dict.get("summary", "")
    if process_view.detail == summary:
        return False

    process_view.detail = summary
    return True


def _apply_health_payload(process_views_by_name, source, payload_dict):
    if payload_dict.get("event") != "health":
        return False

    process_name = SOURCE_PROCESS_NAMES.get(source)
    if process_name is None:
        return False

    process_view = process_views_by_name.get(process_name)
    if process_view is None:
        return False

    changed = False
    ok = payload_dict.get("ok")
    if process_view.health_ok != ok:
        process_view.health_ok = ok
        changed = True

    summary = payload_dict.get("summary", "")
    if process_view.health != summary:
        process_view.health = summary
        changed = True

    return changed


class StatusAggregator:
    """Owns the shared channel / process state and folds status packets into it.

    Both front-ends read ``channels`` and ``process_views`` to render; both feed
    decoded StatusPackets through ``apply_packet``. Keeping the mutation in one
    place is what lets the GUI and the terminal show identical state.
    """

    def __init__(self):
        self.channels: dict[int, ChannelView] = {}
        self.process_views: list[ProcessView] = []
        self._process_views_by_name: dict[str, ProcessView] = {}

    def set_process_views(self, process_views):
        self.process_views = list(process_views)
        self._process_views_by_name = {view.name: view for view in self.process_views}

    def apply_packet(self, packet) -> bool:
        payload_dict = packet.to_dict()
        changed = _apply_service_payload(self._process_views_by_name, packet.source, payload_dict)
        changed = _apply_health_payload(self._process_views_by_name, packet.source, payload_dict) or changed
        changed = _apply_status_payload(self.channels, packet.source, packet.channel_id, payload_dict) or changed
        return changed


class StackSupervisor:
    """Lifecycle of the split stack, independent of how it is rendered.

    Usage is the same from either front-end::

        supervisor = StackSupervisor(repo_root, ...)
        supervisor.resolve()          # locate interpreters, build specs
        supervisor.start()            # bind status socket, spawn children
        while supervisor.exit_message is None:
            supervisor.poll(timeout_ms)   # drain socket, watch for exits
            render(supervisor.process_views, supervisor.channels)
        supervisor.stop()

    The terminal launcher drives ``poll`` from a blocking while loop; the GUI
    drives it from a QTimer with ``timeout_ms=0`` so the event loop never stalls.
    """

    def __init__(
        self,
        repo_root,
        *,
        services_only=False,
        backend_args=(),
        service_python=None,
        backend_python=None,
        passthrough_output=False,
        channel_count=30,
    ):
        self.repo_root = Path(repo_root)
        self.services_only = services_only
        self.backend_args = tuple(backend_args)
        self.service_python_override = service_python
        self.backend_python_override = backend_python
        self.passthrough_output = passthrough_output
        self.channel_count = channel_count

        self.service_python: Path | None = None
        self.backend_python: Path | None = None
        self.status_socket_path: str | None = None
        self.control_socket_path: str | None = None
        self.process_specs: list[ProcessSpec] = []

        self.aggregator = StatusAggregator()
        self._status_receiver = None
        self._control_server = None
        self._processes: list[tuple[str, subprocess.Popen]] = []
        self._process_logs: dict[str, object] = {}
        self.exit_code = 0
        self.exit_message: str | None = None
        self._started = False

    @property
    def mode_label(self) -> str:
        return "services-only" if self.services_only else "full-stack"

    @property
    def channels(self):
        return self.aggregator.channels

    @property
    def process_views(self):
        return self.aggregator.process_views

    def resolve(self):
        """Locate interpreters and build the process specs and initial views.

        Raises RuntimeError if a required interpreter cannot be found, matching
        the launcher's previous behaviour of failing before anything is spawned.
        """
        self.service_python = _resolve_python(
            override=self.service_python_override,
            candidate_paths=[
                self.repo_root / "env/bin/python",
                sys.executable,
                shutil.which("python3"),
                shutil.which("python"),
                "/usr/bin/python",
            ],
            import_checks=SERVICE_IMPORT_CHECKS,
            role_name="protocol/audio services",
        )

        if not self.services_only:
            self.backend_python = _resolve_python(
                override=self.backend_python_override,
                candidate_paths=[
                    "/usr/bin/python",
                    sys.executable,
                    shutil.which("python3"),
                    shutil.which("python"),
                ],
                import_checks=BACKEND_IMPORT_CHECKS,
                role_name="RF backend",
            )

        self.status_socket_path = resolve_status_socket_path(channel_count=self.channel_count)
        self.control_socket_path = (
            None if self.services_only else resolve_control_socket_path()
        )

        self.process_specs = build_process_specs(
            repo_root=self.repo_root,
            service_python=self.service_python,
            status_socket_path=self.status_socket_path,
            backend_python=self.backend_python,
            include_backend=not self.services_only,
            backend_args=self.backend_args,
            control_socket_path=self.control_socket_path,
        )

        self.aggregator.set_process_views([
            ProcessView(
                name=spec.name,
                python_executable=str(spec.python_executable),
                script_name=spec.script_path.name,
            )
            for spec in self.process_specs
        ])
        return self.process_specs

    def dry_run_lines(self) -> list[str]:
        if not self.process_specs:
            self.resolve()
        lines = []
        for spec in self.process_specs:
            extra = " ".join(spec.args)
            if extra:
                lines.append(f"{spec.name}: {spec.python_executable} {spec.script_path} {extra}")
            else:
                lines.append(f"{spec.name}: {spec.python_executable} {spec.script_path}")
        return lines

    def start(self):
        if self._started:
            return
        if not self.process_specs:
            self.resolve()

        self._status_receiver = UdsSeqpacketReceiver(self.status_socket_path)
        if self.control_socket_path:
            self._control_server = UdsSeqpacketServer(self.control_socket_path)

        for spec in self.process_specs:
            process, log_file = _spawn_process(
                spec,
                passthrough_output=self.passthrough_output,
                announce_start=False,
            )
            self._processes.append((spec.name, process))
            self._process_logs[spec.name] = log_file
            process_view = self.aggregator._process_views_by_name[spec.name]
            process_view.pid = process.pid
            # STARTING until the process actually reports in over the status
            # socket -- for the backend that means the SDR is open and the
            # flowgraph is pulling samples, not just that the OS process
            # exists. A slow/contended device bring-up (e.g. a B210 on USB2
            # competing with the secret service's model load) can leave the
            # process alive but silent for a while; a bare "RUNNING" badge
            # would hide exactly that.
            process_view.state = "STARTING"

        self._started = True

    def poll(self, timeout_ms=100) -> bool:
        """Drain the status socket and check the children once.

        Returns True if the rendered state changed. Sets ``exit_message`` and
        ``exit_code`` when a child exits; the caller should stop looping then.
        """
        if not self._started:
            return False

        changed = False

        payload = self._status_receiver.recv(timeout_ms=timeout_ms)
        while payload is not None:
            packet = StatusPacket.decode(payload)
            changed = self.aggregator.apply_packet(packet) or changed

            # Any status packet at all proves this process has gotten far
            # enough to report in (for the backend, that its status publisher
            # is running inside a live work() call -- i.e. actually streaming).
            process_view = self.aggregator._process_views_by_name.get(
                SOURCE_PROCESS_NAMES.get(packet.source)
            )
            if process_view is not None and process_view.state == "STARTING":
                process_view.state = "RUNNING"
                changed = True

            payload = self._status_receiver.recv(timeout_ms=0)

        for name, process in self._processes:
            return_code = process.poll()
            process_view = self.aggregator._process_views_by_name[name]
            if return_code is not None:
                process_view.state = "EXITED"
                self.exit_code = return_code or 1
                self.exit_message = f"{name} service exited with status {return_code}"
                tail = _tail_log(self._process_logs.get(name))
                if tail:
                    self.exit_message += f"\n{tail}"
                changed = True
                break

        return changed

    def set_squelch(self, threshold_db: float) -> bool:
        """Push a live squelch threshold to the running backend, if any.

        Returns False harmlessly if there is no managed backend or it has not
        connected to the control socket yet (e.g. services-only mode, or the
        stack was just started).
        """
        if self._control_server is None:
            return False
        return self._control_server.send(ControlSquelchPacket(threshold_db=threshold_db).encode())

    def stop(self):
        if self._status_receiver is not None:
            self._status_receiver.close()
            self._status_receiver = None
        if self._control_server is not None:
            self._control_server.close()
            self._control_server = None
        _terminate_processes([process for _, process in self._processes])
        self._processes = []
        self._started = False
