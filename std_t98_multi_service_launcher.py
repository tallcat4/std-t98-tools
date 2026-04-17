#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core.pipeline.multi_stack_dashboard import ChannelView, ProcessView, print_stack_dashboard
from ipc.message_schema import STATUS_SOURCE_AUDIO, STATUS_SOURCE_PROTOCOL, STATUS_SOURCE_SECRET, StatusPacket
from ipc.transport.uds_seqpacket import UdsSeqpacketReceiver, resolve_status_socket_path


BACKEND_IMPORT_CHECKS = ("from gnuradio import gr", "from gnuradio import soapy")
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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Launch the split STD-T98 multi-channel stack.")
    parser.add_argument(
        "--services-only",
        action="store_true",
        help="Launch only protocol and audio services, assuming the RF backend is started separately.",
    )
    parser.add_argument(
        "--backend-python",
        help="Path to the Python executable used for the RF backend.",
    )
    parser.add_argument(
        "--service-python",
        help="Path to the Python executable used for protocol and audio services.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without starting any child process.",
    )
    parser.add_argument(
        "--passthrough-output",
        action="store_true",
        help="Let child processes write directly to the terminal for debugging.",
    )
    return parser.parse_args(argv)


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


def build_process_specs(repo_root, service_python, status_socket_path=None, backend_python=None, include_backend=True):
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
        process_specs.append(ProcessSpec("backend", Path(backend_python), repo_root / "std_t98_30ch_multi_rf_backend.py"))

    return process_specs


def _spawn_process(process_spec, passthrough_output=False, announce_start=False):
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [str(process_spec.python_executable), str(process_spec.script_path), *process_spec.args],
        cwd=process_spec.script_path.parent,
        env=child_env,
        stdout=None if passthrough_output else subprocess.DEVNULL,
        stderr=None if passthrough_output else subprocess.STDOUT,
    )
    if announce_start:
        print(
            f"Started {process_spec.name} with {process_spec.python_executable} "
            f"(pid {process.pid})"
        )
    return process


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

    if changed:
        channel.last_update = time.time()
    return changed


def main(argv=None):
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    service_python = _resolve_python(
        override=args.service_python,
        candidate_paths=[
            repo_root / "env/bin/python",
            sys.executable,
            shutil.which("python3"),
            shutil.which("python"),
            "/usr/bin/python",
        ],
        import_checks=SERVICE_IMPORT_CHECKS,
        role_name="protocol/audio services",
    )

    backend_python = None
    if not args.services_only:
        backend_python = _resolve_python(
            override=args.backend_python,
            candidate_paths=[
                "/usr/bin/python",
                sys.executable,
                shutil.which("python3"),
                shutil.which("python"),
            ],
            import_checks=BACKEND_IMPORT_CHECKS,
            role_name="RF backend",
        )

    status_socket_path = resolve_status_socket_path(channel_count=30)

    process_specs = build_process_specs(
        repo_root=repo_root,
        service_python=service_python,
        status_socket_path=status_socket_path,
        backend_python=backend_python,
        include_backend=not args.services_only,
    )

    if args.dry_run:
        for process_spec in process_specs:
            extra = " ".join(process_spec.args)
            if extra:
                print(f"{process_spec.name}: {process_spec.python_executable} {process_spec.script_path} {extra}")
            else:
                print(f"{process_spec.name}: {process_spec.python_executable} {process_spec.script_path}")
        return 0

    status_receiver = UdsSeqpacketReceiver(status_socket_path)

    processes = []
    exit_code = 0
    process_views = []
    channels = {}

    for process_spec in process_specs:
        process_views.append(
            ProcessView(
                name=process_spec.name,
                python_executable=str(process_spec.python_executable),
                script_name=process_spec.script_path.name,
            )
        )

    printed_lines = print_stack_dashboard(
        processes=process_views,
        channels=channels,
        num_lines_last_time=0,
        mode_label="services-only" if args.services_only else "full-stack",
    )

    for process_spec, process_view in zip(process_specs, process_views):
        process = _spawn_process(
            process_spec,
            passthrough_output=args.passthrough_output,
            announce_start=False,
        )
        processes.append((process_spec.name, process))
        process_view.pid = process.pid
        process_view.state = "RUNNING"

    printed_lines = print_stack_dashboard(
        processes=process_views,
        channels=channels,
        num_lines_last_time=printed_lines,
        mode_label="services-only" if args.services_only else "full-stack",
    )

    try:
        while True:
            dashboard_changed = False

            payload = status_receiver.recv(timeout_ms=100)
            while payload is not None:
                packet = StatusPacket.decode(payload)
                dashboard_changed = _apply_status_payload(channels, packet.source, packet.channel_id, packet.to_dict()) or dashboard_changed
                payload = status_receiver.recv(timeout_ms=0)

            for name, process in processes:
                return_code = process.poll()
                process_view = next(view for view in process_views if view.name == name)
                if return_code is not None:
                    process_view.state = "EXITED"
                    printed_lines = print_stack_dashboard(
                        processes=process_views,
                        channels=channels,
                        num_lines_last_time=printed_lines,
                        mode_label="services-only" if args.services_only else "full-stack",
                    )
                    print(f"{name} service exited with status {return_code}", file=sys.stderr)
                    exit_code = return_code or 1
                    return exit_code
                if process_view.state != "RUNNING":
                    process_view.state = "RUNNING"
                    dashboard_changed = True

            if dashboard_changed:
                printed_lines = print_stack_dashboard(
                    processes=process_views,
                    channels=channels,
                    num_lines_last_time=printed_lines,
                    mode_label="services-only" if args.services_only else "full-stack",
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping split multi-channel services...")
        return exit_code
    finally:
        status_receiver.close()
        _terminate_processes([process for _, process in processes])


if __name__ == "__main__":
    raise SystemExit(main())