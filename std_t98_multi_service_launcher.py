#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import sys
import time
from pathlib import Path

from core.pipeline.multi_stack_dashboard import (
    build_stack_dashboard_renderable,
    print_stack_dashboard,
    rich_dashboard_available,
)
from core.pipeline.stack_supervisor import (
    BACKEND_IMPORT_CHECKS,
    IMPORT_CHECK_CODE,
    IMPORT_CHECK_TIMEOUT_SEC,
    SERVICE_IMPORT_CHECKS,
    SOURCE_PROCESS_NAMES,
    ProcessSpec,
    StackSupervisor,
    _apply_service_payload,
    _apply_status_payload,
    _python_supports_import_checks,
    _resolve_python,
    _spawn_process,
    _tail_log,
    _terminate_processes,
    build_process_specs,
)

try:
    from rich.console import Console
    from rich.live import Live
except ModuleNotFoundError:
    Console = None
    Live = None

# Re-exported so existing imports of these names from the launcher module keep
# working; the implementations now live in core.pipeline.stack_supervisor.
__all__ = [
    "BACKEND_IMPORT_CHECKS",
    "IMPORT_CHECK_CODE",
    "IMPORT_CHECK_TIMEOUT_SEC",
    "SERVICE_IMPORT_CHECKS",
    "SOURCE_PROCESS_NAMES",
    "ProcessSpec",
    "StackSupervisor",
    "build_process_specs",
    "main",
]


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
        "--backend-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument to pass to the RF backend; repeat for several. "
        "Mainly for --backend-arg=--replay --backend-arg=capture.cf32 to run "
        "the whole stack from a recording, with no SDR.",
    )
    parser.add_argument(
        "--passthrough-output",
        action="store_true",
        help="Let child processes write directly to the terminal for debugging.",
    )
    parser.add_argument(
        "--show-debug-metrics",
        action="store_true",
        help="Render per-service and per-channel debug metrics published over the status socket.",
    )
    return parser.parse_args(argv)


def _should_use_rich_dashboard(args):
    return (
        Live is not None
        and Console is not None
        and rich_dashboard_available()
        and sys.stdout.isatty()
        and not args.passthrough_output
    )


def main(argv=None):
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent

    supervisor = StackSupervisor(
        repo_root=repo_root,
        services_only=args.services_only,
        backend_args=args.backend_arg,
        service_python=args.service_python,
        backend_python=args.backend_python,
        passthrough_output=args.passthrough_output,
    )
    supervisor.resolve()

    if args.dry_run:
        for line in supervisor.dry_run_lines():
            print(line)
        return 0

    process_views = supervisor.process_views
    channels = supervisor.channels
    mode_label = supervisor.mode_label
    use_rich_dashboard = _should_use_rich_dashboard(args)
    stopped_by_user = False
    printed_lines = 0
    live_dashboard = None

    def render(refresh_live):
        nonlocal printed_lines
        if refresh_live is not None:
            refresh_live.update(
                build_stack_dashboard_renderable(
                    processes=process_views,
                    channels=channels,
                    mode_label=mode_label,
                    show_debug_metrics=args.show_debug_metrics,
                ),
                refresh=True,
            )
        else:
            printed_lines = print_stack_dashboard(
                processes=process_views,
                channels=channels,
                num_lines_last_time=printed_lines,
                mode_label=mode_label,
                show_debug_metrics=args.show_debug_metrics,
            )

    if use_rich_dashboard:
        assert Live is not None
        assert Console is not None
        live_dashboard = Live(
            build_stack_dashboard_renderable(
                processes=process_views,
                channels=channels,
                mode_label=mode_label,
                show_debug_metrics=args.show_debug_metrics,
            ),
            console=Console(),
            refresh_per_second=4,
            transient=False,
        )
    else:
        printed_lines = print_stack_dashboard(
            processes=process_views,
            channels=channels,
            num_lines_last_time=0,
            mode_label=mode_label,
            show_debug_metrics=args.show_debug_metrics,
        )

    with live_dashboard if live_dashboard is not None else contextlib.nullcontext() as live:
        supervisor.start()
        render(live)

        try:
            while True:
                changed = supervisor.poll(timeout_ms=100)

                if live is not None:
                    render(live)
                elif changed:
                    render(None)

                if supervisor.exit_message is not None:
                    break

                time.sleep(0.2)
        except KeyboardInterrupt:
            stopped_by_user = True
        finally:
            supervisor.stop()

    if stopped_by_user:
        print("\nStopping split multi-channel services...")

    if supervisor.exit_message is not None:
        print(supervisor.exit_message, file=sys.stderr)

    return supervisor.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
