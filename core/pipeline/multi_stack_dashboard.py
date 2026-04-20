import sys
import time
from dataclasses import dataclass, field

try:
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError:
    box = None
    Group = None
    Panel = None
    Table = None
    Text = None


@dataclass
class ProcessView:
    name: str
    python_executable: str
    script_name: str
    pid: int | None = None
    state: str = "STARTING"
    detail: str = ""


@dataclass
class ChannelView:
    channel_id: int
    rx_status: str = "CLOSE"
    protocol_status: str = "Waiting"
    audio_status: str = "Idle"
    secret_status: str = "Idle"
    secret_key: int = 0
    secret_cache_keys: tuple[int, ...] = ()
    rf_debug: str = ""
    protocol_debug: str = ""
    audio_debug: str = ""
    csm: str | None = None
    sacch: dict | None = None
    last_update: float = field(default_factory=time.time)


def rich_dashboard_available() -> bool:
    return all(component is not None for component in (box, Group, Panel, Table, Text))


def _format_process_state(state: str) -> str:
    return f"{state:<8}"


def _format_rx_state(state: str) -> str:
    return "OPEN" if state == "OPEN" else "CLOSE"


def _format_sacch(channel: ChannelView) -> str:
    if not channel.sacch:
        return "Waiting..."

    stat_val = channel.sacch.get("CallStat", 0)
    stat_desc = "Normal" if stat_val == 0 else "Secret" if stat_val == 1 else str(stat_val)
    return (
        f"UC: {channel.sacch.get('UserCode', 0):03d} | "
        f"Maker: {channel.sacch.get('MakerCode', 0):03d} | "
        f"Stat: {stat_desc}"
    )


def _format_csm(channel: ChannelView) -> str:
    return channel.csm if channel.csm else "Waiting..."


def _format_secret_cache(channels) -> str:
    cache_keys = next((channel.secret_cache_keys for channel in channels.values() if channel.secret_cache_keys), ())
    return ", ".join(f"{key:05d}" for key in cache_keys) if cache_keys else "(empty)"


def _process_state_text(state: str):
    if not rich_dashboard_available():
        return _format_process_state(state)

    style = "bold yellow"
    if state == "RUNNING":
        style = "bold green"
    elif state == "EXITED":
        style = "bold red"
    return Text(_format_process_state(state), style=style)


def _rx_state_text(state: str):
    if not rich_dashboard_available():
        return _format_rx_state(state)

    style = "bold green" if state == "OPEN" else "grey58"
    return Text(_format_rx_state(state), style=style)


def _channel_border_style(state: str) -> str:
    return "green" if state == "OPEN" else "grey42"


def build_stack_dashboard_renderable(processes, channels, mode_label="full-stack", show_debug_metrics=False):
    if not rich_dashboard_available():
        return _build_plain_dashboard_text(processes, channels, mode_label, show_debug_metrics)

    header = Panel(
        Text(f"Mode: {mode_label} | Ctrl+C to stop", style="bold"),
        title="STD-T98 Multi Receiver",
        border_style="blue",
        expand=True,
    )

    process_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    process_table.add_column("Process", style="cyan", no_wrap=True)
    process_table.add_column("State", no_wrap=True)
    process_table.add_column("PID", justify="right", no_wrap=True)
    process_table.add_column("Python", overflow="fold")
    if show_debug_metrics:
        process_table.add_column("Metrics", overflow="fold")

    for process_view in processes:
        row = [
            process_view.name,
            _process_state_text(process_view.state),
            str(process_view.pid) if process_view.pid is not None else "-",
            process_view.python_executable,
        ]
        if show_debug_metrics:
            row.append(process_view.detail or "-")
        process_table.add_row(*row)

    cache_panel = Panel(
        _format_secret_cache(channels),
        title="Secret Cache",
        border_style="magenta",
        expand=True,
    )

    if not channels:
        channels_renderable = Panel(
            "Waiting for channel activity...",
            title="Channels",
            border_style="grey42",
            expand=True,
        )
    else:
        channel_panels = []
        for channel_id in sorted(channels):
            channel = channels[channel_id]
            age_sec = max(0.0, time.time() - channel.last_update)

            channel_grid = Table.grid(expand=True)
            channel_grid.add_column(style="bold cyan", no_wrap=True)
            channel_grid.add_column(ratio=1)
            channel_grid.add_row("RX", _rx_state_text(channel.rx_status))
            channel_grid.add_row("Protocol", channel.protocol_status)
            channel_grid.add_row("Audio", channel.audio_status)
            channel_grid.add_row("Age", f"{age_sec:4.1f}s")
            channel_grid.add_row("CSM", _format_csm(channel))
            channel_grid.add_row("SACCH", _format_sacch(channel))
            channel_grid.add_row("Secret", f"key={channel.secret_key:05d} | {channel.secret_status}")

            if show_debug_metrics and channel.rf_debug:
                channel_grid.add_row("RF", channel.rf_debug)
            if show_debug_metrics and channel.protocol_debug:
                channel_grid.add_row("Proto", channel.protocol_debug)
            if show_debug_metrics and channel.audio_debug:
                channel_grid.add_row("AudioDbg", channel.audio_debug)

            channel_panels.append(
                Panel(
                    channel_grid,
                    title=f"CH {channel_id:02d}",
                    border_style=_channel_border_style(channel.rx_status),
                    expand=True,
                )
            )

        channels_renderable = Group(*channel_panels)

    return Group(header, process_table, cache_panel, channels_renderable)


def _build_plain_dashboard_text(processes, channels, mode_label="full-stack", show_debug_metrics=False):
    lines = [
        "=" * 70,
        "[STD-T98 Multi Receiver]",
        f"Mode: {mode_label} | Ctrl+C to stop",
        "=" * 70,
    ]

    for process_view in processes:
        pid = process_view.pid if process_view.pid is not None else "-"
        lines.append(
            f"[{process_view.name:<8}] {_format_process_state(process_view.state)} pid={pid} via {process_view.python_executable}"
        )
        if show_debug_metrics and process_view.detail:
            lines.append(f"  metrics: {process_view.detail}")

    lines.append("-" * 70)
    lines.append(f"Secret Cache: {_format_secret_cache(channels)}")
    lines.append("-" * 70)

    if not channels:
        lines.append("  Waiting for channel activity...")
        return "\n".join(lines)

    for channel_id in sorted(channels):
        channel = channels[channel_id]
        age_sec = max(0.0, time.time() - channel.last_update)
        lines.append(
            f"[CH {channel_id:02d}] RX: [{_format_rx_state(channel.rx_status):^5}] | "
            f"Proto: {channel.protocol_status} | Audio: {channel.audio_status} | Age: {age_sec:4.1f}s"
        )
        lines.append(f"  > CSM   : {_format_csm(channel)}")
        lines.append(f"  > SACCH : {_format_sacch(channel)}")
        lines.append(f"  > Secret: key={channel.secret_key:05d} | {channel.secret_status}")
        if show_debug_metrics and channel.rf_debug:
            lines.append(f"  > RF    : {channel.rf_debug}")
        if show_debug_metrics and channel.protocol_debug:
            lines.append(f"  > Proto : {channel.protocol_debug}")
        if show_debug_metrics and channel.audio_debug:
            lines.append(f"  > Audio : {channel.audio_debug}")
        lines.append("-" * 70)

    return "\n".join(lines)


def _colorize_process_state(state: str) -> str:
    if state == "RUNNING":
        return f"\033[1;32m{state:<8}\033[0m"
    if state == "EXITED":
        return f"\033[1;31m{state:<8}\033[0m"
    return f"\033[1;33m{state:<8}\033[0m"


def _colorize_rx_state(state: str) -> str:
    if state == "OPEN":
        return "\033[1;32m OPEN  \033[0m"
    return "\033[1;30m CLOSE \033[0m"


def print_stack_dashboard(processes, channels, num_lines_last_time, mode_label="full-stack", show_debug_metrics=False):
    if rich_dashboard_available():
        rendered = _build_plain_dashboard_text(processes, channels, mode_label, show_debug_metrics)
    else:
        rendered = _build_plain_dashboard_text(processes, channels, mode_label, show_debug_metrics)

    if num_lines_last_time > 0:
        sys.stdout.write(f"\033[{num_lines_last_time}A")

    sys.stdout.write("\033[J")
    line_count = 0
    for line in rendered.splitlines():
        sys.stdout.write(f"\033[K{line}\n")
        line_count += 1

    sys.stdout.flush()
    return line_count