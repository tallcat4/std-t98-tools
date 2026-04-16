import sys
import time
from dataclasses import dataclass, field


@dataclass
class ProcessView:
    name: str
    python_executable: str
    script_name: str
    pid: int | None = None
    state: str = "STARTING"


@dataclass
class ChannelView:
    channel_id: int
    rx_status: str = "CLOSE"
    protocol_status: str = "Waiting"
    audio_status: str = "Idle"
    secret_status: str = "Idle"
    secret_key: int = 0
    secret_cache_keys: tuple[int, ...] = ()
    csm: str | None = None
    sacch: dict | None = None
    last_update: float = field(default_factory=time.time)


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


def print_stack_dashboard(processes, channels, num_lines_last_time, mode_label="full-stack"):
    if num_lines_last_time > 0:
        sys.stdout.write(f"\033[{num_lines_last_time}A")

    sys.stdout.write("\033[J")
    sys.stdout.write("\033[K" + "=" * 70 + "\n")
    sys.stdout.write("\033[K[STD-T98 Multi Receiver]\n")
    sys.stdout.write(f"\033[KMode: {mode_label} | Ctrl+C to stop\n")
    sys.stdout.write("\033[K" + "=" * 70 + "\n")
    lines = 4

    for process_view in processes:
        state = _colorize_process_state(process_view.state)
        pid = process_view.pid if process_view.pid is not None else "-"
        sys.stdout.write(
            f"\033[K[{process_view.name:<8}] {state} pid={pid} via {process_view.python_executable}\n"
        )
        lines += 1

    sys.stdout.write("\033[K" + "-" * 70 + "\n")
    lines += 1

    cache_keys = next((channel.secret_cache_keys for channel in channels.values() if channel.secret_cache_keys), ())
    cache_text = ", ".join(f"{key:05d}" for key in cache_keys) if cache_keys else "(empty)"
    sys.stdout.write(f"\033[KSecret Cache: {cache_text}\n")
    sys.stdout.write("\033[K" + "-" * 70 + "\n")
    lines += 2

    if not channels:
        sys.stdout.write("\033[K  Waiting for channel activity...\n")
        lines += 1
    else:
        for channel_id in sorted(channels):
            channel = channels[channel_id]
            sacch_str = "Waiting..."
            if channel.sacch:
                stat_val = channel.sacch.get("CallStat", 0)
                stat_desc = "Normal" if stat_val == 0 else "Secret" if stat_val == 1 else str(stat_val)
                sacch_str = (
                    f"UC: {channel.sacch.get('UserCode', 0):03d} | "
                    f"Maker: {channel.sacch.get('MakerCode', 0):03d} | "
                    f"Stat: {stat_desc}"
                )

            csm_str = channel.csm if channel.csm else "Waiting..."
            age_sec = max(0.0, time.time() - channel.last_update)

            sys.stdout.write(
                f"\033[K[CH {channel_id:02d}] RX: [{_colorize_rx_state(channel.rx_status)}] | "
                f"Proto: {channel.protocol_status} | Audio: {channel.audio_status} | "
                f"Age: {age_sec:4.1f}s\n"
            )
            sys.stdout.write(f"\033[K  > CSM   : {csm_str}\n")
            sys.stdout.write(f"\033[K  > SACCH : {sacch_str}\n")
            sys.stdout.write(
                f"\033[K  > Secret: key={channel.secret_key:05d} | {channel.secret_status}\n"
            )
            sys.stdout.write("\033[K" + "-" * 70 + "\n")
            lines += 5

    sys.stdout.flush()
    return lines