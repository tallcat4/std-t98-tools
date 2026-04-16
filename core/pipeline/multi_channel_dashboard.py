import sys
import time


class ChannelContext:
    def __init__(self, decoder=None, key_196=None):
        self.decoder = decoder
        self.key_196 = key_196
        self.rx_status = "CLOSE"
        self.csm = None
        self.sacch = None
        self.audio_status = "Idle"
        self.last_update = time.time()


def print_dashboard(channels, num_lines_last_time, title="STD-T98 Multi-Channel Receiver"):
    if num_lines_last_time > 0:
        sys.stdout.write(f"\033[{num_lines_last_time}A")

    sys.stdout.write("\033[K" + "=" * 50 + "\n")
    sys.stdout.write(f"\033[K[{title}]\n")
    sys.stdout.write("\033[K" + "=" * 50 + "\n")
    lines = 3

    if not channels:
        sys.stdout.write("\033[K  Waiting for signals...\n")
        sys.stdout.write("\033[K" + "=" * 50 + "\n")
        lines += 2
    else:
        for channel_id in sorted(channels.keys()):
            ctx = channels[channel_id]
            if ctx.rx_status == "OPEN":
                rx_color = "\033[1;32m OPEN  \033[0m"
            else:
                rx_color = "\033[1;30m CLOSE \033[0m"

            sacch_str = "Waiting..."
            if ctx.sacch:
                stat_val = ctx.sacch.get("CallStat", 0)
                stat_desc = "Normal" if stat_val == 0 else "Secret" if stat_val == 1 else str(stat_val)
                sacch_str = (
                    f"UC: {ctx.sacch.get('UserCode', 0):03d} | "
                    f"Maker: {ctx.sacch.get('MakerCode', 0):03d} | "
                    f"Stat: {stat_desc}"
                )

            csm_str = ctx.csm if ctx.csm else "Waiting..."

            sys.stdout.write(f"\033[K[CH {channel_id:02d}] RX: [{rx_color}] | Audio: {ctx.audio_status}\n")
            sys.stdout.write(f"\033[K  > CSM (PICH) : {csm_str}\n")
            sys.stdout.write(f"\033[K  > SACCH      : {sacch_str}\n")
            sys.stdout.write("\033[K" + "-" * 50 + "\n")
            lines += 4

    sys.stdout.flush()
    return lines