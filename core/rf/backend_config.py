"""RF backend configuration and channelizer rate derivation.

This module keeps device-specific settings (which SDR, at what sample rate,
gains, calibration) out of the GNU Radio flowgraph so the same backend can
drive RTL-SDR, USRP, HackRF and other SoapySDR devices without editing the
flowgraph source.

Two responsibilities live here:

1. :class:`BackendConfig` and its sub-configs describe *what* to receive with.
   They can be built from defaults, a TOML file, and/or argparse overrides.
   The defaults reproduce the historical RTL-SDR / 1.2 MHz behaviour exactly.

2. :func:`derive_rates` computes the resampling / channelizer parameters from
   the input sample rate instead of hard-coding them. STD-T98 uses a 6.25 kHz
   channel raster, so every PFB output bin must be exactly one channel wide;
   given that invariant the stage-1 resampler ratio follows from the input
   rate, and the channel-to-bin map follows from the channel counts.

None of this imports GNU Radio, so it is unit-testable on any machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on old runtimes
    tomllib = None


# ---------------------------------------------------------------------------
# STD-T98 physical-layer constants (protocol-defined, not device-specific).
# ---------------------------------------------------------------------------
CHANNEL_SPACING_HZ = 6250            # 6.25 kHz channel raster
DEFAULT_NUM_CHANNELS = 30
DEFAULT_CENTER_FREQ_HZ = 351_293_750  # 351.29375 MHz
BAUD_RATE = 2400

CONFIG_ENV_VAR = "STD_T98_BACKEND_CONFIG"

# SoapySDR validates stream arguments against what the device advertises and
# raises for anything it does not know, so a stream arg that helps one driver
# is fatal on another. "bufflen" is an rtlsdr/osmosdr-specific knob: passing it
# to e.g. uhd or hackrf aborts source construction with
# "Unsupported stream argument bufflen". Keep the historical RTL-SDR buffer
# size as that driver's default and send nothing to drivers we have no reason
# to tune, while still letting the config file or CLI override either way.
DRIVER_DEFAULT_STREAM_ARGS = {
    "rtlsdr": "bufflen=16384",
}

# Drivers that already track the analog filter to the sample rate, so touching
# the bandwidth would only change long-standing behaviour. Everything else can
# leave its front end wide open -- a USRP B210 sits at its full 56 MHz no
# matter how slowly you sample, folding megahertz of spectrum onto the 187.5
# kHz this actually cares about -- so those get the bandwidth set for them.
DRIVERS_WITH_AUTOMATIC_BANDWIDTH = {"rtlsdr"}

# Resampler ratio search cap. Keeps GNU Radio's rational_resampler taps sane
# when the input rate is not an exact multiple of the channelizer rate; the
# resulting rate error is well under 1 ppm for any realistic SDR rate.
_MAX_RESAMP_DENOMINATOR = 10_000


@dataclass(frozen=True)
class SdrConfig:
    """Device-facing settings passed to the SoapySDR source."""

    driver: str = "rtlsdr"           # SoapySDR driver key, e.g. rtlsdr / uhd / hackrf
    device_args: str = ""            # extra device selection, e.g. "serial=123,type=b200"
    sample_rate: float = 1_200_000.0
    center_freq: float = float(DEFAULT_CENTER_FREQ_HZ)
    freq_offset: float = 0.0          # deliberate tuning offset (Hz)
    freq_err_offset: float = -340.0   # per-device frequency error correction (Hz)
    freq_correction: float = 0.0      # frequency correction in ppm
    tuner_gain: float = 30.0
    agc: bool = True
    bias_tee: bool = False
    gain_element: str = "TUNER"       # SoapySDR gain element; "" => overall gain
    stream_args: Optional[str] = None  # None => this driver's default (see above)
    # RX port. None leaves whatever the driver selects, which is the only
    # sensible default: single-input devices like RTL-SDR have nothing to pick,
    # while a B210 exposes TX/RX and RX2 and defaults to RX2.
    antenna: Optional[str] = None
    # Analog front-end bandwidth in Hz. None asks for the driver-aware default
    # (see above); 0 explicitly leaves whatever the device came up with.
    bandwidth: Optional[float] = None

    def device_string(self) -> str:
        """Build the SoapySDR device string.

        ``device_args`` is appended verbatim so a specific unit can be picked
        out of several identical ones (``serial=...``) or a driver can be told
        what it is talking to (``type=b200``).
        """
        device = f"driver={self.driver}"
        if self.device_args:
            device = f"{device},{self.device_args}"
        return device

    def resolved_stream_args(self) -> str:
        """Stream args actually handed to SoapySDR.

        An explicit setting always wins, including an empty string, which is
        how a user suppresses a driver default.
        """
        if self.stream_args is not None:
            return self.stream_args
        return DRIVER_DEFAULT_STREAM_ARGS.get(self.driver, "")

    def resolved_bandwidth(self) -> Optional[float]:
        """Analog bandwidth to request, or None to leave the device alone."""
        if self.bandwidth is not None:
            return self.bandwidth or None
        if self.driver in DRIVERS_WITH_AUTOMATIC_BANDWIDTH:
            return None
        # Matching the sample rate keeps the whole digitised span usable while
        # filtering out everything that would alias into it.
        return self.sample_rate


@dataclass(frozen=True)
class ChannelizerConfig:
    """Channelizer geometry. Spacing is fixed by the protocol; the rest tune
    how much margin the PFB has and the per-channel demod rate."""

    channel_spacing: int = CHANNEL_SPACING_HZ
    num_channels: int = DEFAULT_NUM_CHANNELS
    pfb_num_channels: int = 48        # >= num_channels; sets the pre-PFB rate
    resamp2_interp: int = 10          # per-channel upsample before demod


@dataclass(frozen=True)
class BackendConfig:
    sdr: SdrConfig = SdrConfig()
    channelizer: ChannelizerConfig = ChannelizerConfig()


@dataclass(frozen=True)
class DerivedRates:
    """Everything the flowgraph needs that used to be hard-coded constants."""

    resamp1_interp: int
    resamp1_decim: int
    samp_rate_post_resamp1: float
    pfb_num_channels: int
    samp_rate_post_pfb: float
    resamp2_interp: int
    resamp2_decim: int
    demod_samp_rate: float
    channel_map: list[int]
    # The stage-1 ratio is a rational approximation, so the rate actually
    # reaching the channelizer can miss the 6.25 kHz raster. Anything but a
    # tiny error mistunes every channel, which looks like a dead receiver
    # rather than a configuration mistake, so it is reported explicitly.
    bin_width_error_hz: float = 0.0


def build_channel_map(pfb_num_channels: int, num_channels: int) -> list[int]:
    """Map flowgraph channel ports to PFB bins, then append unused bins.

    Channels below centre land in the top FFT bins (negative frequencies),
    the centre channel is the DC bin, and channels above centre take the low
    positive bins. Any remaining bins are appended so the caller can attach
    null sinks. For pfb=48, num=30 this reproduces the original hand-written
    map (33..47, 0, 1..14, then 15..32).
    """
    if pfb_num_channels < num_channels:
        raise ValueError(
            f"pfb_num_channels ({pfb_num_channels}) must be >= "
            f"num_channels ({num_channels})"
        )

    neg_count = num_channels // 2
    pos_count = num_channels - neg_count - 1

    mapping = list(range(pfb_num_channels - neg_count, pfb_num_channels))
    mapping.append(0)
    mapping.extend(range(1, pos_count + 1))

    used = set(mapping)
    mapping.extend(bin_index for bin_index in range(pfb_num_channels) if bin_index not in used)
    return mapping


def derive_rates(sdr: SdrConfig, channelizer: ChannelizerConfig) -> DerivedRates:
    """Compute resampling / channelizer rates from the SDR sample rate.

    Invariant: PFB output bin width == channel spacing (6.25 kHz). The rate
    feeding the channelizer is therefore ``pfb_num_channels * channel_spacing``,
    and the stage-1 resampler ratio is whatever gets there from the input rate.
    """
    if channelizer.pfb_num_channels < channelizer.num_channels:
        raise ValueError(
            f"pfb_num_channels ({channelizer.pfb_num_channels}) must be >= "
            f"num_channels ({channelizer.num_channels})"
        )
    if sdr.sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sdr.sample_rate}")

    samp_rate_post_resamp1 = channelizer.pfb_num_channels * channelizer.channel_spacing

    ratio = Fraction(
        int(round(samp_rate_post_resamp1)),
        int(round(sdr.sample_rate)),
    ).limit_denominator(_MAX_RESAMP_DENOMINATOR)
    resamp1_interp = ratio.numerator
    resamp1_decim = ratio.denominator

    samp_rate_post_pfb = samp_rate_post_resamp1 / channelizer.pfb_num_channels
    demod_samp_rate = samp_rate_post_pfb * channelizer.resamp2_interp

    achieved_post_resamp1 = sdr.sample_rate * resamp1_interp / resamp1_decim
    achieved_bin_width = achieved_post_resamp1 / channelizer.pfb_num_channels
    bin_width_error_hz = achieved_bin_width - channelizer.channel_spacing

    return DerivedRates(
        resamp1_interp=resamp1_interp,
        resamp1_decim=resamp1_decim,
        samp_rate_post_resamp1=float(samp_rate_post_resamp1),
        pfb_num_channels=channelizer.pfb_num_channels,
        samp_rate_post_pfb=float(samp_rate_post_pfb),
        resamp2_interp=channelizer.resamp2_interp,
        resamp2_decim=1,
        demod_samp_rate=float(demod_samp_rate),
        channel_map=build_channel_map(
            channelizer.pfb_num_channels, channelizer.num_channels
        ),
        bin_width_error_hz=bin_width_error_hz,
    )


def nearest_sample_rates(
    available: "list[float]",
    near: float,
    count: int = 5,
) -> list[float]:
    """The device's supported rates closest to ``near``.

    Only proximity is used. The stage-1 ratio is a bounded-denominator
    approximation, and measuring it across the plausible rate range puts the
    worst raster error near 0.1 Hz on a 6250 Hz bin, so there is no second
    criterion worth filtering on.
    """
    return sorted(
        (rate for rate in available if rate > 0),
        key=lambda rate: (abs(rate - near), rate),
    )[:count]


# ---------------------------------------------------------------------------
# Loading: defaults -> TOML file -> CLI overrides
# ---------------------------------------------------------------------------
def _coerce_section(cls, defaults, values: Mapping[str, Any]):
    known = {f.name for f in defaults.__dataclass_fields__.values()}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys in config: {sorted(unknown)}"
        )
    return replace(defaults, **{key: values[key] for key in values})


def load_config_file(path: os.PathLike | str) -> BackendConfig:
    """Load a TOML config file into a :class:`BackendConfig`.

    Missing sections/keys fall back to the RTL-SDR defaults. Unknown keys are
    rejected so typos surface immediately instead of being silently ignored.
    """
    if tomllib is None:  # pragma: no cover
        raise RuntimeError(
            "Reading a TOML config requires Python 3.11+ (tomllib). "
            "Use CLI overrides instead, or upgrade Python."
        )

    with open(path, "rb") as config_file:
        raw = tomllib.load(config_file)

    unknown_sections = set(raw) - {"sdr", "channelizer"}
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    return BackendConfig(
        sdr=_coerce_section(SdrConfig, SdrConfig(), raw.get("sdr", {})),
        channelizer=_coerce_section(
            ChannelizerConfig, ChannelizerConfig(), raw.get("channelizer", {})
        ),
    )


def add_config_arguments(parser) -> None:
    """Register the backend's CLI options on an argparse parser."""
    parser.add_argument(
        "--config",
        default=os.environ.get(CONFIG_ENV_VAR),
        help=(
            "Path to a TOML backend config. Defaults to the "
            f"{CONFIG_ENV_VAR} environment variable if set."
        ),
    )
    parser.add_argument("--driver", help="SoapySDR driver key (e.g. rtlsdr, uhd, hackrf).")
    parser.add_argument(
        "--device-args",
        help='Extra SoapySDR device selection appended to the device string, '
        'e.g. "serial=00000001" or "type=b200". Use this to pick one of '
        "several identical SDRs.",
    )
    parser.add_argument(
        "--stream-args",
        help="SoapySDR stream arguments. Defaults to the driver's own default ("
        + ", ".join(f"{k}: {v}" for k, v in DRIVER_DEFAULT_STREAM_ARGS.items())
        + "; none for other drivers). Pass an empty string to send none. "
        "Unsupported args make the device refuse to open.",
    )
    parser.add_argument("--sample-rate", type=float, help="SDR sample rate in Hz.")
    parser.add_argument("--freq", type=float, help="Center frequency in Hz.")
    parser.add_argument("--gain", type=float, help="Tuner gain (used when AGC is off).")
    parser.add_argument("--gain-element", help='SoapySDR gain element name ("" for overall gain).')
    agc_group = parser.add_mutually_exclusive_group()
    agc_group.add_argument("--agc", dest="agc", action="store_true", default=None, help="Enable hardware AGC.")
    agc_group.add_argument("--no-agc", dest="agc", action="store_false", default=None, help="Disable hardware AGC.")
    bias_group = parser.add_mutually_exclusive_group()
    bias_group.add_argument("--bias-tee", dest="bias_tee", action="store_true", default=None, help="Enable bias tee if supported.")
    bias_group.add_argument("--no-bias-tee", dest="bias_tee", action="store_false", default=None, help="Disable bias tee.")
    parser.add_argument(
        "--bandwidth",
        type=float,
        help="Analog front-end bandwidth in Hz. Defaults to the sample rate on "
        "devices that do not track it themselves (RTL-SDR does). Pass 0 to "
        "leave the device's own setting alone.",
    )
    parser.add_argument(
        "--antenna",
        help="RX antenna port to select (e.g. RX2 or TX/RX on a USRP B210). "
        "Omit to leave the driver's own default. Devices with a single input "
        "ignore this.",
    )
    parser.add_argument("--freq-correction", type=float, help="Frequency correction in ppm.")
    parser.add_argument("--pfb-channels", type=int, help="Number of PFB channels (>= num_channels).")


def apply_cli_overrides(config: BackendConfig, args) -> BackendConfig:
    """Return a new config with any provided CLI overrides applied."""
    sdr_overrides: dict[str, Any] = {}
    for attr, field_name in (
        ("driver", "driver"),
        ("device_args", "device_args"),
        ("stream_args", "stream_args"),
        ("sample_rate", "sample_rate"),
        ("freq", "center_freq"),
        ("gain", "tuner_gain"),
        ("gain_element", "gain_element"),
        ("antenna", "antenna"),
        ("bandwidth", "bandwidth"),
        ("agc", "agc"),
        ("bias_tee", "bias_tee"),
        ("freq_correction", "freq_correction"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            sdr_overrides[field_name] = value

    channelizer_overrides: dict[str, Any] = {}
    pfb_channels = getattr(args, "pfb_channels", None)
    if pfb_channels is not None:
        channelizer_overrides["pfb_num_channels"] = pfb_channels

    return BackendConfig(
        sdr=replace(config.sdr, **sdr_overrides) if sdr_overrides else config.sdr,
        channelizer=replace(config.channelizer, **channelizer_overrides)
        if channelizer_overrides
        else config.channelizer,
    )


def load_config(args=None) -> BackendConfig:
    """Resolve the effective config: defaults, then TOML file, then CLI."""
    config = BackendConfig()

    config_path: Optional[str] = getattr(args, "config", None) if args is not None else None
    if config_path:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = load_config_file(config_path)

    if args is not None:
        config = apply_cli_overrides(config, args)

    return config
