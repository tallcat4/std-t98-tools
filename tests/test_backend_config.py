import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rf.backend_config import (
    BackendConfig,
    ChannelizerConfig,
    SdrConfig,
    add_config_arguments,
    apply_cli_overrides,
    build_channel_map,
    derive_rates,
    nearest_sample_rates,
    load_config,
    load_config_file,
)


ORIGINAL_CHANNEL_MAP = [
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]


class DeriveRatesTest(unittest.TestCase):
    def test_defaults_reproduce_original_constants(self):
        config = BackendConfig()
        rates = derive_rates(config.sdr, config.channelizer)

        self.assertEqual(rates.resamp1_interp, 1)
        self.assertEqual(rates.resamp1_decim, 4)
        self.assertEqual(rates.samp_rate_post_resamp1, 300_000.0)
        self.assertEqual(rates.pfb_num_channels, 48)
        self.assertEqual(rates.samp_rate_post_pfb, 6250.0)
        self.assertEqual(rates.resamp2_interp, 10)
        self.assertEqual(rates.resamp2_decim, 1)
        self.assertEqual(rates.demod_samp_rate, 62500.0)
        self.assertEqual(rates.channel_map, ORIGINAL_CHANNEL_MAP)

    def test_channel_map_covers_all_bins_without_duplicates(self):
        mapping = build_channel_map(pfb_num_channels=48, num_channels=30)
        self.assertEqual(sorted(mapping), list(range(48)))
        self.assertEqual(len(mapping), 48)

    def test_bin_width_stays_channel_spacing_for_other_rates(self):
        # A 2 MHz-capable SDR must still land on 6.25 kHz bins.
        sdr = replace_sample_rate(2_000_000.0)
        rates = derive_rates(sdr, ChannelizerConfig())
        self.assertEqual(rates.samp_rate_post_pfb, 6250.0)
        # 48 * 6250 = 300000; 300000/2000000 = 3/20
        self.assertEqual(rates.resamp1_interp, 3)
        self.assertEqual(rates.resamp1_decim, 20)
        self.assertAlmostEqual(
            sdr.sample_rate * rates.resamp1_interp / rates.resamp1_decim,
            300_000.0,
        )

    def test_non_integer_rate_bin_width_within_tolerance(self):
        sdr = replace_sample_rate(2_048_000.0)  # common RTL-SDR rate
        rates = derive_rates(sdr, ChannelizerConfig())
        effective = sdr.sample_rate * rates.resamp1_interp / rates.resamp1_decim
        self.assertAlmostEqual(effective / rates.pfb_num_channels, 6250.0, delta=0.01)

    def test_pfb_channels_must_cover_num_channels(self):
        with self.assertRaises(ValueError):
            derive_rates(SdrConfig(), ChannelizerConfig(pfb_num_channels=20, num_channels=30))

    def test_larger_pfb_still_maps_thirty_channels(self):
        rates = derive_rates(SdrConfig(), ChannelizerConfig(pfb_num_channels=64))
        # First 30 ports are real channels; the rest are null-sink bins.
        self.assertEqual(len(rates.channel_map), 64)
        self.assertEqual(rates.channel_map[15], 0)  # centre channel -> DC bin


class SampleRateSuitabilityTest(unittest.TestCase):
    # A real USRP B210 rate list around this repo's 1.2 MHz default, which the
    # B210 cannot serve.
    B210_RATES = [
        1_000_000.0,
        1_066_666.666667,
        1_142_857.142857,
        1_230_769.230769,
        1_333_333.333333,
        2_000_000.0,
    ]

    def test_defaults_hit_the_raster_exactly(self):
        rates = derive_rates(SdrConfig(), ChannelizerConfig())
        self.assertEqual(rates.bin_width_error_hz, 0.0)

    def test_awkward_rates_still_land_on_the_raster(self):
        # The bounded-denominator approximation keeps the error far below the
        # 6250 Hz bin width even for rates with no common factors.
        for rate in (1_000_003.0, 1_234_567.0, 2_549_918.0):
            with self.subTest(rate=rate):
                rates = derive_rates(SdrConfig(sample_rate=rate), ChannelizerConfig())
                self.assertLess(abs(rates.bin_width_error_hz), 1.0)

    def test_suggestions_are_ordered_by_closeness(self):
        # 1066666.67 and 1333333.33 are equidistant from 1.2 MHz; the lower
        # rate wins the tie, since it costs less to process.
        picked = nearest_sample_rates(self.B210_RATES, 1_200_000.0, count=3)
        self.assertEqual(
            picked, [1_230_769.230769, 1_142_857.142857, 1_066_666.666667]
        )

    def test_suggestions_ignore_nonsense_rates(self):
        self.assertEqual(nearest_sample_rates([0.0, -1.0], 1_200_000.0), [])


class BandwidthTest(unittest.TestCase):
    def test_rtlsdr_is_left_alone(self):
        # RTL-SDR tracks its filter to the rate; touching it would change
        # long-standing behaviour.
        self.assertIsNone(SdrConfig().resolved_bandwidth())

    def test_other_drivers_get_the_sample_rate(self):
        # A B210 otherwise sits at its full 56 MHz and aliases everything in.
        sdr = SdrConfig(driver="uhd", sample_rate=2_000_000.0)
        self.assertEqual(sdr.resolved_bandwidth(), 2_000_000.0)

    def test_explicit_bandwidth_wins(self):
        sdr = SdrConfig(driver="uhd", sample_rate=2_000_000.0, bandwidth=4_000_000.0)
        self.assertEqual(sdr.resolved_bandwidth(), 4_000_000.0)

    def test_zero_leaves_the_device_alone(self):
        sdr = SdrConfig(driver="uhd", sample_rate=2_000_000.0, bandwidth=0)
        self.assertIsNone(sdr.resolved_bandwidth())


class ConfigLoadingTest(unittest.TestCase):
    def _parse(self, argv):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        return parser.parse_args(argv)

    def test_cli_overrides_apply_over_defaults(self):
        args = self._parse(["--driver", "uhd", "--sample-rate", "2000000", "--no-agc", "--gain", "40"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.driver, "uhd")
        self.assertEqual(config.sdr.sample_rate, 2_000_000.0)
        self.assertFalse(config.sdr.agc)
        self.assertEqual(config.sdr.tuner_gain, 40.0)
        # untouched fields keep defaults
        self.assertEqual(config.sdr.center_freq, 351_293_750.0)

    def test_absent_cli_flags_do_not_override(self):
        args = self._parse([])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config, BackendConfig())

    def test_device_string(self):
        self.assertEqual(SdrConfig(driver="hackrf").device_string(), "driver=hackrf")

    def test_device_args_appended_to_device_string(self):
        sdr = SdrConfig(driver="uhd", device_args="type=b200,serial=3164424")
        self.assertEqual(
            sdr.device_string(), "driver=uhd,type=b200,serial=3164424"
        )

    def test_rtlsdr_keeps_historical_stream_args(self):
        self.assertEqual(SdrConfig().resolved_stream_args(), "bufflen=16384")

    def test_other_drivers_get_no_stream_args(self):
        # "bufflen" is rtlsdr-specific; SoapySDR aborts source construction for
        # any driver that does not advertise it, so the default must not leak.
        for driver in ("uhd", "hackrf", "audio"):
            with self.subTest(driver=driver):
                self.assertEqual(
                    SdrConfig(driver=driver).resolved_stream_args(), ""
                )

    def test_explicit_stream_args_override_driver_default(self):
        sdr = SdrConfig(driver="rtlsdr", stream_args="bufflen=65536")
        self.assertEqual(sdr.resolved_stream_args(), "bufflen=65536")

    def test_empty_stream_args_suppresses_driver_default(self):
        sdr = SdrConfig(driver="rtlsdr", stream_args="")
        self.assertEqual(sdr.resolved_stream_args(), "")

    def test_antenna_defaults_to_driver_choice(self):
        # RTL-SDR has one input; never send a port selection it cannot honour.
        self.assertIsNone(SdrConfig().antenna)

    def test_cli_sets_freq_err_offset(self):
        # Per-device calibration: the default is measured on an RTL-SDR and is
        # meaningless on any other radio.
        args = self._parse(["--driver", "uhd", "--freq-err-offset", "1030"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.freq_err_offset, 1030.0)

    def test_cli_sets_bandwidth(self):
        args = self._parse(["--driver", "uhd", "--bandwidth", "3000000"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.resolved_bandwidth(), 3_000_000.0)

    def test_cli_sets_antenna(self):
        args = self._parse(["--driver", "uhd", "--antenna", "TX/RX"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.antenna, "TX/RX")

    def test_cli_sets_device_and_stream_args(self):
        args = self._parse(
            ["--driver", "uhd", "--device-args", "serial=123", "--stream-args", ""]
        )
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.device_string(), "driver=uhd,serial=123")
        self.assertEqual(config.sdr.resolved_stream_args(), "")

    def test_switching_driver_via_cli_drops_rtlsdr_stream_args(self):
        # The whole point of --driver: it must be enough on its own.
        args = self._parse(["--driver", "uhd"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.resolved_stream_args(), "")

    def test_toml_file_roundtrip(self):
        import tempfile

        toml_text = (
            "[sdr]\n"
            'driver = "uhd"\n'
            "sample_rate = 4000000\n"
            "tuner_gain = 20\n"
            "\n"
            "[channelizer]\n"
            "pfb_num_channels = 64\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text(toml_text, encoding="utf-8")
            config = load_config_file(path)

        self.assertEqual(config.sdr.driver, "uhd")
        self.assertEqual(config.sdr.sample_rate, 4_000_000)
        self.assertEqual(config.sdr.tuner_gain, 20)
        self.assertEqual(config.channelizer.pfb_num_channels, 64)
        # defaulted field
        self.assertEqual(config.sdr.center_freq, 351_293_750.0)

    def test_toml_unknown_key_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text('[sdr]\nnope = 1\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config_file(path)

    def test_load_config_file_then_cli(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text('[sdr]\ndriver = "uhd"\nsample_rate = 4000000\n', encoding="utf-8")
            args = self._parse(["--config", str(path), "--sample-rate", "8000000"])
            config = load_config(args)

        self.assertEqual(config.sdr.driver, "uhd")            # from file
        self.assertEqual(config.sdr.sample_rate, 8_000_000.0)  # CLI wins

    def test_missing_config_file_raises(self):
        args = self._parse(["--config", "/no/such/file.toml"])
        with self.assertRaises(FileNotFoundError):
            load_config(args)


def replace_sample_rate(rate):
    from dataclasses import replace

    return replace(SdrConfig(), sample_rate=rate)


if __name__ == "__main__":
    unittest.main()
