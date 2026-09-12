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
