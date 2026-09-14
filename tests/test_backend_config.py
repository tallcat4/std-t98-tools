import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rf.backend_config import (
    BackendConfig,
    DemodConfig,
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
        sdr = replace_sample_rate(2_048_000.0)
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


class FreqErrOffsetTest(unittest.TestCase):
    def test_default_is_zero(self):
        self.assertEqual(SdrConfig().resolved_freq_err_offset(), 0.0)
        self.assertEqual(SdrConfig().tuned_freq(), 351_293_750)

    def test_explicit_calibration_wins(self):
        sdr = SdrConfig(freq_err_offset=1030.0)
        self.assertEqual(sdr.tuned_freq(), 351_293_750 + 1030)

    def test_zero_disables_it_explicitly(self):
        sdr = SdrConfig(freq_err_offset=0)
        self.assertEqual(sdr.resolved_freq_err_offset(), 0.0)


class SquelchTest(unittest.TestCase):
    def test_default_matches_the_historical_value(self):
        self.assertEqual(BackendConfig().demod.squelch_threshold, -25.0)

    def test_cli_overrides_squelch(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        args = parser.parse_args(["--squelch", "-60"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.demod.squelch_threshold, -60.0)

    def test_squelch_survives_a_config_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text("[demod]\nsquelch_threshold = -55\n", encoding="utf-8")
            config = load_config_file(path)
        self.assertEqual(config.demod.squelch_threshold, -55.0)
        # Untouched sections keep their defaults.
        self.assertEqual(config.sdr.device_args, "")


class SyncThresholdTest(unittest.TestCase):
    def test_default_matches_the_historical_hard_coded_value(self):
        self.assertEqual(BackendConfig().demod.sync_error_threshold_ratio, 0.2)

    def test_cli_overrides_sync_threshold_ratio(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        args = parser.parse_args(["--sync-threshold-ratio", "0.35"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.demod.sync_error_threshold_ratio, 0.35)
        # The other [demod] value is untouched by an unrelated override.
        self.assertEqual(config.demod.squelch_threshold, -25.0)

    def test_sync_threshold_ratio_survives_a_config_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text("[demod]\nsync_error_threshold_ratio = 0.3\n", encoding="utf-8")
            config = load_config_file(path)
        self.assertEqual(config.demod.sync_error_threshold_ratio, 0.3)


class BandwidthTest(unittest.TestCase):
    def test_default_follows_sample_rate(self):
        # A USRP otherwise sits at its full front-end bandwidth no matter how
        # slowly you sample, aliasing everything in.
        sdr = SdrConfig(sample_rate=2_000_000.0)
        self.assertEqual(sdr.resolved_bandwidth(), 2_000_000.0)

    def test_explicit_bandwidth_wins(self):
        sdr = SdrConfig(sample_rate=2_000_000.0, bandwidth=4_000_000.0)
        self.assertEqual(sdr.resolved_bandwidth(), 4_000_000.0)

    def test_zero_leaves_the_device_alone(self):
        sdr = SdrConfig(sample_rate=2_000_000.0, bandwidth=0)
        self.assertIsNone(sdr.resolved_bandwidth())


class ConfigLoadingTest(unittest.TestCase):
    def _parse(self, argv):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        return parser.parse_args(argv)

    def test_cli_overrides_apply_over_defaults(self):
        args = self._parse(["--sample-rate", "2000000", "--gain", "40"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.sample_rate, 2_000_000.0)
        self.assertEqual(config.sdr.tuner_gain, 40.0)
        # untouched fields keep defaults
        self.assertEqual(config.sdr.center_freq, 351_293_750.0)

    def test_absent_cli_flags_do_not_override(self):
        args = self._parse([])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config, BackendConfig())

    def test_device_string_is_device_args_verbatim(self):
        self.assertEqual(SdrConfig().device_string(), "")
        sdr = SdrConfig(device_args="type=b200,serial=3164424")
        self.assertEqual(sdr.device_string(), "type=b200,serial=3164424")

    def test_antenna_defaults_to_device_choice(self):
        self.assertIsNone(SdrConfig().antenna)

    def test_cli_sets_freq_err_offset(self):
        args = self._parse(["--freq-err-offset", "1030"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.resolved_freq_err_offset(), 1030.0)

    def test_cli_sets_bandwidth(self):
        args = self._parse(["--bandwidth", "3000000"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.resolved_bandwidth(), 3_000_000.0)

    def test_cli_sets_antenna(self):
        args = self._parse(["--antenna", "TX/RX"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.antenna, "TX/RX")

    def test_cli_sets_device_args(self):
        args = self._parse(["--device-args", "serial=123"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertEqual(config.sdr.device_string(), "serial=123")

    def test_toml_file_roundtrip(self):
        import tempfile

        toml_text = (
            "[sdr]\n"
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

    def test_toml_rejects_removed_soapy_only_keys(self):
        # driver / bias_tee / stream_args / freq_correction do not exist any
        # more now the backend is UHD-only; a leftover profile from the old
        # schema must fail loudly, not be silently ignored. (agc came back
        # once UHD's set_rx_agc was confirmed to exist -- see AgcTest.)
        import tempfile

        for key, value in (("driver", '"uhd"'), ("bias_tee", "false"), ("stream_args", '""')):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "cfg.toml"
                path.write_text(f"[sdr]\n{key} = {value}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config_file(path)

    def test_load_config_file_then_cli(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text('[sdr]\nsample_rate = 4000000\n', encoding="utf-8")
            args = self._parse(["--config", str(path), "--sample-rate", "8000000"])
            config = load_config(args)

        self.assertEqual(config.sdr.sample_rate, 8_000_000.0)  # CLI wins

    def test_missing_config_file_raises(self):
        args = self._parse(["--config", "/no/such/file.toml"])
        with self.assertRaises(FileNotFoundError):
            load_config(args)


class AgcTest(unittest.TestCase):
    def test_default_is_off(self):
        self.assertFalse(SdrConfig().agc)

    def test_toml_sets_agc(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text("[sdr]\nagc = true\n", encoding="utf-8")
            config = load_config_file(path)
        self.assertTrue(config.sdr.agc)

    def test_cli_sets_agc(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        args = parser.parse_args(["--agc"])
        config = apply_cli_overrides(BackendConfig(), args)
        self.assertTrue(config.sdr.agc)

    def test_cli_no_agc_overrides_file(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        args = parser.parse_args(["--no-agc"])
        config = apply_cli_overrides(BackendConfig(sdr=SdrConfig(agc=True)), args)
        self.assertFalse(config.sdr.agc)

    def test_absent_cli_flag_does_not_override(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        args = parser.parse_args([])
        config = apply_cli_overrides(BackendConfig(sdr=SdrConfig(agc=True)), args)
        self.assertTrue(config.sdr.agc)


def replace_sample_rate(rate):
    from dataclasses import replace

    return replace(SdrConfig(), sample_rate=rate)


if __name__ == "__main__":
    unittest.main()
