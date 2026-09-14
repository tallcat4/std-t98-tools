import numpy as np
import pytest

pytest.importorskip("gnuradio")

import core.rf.sync_word_correlator as correlator_mod  # noqa: E402
from core.rf.sync_word_correlator import SyncWordCorrelator  # noqa: E402


class _SilentPublisher:
    def __init__(self, *args, **kwargs):
        pass

    def publish(self, *args, **kwargs):
        return False

    def close(self):
        pass


@pytest.fixture
def make_correlator(tmp_path, monkeypatch):
    # No status socket exists here; skip the publisher's connect timeout.
    monkeypatch.setattr(correlator_mod, "StatusPublisher", _SilentPublisher)
    made = []

    def factory(ratio):
        block = SyncWordCorrelator(
            channel_count=1,
            error_threshold_ratio=ratio,
            socket_path=str(tmp_path / "frame.sock"),
        )
        made.append(block)
        return block

    yield factory
    for block in made:
        block.stop()


def _feed(block, symbols):
    """Push one channel's float symbols through work() and return how many
    sync detections it produced."""
    before = block.sync_detect_count[0]
    block.work([np.asarray(symbols, dtype=np.float32)], [])
    return block.sync_detect_count[0] - before


def test_set_threshold_ratio_rederives_the_absolute_threshold(make_correlator):
    block = make_correlator(0.2)
    energy = float(np.dot(block.sync_word, block.sync_word))
    assert block.match_threshold == pytest.approx(energy * 0.2)

    block.set_threshold_ratio(0.5)

    assert block.threshold_ratio == 0.5
    assert block.match_threshold == pytest.approx(energy * 0.5)


def test_set_threshold_ratio_rejects_non_positive(make_correlator):
    block = make_correlator(0.2)
    with pytest.raises(ValueError):
        block.set_threshold_ratio(0)
    assert block.threshold_ratio == 0.2


def test_live_ratio_change_alters_what_counts_as_a_sync(make_correlator):
    block = make_correlator(0.05)
    energy = float(np.dot(block.sync_word, block.sync_word))
    # A sync word with every symbol off by 1.0: SSE = 10, i.e. ~0.135 of the
    # 74-unit energy -- rejected by a strict ratio, accepted by a loose one.
    noisy_sync = block.sync_word + 1.0
    assert float(np.sum((noisy_sync - block.sync_word) ** 2)) / energy == pytest.approx(10 / 74)

    assert _feed(block, noisy_sync) == 0

    block.set_threshold_ratio(0.2)
    assert _feed(block, noisy_sync) == 1
