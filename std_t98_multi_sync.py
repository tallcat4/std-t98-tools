#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from core.rf.sync_word_correlator import SyncWordCorrelator


class sync_word_correlator(SyncWordCorrelator):
    def __init__(
        self,
        num_channels,
        sync_word=None,
        error_threshold_ratio=0.2,
        packet_len=192,
        socket_path=None,
    ):
        super().__init__(
            channel_count=num_channels,
            sync_word=sync_word,
            packet_len=packet_len,
            socket_path=socket_path,
            error_threshold_ratio=error_threshold_ratio,
        )