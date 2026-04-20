import time

import numpy as np
from gnuradio import gr

from core.pipeline.runtime_status import StatusPublisher
from core.protocol.frame_layout import quantize_float_symbols
from ipc.message_schema import FRAME_FLAG_SYNC_DETECTED, STATUS_SOURCE_RF, FramePacket
from ipc.transport.uds_seqpacket import UdsSeqpacketServer, resolve_frame_socket_path, resolve_status_socket_path


RF_DEBUG_REPORT_INTERVAL_SEC = 1.0
RF_DEBUG_ACTIVE_WINDOW_SEC = 2.0


class SyncWordCorrelator(gr.sync_block):
    def __init__(
        self,
        channel_count,
        sync_word=None,
        packet_len=192,
        socket_path=None,
        abs_limit=None,
        threshold_ratio=None,
        error_threshold_ratio=None,
    ):
        self.channel_count = int(channel_count)
        gr.sync_block.__init__(
            self,
            name="Sync Word Correlator",
            in_sig=[np.float32] * self.channel_count,
            out_sig=None,
        )

        if threshold_ratio is None and error_threshold_ratio is None:
            raise ValueError("Either threshold_ratio or error_threshold_ratio must be provided.")
        if threshold_ratio is not None and error_threshold_ratio is not None:
            raise ValueError("threshold_ratio and error_threshold_ratio are mutually exclusive.")

        if sync_word is None:
            sync_word = [-3.0, +1.0, -3.0, +3.0, -3.0, -3.0, +3.0, +3.0, -1.0, +3.0]
        self.sync_word = np.asarray(sync_word, dtype=np.float32)
        self.sync_word_length = len(self.sync_word)

        self.packet_len = int(packet_len)
        if self.packet_len < self.sync_word_length:
            raise ValueError("packet_len must be >= length of sync_word")

        sync_word_energy = float(np.dot(self.sync_word, self.sync_word))
        if threshold_ratio is not None:
            self.detection_mode = "correlation"
            self.match_threshold = sync_word_energy * float(threshold_ratio)
            self.abs_limit = float(abs_limit if abs_limit is not None else 4.0)
        else:
            self.detection_mode = "sse"
            self.match_threshold = sync_word_energy * float(error_threshold_ratio)
            self.abs_limit = None

        self.shift_registers = [np.zeros(self.sync_word_length, dtype=np.float32) for _ in range(self.channel_count)]
        self.collecting = [False] * self.channel_count
        self.packet_bufs = [None] * self.channel_count
        self.samples_collected = [0] * self.channel_count
        self.sequence = 0

        resolved_socket_path = resolve_frame_socket_path(channel_count=self.channel_count, socket_path=socket_path)
        self.transport = UdsSeqpacketServer(resolved_socket_path)
        self.status_publisher = StatusPublisher(
            socket_path=resolve_status_socket_path(channel_count=self.channel_count),
            source=STATUS_SOURCE_RF,
        )
        self.next_debug_report_at = time.monotonic() + RF_DEBUG_REPORT_INTERVAL_SEC
        self.sync_detect_count = [0] * self.channel_count
        self.frame_send_ok = [0] * self.channel_count
        self.frame_send_fail = [0] * self.channel_count
        self.last_metric = [None] * self.channel_count
        self.best_metric = [None] * self.channel_count
        self.last_detect_metric = [None] * self.channel_count
        self.last_detect_at = [0.0] * self.channel_count

    def stop(self):
        self.transport.close()
        self.status_publisher.close()
        return True

    def _start_collection(self, channel_index, initial_window):
        self.packet_bufs[channel_index] = np.empty(self.packet_len, dtype=np.float32)
        self.packet_bufs[channel_index][: self.sync_word_length] = initial_window
        self.samples_collected[channel_index] = self.sync_word_length
        self.collecting[channel_index] = True

    def _sync_metric(self, channel_index):
        if self.detection_mode == "correlation":
            return float(np.dot(self.shift_registers[channel_index], self.sync_word))

        return float(np.sum((self.shift_registers[channel_index] - self.sync_word) ** 2))

    def _update_metric_stats(self, channel_index, metric):
        self.last_metric[channel_index] = metric
        best_metric = self.best_metric[channel_index]
        if best_metric is None:
            self.best_metric[channel_index] = metric
            return

        if self.detection_mode == "correlation":
            if metric > best_metric:
                self.best_metric[channel_index] = metric
            return

        if metric < best_metric:
            self.best_metric[channel_index] = metric

    def _sync_detected(self, channel_index, metric=None):
        if metric is None:
            metric = self._sync_metric(channel_index)

        if self.detection_mode == "correlation":
            return metric >= self.match_threshold

        return metric <= self.match_threshold

    def _format_metric(self, metric):
        if metric is None:
            return "n/a"
        return f"{metric:.1f}"

    def _channel_debug_summary(self, channel_index):
        metric_name = "corr" if self.detection_mode == "correlation" else "sse"
        return (
            f"sync={self.sync_detect_count[channel_index]} ipc={self.frame_send_ok[channel_index]}/{self.frame_send_fail[channel_index]} "
            f"{metric_name}={self._format_metric(self.last_detect_metric[channel_index])} "
            f"thr={self.match_threshold:.1f} best={self._format_metric(self.best_metric[channel_index])}"
        )

    def _maybe_publish_debug_metrics(self):
        current_time = time.monotonic()
        if current_time < self.next_debug_report_at:
            return

        active_channels = sum(
            1
            for last_detect_at in self.last_detect_at
            if last_detect_at and (current_time - last_detect_at) <= RF_DEBUG_ACTIVE_WINDOW_SEC
        )
        self.status_publisher.publish(
            payload_dict={
                "event": "service_metrics",
                "summary": (
                    f"sync={sum(self.sync_detect_count)} ipc={sum(self.frame_send_ok)}/{sum(self.frame_send_fail)} "
                    f"mode={self.detection_mode} thr={self.match_threshold:.1f} active={active_channels}"
                ),
            }
        )

        for channel_index in range(self.channel_count):
            if self.sync_detect_count[channel_index] <= 0 and self.frame_send_fail[channel_index] <= 0:
                continue
            if self.last_detect_at[channel_index] and (current_time - self.last_detect_at[channel_index]) > RF_DEBUG_ACTIVE_WINDOW_SEC:
                if self.frame_send_fail[channel_index] <= 0:
                    continue
            self.status_publisher.publish(
                channel_id=channel_index + 1,
                payload_dict={
                    "event": "channel_state",
                    "rf_debug": self._channel_debug_summary(channel_index),
                },
            )

        self.next_debug_report_at = current_time + RF_DEBUG_REPORT_INTERVAL_SEC

    def _append_symbol(self, channel_index, symbol):
        if not self.collecting[channel_index]:
            return

        collected = self.samples_collected[channel_index]
        if collected < self.packet_len:
            self.packet_bufs[channel_index][collected] = float(symbol)
            self.samples_collected[channel_index] += 1

        if self.samples_collected[channel_index] < self.packet_len:
            return

        packet = FramePacket(
            sequence=self.sequence,
            monotonic_ns=time.monotonic_ns(),
            channel_id=channel_index + 1,
            flags=FRAME_FLAG_SYNC_DETECTED,
            symbols=quantize_float_symbols(self.packet_bufs[channel_index]),
        )
        self.sequence += 1
        if self.transport.send(packet.encode()):
            self.frame_send_ok[channel_index] += 1
        else:
            self.frame_send_fail[channel_index] += 1

        self.collecting[channel_index] = False
        self.packet_bufs[channel_index] = None
        self.samples_collected[channel_index] = 0

    def work(self, input_items, output_items):
        sample_count = len(input_items[0])

        for channel_index in range(self.channel_count):
            in_data = input_items[channel_index]
            for sample_index in range(sample_count):
                symbol = float(in_data[sample_index])
                if self.abs_limit is not None:
                    if symbol > self.abs_limit:
                        symbol = self.abs_limit
                    elif symbol < -self.abs_limit:
                        symbol = -self.abs_limit

                if self.collecting[channel_index]:
                    self._append_symbol(channel_index, symbol)
                    continue

                self.shift_registers[channel_index][:-1] = self.shift_registers[channel_index][1:]
                self.shift_registers[channel_index][-1] = symbol
                metric = self._sync_metric(channel_index)
                self._update_metric_stats(channel_index, metric)
                if self._sync_detected(channel_index, metric):
                    self.sync_detect_count[channel_index] += 1
                    self.last_detect_metric[channel_index] = metric
                    self.last_detect_at[channel_index] = time.monotonic()
                    self._start_collection(channel_index, self.shift_registers[channel_index].copy())

        self._maybe_publish_debug_metrics()

        return sample_count