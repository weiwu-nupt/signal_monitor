import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6 import QtWidgets
from fpga_closed_loop import Target, TargetFilter
from signal_tracks import SignalTracker
from receiver.file_replay import FileReplayPage


def target(center, width=100000):
    return Target('narrow', 0, center, width, 320000000)


def row(t, kind='未分类', confidence='—'):
    return [f'{700+t.center_hz/1e6:.6f}', f'{t.bandwidth_hz/1e6:.6f}',
            kind, confidence, t.target_type, '1', '0.400000', '2.621440']


class SignalTrackTests(unittest.TestCase):
    def test_strength_order_and_small_drift_keep_ids_in_frequency_order(self):
        tracker = SignalTracker()
        first = [target(2_000_000), target(1_000_000)]
        tracker.observe(1, first, list(map(row, first)), 10000)
        second = [target(1_010_000), target(1_990_000)]
        tracker.observe(2, second, list(map(row, second)), 10000)
        self.assertEqual([t.identifier for t in tracker.visible(TargetFilter())], [1, 2])
        self.assertEqual(len(tracker.tracks), 2)

    def test_short_absence_retains_id_expiry_does_not_reuse_id(self):
        tracker = SignalTracker(keep_missing_frames=2)
        t = target(1000000)
        tracker.observe(1, [t], [row(t)], 10000)
        tracker.observe(2, [], [], 10000)
        self.assertEqual(tracker.tracks[1].missed, 1)
        tracker.observe(3, [t], [row(t)], 10000)
        self.assertEqual(tracker.tracks[1].missed, 0)
        for frame in (4, 5, 6):
            tracker.observe(frame, [], [], 10000)
        self.assertFalse(tracker.tracks)
        tracker.observe(7, [t], [row(t)], 10000)
        self.assertEqual(list(tracker.tracks), [2])

    def test_neighbours_and_nested_bands_have_distinct_ids(self):
        tracker = SignalTracker()
        first = [target(1000000), target(1050000), target(1000000, 25000000)]
        tracker.observe(1, first, list(map(row, first)), 10000)
        second = [target(1055000), target(1005000, 25000000), target(1005000)]
        tracker.observe(2, second, list(map(row, second)), 10000)
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(set(tracker._assignments.values())), 3)
        self.assertEqual(tracker.tracks[2].target.bandwidth_hz, 25000000)

    def test_same_frame_results_do_not_age_tracks_and_filter_removes_held_rows(self):
        tracker = SignalTracker()
        a, b = target(1000000), target(3000000, 300000)
        tracker.observe(1, [a, b], [row(a), row(b)], 10000)
        tracker.observe(2, [a], [row(a, 'WiFi', '95%')], 10000)
        tracker.observe(2, [a], [row(a, 'WiFi', '96%')], 10000)
        self.assertEqual(tracker.tracks[2].missed, 1)
        tracker.observe(3, [a], [row(a, '待识别')], 10000)
        self.assertEqual(tracker.tracks[1].row[2:4], ['WiFi（上次）', '96%'])
        tracker.apply_filter(TargetFilter(None, 200000))
        self.assertEqual(list(tracker.tracks), [1])

    def test_table_throttles_updates_and_keeps_selected_id_after_insert(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        page = FileReplayPage()
        def show(frame, targets, now):
            state = dict(frame=frame, axis=np.linspace(550e6,850e6,32768), center=700e6,
                         targets=targets, rows=list(map(row, targets)))
            page._last_state = state
            with patch('receiver.file_replay.time.monotonic', return_value=now):
                page._render_targets(state)
        try:
            show(1, [target(2e6), target(1e6)], 10)
            self.assertEqual(page.table.item(0, 0).text(), '701.000')
            page.table.selectRow(1)
            old_item = page.table.item(0, 0)
            show(2, [target(2e6+10000), target(1e6)], 10.2)
            self.assertEqual(page.table.item(1, 0).text(), '702.000')
            self.assertIs(page.table.item(0, 0), old_item)
            show(3, [target(2e6), target(1e6), target(0)], 11.1)
            self.assertEqual(page.table.item(2, 0).text(), '700.000')
            self.assertEqual(page._table_ids[page.table.currentRow()], 2)
            self.assertEqual(page.table.rowCount(), 3)
        finally:
            page.shutdown()
            page.close()


if __name__ == '__main__':
    unittest.main()
