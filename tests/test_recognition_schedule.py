import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6 import QtWidgets

from fpga_closed_loop import (Target, TargetFilter, IQPacket, IQ_BYTES,
                              SpectrumDDCClosedLoop, build_target_batch)
from recognition_schedule import RecognitionSchedule, target_batches
from receiver.recognition_controls import RecognitionBandwidth
from iq_file_source import inspect_iq_file, read_channel_packet
from test_iq_file_replay import write_rf64


class ScheduleTests(unittest.TestCase):
    def candidates(self):
        return ([Target('narrow', i, i*1000, 1_000_000, 320_000_000) for i in range(300)]
                + [Target('narrow', i, 10_000_000+i*1000, 21_000_000, 320_000_000) for i in range(5)])

    def packet(self, target):
        samples = np.zeros(IQ_BYTES//4, dtype=np.int16)
        return IQPacket(target.target_type, target.sequence, target.center_hz,
                        target.bandwidth_hz, target.sample_rate_hz, samples, samples)

    def test_all_targets_covered_without_exceeding_channel_capacity(self):
        batches = target_batches(self.candidates())
        self.assertEqual([len(b) for b in batches], [130, 130, 45])
        self.assertEqual(sum(map(len, batches)), 305)
        for batch in batches:
            self.assertLessEqual(sum(t.target_type == 'wide' for t in batch), 2)
            self.assertLessEqual(sum(t.target_type == 'narrow' for t in batch), 128)
            self.assertEqual(len(build_target_batch(list(batch))), 32*len(batch))

    def test_threshold_and_optional_inclusive_filter(self):
        targets = [Target('wide', i, i*1000, bw, 320000000)
                   for i, bw in enumerate((1, 20000000, 20000001))]
        selected = target_batches(targets, TargetFilter(20000000, 20000001))[0]
        self.assertEqual([t.target_type for t in selected], ['narrow', 'wide'])
        self.assertEqual(len(target_batches(targets)[0]), 3)
        self.assertEqual(target_batches(targets, TargetFilter(2, 3)), [])

    def test_completion_barrier_and_snapshot_survive_spectrum_changes(self):
        schedule = RecognitionSchedule()
        batch = schedule.next_batch(self.candidates())
        self.assertIsNone(schedule.next_batch([]))
        for target in batch[:-1]:
            packet = self.packet(target)
            self.assertTrue(schedule.complete(packet))
            self.assertFalse(schedule.complete(packet))
        self.assertIsNone(schedule.next_batch([]))
        self.assertTrue(schedule.complete(self.packet(batch[-1])))
        second = schedule.next_batch([])
        self.assertEqual(len(second), 130)
        self.assertEqual(schedule.round, 1)
        for target in second:
            schedule.complete(self.packet(target))
        third = schedule.next_batch([])
        for target in third:
            schedule.complete(self.packet(target))
        self.assertEqual(schedule.next_batch([]), ())
        schedule.next_batch(self.candidates())
        self.assertEqual(schedule.round, 2)

    def test_short_wrong_and_timeout_packets_cannot_advance(self):
        schedule = RecognitionSchedule(timeout_s=1)
        target = schedule.next_batch(self.candidates())[0]
        packet = self.packet(target)
        self.assertFalse(schedule.complete(replace(packet, center_hz=packet.center_hz+1)))
        self.assertFalse(schedule.complete(replace(packet, i_data=packet.i_data[:-1])))
        self.assertFalse(schedule.complete(replace(packet, sample_rate_hz=packet.sample_rate_hz//2)))
        with self.assertRaises(TimeoutError):
            schedule.check_timeout(schedule.started+2)
        self.assertIsNone(schedule.next_batch(self.candidates()))
        self.assertFalse(schedule.complete(packet))

    def test_fpga_only_sends_next_batch_after_completions(self):
        source = SpectrumDDCClosedLoop()
        source._running = True
        source._latest_detected[('narrow', 0)] = self.candidates()
        with patch('fpga_closed_loop.send_target_batch') as send:
            source._send_current_targets()
            source._send_current_targets()
            self.assertEqual(send.call_count, 1)
            for target in list(source._schedule.pending.values()):
                source._schedule.complete(self.packet(target))
            source._send_current_targets()
            self.assertEqual(send.call_count, 2)
            self.assertEqual(len(send.call_args.args[0]), 130)
        source._running = False

    def test_queued_iq_waits_for_explicit_recognition_acknowledgement(self):
        source = SpectrumDDCClosedLoop(target_interval_s=0)
        source._running = True
        source._latest_detected[('narrow', 0)] = self.candidates()
        with patch('fpga_closed_loop.send_target_batch') as send:
            source._send_current_targets()
            targets = list(source._schedule.pending.values())
            for target in targets[:-1]:
                packet = self.packet(target)
                self.assertTrue(source._schedule.receive(packet))
                self.assertTrue(source.complete_iq_recognition(packet))
            packet = self.packet(targets[-1])
            self.assertTrue(source._schedule.receive(packet))
            self.assertFalse(source._schedule.receive(packet))  # Duplicate reception.
            source._iq_queue.put(packet)
            self.assertIs(source.get_iq_packet(), packet)
            source._schedule.check_timeout(source._schedule.started+1000)
            source._send_current_targets()
            self.assertEqual(send.call_count, 1)  # All IQ received; last result still missing.
            self.assertFalse(source.complete_iq_recognition(replace(packet, center_hz=packet.center_hz+1)))
            self.assertTrue(source.complete_iq_recognition(packet))
            self.assertEqual(send.call_count, 2)
            self.assertFalse(source.complete_iq_recognition(packet))
        source._running = False

    def test_delayed_software_ack_cannot_finish_an_identical_target_in_a_new_round(self):
        source = SpectrumDDCClosedLoop()
        target = Target('narrow', 0, 0, 1000000, 4000000)
        source._schedule.next_batch([target])
        old = self.packet(target)
        source._schedule.receive(old)
        self.assertTrue(source.complete_iq_recognition(old))
        source._schedule.next_batch([target])
        current = self.packet(target)
        source._schedule.receive(current)
        self.assertFalse(source.complete_iq_recognition(old))
        self.assertTrue(source._schedule.active)
        self.assertTrue(source.complete_iq_recognition(current))

    def test_file_ddc_assembles_exactly_four_mib(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.wav'
            raw = np.tile(np.array([[1200, -500]], dtype=np.int16), (2048, 1))
            write_rf64(path, raw, rate=102400)
            info = inspect_iq_file(path)
            target = Target('narrow', 0, 0, 10000, 40000)
            data, rate = read_channel_packet(info, 1700, target)
            self.assertEqual(data.nbytes, IQ_BYTES)
            self.assertEqual(rate, 40000)
            np.testing.assert_array_equal(data[[0, -1]], [[1200, -500], [1200, -500]])
            self.assertIsNone(read_channel_packet(info, 0, target, lambda: True))

    def test_ui_blank_and_invalid_ranges(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        control = RecognitionBandwidth()
        self.assertEqual(control.value(), TargetFilter())
        control.minimum.setText('20')
        self.assertEqual(control.value(), TargetFilter(20e6, None))
        control.maximum.setText('10')
        with self.assertRaises(ValueError):
            control.value()
        control.minimum.setText('nan')
        with self.assertRaises(ValueError):
            control.value()
        control.close()


if __name__ == '__main__':
    unittest.main()
