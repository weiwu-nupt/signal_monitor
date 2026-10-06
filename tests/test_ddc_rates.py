import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

from ddc_rates import output_sample_rate, prepare_targets
from fpga_closed_loop import Target, build_target_batch
from iq_file_source import extract_channel, inspect_iq_file, read_channel_packet
from test_iq_file_replay import write_rf64
from recognition_schedule import target_batches, RecognitionSchedule


class DDCRateTests(unittest.TestCase):
    def test_policy_three_four_times_and_input_limit(self):
        self.assertEqual(output_sample_rate(1_000_000, 320_000_000), 4_000_000)
        self.assertEqual(output_sample_rate(1_000_000, 320_000_000, 3), 3_000_000)
        self.assertEqual(output_sample_rate(1_000_000, 320_000_000, 3.5), 3_500_000)
        self.assertEqual(output_sample_rate(100_000_000, 320_000_000), 320_000_000)
        for width in (19531, 332031, 29179688):
            rate = output_sample_rate(width, 320_000_000)
            self.assertGreaterEqual(rate, 3*width)
            self.assertLessEqual(rate, 4*width)
        with self.assertRaises(ValueError):
            output_sample_rate(110_000_000, 320_000_000)

    def test_h2c_field_carries_output_rate_and_slots_still_obey_limits(self):
        targets = [Target('narrow', i, i*10000, 1_000_000, 320_000_000) for i in range(129)]
        prepared, rejected = prepare_targets(targets, 320_000_000)
        self.assertFalse(rejected)
        batches = target_batches(prepared)
        self.assertEqual([len(b) for b in batches], [128, 1])
        data = build_target_batch(list(batches[0]))
        for offset in range(0, len(data), 32):
            self.assertEqual(struct.unpack_from('<I', data, offset+18)[0], 4_000_000)

    def test_ddc_preserves_tone_and_rejects_aliasing_neighbour(self):
        input_rate, output_rate = 3_200_000, 400_000
        count = input_rate//50  # 20 ms with 5 ms guards either side
        center, wanted_offset, interferer_offset = 30037, 12300, 1212300
        start = 12345
        time = (start+np.arange(count))/input_rate
        iq = (5000*np.exp(2j*np.pi*(center+wanted_offset)*time)
              + 5000*np.exp(2j*np.pi*(center+interferer_offset)*time))
        raw = np.rint(np.column_stack((iq.real, iq.imag))).astype(np.int16)
        channel, rate = extract_channel(raw, input_rate, center, 100000,
                                        output_rate_hz=output_rate, input_start=start)
        self.assertEqual(rate, output_rate)
        self.assertEqual(len(channel), output_rate//50)
        output = channel[:, 0].astype(float)+1j*channel[:, 1]
        output_time = start/input_rate+np.arange(len(output))/output_rate
        expected = 5000*np.exp(2j*np.pi*wanted_offset*output_time)
        trim = output_rate//200
        # Near-neighbour aliases onto the wanted tone without a low-pass filter.
        error = np.sqrt(np.mean(np.abs(output[trim:-trim]-expected[trim:-trim])**2))
        self.assertLess(error, 30)

    def test_low_rate_batch_timeout_includes_acquisition_time(self):
        schedule = RecognitionSchedule(timeout_s=1)
        t = Target('narrow', 0, 0, 1000, 4000)
        schedule.next_batch([t])
        schedule.check_timeout(schedule.started+2)
        with self.assertRaises(TimeoutError):
            schedule.check_timeout(schedule.started+264)

    def test_full_packet_retains_phase_through_blocks_and_file_loops(self):
        input_rate, output_rate = 102400, 40000
        center, offset, start = 3101, 3000, 1700
        iq = 5000*np.exp(2j*np.pi*(center+offset)*np.arange(input_rate)/input_rate)
        raw = np.rint(np.column_stack((iq.real, iq.imag))).astype(np.int16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'phase.wav'
            write_rf64(path, raw, rate=input_rate)
            target = Target('narrow', 0, center, 10000, output_rate)
            packet, rate = read_channel_packet(inspect_iq_file(path), start, target)
        self.assertEqual(packet.nbytes, 4*1024*1024)
        self.assertEqual(rate, output_rate)
        # Samples on both sides of many output-block and file-loop boundaries.
        indices = np.r_[np.arange(390, 411), np.arange(39990, 40011),
                        np.arange(79990, 80011), np.arange(len(packet)-21, len(packet))]
        expected = 5000*np.exp(2j*np.pi*offset*(start/input_rate+indices/output_rate))
        actual = packet[indices, 0].astype(float)+1j*packet[indices, 1]
        self.assertLess(float(np.max(np.abs(actual-expected))), 30)


if __name__ == '__main__':
    unittest.main()
