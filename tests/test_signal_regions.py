"""Regression cases for occupied-band edges, internal notches and adjacent carriers."""
import unittest
import numpy as np

from fpga_closed_loop import EnergyDetector


class SignalRegionTests(unittest.TestCase):
    fs = 100_000_000
    count = 10000

    def signal(self):
        return np.full(self.count, 1e-10)

    def detect(self, power, **kwargs):
        return EnergyDetector(smoothing=0, **kwargs).detect(
            power, 0, self.fs, 'narrow', len(power))

    def test_narrow_notch_between_similar_plateaus_is_one_signal(self):
        power = self.signal()
        power[2000:3600] = 1e-7
        power[2790:2810] = 1e-10  # 200 kHz notch in a 16 MHz signal
        targets = self.detect(power)
        large = [t for t in targets if t.bandwidth_hz > 1e6]
        self.assertEqual(len(large), 1)
        self.assertAlmostEqual(large[0].bandwidth_hz, 16e6, delta=10000)
        self.assertAlmostEqual(large[0].center_hz, -22.005e6, delta=10000)

    def test_real_gap_and_unequal_plateaus_are_not_merged(self):
        for gap, second_power in ((100, 1e-7), (20, 1e-6)):
            with self.subTest(gap=gap):
                power = self.signal()
                power[2000:2800] = 1e-7
                power[2800+gap:3600+gap] = second_power
                large = [t for t in self.detect(power) if t.bandwidth_hz > 1e6]
                self.assertEqual(len(large), 2)

    def test_nearby_strong_peak_cannot_move_band_centre(self):
        power = self.signal()
        power[2000:3600] = 1e-7
        power[3630:3650] = 1.0  # inside the former ten-percent guard
        broad = [t for t in self.detect(power) if t.bandwidth_hz > 1e6]
        self.assertEqual(len(broad), 1)
        target = broad[0]
        self.assertAlmostEqual(target.center_hz, -22.005e6, delta=10000)
        self.assertAlmostEqual(target.bandwidth_hz, 16e6, delta=10000)

    def test_weak_skirt_does_not_bridge_wide_signal_and_neighbour(self):
        power = self.signal()
        power[1000:5000] = 3e-10  # about 4.8 dB above noise
        power[1200:1350] = 1e-6
        power[1800:4800] = 1e-5
        large = [t for t in self.detect(power) if t.bandwidth_hz > 1e6]
        self.assertEqual(len(large), 2)
        self.assertEqual(sorted(t.bandwidth_hz for t in large), [1_500_000, 30_000_000])

    def test_capture_guards_do_not_change_measured_bandwidth(self):
        power = self.signal()
        power[2000:2200] = 1e-6
        a = self.detect(power, relative_guard=0, min_guard_hz=0)
        b = self.detect(power, relative_guard=.5, min_guard_hz=200000)
        self.assertEqual(a, b)
        self.assertEqual(a[0].bandwidth_hz, 2_000_000)

    def test_low_snr_wide_plateau_is_retained(self):
        power = self.signal()
        power[2000:3600] = 5e-10  # 7 dB, below a fixed 12 dB plateau threshold
        targets = self.detect(power)
        self.assertTrue(any(t.bandwidth_hz >= 15e6 for t in targets))


if __name__ == '__main__':
    unittest.main()
