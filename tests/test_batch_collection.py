import tempfile
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from fpga_closed_loop import Target
from iq_file_source import inspect_iq_file, read_channel_packet, collect_channel_batch
from test_iq_file_replay import write_rf64


class BatchCollectionTests(unittest.TestCase):
    def test_gpu_batch_matches_cpu_ddc_across_file_loops_and_keeps_packets_on_gpu(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        rate = 102400
        n = np.arange(rate//10)
        iq = 5000*np.exp(2j*np.pi*6100*n/rate)+2000*np.exp(-2j*np.pi*11800*n/rate)
        raw = np.rint(np.column_stack((iq.real, iq.imag))).astype(np.int16)
        targets = [Target('narrow', 0, -12000, 1000, 4000),
                   Target('narrow', 1, 3100, 10000, 40000)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'gpu.wav'
            write_rf64(path, raw, rate)
            info = inspect_iq_file(path)
            packets = []
            with patch('fpga_closed_loop.IQ_BYTES', 4096):
                collect_channel_batch(info, 1700, [(t, i) for i, t in enumerate(targets)],
                    lambda t, i, data, fs: packets.append((i, data, fs)), device='cuda:0')
                self.assertEqual([p[0] for p in packets], [1, 0])
                for identifier, packet, output_rate in packets:
                    self.assertEqual(packet.device.type, 'cuda')
                    self.assertEqual(packet.shape, (1024, 2))
                    expected, rate = read_channel_packet(info, 1700, targets[identifier])
                    delta = np.abs(packet.cpu().numpy().astype(int)-expected.astype(int))
                    self.assertLessEqual(delta.max(), 1)
                    self.assertEqual(output_rate, rate)

    def test_fast_lane_is_delivered_before_slow_lane_and_shared_fft_preserves_iq(self):
        rate = 102400
        n = np.arange(rate//10)
        iq = (5000*np.exp(2j*np.pi*6100*n/rate)
              + 2000*np.exp(-2j*np.pi*11800*n/rate))
        raw = np.rint(np.column_stack((iq.real, iq.imag))).astype(np.int16)
        slow = Target('narrow', 0, -12000, 1000, 4000)
        fast = Target('narrow', 1, 3100, 10000, 40000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lanes.wav'
            write_rf64(path, raw, rate)
            info = inspect_iq_file(path)
            progress, packets, incomplete_at_first = {}, [], []
            def ready(target, identifier, packet, output_rate):
                if not packets:
                    incomplete_at_first.append(progress[1] < 1024)
                packets.append((identifier, packet, output_rate))
            with patch('fpga_closed_loop.IQ_BYTES', 4096):
                with patch('iq_file_source.np.fft.fft', wraps=np.fft.fft) as fft:
                    collect_channel_batch(info, 1700, [(slow, 1), (fast, 2)], ready,
                        progress=lambda identifier, done, total: progress.update({identifier: done}))
                    self.assertEqual(fft.call_count, 26)  # One FFT per shared input step.
                self.assertEqual([p[0] for p in packets], [2, 1])
                self.assertEqual(incomplete_at_first, [True])
                for identifier, packet, output_rate in packets:
                    self.assertEqual(packet.nbytes, 4096)
                    target = fast if identifier == 2 else slow
                    expected, rate = read_channel_packet(info, 1700, target)
                    np.testing.assert_array_equal(packet, expected)
                    self.assertEqual(output_rate, rate)

    def test_cancel_after_one_packet_does_not_fabricate_remaining_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cancel.wav'
            write_rf64(path, np.zeros((10240, 2), dtype=np.int16), 102400)
            info = inspect_iq_file(path)
            cancel = threading.Event()
            packets = []
            def ready(target, identifier, packet, rate):
                packets.append(identifier)
                cancel.set()
            with patch('fpga_closed_loop.IQ_BYTES', 4096):
                collect_channel_batch(info, 0,
                    [(Target('narrow', 0, 0, 1000, 4000), 1),
                     (Target('narrow', 1, 20000, 10000, 40000), 2)], ready, cancel.is_set)
            self.assertEqual(packets, [2])


if __name__ == '__main__':
    unittest.main()
