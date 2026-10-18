import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import struct
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PyQt6 import QtWidgets

from iq_file_source import (inspect_iq_file, IQReplayReader, spectrum_power,
                            extract_channel, detect_signals)
from fpga_closed_loop import EnergyDetector
from receiver.file_replay import FileReplayPage, smooth_spectrum
from receiver.recognition_history import RecordedSpectrumDialog


def write_rf64(path, raw, rate=320000):
    def chunk(tag, data):
        return tag + struct.pack('<I', len(data)) + data + b'\0' * (len(data) % 2)
    data = raw.astype('<i2').tobytes()
    body = chunk(b'ds64', struct.pack('<QQQI', 0, len(data), len(raw), 0))
    body += chunk(b'JUNK', b'abc')
    body += chunk(b'fmt ', struct.pack('<HHIIHH', 1, 2, rate, rate*4, 4, 16))
    body += b'data' + struct.pack('<I', 0xffffffff) + data
    path.write_bytes(b'RF64\xff\xff\xff\xffWAVE' + body)


class FileTests(unittest.TestCase):
    def test_rf64_chunks_and_loop_preserve_iq(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test_fc700MHz.wav'
            raw = np.arange(22, dtype=np.int16).reshape(-1, 2)
            write_rf64(path, raw, rate=1000)
            info = inspect_iq_file(path)
            self.assertEqual((info.samples, info.sample_rate, info.center_hz), (11, 1000, 700e6))
            reader = IQReplayReader(info, .006)
            try:
                np.testing.assert_array_equal(reader.read()[0], raw[:6])
                np.testing.assert_array_equal(reader.read()[0], raw[6:])
                np.testing.assert_array_equal(reader.read()[0], raw[:6])
                self.assertEqual(reader.cycle, 2)
            finally:
                reader.close()
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, '截断'):
                inspect_iq_file(path)

    def test_fft_iq_order_and_channel_translation(self):
        rate = 3276800
        count = 32768
        tone = 10000*np.exp(2j*np.pi*400000*np.arange(count)/rate)
        raw = np.column_stack((tone.real, tone.imag)).astype(np.int16)
        power = spectrum_power(raw)
        frequency = np.fft.fftshift(np.fft.fftfreq(count, 1/rate))
        self.assertAlmostEqual(frequency[np.argmax(power)], 400000)
        channel, output_rate = extract_channel(raw, rate, 400000, 100000)
        self.assertGreater(output_rate, 100000)
        self.assertAlmostEqual(float(channel[:, 0].mean()), 10000, delta=2)
        self.assertLess(np.abs(channel[:, 1]).max(), 2)

    def test_detector_frequency_is_invariant_to_power_scale(self):
        power = np.full(8192, 1e-10)
        power[5000:5040] = 1e-7
        a = detect_signals(power, 320e6, 300e6, EnergyDetector(smoothing=0))
        b = detect_signals(power*1e12, 320e6, 300e6, EnergyDetector(smoothing=0))
        self.assertTrue(a)
        self.assertEqual(a[0].center_hz, b[0].center_hz)
        self.assertGreater(a[0].center_hz, 30e6)

    def test_display_smoothing_preserves_length_and_reduces_single_bin_spike(self):
        source = np.array([-80, -80, -20, -80, -80], dtype=np.float32)
        smoothed = smooth_spectrum(source, 3)
        self.assertEqual(smoothed.shape, source.shape)
        self.assertLess(smoothed[2], source[2])
        self.assertGreater(smoothed[2], source[1])
        np.testing.assert_array_equal(smooth_spectrum(source, 1), source)

    def test_recorded_spectrum_replay_reads_saved_source_position(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            samples = np.zeros((6400, 2), dtype=np.int16)
            samples[:, 0] = 1000
            packet = Path(directory) / 'iq_packets' / 'packet-1.npz'
            packet.parent.mkdir()
            np.savez_compressed(packet, iq=samples, sample_rate_hz=np.int64(320000))
            dialog = RecordedSpectrumDialog({
                'center_hz': 700100000.0, 'bandwidth_hz': 100000,
                'iq_path': 'iq_packets/packet-1.npz',
            }, directory)
            try:
                frequency, values = dialog.curve.getData()
                self.assertEqual(len(frequency), len(values))
                self.assertEqual(len(frequency), 32768)
                self.assertAlmostEqual(sum(dialog.band.getRegion()) / 2, 700100000.0)
                self.assertEqual(dialog.frame_count, 2)
                before = dialog.plot.getViewBox().viewRange()
                dialog.next_button.click()
                self.assertEqual(dialog.frame_slider.value(), 1)
                self.assertIn('第 2/2 帧', dialog.frame_label.text())
                self.assertEqual(dialog.plot.getViewBox().viewRange(), before)
            finally:
                dialog.close()
                app.processEvents()

    def test_page_worker_pause_stop_and_restart(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test_fc700MHz.wav'
            write_rf64(path, np.zeros((6400, 2), dtype=np.int16))
            page = FileReplayPage()
            page.checkpoint = ''  # This lifecycle test intentionally exercises detection without a model.
            page.info = inspect_iq_file(path)
            page.center.setValue(700)
            page.band.setValue(.3)
            try:
                page.start()
                low, high = 699850000, 700150000
                for plot in (page.spectrum, page.waterfall):
                    view = plot.getViewBox()
                    view.setXRange(low-1e6, high+1e6, padding=0)
                    view.scaleBy((10, 1))
                    view.translateBy(x=-1e9)
                    self.assertGreaterEqual(view.viewRange()[0][0], low-.01)
                    self.assertLessEqual(view.viewRange()[0][1], high+.01)
                    view.setXRange(low+10000, low+20000, padding=0)
                    view.translateBy(x=1e9)
                    self.assertGreaterEqual(view.viewRange()[0][0], low-.01)
                    self.assertLessEqual(view.viewRange()[0][1], high+.01)
                deadline = time.monotonic()+5
                while page._frame is None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(.02)
                self.assertIsNotNone(page._frame)
                page.pause_button.setChecked(True)
                frame = page._frame
                for _ in range(10):
                    app.processEvents()
                    time.sleep(.02)
                self.assertEqual(page._frame, frame)
                page.stop()
                page.worker.thread.join(2)
                page.poll()
                self.assertIsNone(page.worker)
                page.start()
                self.assertIsNotNone(page.worker)
            finally:
                page.shutdown()
                page.close()


if __name__ == '__main__':
    unittest.main()
