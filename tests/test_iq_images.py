import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6 import QtWidgets
from fpga_closed_loop import Target
from iq_images import preview_images
from recognition_results import IQ_PACKET_SAMPLES
from recognition_settings import remembered_checkpoint, remember_checkpoint
from receiver.file_replay import ReplayWorker, FileReplayPage
from receiver.iq_image_preview import IQImagePreviewDialog


class IQImageTests(unittest.TestCase):
    def test_real_iq_image_orientation_and_bounded_packet_preview(self):
        n = np.arange(1024)
        tone = 18000*np.exp(2j*np.pi*.25*n)
        positive = np.column_stack((tone.real, tone.imag)).astype(np.int16)
        negative = np.column_stack((tone.real, -tone.imag)).astype(np.int16)
        top, _, _ = preview_images(positive, 1024000, 1)
        bottom, _, _ = preview_images(negative, 1024000, 1)
        top_peak = np.argmax(top[0]['rgb'].sum(axis=2).mean(axis=1))
        bottom_peak = np.argmax(bottom[0]['rgb'].sum(axis=2).mean(axis=1))
        self.assertLess(top_peak, 112)
        self.assertGreater(bottom_peak, 112)
        packet = np.zeros((IQ_PACKET_SAMPLES, 2), dtype=np.int16)
        images, total, nfft = preview_images(packet, 116718600, 1)
        self.assertEqual(total, 9)
        self.assertEqual(len(images), 8)
        self.assertEqual(images[0]['start_sample'], 0)
        self.assertEqual(images[-1]['start_sample']+images[-1]['sample_count'], IQ_PACKET_SAMPLES)
        self.assertEqual(nfft, 1024)
        self.assertNotIn('class_name', images[0])  # Preview cannot claim neural inference.
        with self.assertRaises(InterruptedError):
            preview_images(packet, 116718600, cancelled=lambda: True)

    def test_checkpoint_preferences_ignore_missing_or_invalid_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory)/'settings.json'
            weights = Path(directory)/'model.pth'
            self.assertEqual(remembered_checkpoint(settings), '')
            weights.write_bytes(b'test-only')
            remember_checkpoint(weights, settings)
            self.assertEqual(remembered_checkpoint(settings), str(weights.resolve()))
            weights.unlink()
            self.assertEqual(remembered_checkpoint(settings), '')
            settings.write_text('[]', encoding='utf-8')
            self.assertEqual(remembered_checkpoint(settings), '')

    def test_no_model_and_failed_model_are_explicit_and_do_not_infer(self):
        from gpu_iq_recognizer import SigRecGpuRecognizer
        for selected, failed in (('', False), ('missing.pth', True)):
            info = SimpleNamespace(path=Path('capture.wav'), sample_rate=320000000, samples=320000000)
            worker = ReplayWorker(info, 700e6, 300e6, 4, selected)
            class Reader:
                cycle = 1
                def __init__(self, info):
                    pass
                def read(self):
                    worker.cancel.set()
                    return np.zeros((32768, 2), dtype=np.int16), 6400000
                def close(self):
                    pass
            target = Target('narrow', 0, 1000000, 100000, 400000)
            with patch.object(SigRecGpuRecognizer, 'from_checkpoint', side_effect=RuntimeError('test model load failure')), \
                 patch('receiver.file_replay.IQReplayReader', Reader), \
                 patch('receiver.file_replay.spectrum_power', return_value=np.ones(32768)), \
                 patch('receiver.file_replay.detect_signals', return_value=[target]), \
                 patch('receiver.file_replay.collect_channel_batch') as collect:
                worker.run()
                collect.assert_not_called()
            state = worker.mailbox.get_nowait()
            self.assertEqual(state['source_start_sample'], 6400000)
            self.assertEqual(state['rows'][0][2], '模型加载失败' if failed else '未加载模型')
            self.assertEqual(state['rows'][0][8:11], ['未生成', '未识别', '未保存'])
            self.assertIn('加载失败' if failed else '未启用', state['nn_state'])

    def test_cached_network_images_are_visible_with_real_window_class_labels(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        target = Target('narrow', 0, 1000000, 100000, 400000)
        image = dict(image_index=1, start_sample=0, sample_count=400,
                     rgb=np.zeros((224, 224, 3), dtype=np.uint8), class_name='WiFi', confidence=.9)
        cached = dict(images=[image], center_hz=701e6, sample_rate_hz=400000, frame=3)
        dialog = IQImagePreviewDialog(None, 0, target, 701e6, 1, cached=cached)
        try:
            self.assertIsNone(dialog.thread)
            self.assertFalse(dialog._image_labels[0].pixmap().isNull())
            captions = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
            self.assertTrue(any('WiFi：90.0%' in text for text in captions))
        finally:
            dialog.reject()

    def test_preview_pauses_file_worker_and_restores_previous_pause_state(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        page = FileReplayPage()
        target = Target('narrow', 0, 1000000, 100000, 400000)
        state = dict(frame=1, axis=np.linspace(550e6, 850e6, 32768), center=700e6,
                     targets=[target], rows=[['701.000000', '.100000', '未加载模型', '—',
                                            '窄带', '1', '.400000', '2.621440']])
        page.info = SimpleNamespace()
        page._last_state = state
        page._render_targets(state, force_table=True)
        page.table.selectRow(0)
        worker = SimpleNamespace(paused=threading.Event())
        page.worker = worker
        try:
            with patch('receiver.iq_image_preview.IQImagePreviewDialog') as constructor:
                constructor.return_value.exec.side_effect = lambda: self.assertTrue(worker.paused.is_set())
                page.show_images()
                self.assertFalse(worker.paused.is_set())
                worker.paused.set()
                page.show_images()
                self.assertTrue(worker.paused.is_set())
        finally:
            page.worker = None
            page.shutdown()
            page.close()


if __name__ == '__main__':
    unittest.main()
