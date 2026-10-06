import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6 import QtWidgets

from fpga_closed_loop import Target, TargetFilter
from receiver.file_replay import ReplayWorker, FileReplayPage
from receiver.recognition_history import RecognitionHistoryDialog
from recognition_results import WindowPrediction, majority_result, IQ_PACKET_SAMPLES
from recognition_store import RecognitionStore


class ReplayPersistenceTests(unittest.TestCase):
    def test_worker_displays_persisted_vote_and_same_id_in_history(self):
        import torch
        from gpu_iq_recognizer import SigRecGpuRecognizer
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)/'scripted-test-model.pth'
            model_path.write_bytes(b'not a production checkpoint; loader is mocked')
            info = SimpleNamespace(path=Path(directory)/'capture.wav', sample_rate=320000000,
                                   samples=320000000)
            worker = ReplayWorker(info, 700e6, 300e6, 4, model_path)
            worker.database_path = Path(directory)/'recognition.sqlite3'
            class Reader:
                cycle = 1
                def __init__(self, info):
                    pass
                def read(self):
                    return np.zeros((32768, 2), dtype=np.int16), 3200000
                def close(self):
                    pass
            class Recognizer:
                device = 'cpu'
                class_names = ['4G', '5G', 'DVB', 'GSM', 'WiFi']
                def infer_packet_iq(self, iq, rate, **kwargs):
                    self_iq_shape = tuple(iq.shape)
                    if self_iq_shape != (IQ_PACKET_SAMPLES, 2):
                        raise AssertionError('incorrect packet length')
                    p = dict.fromkeys(self.class_names, .025)
                    p['WiFi'] = .9
                    half = IQ_PACKET_SAMPLES//2
                    windows = [WindowPrediction(i*half, half, 'WiFi', .9, dict(p)) for i in (0, 1)]
                    kwargs['progress'](2, 2)
                    return majority_result(windows, self.class_names, IQ_PACKET_SAMPLES, rate)
            def collect(info, start, channels, ready, cancelled, progress=None, accepts=None, device=None):
                for target, identifier in channels:
                    if cancelled():
                        return
                    ready(target, identifier, np.zeros((IQ_PACKET_SAMPLES, 2), dtype=np.int16), 4000000)
            original_save = RecognitionStore.save
            def save_then_stop(store, *args):
                identifier = original_save(store, *args)
                worker.cancel.set()  # Stop after exactly one committed test packet.
                return identifier
            target = Target('narrow', 0, 1000000, 1000000, 320000000)
            with patch.object(SigRecGpuRecognizer, 'from_checkpoint', return_value=Recognizer()), \
                 patch('receiver.file_replay.IQReplayReader', Reader), \
                 patch('receiver.file_replay.spectrum_power', return_value=np.ones(32768)*1e-10), \
                 patch('receiver.file_replay.detect_signals', return_value=[target]), \
                 patch('receiver.file_replay.collect_channel_batch', side_effect=collect), \
                 patch.object(RecognitionStore, 'save', save_then_stop):
                worker.run()
            state = worker.mailbox.get_nowait()
            self.assertNotIn('error', state)
            self.assertEqual(state['rows'][0][2:4], ['WiFi', '90.0%'])
            self.assertEqual(state['rows'][0][8:11], ['2', '100.0%', '1'])
            self.assertEqual(state['tracks'][0].identifier, 1)
            store = RecognitionStore(worker.database_path)
            try:
                packet = store.connection.execute('SELECT signal_id,frame,source_start_sample,sample_rate_hz FROM packets').fetchone()
                self.assertEqual(packet, (1, 1, 3200000, 4000000))
                self.assertEqual(store.connection.execute('SELECT count(*) FROM windows').fetchone()[0], 2)
            finally:
                store.close()
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            page = FileReplayPage()
            history = RecognitionHistoryDialog(worker.database_path)
            try:
                page._last_state = state
                page._render_targets(state, force_table=True)
                self.assertEqual(page.table.item(0, 0).text(), '701.000')
                self.assertEqual(page.table.item(0, 8).text(), '2')
                self.assertEqual(page.table.item(0, 9).text(), '100.0%')
                self.assertEqual(page.table.item(0, 10).text(), '1')
                self.assertEqual(history.table.item(0, 2).text(), 'S0001')
                history.table.selectRow(0)
                history.show_details()
                self.assertIn('图 2', history.details.toPlainText())
                self.assertIn('WiFi', history.details.toPlainText())
            finally:
                page.shutdown()
                page.close()
                history.close()

    def test_skipped_gui_frames_do_not_change_worker_signal_identity(self):
        info = SimpleNamespace()
        worker = ReplayWorker(info, 700e6, 300e6, 4, '')
        axis = np.linspace(550e6, 850e6, 32768)
        def publish(frame, centers):
            targets = [Target('narrow', 0, c, 100000, 400000) for c in centers]
            rows = [[f'{700+c/1e6:.6f}', '.100000', '未分类', '—', '窄带', '1', '.400000', '2.621440'] for c in centers]
            worker.publish(dict(frame=frame, axis=axis, targets=targets, rows=rows, center=700e6))
        publish(1, [2e6, 1e6])
        publish(2, [1e6+1000, 2e6-1000])  # GUI did not consume frame 1.
        latest = worker.mailbox.get_nowait()
        tracks = sorted(latest['tracks'], key=lambda t: t.target.center_hz)
        self.assertEqual([t.identifier for t in tracks], [1, 2])
        self.assertEqual(worker.tracker.identifier_for(latest['targets'][0]), 1)


if __name__ == '__main__':
    unittest.main()
