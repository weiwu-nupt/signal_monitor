"""Playback must progress independently of slow IQ collection and GPU work."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from fpga_closed_loop import Target
from receiver.file_replay import ReplayWorker
from recognition_results import IQ_PACKET_SAMPLES, WindowPrediction, majority_result
from recognition_store import RecognitionStore


def wait_until(predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return False


class AsyncReplayTests(unittest.TestCase):
    def test_new_detections_show_next_round_instead_of_claiming_current_batch(self):
        worker = ReplayWorker(SimpleNamespace(), 700e6, 300e6, 4, '')
        targets = [Target('narrow', i, i*2000000, 1000000, 4000000) for i in range(3)]
        rows = [['700', '1', '待识别', '—', '窄带', '1', '4', '.26',
                 '待采集', '待识别', '未保存'] for _ in targets]
        state = dict(frame=1, axis=np.array([0, 10000]), targets=targets, rows=rows)
        worker.publish(state)
        worker._round_busy = True
        worker._active_batch = 1
        worker._scheduled_ids = {1: 1, 2: 2}
        worker.publish(state)
        tracks = worker.mailbox.get_nowait()['tracks']
        self.assertEqual(tracks[1].row[2], '等待第2批')
        self.assertEqual(tracks[1].row[5], '2')
        self.assertEqual(tracks[2].row[2], '等待下一轮')
        self.assertEqual(tracks[2].row[5], '本轮未排入')

    def test_active_round_keeps_t0_targets_until_all_batches_finish(self):
        worker = ReplayWorker(SimpleNamespace(), 700e6, 300e6, 4, '')
        first = Target('narrow', 0, 1_000_000, 1_000_000, 4_000_000)
        later = Target('narrow', 0, 9_000_000, 1_000_000, 4_000_000)
        def state(frame, targets):
            return dict(frame=frame, axis=np.array([0, 10_000]), targets=targets,
                        rows=[['700', '1', '待识别', '—', '窄带', '1', '4', '.26',
                               '待采集', '待识别', '未保存'] for _ in targets])

        worker.publish(state(1, [first]))
        worker._round_busy = True
        worker._round_ids = (1,)
        worker._scheduled_ids = {1: 1}
        worker.publish(state(2, [later]))

        frozen = worker.mailbox.get_nowait()
        self.assertEqual([target.center_hz for target in frozen['targets']], [1_000_000])
        self.assertEqual([track.identifier for track in frozen['tracks']], [1])
        self.assertNotIn(2, worker.tracker.tracks)

    def test_failed_inference_does_not_start_the_next_batch(self):
        worker = ReplayWorker(SimpleNamespace(), 700e6, 300e6, 4, '')
        target = Target('narrow', 0, 0, 1000000, 4000000)
        job = dict(batches=[[(target, 1)], [(target, 2)]], frame=1, cycle=1,
                   start=0, multiplier=4, filter=worker.bandwidth_filter)
        started = []
        packet = np.zeros((IQ_PACKET_SAMPLES, 2), dtype=np.int16)
        def collect(info, start, channels, ready, cancelled, progress, accepts, device=None):
            started.append(channels[0][1])
            ready(target, channels[0][1], packet, target.sample_rate_hz)
        class Pipeline:
            def recognize_iq(self, *args, **kwargs):
                raise ValueError('test persistence failure')
        with patch('receiver.file_replay.collect_channel_batch', collect):
            with self.assertRaisesRegex(RuntimeError, '轮询已暂停'):
                for index, batch in enumerate(job['batches'], 1):
                    worker._recognize_batch(job, index, batch, Pipeline(), SimpleNamespace(device='cpu'))
        self.assertEqual(started, [1])
        self.assertEqual(worker._updates[1]['fields'][10], '未保存')

    def test_ready_packet_is_inferred_while_other_lane_collects_and_batch_waits_for_result(self):
        worker = ReplayWorker(SimpleNamespace(), 700e6, 300e6, 4, '')
        slow = Target('narrow', 0, -1000000, 1000000, 4000000)
        fast = Target('wide', 0, 50000000, 25000000, 100000000)
        next_target = Target('narrow', 0, 2000000, 1000000, 4000000)
        job = dict(batches=[[(slow, 1), (fast, 2)], [(next_target, 3)]],
                   frame=1, cycle=1, start=0, multiplier=4, filter=worker.bandwidth_filter)
        first_result, release_capture, slow_inference, release_inference = [threading.Event() for _ in range(4)]
        started_batches, results, errors = [], [], []
        packet = np.zeros((IQ_PACKET_SAMPLES, 2), dtype=np.int16)
        window = WindowPrediction(0, IQ_PACKET_SAMPLES, 'GSM', 1, {'GSM': 1})
        class Pipeline:
            def recognize_iq(self, iq, rate, metadata, **kwargs):
                identifier = metadata['signal_id']
                if identifier == 1:
                    slow_inference.set()
                    while not release_inference.wait(.01):
                        if worker.cancel.is_set():
                            raise InterruptedError()
                results.append(identifier)
                if identifier == 2:
                    first_result.set()
                return majority_result([window], ['GSM'], IQ_PACKET_SAMPLES, rate), identifier
        def collect(info, start, channels, ready, cancelled, progress, accepts, device=None):
            started_batches.append([identifier for _, identifier in channels])
            if len(channels) == 2:
                ready(fast, 2, packet, fast.sample_rate_hz)
                while not release_capture.wait(.01):
                    if cancelled():
                        return
                ready(slow, 1, packet, slow.sample_rate_hz)
            else:
                ready(next_target, 3, packet, next_target.sample_rate_hz)
        def process():
            try:
                for index, batch in enumerate(job['batches'], 1):
                    if not worker._recognize_batch(job, index, batch, Pipeline(), SimpleNamespace(device='cpu')):
                        break
            except Exception as exc:
                errors.append(exc)
        with patch('receiver.file_replay.collect_channel_batch', collect):
            thread = threading.Thread(target=process)
            thread.start()
            try:
                self.assertTrue(first_result.wait(5))
                self.assertEqual(results, [2])  # First result precedes the slow lane's ready callback.
                self.assertEqual(len(started_batches), 1)
                release_capture.set()
                self.assertTrue(slow_inference.wait(5))
                time.sleep(.1)
                self.assertEqual(len(started_batches), 1)  # All IQ arrived, last inference still running.
                release_inference.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(results, [2, 1, 3])
                self.assertEqual(started_batches, [[1, 2], [3]])
                self.assertEqual(errors, [])
            finally:
                worker.cancel.set()
                release_capture.set()
                release_inference.set()
                thread.join(5)

    def test_next_capture_preserves_last_vote_and_database_record(self):
        worker = ReplayWorker(SimpleNamespace(), 700e6, 300e6, 4, '')
        target = Target('narrow', 0, 0, 1000000, 4000000)
        state = dict(frame=1, axis=np.array([0, 10000]), targets=[target],
            rows=[['700', '1', '待识别', '—', '窄带', '1', '4', '.26',
                   '待采集', '待识别', '未保存']])
        worker.publish(state)
        worker._post_recognition(1, {2: 'WiFi', 3: '90%', 8: '9', 9: '88.9%', 10: '7'}, '')
        worker.publish(state)
        worker._post_recognition(1, {2: '采集 4 MiB IQ', 3: '—', 8: '采集 25.0%',
                                    9: '待识别', 10: '未保存'}, '')
        worker.publish({**state, 'frame': 2})
        row = worker.mailbox.get_nowait()['tracks'][0].row
        self.assertEqual(row[2:4], ['WiFi（上次）', '90%'])
        self.assertEqual(row[8:11], ['采集 25.0%', '88.9%', '7'])

    def test_slow_capture_keeps_frames_moving_and_finishes_frozen_batches(self):
        from gpu_iq_recognizer import SigRecGpuRecognizer
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            weights = Path(directory)/'test.pth'
            weights.write_bytes(b'mocked weights')
            info = SimpleNamespace(path=Path(directory)/'capture.wav', sample_rate=320e6,
                                   samples=320000000)
            worker = ReplayWorker(info, 700e6, 300e6, 4, weights)
            worker.database_path = Path(directory)/'results.sqlite3'
            entered, release = threading.Event(), threading.Event()
            read_frames, collected, saved = [], [], []
            targets = [Target('narrow', 0, i*2000000-128000000, 1000000, 4000000)
                       for i in range(129)]
            packet = np.zeros((IQ_PACKET_SAMPLES, 2), dtype=np.int16)

            class Reader:
                cycle = 1
                def __init__(self, info):
                    pass
                def read(self):
                    read_frames.append(len(read_frames)+1)
                    return np.zeros((32768, 2), dtype=np.int16), 3200000*len(read_frames)
                def close(self):
                    pass

            class Recognizer:
                device = 'cpu'
                class_names = ['GSM']
                def infer_packet_iq(self, iq, rate, **kwargs):
                    kwargs['progress'](1, 1)
                    window = WindowPrediction(0, IQ_PACKET_SAMPLES, 'GSM', 1, {'GSM': 1})
                    return majority_result([window], self.class_names, IQ_PACKET_SAMPLES, rate)

            def collect(info, start, channels, ready, cancelled, progress=None, accepts=None, device=None):
                for target, identifier in channels:
                    collected.append((start, target.center_hz, target.sequence))
                    if len(collected) == 1:
                        progress(identifier, IQ_PACKET_SAMPLES//4, IQ_PACKET_SAMPLES)
                        entered.set()
                        while not release.wait(.01):
                            if cancelled():
                                return
                    if cancelled():
                        return
                    ready(target, identifier, packet, target.sample_rate_hz)

            def detections(*args):
                # Spectrum candidates drift during the pending capture. The job
                # must retain its original source position, IDs, targets and slots.
                drift = 5000 if len(read_frames) > 1 else 0
                return [Target(t.target_type, t.sequence, t.center_hz+drift,
                               t.bandwidth_hz, t.sample_rate_hz) for t in targets]

            original_save = RecognitionStore.save
            def save(store, *args):
                result = original_save(store, *args)
                saved.append(result)
                if len(saved) == 129:
                    worker.cancel.set()
                return result

            for name, value in [('IQReplayReader', Reader), ('spectrum_power', lambda raw: np.ones(32768)),
                                ('detect_signals', detections), ('collect_channel_batch', collect)]:
                patches.enter_context(patch('receiver.file_replay.'+name, value))
            patches.enter_context(patch.object(SigRecGpuRecognizer, 'from_checkpoint', return_value=Recognizer()))
            patches.enter_context(patch.object(RecognitionStore, 'save', save))
            worker.thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(wait_until(lambda: len(read_frames) >= 4))
                deadline = time.monotonic()+5
                state = worker.mailbox.get(timeout=1)
                while state['frame'] < 4 and time.monotonic() < deadline:
                    state = worker.mailbox.get(timeout=1)
                self.assertGreaterEqual(state['frame'], 4)
                self.assertEqual(state['tracks'][0].row[8], '采集 25.0%')
                self.assertEqual(len(collected), 1)
                self.assertTrue(worker._round_busy)
                self.assertEqual(worker._jobs.qsize(), 0)
                worker.paused.set()
                time.sleep(.2)  # Allow any already-running spectrum computation to finish.
                count = len(read_frames)
                time.sleep(.2)
                self.assertEqual(len(read_frames), count)
                worker.paused.clear()
                release.set()
                worker.thread.join(10)
                self.assertFalse(worker.thread.is_alive())
                self.assertEqual(len(collected), 129)
                self.assertEqual({start for start, _, _ in collected}, {3200000})
                self.assertEqual(collected[-1][2], 0)  # Second batch reuses slot 0.
                with_store = RecognitionStore(worker.database_path)
                try:
                    records = with_store.connection.execute(
                        'SELECT signal_id, frame, batch, channel_slot, center_hz '
                        'FROM packets ORDER BY id').fetchall()
                    self.assertEqual([r[0] for r in records], list(range(1, 130)))
                    self.assertEqual({r[1] for r in records}, {1})
                    self.assertEqual(records[-1][2:4], (2, 0))
                    self.assertEqual(records[-1][4], 700e6+targets[-1].center_hz)
                finally:
                    with_store.close()
                latest = worker.mailbox.get_nowait()
                self.assertEqual(latest['tracks'][-1].row[2], 'GSM')
                self.assertEqual(latest['tracks'][-1].row[10], '129')
            finally:
                worker.cancel.set()
                release.set()
                worker.thread.join(5)

    def test_result_from_expired_id_cannot_label_a_new_signal(self):
        worker = ReplayWorker(SimpleNamespace(), 700e6, 300e6, 4, '')
        def state(frame, center):
            target = Target('narrow', 0, center, 1000000, 4000000)
            return dict(frame=frame, axis=np.array([0, 10000]), targets=[target],
                        rows=[['700', '1', '待识别', '—', '窄带', '1', '4', '.26',
                               '待采集', '待识别', '未保存']])
        worker.publish(state(1, 0))
        for frame in range(2, 13):
            worker.publish(state(frame, 10000000))
        self.assertNotIn(1, worker.tracker.tracks)
        worker._post_recognition(1, {2: 'WiFi', 3: '90%', 10: '7'}, ' | saved')
        worker.publish(state(13, 10000000))
        tracks = worker.mailbox.get_nowait()['tracks']
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].identifier, 2)
        self.assertEqual(tracks[0].row[2], '待识别')


if __name__ == '__main__':
    unittest.main()
