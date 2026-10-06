import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from recognition_results import (WindowPrediction, majority_result, window_spans,
                                 IQ_PACKET_SAMPLES, effective_window_ms)
from recognition_store import RecognitionStore
from recognition_pipeline import PacketRecognitionPipeline


def prediction(start, count, a, b):
    probs = {'A': a, 'B': b}
    winner = max(probs, key=probs.get)
    return WindowPrediction(start, count, winner, probs[winner], probs)


class PacketRecognitionTests(unittest.TestCase):
    def test_automatic_windows_start_at_ten_ms_cap_images_and_cover_the_entire_packet(self):
        for rate in (156200, 859200, 4_000_000, 116718600, 320_000_000):
            for maximum in (1, 63, 64, 512):
                duration = effective_window_ms(IQ_PACKET_SAMPLES, rate, 10, 1024, maximum)
                self.assertGreaterEqual(duration, 10)
                spans = window_spans(IQ_PACKET_SAMPLES, rate, duration, 1024)
                self.assertLessEqual(len(spans), maximum)
                cursor = 0
                for start, count in spans:
                    self.assertEqual(start, cursor)
                    cursor += count
                self.assertEqual(cursor, IQ_PACKET_SAMPLES)
        self.assertAlmostEqual(effective_window_ms(IQ_PACKET_SAMPLES, 859200, 10, 64, 64),
                               19.068901303538173)
        self.assertEqual(effective_window_ms(IQ_PACKET_SAMPLES, 859200, 10, 64, None), 10)
        self.assertEqual(len(window_spans(IQ_PACKET_SAMPLES, 116718600, 10, 1024)), 1)

    def test_windows_cover_packet_including_short_tail_without_overlap(self):
        for rate in (40_000, 4_000_000, 63_554_600, 116_718_600, 409_600_000):
            spans = window_spans(IQ_PACKET_SAMPLES, rate, 1, 1024)
            self.assertGreater(len(spans), 1)
            cursor = 0
            for start, count in spans:
                self.assertEqual(start, cursor)
                self.assertGreaterEqual(count, 1024)
                cursor += count
            self.assertEqual(cursor, IQ_PACKET_SAMPLES)
        self.assertEqual(window_spans(2150, 1e6, 1, 1024), [(0, 1024), (1024, 1126)])
        self.assertEqual(window_spans(1048576, 320e6, 10, 1024), [(0, 1048576)])
        for duration in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                window_spans(IQ_PACKET_SAMPLES, 1e6, duration)

    def test_majority_overrides_mean_probabilities_and_reports_both_metrics(self):
        windows = [prediction(0, 100, .51, .49), prediction(100, 100, .51, .49),
                   prediction(200, 100, .01, .99)]
        result = majority_result(windows, ['A', 'B'], 300, 100000)
        self.assertEqual(result.class_name, 'A')
        self.assertEqual(result.vote_counts, {'A': 2, 'B': 1})
        self.assertAlmostEqual(result.vote_fraction, 2/3)
        self.assertAlmostEqual(result.confidence, 1.03/3)
        self.assertLess(result.confidence, result.probabilities['B'])
        self.assertFalse(result.tied)

    def test_tie_uses_probabilities_then_stable_class_order(self):
        windows = [prediction(0, 100, .6, .4), prediction(100, 100, .1, .9)]
        result = majority_result(windows, ['A', 'B'], 200, 100000)
        self.assertTrue(result.tied)
        self.assertEqual(result.class_name, 'B')
        self.assertEqual(result.vote_fraction, .5)
        equal = [prediction(0, 100, .6, .4), prediction(100, 100, .4, .6)]
        self.assertEqual(majority_result(equal, ['A', 'B'], 200, 100000).class_name, 'A')
        # Full coverage validation prevents an incomplete inference becoming a packet result.
        with self.assertRaises(ValueError):
            majority_result(windows[:1], ['A', 'B'], 200, 100000)

    def test_database_commits_summary_and_votes_and_supports_concurrent_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'recognition.sqlite3'
            store = RecognitionStore(path)
            try:
                session = store.start_session('iq.wav', 'test-model.pth', {'window_ms': 1}, model_sha256='test')
                windows = [prediction(0, 100, .6, .4), prediction(100, 100, .1, .9)]
                result = majority_result(windows, ['A', 'B'], 200, 100000)
                metadata = dict(center_hz=700123000, bandwidth_hz=25000,
                                signal_id=1, frame=2, cycle=3, source_start_sample=3200000)
                identifier = store.save(session, result, metadata)
                reader = sqlite3.connect(path)
                try:
                    saved = reader.execute('SELECT class_name,image_count,sample_rate_hz FROM packets').fetchone()
                    self.assertEqual(saved, ('B', 2, 100000))
                    self.assertEqual(reader.execute('SELECT count(*) FROM windows').fetchone()[0], 2)
                    self.assertEqual(store.recent()[0]['id'], identifier)
                    probs = reader.execute('SELECT probabilities_json FROM windows WHERE image_index=2').fetchone()[0]
                    self.assertEqual(json.loads(probs), {'A': .1, 'B': .9})
                finally:
                    reader.close()
            finally:
                store.close()
            # Results survive connection close/reopen.
            reopened = RecognitionStore(path)
            try:
                self.assertEqual(reopened.recent()[0]['class_name'], 'B')
            finally:
                reopened.close()

    def test_failed_window_write_rolls_back_packet(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as directory:
            store = RecognitionStore(Path(directory)/'results.db')
            try:
                session = store.start_session('iq', 'model', {}, model_sha256='test')
                result = majority_result([prediction(0, 100, .6, .4)], ['A', 'B'], 100, 100000)
                # Summary is valid, but per-image serialization fails inside the transaction.
                broken = replace(result, windows=(replace(result.windows[0], probabilities={'A': float('nan')}),))
                with self.assertRaises(ValueError):
                    store.save(session, broken, dict(center_hz=700e6, bandwidth_hz=25000))
                self.assertEqual(store.connection.execute('SELECT count(*) FROM packets').fetchone()[0], 0)
                self.assertEqual(store.connection.execute('SELECT count(*) FROM windows').fetchone()[0], 0)
                self.assertEqual(store.connection.execute('SELECT count(*) FROM sessions').fetchone()[0], 1)
            finally:
                store.close()

    def test_cancelled_inference_never_persists_a_partial_packet(self):
        class CancelledRecognizer:
            def infer_packet_iq(self, *args, **kwargs):
                raise InterruptedError('cancelled')
        class Store:
            def save(self, *args):
                raise AssertionError('must not save incomplete inference')
        pipeline = PacketRecognitionPipeline(CancelledRecognizer(), Store(), 'session')
        with self.assertRaises(InterruptedError):
            pipeline.recognize_iq(None, 1e6, {})


if __name__ == '__main__':
    unittest.main()
