from pathlib import Path
import tempfile
import unittest

from sigrec_model import discover_sigrec_checkpoint, training_paths


class SigRecDiscoveryTests(unittest.TestCase):
    def test_discovers_training_output_outside_reference_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root/'sigRec'
            reference.mkdir()
            output = root/'trained-output'
            output.mkdir()
            checkpoint = output/'best_resnet_openset.pth'
            checkpoint.write_bytes(b'test-only')
            script = reference/'signal_openmax.py'
            script.write_text('import os\nOUTPUT_DIR = '+repr(str(output))+
                '\nMODEL_SAVE = os.path.join(OUTPUT_DIR, "best_resnet_openset.pth")\n'
                'raise RuntimeError("Discovery must not execute this script")\n', encoding='utf-8')
            self.assertEqual(discover_sigrec_checkpoint(reference), str(checkpoint.resolve()))
            self.assertEqual(training_paths(script)['MODEL_SAVE'], str(checkpoint))
            checkpoint.unlink()
            self.assertEqual(discover_sigrec_checkpoint(reference), '')


class SigRecCheckpointTests(unittest.TestCase):
    def test_class_count_and_metadata_order_control_network_outputs(self):
        import torch
        from gpu_iq_recognizer import _checkpoint_model, DEFAULT_CLASS_NAMES
        five = {'fc.1.weight': torch.zeros(5, 2048)}
        six = {'fc.1.weight': torch.zeros(6, 2048)}
        self.assertEqual(_checkpoint_model(five)[1], DEFAULT_CLASS_NAMES)
        self.assertEqual(_checkpoint_model(six)[1], (*DEFAULT_CLASS_NAMES, 'Unknown'))
        reordered = ['WiFi', '4G', '5G', 'DVB', 'GSM']
        self.assertEqual(_checkpoint_model({'model': five, 'known_classes': reordered})[1], tuple(reordered))
        self.assertEqual(_checkpoint_model({'model': six, 'known_classes': reordered})[1], (*reordered, 'Unknown'))
        with self.assertRaises(ValueError):
            _checkpoint_model(six, DEFAULT_CLASS_NAMES)
        with self.assertRaises(ValueError):
            _checkpoint_model({'fc.1.weight': torch.zeros(7, 2048)})

    def test_unknown_votes_are_preserved_in_packet_majority(self):
        from recognition_results import WindowPrediction, majority_result
        from gpu_iq_recognizer import DEFAULT_OPENSET_CLASS_NAMES
        names = DEFAULT_OPENSET_CLASS_NAMES
        windows = []
        for i, name in enumerate(('WiFi', 'Unknown', 'Unknown')):
            p = dict.fromkeys(names, .02)
            p[name] = .9
            windows.append(WindowPrediction(i*1024, 1024, name, .9, p))
        result = majority_result(windows, names, 3072, 1024000)
        self.assertEqual(result.class_name, 'Unknown')
        self.assertEqual(result.vote_counts['Unknown'], 2)
        self.assertAlmostEqual(result.vote_fraction, 2/3)


if __name__ == '__main__':
    unittest.main()
