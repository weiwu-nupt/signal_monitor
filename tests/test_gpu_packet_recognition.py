"""Real CUDA preprocessing; scripted logits verify routing, never model accuracy."""
import struct
import unittest
import threading
from unittest.mock import patch

try:
    import torch
    import numpy as np
    from gpu_iq_recognizer import (SigRecGpuRecognizer, FPGA_IQ_PACKET_BYTES,
                                  FPGA_SYNC_WORD, FPGA_IQ_MESSAGE_LENGTH)
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

from recognition_results import IQ_PACKET_SAMPLES


@unittest.skipUnless(CUDA_AVAILABLE, 'CUDA/PyTorch is unavailable')
class GpuPacketRecognitionTests(unittest.TestCase):
    def test_default_automatic_mode_votes_on_all_64_real_windows(self):
        class Model(torch.nn.Module):
            def forward(self, images):
                logits = torch.zeros((len(images), 5), device=images.device)
                logits[:, 3] = 5
                return logits
        recognizer = SigRecGpuRecognizer(Model())
        self.assertEqual(recognizer.window_ms, 10)
        self.assertEqual(recognizer.max_images_per_packet, 64)
        iq = torch.zeros((IQ_PACKET_SAMPLES, 2), dtype=torch.int16, device=recognizer.device)
        result = recognizer.infer_packet_iq(iq, 859200)
        self.assertEqual(len(result.windows), 64)
        self.assertEqual(result.vote_counts['GSM'], 64)
        self.assertGreater(result.window_ms, 10)
        self.assertEqual(sum(w.sample_count for w in result.windows), IQ_PACKET_SAMPLES)

    def test_vectorized_images_match_individual_windows_including_a_ragged_tail(self):
        recognizer = SigRecGpuRecognizer(nfft=64)
        torch.manual_seed(123)
        components = torch.randn((2*859+110, 2), device=recognizer.device)*.1
        spans = [(0, 859), (859, 859), (1718, 110)]
        together = recognizer._window_images(components, spans, 64)
        individual = torch.cat([recognizer._image_from_normalized(components[a:a+b], 64)
                                for a, b in spans])
        self.assertTrue(torch.allclose(together, individual, atol=1e-6))

    def test_image_preparation_advances_while_inference_is_busy_and_all_votes_are_kept(self):
        second_prepared = threading.Event()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.overlapped = False
            def forward(self, images):
                self.calls += 1
                if self.calls == 1:
                    self.overlapped = second_prepared.wait(3)
                return torch.zeros((len(images), 5), device=images.device)
        model = Model()
        recognizer = SigRecGpuRecognizer(model, window_ms=1, max_images_per_packet=None)
        original = recognizer._window_images
        prepared = []
        def images(*args):
            result = original(*args)
            prepared.append(1)
            if len(prepared) == 2:
                second_prepared.set()
            return result
        iq = torch.zeros((IQ_PACKET_SAMPLES, 2), dtype=torch.int16, device=recognizer.device)
        with patch.object(recognizer, '_window_images', side_effect=images):
            result = recognizer.infer_packet_iq(iq, 859200)
        self.assertTrue(model.overlapped)
        self.assertEqual(len(result.windows), 1221)
        self.assertEqual(result.vote_counts['4G'], 1221)
        self.assertEqual(sum(w.sample_count for w in result.windows), IQ_PACKET_SAMPLES)

    def test_real_gpu_windows_cover_full_packet_and_model_votes_include_last_image(self):
        class ScriptedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = 0
                self.batch_sizes = []

            def forward(self, images):
                self.batch_sizes.append(len(images))
                logits = torch.full((len(images), 5), -2., device=images.device)
                for i in range(len(images)):
                    # 4 GSM then 5 WiFi images. A first-window-only path fails.
                    logits[i, 3 if self.seen < 4 else 4] = 4.
                    self.seen += 1
                return logits

        model = ScriptedModel()
        recognizer = SigRecGpuRecognizer(model, window_ms=1, inference_batch_size=8)
        iq = torch.zeros((IQ_PACKET_SAMPLES, 2), dtype=torch.int16, device=recognizer.device)
        progress = []
        previews = []
        result = recognizer.infer_packet_iq(iq, 116_718_600,
            progress=lambda a, b: progress.append((a, b)), preview=previews.extend)
        self.assertEqual(result.class_name, 'WiFi')
        self.assertEqual(result.vote_counts['GSM'], 4)
        self.assertEqual(result.vote_counts['WiFi'], 5)
        self.assertEqual(result.sample_count, IQ_PACKET_SAMPLES)
        self.assertEqual(sum(w.sample_count for w in result.windows), IQ_PACKET_SAMPLES)
        self.assertAlmostEqual(result.vote_fraction, 5/9)
        self.assertEqual(model.batch_sizes, [8, 1])
        self.assertEqual(progress, [(8, 9), (9, 9)])
        self.assertEqual(len(previews), 8)
        self.assertEqual([p['image_index'] for p in previews], list(range(1, 9)))
        self.assertEqual([p['class_name'] for p in previews], ['GSM']*4+['WiFi']*4)
        self.assertEqual(previews[0]['rgb'].shape, (224, 224, 3))
        self.assertEqual(previews[0]['rgb'].dtype, np.uint8)
        with self.assertRaises(ValueError):
            recognizer.infer_packet_iq(iq[:-1], 116_718_600)
        with self.assertRaises(InterruptedError):
            recognizer.infer_packet_iq(iq, 116_718_600, cancelled=lambda: True)

        # The actual FPGA GPU packet follows the identical multi-image path.
        header = bytearray(32)
        struct.pack_into('<I', header, 0, FPGA_SYNC_WORD)
        header[4] = 4
        struct.pack_into('<I', header, 5, FPGA_IQ_MESSAGE_LENGTH)
        struct.pack_into('<I', header, 14, 29_179_688)
        struct.pack_into('<I', header, 18, 116_718_600)
        packet = torch.zeros(FPGA_IQ_PACKET_BYTES, dtype=torch.uint8, device=recognizer.device)
        packet[:32] = torch.tensor(list(header), dtype=torch.uint8, device=recognizer.device)
        model.seen = 0
        hardware_result = recognizer.infer_fpga_iq_packet(packet)
        self.assertEqual(hardware_result.vote_counts, result.vote_counts)
        self.assertEqual(hardware_result.sample_count, IQ_PACKET_SAMPLES)

    def test_gpu_rgb_matches_reference_gui_numpy_pil_preprocessing(self):
        import matplotlib
        from PIL import Image
        from torchvision import transforms
        rng = np.random.default_rng(1234)
        raw = rng.integers(-20000, 20000, (4096, 2), dtype=np.int16)
        iq = (raw[:, 0].astype(np.float32)+1j*raw[:, 1].astype(np.float32))/32768
        frames = np.stack([iq[start:start+256]*np.hanning(256)
                           for start in range(0, len(iq)-256+1, 64)], axis=1)
        power = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=0), axes=0))**2
        db = 10*np.log10(power+1e-12)
        low, high = np.percentile(db, [1, 99])
        rgb = (matplotlib.colormaps['viridis'](np.clip((db-low)/(high-low+1e-12), 0, 1)[::-1])[:, :, :3]*255).astype(np.uint8)
        reference = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])(Image.fromarray(rgb))
        recognizer = SigRecGpuRecognizer(nfft=256, window_ms=10)
        image = recognizer.image_from_iq(torch.from_numpy(raw).to(recognizer.device), 1_280_000)[0].cpu()
        difference = (image-reference).abs()
        self.assertLess(float(difference.mean()), .006)
        self.assertLess(float(difference.max()), .06)


if __name__ == '__main__':
    unittest.main()
