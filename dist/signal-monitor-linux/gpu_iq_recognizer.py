"""GPU-resident IQ recognition pipeline compatible with ``C:\\VS\\sigRec``.

The input is an already-resident CUDA tensor containing interleaved signed
16-bit IQ samples, shape ``[sample_count, 2]`` (I then Q).  No IQ data is
written to disk or copied through host memory by this module.  This is the
interface a GPUDirect/P2P receiver should call after it has made a contiguous
view of its DMA ring buffer.

The pre-processing follows sigRec's recognition GUI: configurable IQ windows, Hann
STFT, power in dB, 1st/99th percentile normalisation, viridis RGB image, and
ImageNet-normalised 224x224 ResNet input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Literal, Sequence
import math
import threading
from queue import Queue, Empty, Full

from recognition_results import (RecognitionResult, WindowPrediction,
                                 window_spans, majority_result, IQ_PACKET_SAMPLES, effective_window_ms)

import torch
import torch.nn.functional as F


DEFAULT_CLASS_NAMES = ("4G", "5G", "DVB", "GSM", "WiFi")
DEFAULT_OPENSET_CLASS_NAMES = (*DEFAULT_CLASS_NAMES, 'Unknown')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FPGA_SYNC_WORD = 0x1ACF_FC1D
FPGA_IQ_HEADER_BYTES = 32
FPGA_IQ_PAYLOAD_BYTES = 4 * 1024 * 1024
FPGA_IQ_PACKET_BYTES = FPGA_IQ_HEADER_BYTES + FPGA_IQ_PAYLOAD_BYTES
FPGA_IQ_MESSAGE_LENGTH = 0x0040000D


@dataclass(frozen=True)
class GpuIQFormat:
    """Description of one contiguous IQ payload already resident on GPU.

    ``payload_offset_bytes`` includes every transport header before IQ data.
    For the current XDMA IQ packet it is 32.  The vendor P2P example is
    different: it supplies an arbitrary C2H byte-stream fragment containing
    16-byte channel frames, so it must first be stream-parsed on GPU and this
    descriptor then describes the resulting *payload*, not the raw fragment.
    Only this descriptor changes when the actual IQ payload layout changes;
    the STFT and neural-network path remains unchanged.
    """

    sample_rate_hz: int
    payload_offset_bytes: int = 32
    sample_count: int | None = None
    component_type: Literal["int8", "int16", "float32"] = "int16"
    iq_order: Literal["IQ", "QI"] = "IQ"
    scale: float | None = None

    @property
    def component_bytes(self) -> int:
        return {"int8": 1, "int16": 2, "float32": 4}[self.component_type]

    @property
    def normalisation(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        return {"int8": 128.0, "int16": 32768.0, "float32": 1.0}[self.component_type]


@dataclass(frozen=True)
class FPGAIQHeader:
    """Decoded metadata from the 32-byte FPGA Type-0x04 IQ packet header."""

    sequence: int
    center_hz: int
    bandwidth_hz: int
    sample_rate_hz: int


def parse_fpga_iq_header(header: bytes) -> FPGAIQHeader:
    """Validate the fixed FPGA IQ header; callers pass exactly 32 bytes."""
    if len(header) != FPGA_IQ_HEADER_BYTES:
        raise ValueError(f"IQ header must be {FPGA_IQ_HEADER_BYTES} bytes")
    sync = struct.unpack_from("<I", header, 0)[0]
    message_type = header[4]
    message_length = struct.unpack_from("<I", header, 5)[0]
    if sync != FPGA_SYNC_WORD or message_type != 0x04:
        raise ValueError(f"invalid FPGA IQ sync/type: 0x{sync:08X}/0x{message_type:02X}")
    if message_length != FPGA_IQ_MESSAGE_LENGTH or any(header[22:FPGA_IQ_HEADER_BYTES]):
        raise ValueError("invalid FPGA IQ message length or reserved bytes")
    return FPGAIQHeader(
        sequence=header[9],
        center_hz=struct.unpack_from("<i", header, 10)[0],
        bandwidth_hz=struct.unpack_from("<I", header, 14)[0],
        sample_rate_hz=struct.unpack_from("<I", header, 18)[0],
    )


def _build_resnet(num_classes: int, backbone: str):
    from torch import nn
    from torchvision import models

    if backbone == "resnet18":
        model, features = models.resnet18(weights=None), 512
    elif backbone == "resnet34":
        model, features = models.resnet34(weights=None), 512
    elif backbone == "resnet50":
        model, features = models.resnet50(weights=None), 2048
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(features, num_classes))
    return model


def _detect_backbone(state: dict[str, torch.Tensor]) -> str:
    for key, value in state.items():
        if key.endswith("fc.1.weight") or key.endswith("fc.weight"):
            if value.shape[1] == 512:
                return 'resnet34' if 'layer3.5.conv2.weight' in state else 'resnet18'
            return {2048: "resnet50"}.get(value.shape[1], "resnet50")
    return "resnet50"


def _checkpoint_model(saved, class_names=None):
    """Resolve saved class metadata before unwrapping the state dictionary."""
    if not isinstance(saved, dict):
        raise ValueError('checkpoint must contain a ResNet state dictionary')
    state = saved.get('state_dict', saved.get('model', saved))
    if not isinstance(state, dict):
        raise ValueError('checkpoint model must be a state dictionary')
    classifier = next((value for key, value in state.items()
                       if key in ('fc.1.weight', 'fc.weight')), None)
    if classifier is None or classifier.ndim != 2:
        raise ValueError('checkpoint does not contain a supported ResNet classifier')
    count = classifier.shape[0]
    names = class_names
    if names is None:
        names = saved.get('class_names', saved.get('classes', saved.get('known_classes')))
        if (isinstance(names, (list, tuple)) and 'known_classes' in saved
                and len(names)+1 == count):
            names = (*names, 'Unknown')
        if names is None:
            if count == 5:
                names = DEFAULT_CLASS_NAMES
            elif count == 6:
                names = DEFAULT_OPENSET_CLASS_NAMES
            else:
                raise ValueError(f'模型输出{count}类，但没有类别名称；无法确定标签顺序')
    if (not isinstance(names, (list, tuple)) or len(names) != count
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != count):
        raise ValueError(f'模型输出{count}类，类别名称/顺序配置不匹配')
    return state, tuple(names)


def _viridis_lut(device: torch.device) -> torch.Tensor:
    """Build the same 256-entry viridis lookup table used by sigRec once."""
    try:
        import matplotlib

        values = matplotlib.colormaps["viridis"](range(256))[:, :3]
    except (ImportError, AttributeError, KeyError) as exc:
        raise RuntimeError('matplotlib viridis is required for sigRec preprocessing') from exc
    return torch.tensor(values, dtype=torch.float32, device=device)


class SigRecGpuRecognizer:
    """Run the sigRec ResNet entirely from a CUDA IQ tensor."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        device: str | torch.device = "cuda",
        nfft: int | None = None,
        window_ms: float = 10.0,
        inference_batch_size: int = 64,
        max_images_per_packet: int | None = 64,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for direct GPU IQ recognition")
        self.adaptive_nfft = nfft is None
        nfft = 1024 if nfft is None else nfft
        if nfft < 16 or nfft & (nfft - 1):
            raise ValueError("nfft must be a power of two and at least 16")
        if not math.isfinite(window_ms) or window_ms <= 0 or inference_batch_size < 1:
            raise ValueError('window duration and inference batch size must be positive')
        self.class_names = tuple(class_names)
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError('class names must be unique and nonempty')
        self.nfft = nfft
        self.window_ms = float(window_ms)
        self.inference_batch_size = int(inference_batch_size)
        if max_images_per_packet is not None and (not isinstance(max_images_per_packet, int)
                                                  or max_images_per_packet < 1):
            raise ValueError('maximum image count must be a positive integer')
        self.max_images_per_packet = max_images_per_packet
        self.model = model.to(self.device).eval() if model is not None else None
        self._windows = {}
        self._viridis = _viridis_lut(self.device)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        class_names: Sequence[str] | None = None,
        device: str | torch.device = "cuda",
        strict: bool = True,
        window_ms: float = 10.0,
        max_images_per_packet: int | None = 64,
    ) -> "SigRecGpuRecognizer":
        target = torch.device(device)
        saved = torch.load(Path(checkpoint), map_location=target, weights_only=False)
        saved, class_names = _checkpoint_model(saved, class_names)
        backbone = _detect_backbone(saved)
        model = _build_resnet(len(class_names), backbone)
        missing, unexpected = model.load_state_dict(saved, strict=strict)
        if missing or unexpected:
            print(
                f"[GPU NN] loaded {Path(checkpoint).name}: {backbone}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}",
                flush=True,
            )
        return cls(model, class_names=class_names, device=target, window_ms=window_ms,
                   max_images_per_packet=max_images_per_packet)

    def _nfft_for_rate(self, sample_rate_hz):
        if sample_rate_hz <= 0:
            raise ValueError('sample_rate_hz must be positive')
        if not self.adaptive_nfft:
            return self.nfft
        return max(64, min(1024, 1 << max(0, math.ceil(math.log2(sample_rate_hz*.00005)))))

    def _image_from_normalized(self, components, nfft):
        return self._images_from_normalized(components.unsqueeze(0), nfft)

    def _images_from_normalized(self, components, nfft):
        """Vectorized STFT, per-image quantiles and RGB for equal-length windows."""
        if nfft not in self._windows:
            self._windows[nfft] = torch.hann_window(nfft, periodic=False, device=self.device)
        iq = torch.complex(components[:, :, 0], components[:, :, 1])
        stft = torch.stft(iq, n_fft=nfft, hop_length=max(1, nfft//4),
                          window=self._windows[nfft], center=False, return_complex=True)
        # Include a final end-aligned FFT when STFT's stride leaves a tail.
        # This uses real samples, with no padding, and covers the entire window.
        if (iq.shape[1]-nfft) % max(1, nfft//4):
            last = torch.fft.fft(iq[:, -nfft:]*self._windows[nfft]).unsqueeze(2)
            stft = torch.cat((stft, last), dim=2)
        power_db = torch.fft.fftshift(10*torch.log10(stft.abs().square()+1e-12), dim=1)
        low, high = torch.quantile(power_db.flatten(1),
                                  torch.tensor([.01, .99], device=self.device), dim=1)
        low, high = low[:, None, None], high[:, None, None]
        levels = ((power_db-low)/(high-low+1e-12)).clamp(0, 1)
        # Matplotlib's 256-entry map uses floor(level*256), then RGB uint8.
        indices = (levels*256).long().clamp(max=255)
        rgb = (self._viridis[indices.flip(1)]*255).to(torch.uint8).float()/255
        image = F.interpolate(rgb.permute(0, 3, 1, 2), size=(224, 224),
                              mode='bilinear', align_corners=False, antialias=True)
        # Match PIL RGB resizing/ToTensor quantisation in sigRec's GUI.
        image = (image*255).round().clamp(0, 255)/255
        return (image-self._mean)/self._std

    def _window_images(self, components, spans, nfft):
        # The last real window may have a different length. Group by length
        # without padding, duplicating votes or dropping the tail.
        images = []
        index = 0
        while index < len(spans):
            start, size = spans[index]
            end = index+1
            while end < len(spans) and spans[end][1] == size:
                end += 1
            windows = components[start:start+(end-index)*size].reshape(end-index, size, 2)
            images.append(self._images_from_normalized(windows, nfft))
            index = end
        return torch.cat(images)

    def image_from_iq(self, iq_i16: torch.Tensor, sample_rate_hz: int) -> torch.Tensor:
        """Convert a GPU ``int16 [N,2]`` IQ buffer to a GPU ResNet tensor."""
        if iq_i16.device.type != "cuda":
            raise ValueError("IQ input must already be resident on a CUDA GPU")
        if iq_i16.device != self.device:
            raise ValueError(f"IQ device {iq_i16.device} does not match {self.device}")
        if iq_i16.dtype != torch.int16 or iq_i16.ndim != 2 or iq_i16.shape[1] != 2:
            raise ValueError("IQ input must have dtype int16 and shape [sample_count, 2]")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        nfft = self._nfft_for_rate(sample_rate_hz)
        wanted = max(nfft, round(sample_rate_hz * self.window_ms / 1000.0))
        count = min(int(iq_i16.shape[0]), wanted)
        if count < nfft:
            raise ValueError(f"need at least {nfft} IQ samples, received {count}")
        raw = iq_i16[:count].to(torch.float32)
        return self._image_from_normalized(raw/32768.0, nfft)

    def iq_from_dma_bytes(self, dma_bytes: torch.Tensor, fmt: GpuIQFormat) -> torch.Tensor:
        """Expose a variable-format DMA payload as normalized GPU IQ samples.

        ``dma_bytes`` is a contiguous one-dimensional ``torch.uint8`` CUDA
        tensor.  Reinterpretation and conversion occur on GPU; no IQ payload
        crosses to host memory.  Hardware-specific code only needs to provide
        the small ``GpuIQFormat`` descriptor alongside the GPU buffer.
        """
        if (
            dma_bytes.device != self.device
            or dma_bytes.device.type != "cuda"
            or dma_bytes.dtype != torch.uint8
            or dma_bytes.ndim != 1
            or not dma_bytes.is_contiguous()
        ):
            raise ValueError("DMA input must be a contiguous 1-D uint8 tensor on the recognizer GPU")
        if fmt.sample_rate_hz <= 0 or fmt.payload_offset_bytes < 0:
            raise ValueError("invalid GPU IQ format descriptor")
        offset = fmt.payload_offset_bytes
        available = int(dma_bytes.numel()) - offset
        bytes_per_sample = 2 * fmt.component_bytes
        if available < 0 or available % bytes_per_sample:
            raise ValueError("DMA byte count is not aligned to the declared IQ format")
        count = available // bytes_per_sample if fmt.sample_count is None else fmt.sample_count
        if count < 0 or count * bytes_per_sample > available:
            raise ValueError("GPU IQ format sample_count exceeds DMA payload")

        payload = dma_bytes.narrow(0, offset, count * bytes_per_sample)
        dtype = {"int8": torch.int8, "int16": torch.int16, "float32": torch.float32}[fmt.component_type]
        components = payload.view(dtype).reshape(count, 2).to(torch.float32)
        if fmt.iq_order == "QI":
            components = components[:, [1, 0]]
        elif fmt.iq_order != "IQ":
            raise ValueError(f"unsupported IQ order: {fmt.iq_order}")
        return components / fmt.normalisation

    def image_from_dma_bytes(self, dma_bytes: torch.Tensor, fmt: GpuIQFormat) -> torch.Tensor:
        """Preprocess a variable-format FPGA DMA packet directly on GPU."""
        iq = self.iq_from_dma_bytes(dma_bytes, fmt)
        nfft = self._nfft_for_rate(fmt.sample_rate_hz)
        wanted = max(nfft, round(fmt.sample_rate_hz * self.window_ms / 1000.0))
        iq = iq[:min(int(iq.shape[0]), wanted)]
        if iq.shape[0] < nfft:
            raise ValueError(f"need at least {nfft} IQ samples, received {iq.shape[0]}")
        return self._image_from_normalized(iq, nfft)

    def fpga_iq_packet_from_gpu(self, packet: torch.Tensor) -> tuple[FPGAIQHeader, torch.Tensor]:
        """Parse one fixed FPGA IQ packet while retaining all samples on GPU.

        Only the 32-byte metadata header is copied to host for validation and
        sample-rate metadata.  The 4 MiB IQ payload is exposed as a CUDA
        ``int16 [1048576, 2]`` tensor without a host-side payload copy.
        """
        if (
            packet.device != self.device
            or packet.dtype != torch.uint8
            or packet.ndim != 1
            or not packet.is_contiguous()
            or packet.numel() != FPGA_IQ_PACKET_BYTES
        ):
            raise ValueError(
                "FPGA packet must be a contiguous CUDA uint8 tensor of "
                f"{FPGA_IQ_PACKET_BYTES} bytes"
            )
        header = parse_fpga_iq_header(bytes(packet[:FPGA_IQ_HEADER_BYTES].cpu().tolist()))
        iq_i16 = packet[FPGA_IQ_HEADER_BYTES:].view(torch.int16).reshape(-1, 2)
        return header, iq_i16

    def image_from_fpga_iq_packet(self, packet: torch.Tensor) -> tuple[FPGAIQHeader, torch.Tensor]:
        """Create a ResNet input from the FPGA's exact 32B + 4MiB IQ format."""
        header, iq_i16 = self.fpga_iq_packet_from_gpu(packet)
        return header, self.image_from_iq(iq_i16, header.sample_rate_hz)

    @torch.inference_mode()
    def _infer_windows(self, components, sample_rate_hz, *, cancelled=None, progress=None, preview=None):
        if self.model is None:
            raise RuntimeError("no neural-network checkpoint has been loaded")
        nfft = self._nfft_for_rate(sample_rate_hz)
        count = int(components.shape[0])
        actual_window_ms = effective_window_ms(count, sample_rate_hz, self.window_ms, nfft,
                                               self.max_images_per_packet)
        spans = window_spans(count, sample_rate_hz, actual_window_ms, nfft)
        predictions = []
        prepared = Queue(maxsize=2)
        stop_preparing = threading.Event()

        def prepare():
            def send(value):
                while not stop_preparing.is_set():
                    try:
                        prepared.put(value, timeout=.05)
                        return
                    except Full:
                        pass
            try:
                with torch.inference_mode(), torch.cuda.device(self.device):
                    stream = torch.cuda.Stream(device=self.device)
                    with torch.cuda.stream(stream):
                        for offset in range(0, len(spans), self.inference_batch_size):
                            if stop_preparing.is_set() or (cancelled and cancelled()):
                                return
                            batch = spans[offset:offset+self.inference_batch_size]
                            images = self._window_images(components, batch, nfft)
                            stream.synchronize()
                            send(('images', (offset, batch, images)))
            except Exception as exc:
                send(('error', exc))
            finally:
                send(('done', None))

        # Input normalization may still be queued on the caller's CUDA stream.
        # Establish that dependency before a separate preparation stream reads it.
        torch.cuda.current_stream(self.device).synchronize()
        producer = threading.Thread(target=prepare, daemon=True)
        producer.start()
        try:
            while True:
                if cancelled and cancelled():
                    raise InterruptedError('IQ recognition cancelled')
                try:
                    kind, value = prepared.get(timeout=.05)
                except Empty:
                    continue
                if kind == 'done':
                    break
                if kind == 'error':
                    raise value
                offset, batch, images = value
                images.record_stream(torch.cuda.current_stream(self.device))
                self._consume_images(images, batch, offset, predictions, preview, progress, len(spans))
        finally:
            stop_preparing.set()
            producer.join()
        if cancelled and cancelled():
            raise InterruptedError('IQ recognition cancelled')
        return majority_result(predictions, self.class_names, count, int(sample_rate_hz),
                               actual_window_ms, nfft)

    def _consume_images(self, images, batch, offset, predictions, preview, progress, total):
        """Inference consumer; the next image batch can prepare concurrently."""
        logits = self.model(images)
        if logits.shape != (len(batch), len(self.class_names)):
            raise ValueError('model output shape/values do not match configured classes')
        # One host synchronization per inference batch, rather than another
        # scalar CUDA synchronization just to validate finite values.
        outputs = torch.cat((torch.softmax(logits, dim=1),
                             torch.isfinite(logits).all(dim=1, keepdim=True).to(logits.dtype)), dim=1).cpu().tolist()
        probabilities = [row[:-1] for row in outputs]
        if any(row[-1] != 1 for row in outputs) or any(not math.isfinite(p) for row in probabilities for p in row):
            raise ValueError('model output shape/values do not match configured classes')
        for (start, size), values in zip(batch, probabilities):
            probs = dict(zip(self.class_names, map(float, values)))
            winner = max(self.class_names, key=lambda c: probs[c])
            predictions.append(WindowPrediction(start, size, winner, probs[winner], probs))
        if preview and offset < 8:
            # Only eight small display images cross to CPU; IQ remains on
            # GPU. These are the exact RGB inputs supplied to the network.
            count_preview = min(len(batch), 8-offset)
            rgb = ((images[:count_preview]*self._std+self._mean)*255).round().clamp(0, 255)
            rgb = rgb.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
            preview([dict(image_index=offset+i+1, start_sample=start, sample_count=size,
                rgb=rgb[i], class_name=predictions[offset+i].class_name,
                confidence=predictions[offset+i].confidence)
                for i, (start, size) in enumerate(batch[:count_preview])])
        if progress:
            progress(len(predictions), total)

    @torch.inference_mode()
    def infer_iq(self, iq_i16: torch.Tensor, sample_rate_hz: int, *,
                 cancelled=None, progress=None, preview=None) -> RecognitionResult:
        """Classify every sample using disjoint images and majority voting."""
        if (iq_i16.device != self.device or iq_i16.dtype != torch.int16
                or iq_i16.ndim != 2 or iq_i16.shape[1] != 2):
            raise ValueError('IQ must be int16 [N,2] on the recognizer GPU')
        return self._infer_windows(iq_i16.float()/32768, sample_rate_hz,
                                   cancelled=cancelled, progress=progress, preview=preview)

    def infer_packet_iq(self, iq_i16, sample_rate_hz, **kwargs):
        if iq_i16.shape != (IQ_PACKET_SAMPLES, 2):
            raise ValueError('recognition requires exactly 4 MiB int16 IQ')
        return self.infer_iq(iq_i16, sample_rate_hz, **kwargs)

    @torch.inference_mode()
    def infer_dma_bytes(self, dma_bytes: torch.Tensor, fmt: GpuIQFormat) -> RecognitionResult:
        """Run a loaded model using a variable-format GPU DMA packet."""
        return self._infer_windows(self.iq_from_dma_bytes(dma_bytes, fmt), fmt.sample_rate_hz)

    @torch.inference_mode()
    def infer_fpga_iq_packet(self, packet: torch.Tensor) -> RecognitionResult:
        """Run inference for one complete FPGA ``32B + 4MiB`` GPU packet."""
        header, iq = self.fpga_iq_packet_from_gpu(packet)
        return self.infer_packet_iq(iq, header.sample_rate_hz)


def run_gpu_cache_loopback_test(sample_rate_hz: int = 12_800_000) -> dict[str, int | str]:
    """Exercise GPU-buffer write -> read -> STFT -> RGB without a board.

    This is deliberately a GPU-only loopback: synthetic interleaved IQ is
    created and copied on CUDA, then consumed by ``image_from_iq``.  It proves
    the recognizer accepts a pre-existing device buffer and no IQ payload is
    read from disk or staged through CPU memory.
    """
    pipeline = SigRecGpuRecognizer()
    samples = max(pipeline.nfft * 2, int(sample_rate_hz * 0.010))
    index = torch.arange(samples, device=pipeline.device, dtype=torch.float32)
    phase = 2.0 * torch.pi * 0.071 * index
    source = torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)
    source = (source * 20_000.0).round().to(torch.int16)
    gpu_cache = torch.empty_like(source)
    gpu_cache.copy_(source)  # models the producer-written P2P GPU cache
    before = int(gpu_cache.to(torch.int32).sum().item())
    image = pipeline.image_from_iq(gpu_cache, sample_rate_hz)
    after = int(gpu_cache.to(torch.int32).sum().item())
    if before != after or image.device.type != "cuda" or not bool(torch.isfinite(image).all()):
        raise RuntimeError("GPU IQ cache loopback verification failed")
    return {
        "status": "OK",
        "device": str(pipeline.device),
        "samples": samples,
        "image_elements": int(image.numel()),
        "checksum": before,
    }


def run_fpga_iq_packet_loopback_test(sample_rate_hz: int = 12_800_000) -> dict[str, int | str]:
    """Verify the exact FPGA ``32-byte header + 4 MiB IQ`` GPU contract."""
    pipeline = SigRecGpuRecognizer()
    header = bytearray(FPGA_IQ_HEADER_BYTES)
    struct.pack_into("<I", header, 0, FPGA_SYNC_WORD)
    header[4] = 0x04
    struct.pack_into("<I", header, 5, FPGA_IQ_MESSAGE_LENGTH)
    header[9] = 7
    struct.pack_into("<i", header, 10, -6_400_000)
    struct.pack_into("<I", header, 14, 12_800_000)
    struct.pack_into("<I", header, 18, sample_rate_hz)
    iq_count = FPGA_IQ_PAYLOAD_BYTES // 4
    index = torch.arange(iq_count, device=pipeline.device, dtype=torch.float32)
    phase = 2.0 * torch.pi * 0.071 * index
    iq = (20_000.0 * torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)).round().to(torch.int16)
    packet = torch.empty(FPGA_IQ_PACKET_BYTES, dtype=torch.uint8, device=pipeline.device)
    packet[:FPGA_IQ_HEADER_BYTES].copy_(torch.tensor(list(header), dtype=torch.uint8, device=pipeline.device))
    packet[FPGA_IQ_HEADER_BYTES:].copy_(iq.view(torch.uint8).reshape(-1))
    decoded, image = pipeline.image_from_fpga_iq_packet(packet)
    if decoded.sequence != 7 or decoded.center_hz != -6_400_000 or image.device.type != "cuda":
        raise RuntimeError("FPGA IQ packet GPU loopback verification failed")
    return {
        "status": "OK",
        "packet_bytes": int(packet.numel()),
        "iq_samples": iq_count,
        "sequence": decoded.sequence,
        "image_elements": int(image.numel()),
    }


if __name__ == "__main__":
    print(run_gpu_cache_loopback_test())
    print(run_fpga_iq_packet_loopback_test())
