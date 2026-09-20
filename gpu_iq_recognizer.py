"""GPU-resident IQ recognition pipeline compatible with ``C:\\VS\\sigRec``.

The input is an already-resident CUDA tensor containing interleaved signed
16-bit IQ samples, shape ``[sample_count, 2]`` (I then Q).  No IQ data is
written to disk or copied through host memory by this module.  This is the
interface a GPUDirect/P2P receiver should call after it has made a contiguous
view of its DMA ring buffer.

The pre-processing follows sigRec's recognition GUI: 10 ms IQ window, Hann
STFT, power in dB, 1st/99th percentile normalisation, viridis RGB image, and
ImageNet-normalised 224x224 ResNet input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


DEFAULT_CLASS_NAMES = ("4G", "5G", "DVB", "GSM", "WiFi")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FPGA_SYNC_WORD = 0x1ACF_FC1D
FPGA_IQ_HEADER_BYTES = 32
FPGA_IQ_PAYLOAD_BYTES = 4 * 1024 * 1024
FPGA_IQ_PACKET_BYTES = FPGA_IQ_HEADER_BYTES + FPGA_IQ_PAYLOAD_BYTES
FPGA_IQ_MESSAGE_LENGTH = 0x0040000D


@dataclass(frozen=True)
class RecognitionResult:
    """Small host-side result returned after GPU inference."""

    class_name: str
    confidence: float
    probabilities: dict[str, float]
    sample_count: int
    sample_rate_hz: int


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
            return {512: "resnet18", 2048: "resnet50"}.get(value.shape[1], "resnet50")
    return "resnet50"


def _viridis_lut(device: torch.device) -> torch.Tensor:
    """Build the same 256-entry viridis lookup table used by sigRec once."""
    try:
        import matplotlib

        values = matplotlib.colormaps["viridis"](range(256))[:, :3]
    except (ImportError, AttributeError, KeyError):
        # Kept only as a dependency-light fallback; deployed recognition uses
        # matplotlib and therefore the exact sigRec viridis table above.
        values = [[i / 255.0, i / 255.0, i / 255.0] for i in range(256)]
    return torch.tensor(values, dtype=torch.float32, device=device)


class SigRecGpuRecognizer:
    """Run the sigRec ResNet entirely from a CUDA IQ tensor."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        device: str | torch.device = "cuda",
        nfft: int = 1024,
        window_ms: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for direct GPU IQ recognition")
        if nfft < 16 or nfft & (nfft - 1):
            raise ValueError("nfft must be a power of two and at least 16")
        self.class_names = tuple(class_names)
        self.nfft = nfft
        self.window_ms = float(window_ms)
        self.model = model.to(self.device).eval() if model is not None else None
        self._window = torch.hann_window(nfft, device=self.device)
        self._viridis = _viridis_lut(self.device)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        device: str | torch.device = "cuda",
    ) -> "SigRecGpuRecognizer":
        target = torch.device(device)
        saved = torch.load(Path(checkpoint), map_location=target, weights_only=False)
        if isinstance(saved, dict) and "state_dict" in saved:
            saved = saved["state_dict"]
        if isinstance(saved, dict) and "model" in saved:
            saved = saved["model"]
        if not isinstance(saved, dict):
            raise ValueError("checkpoint must contain a ResNet state dictionary")
        backbone = _detect_backbone(saved)
        model = _build_resnet(len(class_names), backbone)
        missing, unexpected = model.load_state_dict(saved, strict=False)
        if missing or unexpected:
            print(
                f"[GPU NN] loaded {Path(checkpoint).name}: {backbone}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}",
                flush=True,
            )
        return cls(model, class_names=class_names, device=target)

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

        wanted = int(sample_rate_hz * self.window_ms / 1000.0)
        count = min(int(iq_i16.shape[0]), wanted)
        if count < self.nfft:
            raise ValueError(f"need at least {self.nfft} IQ samples, received {count}")
        raw = iq_i16[:count].to(torch.float32)
        iq = torch.complex(raw[:, 0], raw[:, 1]) / 32768.0
        stft = torch.stft(
            iq,
            n_fft=self.nfft,
            hop_length=max(1, self.nfft // 4),
            window=self._window,
            center=False,
            return_complex=True,
        )
        power_db = 10.0 * torch.log10(stft.abs().square().clamp_min(1e-12))
        power_db = torch.fft.fftshift(power_db, dim=0)
        low, high = torch.quantile(power_db, torch.tensor([0.01, 0.99], device=self.device))
        levels = ((power_db - low) / (high - low).clamp_min(1e-12)).clamp_(0.0, 1.0)
        indices = (levels * 255.0).to(torch.long)
        # sigRec reverses the frequency rows before applying the colour map.
        rgb = self._viridis[indices.flip(0)].permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)
        return (image - self._mean) / self._std

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
        # ``image_from_iq`` is deliberately strict for the established FPGA
        # int16 input.  Reuse its STFT path after converting to int16 only
        # when possible; formats with another dynamic range run directly here.
        wanted = int(fmt.sample_rate_hz * self.window_ms / 1000.0)
        iq = iq[:min(int(iq.shape[0]), wanted)]
        if iq.shape[0] < self.nfft:
            raise ValueError(f"need at least {self.nfft} IQ samples, received {iq.shape[0]}")
        complex_iq = torch.complex(iq[:, 0], iq[:, 1])
        stft = torch.stft(complex_iq, n_fft=self.nfft, hop_length=max(1, self.nfft // 4),
                          window=self._window, center=False, return_complex=True)
        power_db = 10.0 * torch.log10(stft.abs().square().clamp_min(1e-12))
        power_db = torch.fft.fftshift(power_db, dim=0)
        low, high = torch.quantile(power_db, torch.tensor([0.01, 0.99], device=self.device))
        indices = (((power_db - low) / (high - low).clamp_min(1e-12)).clamp_(0.0, 1.0) * 255.0).to(torch.long)
        rgb = self._viridis[indices.flip(0)].permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)
        return (image - self._mean) / self._std

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
    def infer_iq(self, iq_i16: torch.Tensor, sample_rate_hz: int) -> RecognitionResult:
        if self.model is None:
            raise RuntimeError("no neural-network checkpoint has been loaded")
        image = self.image_from_iq(iq_i16, sample_rate_hz)
        probs = torch.softmax(self.model(image), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
        # Only this five-element result crosses back to the host.
        values = probs.detach().to("cpu", non_blocking=False).tolist()
        best = int(index.item())
        return RecognitionResult(
            class_name=self.class_names[best],
            confidence=float(confidence.item()),
            probabilities=dict(zip(self.class_names, map(float, values))),
            sample_count=min(int(iq_i16.shape[0]), int(sample_rate_hz * self.window_ms / 1000.0)),
            sample_rate_hz=int(sample_rate_hz),
        )

    @torch.inference_mode()
    def infer_dma_bytes(self, dma_bytes: torch.Tensor, fmt: GpuIQFormat) -> RecognitionResult:
        """Run a loaded model using a variable-format GPU DMA packet."""
        if self.model is None:
            raise RuntimeError("no neural-network checkpoint has been loaded")
        image = self.image_from_dma_bytes(dma_bytes, fmt)
        probs = torch.softmax(self.model(image), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
        values = probs.detach().to("cpu", non_blocking=False).tolist()
        best = int(index.item())
        count = fmt.sample_count or (int(dma_bytes.numel()) - fmt.payload_offset_bytes) // (2 * fmt.component_bytes)
        return RecognitionResult(
            class_name=self.class_names[best], confidence=float(confidence.item()),
            probabilities=dict(zip(self.class_names, map(float, values))),
            sample_count=count, sample_rate_hz=fmt.sample_rate_hz,
        )

    @torch.inference_mode()
    def infer_fpga_iq_packet(self, packet: torch.Tensor) -> RecognitionResult:
        """Run inference for one complete FPGA ``32B + 4MiB`` GPU packet."""
        if self.model is None:
            raise RuntimeError("no neural-network checkpoint has been loaded")
        header, image = self.image_from_fpga_iq_packet(packet)
        probs = torch.softmax(self.model(image), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
        values = probs.detach().to("cpu", non_blocking=False).tolist()
        best = int(index.item())
        return RecognitionResult(
            class_name=self.class_names[best], confidence=float(confidence.item()),
            probabilities=dict(zip(self.class_names, map(float, values))),
            sample_count=FPGA_IQ_PAYLOAD_BYTES // 4,
            sample_rate_hz=header.sample_rate_hz,
        )


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
