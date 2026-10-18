"""CPU spectrogram preview of real IQ, using sigRec GUI's image convention."""
import math
import numpy as np
from recognition_results import window_spans, effective_window_ms


def preview_images(iq_i16, sample_rate_hz, window_ms=1.0, limit=8, cancelled=lambda: False,
                   max_images=None):
    """Preview evenly spaced windows; no neural model or classification is implied."""
    if iq_i16.dtype != np.int16 or iq_i16.ndim != 2 or iq_i16.shape[1] != 2:
        raise ValueError('IQ must be int16 [N,2]')
    if sample_rate_hz <= 0 or limit < 1:
        raise ValueError('sample rate and preview limit must be positive')
    import matplotlib
    from PIL import Image
    nfft = max(64, min(1024, 1 << max(0, math.ceil(math.log2(sample_rate_hz*.00005)))))
    window_ms = effective_window_ms(len(iq_i16), sample_rate_hz, window_ms, nfft, max_images)
    spans = window_spans(len(iq_i16), sample_rate_hz, window_ms, nfft)
    indices = np.unique(np.linspace(0, len(spans)-1, min(limit, len(spans)), dtype=int))
    window = np.hanning(nfft)
    images = []
    for index in indices:
        if cancelled():
            raise InterruptedError('preview cancelled')
        start, size = spans[index]
        raw = iq_i16[start:start+size].astype(np.float32)/32768
        iq = raw[:, 0]+1j*raw[:, 1]
        frames = np.lib.stride_tricks.sliding_window_view(iq, nfft)[::nfft//4]
        if (len(iq)-nfft) % (nfft//4):
            frames = np.concatenate((frames, iq[-nfft:][None]), axis=0)
        spec = np.fft.fftshift(np.fft.fft(frames*window, axis=1), axes=1)
        db = (10*np.log10(np.abs(spec)**2+1e-12)).T
        low, high = np.percentile(db, [1, 99])
        norm = np.clip((db-low)/(high-low+1e-12), 0, 1)
        rgb = (matplotlib.colormaps['viridis'](norm[::-1])[:, :, :3]*255).astype(np.uint8)
        rgb = np.array(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR))
        images.append(dict(image_index=int(index)+1, start_sample=start, sample_count=size, rgb=rgb))
    return images, len(spans), nfft
