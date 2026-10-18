"""Packet window planning and auditable majority voting, independent of GPU/UI."""
from dataclasses import dataclass, field
import math

IQ_PACKET_SAMPLES = 4 * 1024 * 1024 // 4


def effective_window_ms(sample_count, sample_rate_hz, window_ms, nfft, max_images=None):
    """Enlarge real windows to cap image count; never sample/drop packet portions."""
    if max_images is None:
        return window_ms
    if not isinstance(max_images, int) or max_images < 1:
        raise ValueError('maximum image count must be a positive integer')
    if sample_rate_hz <= 0 or not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError('sample rate and window duration must be positive')
    size = max(nfft, round(sample_rate_hz*window_ms/1000))
    if math.ceil(sample_count/size) > max_images:
        return max(size, math.ceil(sample_count/max_images))/sample_rate_hz*1000
    return window_ms


@dataclass(frozen=True)
class WindowPrediction:
    start_sample: int
    sample_count: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class RecognitionResult:
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    sample_count: int
    sample_rate_hz: int
    windows: tuple[WindowPrediction, ...] = ()
    vote_counts: dict[str, int] = field(default_factory=dict)
    vote_fraction: float = 0.0
    tied: bool = False
    window_ms: float = 1.0
    nfft: int = 1024


def window_spans(sample_count, sample_rate_hz, window_ms=1.0, nfft=1024):
    """Disjoint windows cover every sample; merge a sub-FFT tail into its predecessor."""
    if sample_rate_hz <= 0 or not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError('sample rate and window duration must be positive')
    if nfft < 16 or sample_count < nfft:
        raise ValueError(f'need at least {nfft} IQ samples')
    size = max(nfft, round(sample_rate_hz * window_ms / 1000))
    spans = []
    start = 0
    while start < sample_count:
        count = min(size, sample_count-start)
        if count < nfft and spans:
            previous_start, previous_count = spans[-1]
            spans[-1] = (previous_start, previous_count+count)
        else:
            spans.append((start, count))
        start += count
    return spans


def majority_result(windows, class_names, sample_count, sample_rate_hz,
                    window_ms=1.0, nfft=1024):
    """Each image casts one vote. Ties use mean probabilities then class order."""
    windows, classes = tuple(windows), tuple(class_names)
    if not windows or not classes or len(set(classes)) != len(classes):
        raise ValueError('nonempty windows and unique classes are required')
    votes = dict.fromkeys(classes, 0)
    sums = dict.fromkeys(classes, 0.0)
    position = 0
    for window in windows:
        if window.start_sample != position or window.sample_count <= 0:
            raise ValueError('windows must cover the packet without gaps or overlaps')
        position += window.sample_count
        probs = window.probabilities
        if set(probs) != set(classes) or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
            raise ValueError('invalid window probabilities')
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
            raise ValueError('window probabilities must sum to one')
        winner = max(classes, key=lambda c: probs[c])
        if winner != window.class_name or not math.isclose(window.confidence, probs[winner], abs_tol=1e-5):
            raise ValueError('window label/confidence does not match its probabilities')
        votes[winner] += 1
        for c in classes:
            sums[c] += probs[c]
    if position != sample_count or sample_rate_hz <= 0:
        raise ValueError('window coverage does not match packet sample count')
    mean = {c: sums[c]/len(windows) for c in classes}
    candidates = [c for c in classes if votes[c] == max(votes.values())]
    winner = max(candidates, key=lambda c: mean[c])
    return RecognitionResult(winner, mean[winner], mean, sample_count, sample_rate_hz,
                             windows, votes, votes[winner]/len(windows),
                             len(candidates) > 1, window_ms, nfft)
