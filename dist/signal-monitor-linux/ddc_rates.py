"""IQ output-rate policy shared by software DDC and FPGA target records."""
from dataclasses import replace
import math

DEFAULT_RATE_MULTIPLIER = 4.0
RATE_QUANTUM_HZ = 200  # Exact even sample counts in 10 ms software DDC blocks.


def output_sample_rate(bandwidth_hz, input_rate_hz, multiplier=DEFAULT_RATE_MULTIPLIER):
    if not math.isfinite(multiplier) or not 3 <= multiplier <= 4:
        raise ValueError('IQ采样率倍数必须在3～4之间')
    if bandwidth_hz <= 0 or input_rate_hz <= 0:
        raise ValueError('带宽和输入采样率必须大于0')
    minimum = math.ceil(3*bandwidth_hz/RATE_QUANTUM_HZ)*RATE_QUANTUM_HZ
    maximum = math.floor(min(4*bandwidth_hz, input_rate_hz)/RATE_QUANTUM_HZ)*RATE_QUANTUM_HZ
    if minimum > maximum:
        raise ValueError('原始采样率不足以提供带宽3～4倍的IQ输出采样率')
    requested = math.floor(multiplier*bandwidth_hz/RATE_QUANTUM_HZ)*RATE_QUANTUM_HZ
    return int(min(maximum, max(minimum, requested)))


def prepare_targets(targets, input_rate_hz, multiplier=DEFAULT_RATE_MULTIPLIER):
    accepted, rejected = [], []
    for target in targets:
        try:
            rate = output_sample_rate(target.bandwidth_hz, input_rate_hz, multiplier)
        except ValueError:
            rejected.append(target)
        else:
            accepted.append(replace(target, sample_rate_hz=rate))
    return accepted, rejected
