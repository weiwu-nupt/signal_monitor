"""Shared, completion-driven FPGA/file target scheduling (caller serializes access)."""
from dataclasses import replace
import math
import time

from fpga_closed_loop import TargetFilter, IQ_BYTES

WIDE_THRESHOLD_HZ = 20_000_000
CAPACITY = {'narrow': 128, 'wide': 2}


def classify_target(target):
    return replace(target, target_type='wide' if target.bandwidth_hz > WIDE_THRESHOLD_HZ else 'narrow')


def target_batches(targets, bandwidth_filter=None):
    """No candidate truncation; sequences are physical slots, scoped to each batch."""
    bandwidth_filter = bandwidth_filter or TargetFilter()
    groups = {kind: [] for kind in CAPACITY}
    seen = set()
    for target in targets:
        target = classify_target(target)
        key = (target.target_type, target.center_hz, target.bandwidth_hz)
        if bandwidth_filter.allows(target) and key not in seen:
            groups[target.target_type].append(target)
            seen.add(key)
    count = max(((len(group)+CAPACITY[kind]-1)//CAPACITY[kind]
                 for kind, group in groups.items()), default=0)
    return [tuple(replace(target, sequence=slot)
                  for kind, limit in CAPACITY.items()
                  for slot, target in enumerate(groups[kind][index*limit:(index+1)*limit]))
            for index in range(count)]


class RecognitionSchedule:
    def __init__(self, timeout_s=30):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('IQ timeout must be finite and positive')
        self.timeout_s = timeout_s
        self.round = 0
        self.batches = []
        self.index = 0
        self.pending = {}
        self.received = {}
        self.started = None
        self.halted = False

    @property
    def active(self):
        return bool(self.pending)

    def next_batch(self, latest_targets, bandwidth_filter=None):
        if self.halted or self.pending:
            return None
        if self.index >= len(self.batches):
            self.batches = target_batches(latest_targets, bandwidth_filter)
            self.index = 0
            if not self.batches:
                return ()
            self.round += 1
        batch = self.batches[self.index]
        self.index += 1
        self.pending = {(t.target_type, t.sequence): t for t in batch}
        self.received.clear()
        self.started = time.monotonic()
        return batch

    def matches(self, iq):
        target = self.pending.get((iq.target_type, iq.sequence))
        return (not self.halted and target is not None
                and iq.center_hz == target.center_hz
                and iq.bandwidth_hz == target.bandwidth_hz
                and iq.sample_rate_hz == target.sample_rate_hz
                and iq.i_data.dtype == iq.q_data.dtype == 'int16'
                and iq.i_data.size == iq.q_data.size == IQ_BYTES//4)

    def complete(self, iq):
        if not self.matches(iq):
            return False
        key = (iq.target_type, iq.sequence)
        del self.pending[key]
        self.received.pop(key, None)
        return True

    def receive(self, iq):
        """Accept one full lane packet; reception alone does not finish recognition."""
        key = (iq.target_type, iq.sequence)
        if not self.matches(iq) or key in self.received:
            return False
        self.received[key] = iq
        return True

    def check_timeout(self, now=None):
        now = time.monotonic() if now is None else now
        # Acquisition is longer at lower DDC rates; timeout allowance starts
        # after the slowest lane has had time to collect 1,048,576 IQ points.
        awaiting_iq = [t for key, t in self.pending.items() if key not in self.received]
        acquisition = max((IQ_BYTES/4/t.sample_rate_hz for t in awaiting_iq), default=0)
        if not self.halted and awaiting_iq and now-self.started >= acquisition+self.timeout_s:
            self.halted = True
            raise TimeoutError(f'IQ 接收超时：第 {self.round} 轮第 {self.index} 批仍缺 '
                               f'{len(awaiting_iq)} 路 4 MiB IQ；轮询已暂停，请停止并检查 FPGA。')
