"""Display-only signal association; IDs are independent of FPGA slots/batches."""
from dataclasses import dataclass
import math
from bisect import bisect_left, bisect_right


@dataclass
class SignalTrack:
    identifier: int
    target: object
    row: list
    missed: int = 0
    observed_frames: int = 1
    last_result: tuple | None = None

    def update_row(self, row):
        self.row = list(row)
        if row[3] != '—':
            self.last_result = (row[2], row[3], list(row[8:]))
        elif self.last_result is not None and (row[2] in ('待识别', '采集 4 MiB IQ')
                                               or row[2].startswith('识别图像')):
            self.row[2] = f'{self.last_result[0]}（上次）'
            self.row[3] = self.last_result[1]
            self.row[8:] = self.last_result[2]


class SignalTracker:
    def __init__(self, keep_missing_frames=10):
        self.keep_missing_frames = keep_missing_frames
        self.tracks = {}
        self._next_id = 1
        self._frame = None
        self._assignments = {}

    @staticmethod
    def key(target):
        return (target.center_hz, target.bandwidth_hz)

    def observe(self, frame, targets, rows, bin_hz):
        if frame != self._frame:
            # Match globally closest compatible intervals one-to-one. A broad
            # envelope cannot steal the ID of a narrow carrier inside it.
            candidates = []
            ordered = sorted(self.tracks.items(), key=lambda item: (item[1].target.center_hz, item[0]))
            frequencies = [track.target.center_hz for _, track in ordered]
            maximum_distance = max(3*bin_hz, 1_000_000)
            for index, target in enumerate(targets):
                left = bisect_left(frequencies, target.center_hz-maximum_distance)
                right = bisect_right(frequencies, target.center_hz+maximum_distance)
                for identifier, track in ordered[left:right]:
                    a, b = target.bandwidth_hz, track.target.bandwidth_hz
                    if min(a, b) <= 0 or max(a, b) > 2*min(a, b):
                        continue
                    tolerance = max(3*bin_hz, min(1_000_000, .1*min(a, b)))
                    distance = abs(target.center_hz-track.target.center_hz)
                    if distance <= tolerance:
                        score = distance/tolerance + .2*abs(math.log(a/b))
                        candidates.append((score, identifier, index))
            assigned_ids, assigned_indices = set(), set()
            assignments = {}
            for _, identifier, index in sorted(candidates):
                if identifier not in assigned_ids and index not in assigned_indices:
                    assignments[index] = identifier
                    assigned_ids.add(identifier)
                    assigned_indices.add(index)
            for identifier, track in list(self.tracks.items()):
                if identifier not in assigned_ids:
                    track.missed += 1
                    if track.missed > self.keep_missing_frames:
                        del self.tracks[identifier]
            # Initial/new IDs follow frequency order, not fluctuating peak power.
            for index in sorted(range(len(targets)), key=lambda i: targets[i].center_hz):
                if index not in assignments:
                    identifier = self._next_id
                    self._next_id += 1
                    self.tracks[identifier] = SignalTrack(identifier, targets[index], [])
                    assignments[index] = identifier
                track = self.tracks[assignments[index]]
                if index in assigned_indices:
                    track.observed_frames += 1
                track.target = targets[index]
                track.missed = 0
            self._assignments = {self.key(targets[index]): identifier
                                 for index, identifier in assignments.items()}
            self._frame = frame
        for target, row in zip(targets, rows):
            identifier = self._assignments.get(self.key(target))
            if identifier in self.tracks:
                self.tracks[identifier].update_row(row)

    def visible(self, selected_filter):
        return sorted((track for track in self.tracks.values()
                       if selected_filter.allows(track.target)),
                      # 连续检出的稳定信号保持在前面并按频率排序；刚出现、
                      # 短暂漏检或尚未稳定的信号统一收在列表末尾。
                      key=lambda track: (
                          not (track.observed_frames >= 2 and track.missed == 0),
                          track.target.center_hz,
                          track.identifier,
                      ))

    def identifier_for(self, target):
        return self._assignments.get(self.key(target))

    def apply_filter(self, selected_filter):
        self.tracks = {identifier: track for identifier, track in self.tracks.items()
                       if selected_filter.allows(track.target)}
