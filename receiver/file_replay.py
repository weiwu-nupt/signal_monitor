"""Offline IQ test page, independent of receiver/FPGA connections."""
from pathlib import Path
from queue import Queue, Empty, Full
import threading
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from iq_file_source import (inspect_iq_file, IQReplayReader, spectrum_power,
                            detect_signals, collect_channel_batch)
from fpga_closed_loop import EnergyDetector
from recognition_schedule import target_batches, classify_target
from fpga_closed_loop import TargetFilter
from .recognition_controls import RecognitionBandwidth
from .mode_views import WaterfallPlot
from .ffm_spectrum import TraceProcessor
from signal_tracks import SignalTracker
from ddc_rates import prepare_targets
from dataclasses import replace
from collections import OrderedDict
from recognition_store import RecognitionStore, DEFAULT_DATABASE
from recognition_pipeline import PacketRecognitionPipeline
from recognition_settings import remembered_checkpoint, remember_checkpoint
from sigrec_model import discover_sigrec_checkpoint


class ReplayWorker:
    """Spectrum producer plus a bounded, independent recognition consumer."""
    def __init__(self, info, center, usable, threshold, checkpoint):
        self.info, self.center, self.usable = info, center, usable
        self.threshold, self.checkpoint = threshold, checkpoint
        self.bandwidth_filter = TargetFilter()
        self.rate_multiplier = 4.0
        self.image_window_ms = 10.0
        self.max_images_per_packet = 64
        self.database_path = DEFAULT_DATABASE
        self.tracker = SignalTracker()
        self.previews = OrderedDict()
        self.cancel = threading.Event()
        self.paused = threading.Event()
        self.mailbox = Queue(maxsize=1)
        self._jobs = Queue(maxsize=1)
        self._round_busy = False
        # 当前识别轮在 t0 冻结的信号 ID。None 仅用于尚未启动轮次时。
        self._round_ids = None
        self._round_display_targets = None
        self._scheduled_ids = {}
        self._active_batch = 0
        self._round_finished = threading.Event()
        self._updates_lock = threading.Lock()
        self._updates = {}
        self._recognition_rows = {}
        self._recognition_status = ''
        self._recognition_error = ''
        self.thread = threading.Thread(target=self.run, daemon=True)

    def publish(self, value, observe=True):
        if 'frame' in value:
            axis = value['axis']
            bin_hz = abs(float(axis[1]-axis[0])) if len(axis) > 1 else 1.0
            self.tracker.apply_filter(self.bandwidth_filter)
            # 完成上一轮后，才能让新检测结果成为下一轮候选；识别期间
            # 绝不用后续频谱帧替换 t0 时刻冻结的信号集合。
            self._drain_recognition()
            if observe and not self._round_busy:
                self.tracker.observe(value['frame'], value['targets'], value['rows'], bin_hz)
            for identifier, fields in self._recognition_rows.items():
                track = self.tracker.tracks.get(identifier)
                if track:
                    for column, text in fields.items():
                        track.row[column] = text
                    if track.row[3] != '—':
                        track.last_result = (track.row[2], track.row[3], list(track.row[8:]))
                    elif track.last_result is not None and (track.row[2] in ('采集 4 MiB IQ', '等待神经网络')
                                                           or track.row[2].startswith('识别图像')):
                        track.row[2] = f'{track.last_result[0]}（上次）'
                        track.row[3] = track.last_result[1]
                        track.row[9:11] = track.last_result[2][1:3]
            if self._round_busy:
                round_ids = self._round_ids or tuple(self.tracker.tracks)
                for identifier in round_ids:
                    track = self.tracker.tracks.get(identifier)
                    if track is None:
                        continue
                    batch = self._scheduled_ids.get(identifier)
                    if batch is None:
                        track.row[5] = '本轮未排入'
                        if identifier not in self._recognition_rows:
                            track.row[2], track.row[8] = '等待下一轮', '未排入当前轮'
                    elif batch > self._active_batch:
                        track.row[5] = str(batch)
                        if identifier not in self._recognition_rows:
                            track.row[2], track.row[8] = f'等待第{batch}批', '待换批'
            if self._round_busy:
                round_ids = self._round_ids or tuple(self.tracker.tracks)
                snapshot_tracks = [self.tracker.tracks[identifier]
                                   for identifier in round_ids
                                   if identifier in self.tracker.tracks]
                value = {**value,
                         'targets': [track.target for track in snapshot_tracks],
                         'rows': [list(track.row) for track in snapshot_tracks],
                         'display_targets': self._round_display_targets or value.get('display_targets', [])}
            else:
                rows = [list(self.tracker.tracks[self.tracker.identifier_for(t)].row)
                        if self.tracker.identifier_for(t) in self.tracker.tracks else list(row)
                        for t, row in zip(value['targets'], value['rows'])]
                value = {**value, 'rows': rows}
            value = {**value,
                     'status': value.get('status', '') + self._recognition_status}
            if self._recognition_error:
                value['nn_state'] = '识别流程失败：' + self._recognition_error[:240]
                value['model_error'] = self._recognition_error
            # Stable IDs are assigned before persistence, on the worker. The
            # GUI may skip mailbox frames without changing database identity.
            tracks_for_display = (snapshot_tracks if self._round_busy
                                  else self.tracker.tracks.values())
            value = {**value, 'previews': dict(self.previews), 'tracks': [replace(t, row=list(t.row))
                                        for t in tracks_for_display]}
        try:
            self.mailbox.get_nowait()
        except Empty:
            pass
        try:
            self.mailbox.put_nowait(value)
        except Full:
            pass

    def wait_pause(self):
        while self.paused.is_set() and not self.cancel.wait(.05):
            pass
        return self.cancel.is_set()

    def _post_recognition(self, identifier, fields, status, preview=None):
        # Coalesce progress by ID; final results replace progress and are never
        # lost through the GUI's latest-frame mailbox. Only the producer owns tracks.
        with self._updates_lock:
            update = self._updates.setdefault(identifier, {'fields': {}})
            update['fields'].update(fields)
            if preview is not None:
                update['preview'] = preview
            self._recognition_status = status

    def _drain_recognition(self):
        with self._updates_lock:
            updates, self._updates = self._updates, {}
        for identifier, update in updates.items():
            if identifier not in self.tracker.tracks:
                continue  # An expired/filtered ID must not label another signal.
            self._recognition_rows.setdefault(identifier, {}).update(update['fields'])
            if 'preview' in update:
                self.previews[identifier] = update['preview']
                self.previews.move_to_end(identifier)
                while len(self.previews) > 8:
                    self.previews.popitem(last=False)
        self._recognition_rows = {i: fields for i, fields in self._recognition_rows.items()
                                  if i in self.tracker.tracks}
        if self._round_finished.is_set():
            self._round_finished.clear()
            self._round_busy = False
            self._round_ids = None
            self._round_display_targets = None

    def _recognize_rounds(self, recognizer):
        """Keep SQLite and slow DDC/inference on this thread, outside playback."""
        store = None
        try:
            import torch
            store = RecognitionStore(self.database_path)
            session_id = store.start_session(self.info.path, self.checkpoint, {
                'source_sample_rate_hz': self.info.sample_rate, 'center_hz': self.center,
                'usable_bandwidth_hz': self.usable, 'threshold_db': self.threshold,
                'window_ms': self.image_window_ms, 'classes': list(recognizer.class_names),
                'max_images_per_packet': self.max_images_per_packet,
                'window_mode': 'automatic' if self.max_images_per_packet else 'fixed',
                'preprocessing': 'sigRec GUI: symmetric Hann, adaptive FFT<=1024, '
                    '75% STFT overlap, percentiles 1/99, flipped viridis, '
                    'RGB uint8, bilinear 224x224, ImageNet normalization'})
            pipeline = PacketRecognitionPipeline(recognizer, store, session_id)
            while not self.cancel.is_set():
                try:
                    job = self._jobs.get(timeout=.05)
                except Empty:
                    continue
                # Finish every batch of this frozen round before accepting the
                # latest detections. New spectrum frames cannot starve batch 2+.
                for batch_index, batch in enumerate(job['batches'], 1):
                    if not self._recognize_batch(job, batch_index, batch, pipeline, recognizer):
                        return
                self._round_finished.set()
        except Exception as exc:
            self._recognition_error = str(exc)
        finally:
            if store:
                store.close()

    def _recognize_batch(self, job, batch_index, batch, pipeline, recognizer):
        """One source/DDC producer; GPU consumes each ready lane immediately.

        Join the collector AND finish all successful results before reusing slots.
        A failed lane halts recognition rather than silently advancing the batch.
        """
        import torch
        self._active_batch = batch_index
        packets = Queue(maxsize=4)
        stop_capture = threading.Event()
        prefix = f' | 当前识别批 {batch_index}/{len(job["batches"])}'
        channels = [(t, i) for t, i in batch if self.bandwidth_filter.allows(t)]
        for target, identifier in channels:
            self._post_recognition(identifier, {2: '采集 4 MiB IQ', 3: '—',
                5: str(batch_index), 8: '采集 0.0%', 9: '待识别', 10: '未保存'},
                prefix+f' | {len(channels)} 路同时采集')

        def cancelled():
            return stop_capture.is_set() or self.wait_pause()

        def send(value):
            while not cancelled():
                try:
                    packets.put(value, timeout=.05)
                    return
                except Full:
                    pass

        def ready(target, identifier, channel, rate):
            self._post_recognition(identifier, {2: '等待神经网络', 8: '已采集'},
                prefix+f' | S{identifier:04d} 已满 4 MiB')
            send(('packet', (target, identifier, channel, rate)))

        last_progress = {}
        def capture_progress(identifier, done, total):
            now = time.monotonic()
            if now-last_progress.get(identifier, 0) >= .25 or done == total:
                last_progress[identifier] = now
                text = f'采集 {done/total:.1%}'
                self._post_recognition(identifier, {8: text},
                    prefix+f' | S{identifier:04d} '+text)

        def collect():
            try:
                collect_channel_batch(self.info, job['start'], channels, ready,
                    cancelled, capture_progress, lambda t: self.bandwidth_filter.allows(t),
                    device=recognizer.device)
            except Exception as exc:
                send(('error', exc))
            finally:
                send(('done', None))

        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        try:
            while not self.wait_pause():
                try:
                    kind, value = packets.get(timeout=.05)
                except Empty:
                    continue
                if kind == 'done':
                    return True
                if kind == 'error':
                    raise RuntimeError(f'第 {batch_index} 批采集失败，轮询已暂停：{value}')
                target, identifier, channel, rate = value
                if not self.bandwidth_filter.allows(target):
                    continue
                status = prefix+f' | S{identifier:04d}'
                def progress(done, total):
                    text = f'识别图像 {done}/{total}'
                    self._post_recognition(identifier, {2: text, 8: f'{done}/{total}'},
                                           status+' | '+text)
                metadata = dict(signal_id=identifier,
                    frame=job['frame'], cycle=job['cycle'], source_start_sample=job['start'],
                    batch=batch_index, channel_type=target.target_type,
                    channel_slot=target.sequence, center_hz=self.center+target.center_hz,
                    baseband_center_hz=target.center_hz, bandwidth_hz=target.bandwidth_hz,
                    rate_multiplier=job['multiplier'],
                    filter_min_hz=job['filter'].bandwidth_min_hz,
                    filter_max_hz=job['filter'].bandwidth_max_hz)
                metadata.update(image_window_mode='automatic' if self.max_images_per_packet else 'fixed',
                    max_images_per_packet=self.max_images_per_packet, minimum_window_ms=self.image_window_ms)
                preview_state = dict(images=[], frame=job['frame'],
                    center_hz=self.center+target.center_hz, sample_rate_hz=rate)
                def preview(images):
                    if images:
                        preview_state.setdefault('window_ms', images[0]['sample_count']/rate*1000)
                    preview_state['images'] = preview_state['images']+images
                    self._post_recognition(identifier, {}, status, dict(preview_state))
                try:
                    result, packet_id = pipeline.recognize_iq(
                        (channel if isinstance(channel, torch.Tensor) else
                         torch.from_numpy(channel).to(recognizer.device)), rate, metadata,
                        cancelled=self.wait_pause, progress=progress, preview=preview)
                except InterruptedError:
                    return False
                except Exception as exc:
                    self._post_recognition(identifier, {2: f'识别/保存失败：{exc}',
                        3: '—', 8: '—', 9: '—', 10: '未保存'}, status+' | 轮询已暂停')
                    raise RuntimeError(f'第 {batch_index} 批未全部识别成功，轮询已暂停：{exc}') from exc
                self._post_recognition(identifier, {2: result.class_name,
                    3: f'{result.confidence:.1%}', 8: str(len(result.windows)),
                    9: f'{result.vote_fraction:.1%}'+('（平票）' if result.tied else ''),
                    10: str(packet_id)}, status+' | 已识别并保存')
            return False
        finally:
            stop_capture.set()
            collector.join()

    def run(self):
        reader = None
        recognition_thread = None
        state = None
        try:
            recognizer = None
            model_status = '未加载模型：仅检测，未分类'
            nn_state = '神经网络未启用：尚未选择训练权重。请点击“加载识别模型”。'
            model_error = ''
            unclassified = '未加载模型'
            if self.checkpoint:
                self.publish({'status': '正在加载识别模型…', 'nn_state': '正在加载神经网络权重…'})
                try:
                    from gpu_iq_recognizer import SigRecGpuRecognizer
                    recognizer = SigRecGpuRecognizer.from_checkpoint(
                        self.checkpoint, strict=True, window_ms=self.image_window_ms,
                        max_images_per_packet=self.max_images_per_packet)
                    model_status = 'GPU 模型已加载'
                    nn_state = (f'神经网络已启用：{Path(self.checkpoint).name} | '
                        f'{len(recognizer.class_names)}类 | GPU下采样 / 批量制图 / GPU推理 | '
                        +(f'自动窗口 {self.image_window_ms:g} ms起步，每包最多{self.max_images_per_packet}图'
                          if self.max_images_per_packet else f'固定窗口 {self.image_window_ms:g} ms/图'))
                except Exception as exc:
                    model_error = str(exc)
                    model_status = '模型加载失败，详细原因见上方神经网络状态'
                    nn_state = f'神经网络加载失败：{model_error.replace(chr(10), " ")[:240]}'
                    unclassified = '模型加载失败'
            if recognizer:
                recognition_thread = threading.Thread(target=self._recognize_rounds,
                    args=(recognizer,), daemon=True)
                recognition_thread.start()
                model_status += (' | '+(f'自动窗口，{self.image_window_ms:g} ms起步，最多{self.max_images_per_packet}图/包'
                    if self.max_images_per_packet else f'固定窗口{self.image_window_ms:g} ms/图')
                    +f' | 数据库：{self.database_path}')
            # File recordings often contain multi-carrier/modulated emissions:
            # retain one target when their valley remains above the measured
            # noise floor, instead of issuing one IQ capture per carrier.
            detector = EnergyDetector(snr_db=self.threshold,
                                      component_merge_gap_hz=25_000_000)
            reader = IQReplayReader(self.info)
            frame = 0
            while not self.cancel.is_set():
                if self.wait_pause():
                    break
                tick = time.monotonic()
                raw, start = reader.read()
                power = spectrum_power(raw)
                detected_targets = detect_signals(power, self.info.sample_rate, self.usable, detector)
                detected_targets = [classify_target(t) for t in detected_targets]
                detected_count = len(detected_targets)
                frame_filter, frame_multiplier = self.bandwidth_filter, self.rate_multiplier
                targets, rejected = prepare_targets(detected_targets, self.info.sample_rate, frame_multiplier)
                batches = target_batches(targets, frame_filter)
                assignments = {(t.target_type, t.center_hz, t.bandwidth_hz): batch_index
                               for batch_index, batch in enumerate(batches, 1) for t in batch}
                targets = [t for t in targets if (t.target_type, t.center_hz, t.bandwidth_hz) in assignments]
                axis = self.center + np.fft.fftshift(np.fft.fftfreq(len(power), 1/self.info.sample_rate))
                rows = [[f'{self.center/1e6 + t.center_hz/1e6:.6f}',
                         f'{t.bandwidth_hz/1e6:.6f}',
                         '待识别' if recognizer else unclassified, '—',
                         '宽带' if t.target_type == 'wide' else '窄带',
                         str(assignments.get((t.target_type, t.center_hz, t.bandwidth_hz), '—')),
                         f'{t.sample_rate_hz/1e6:.6f}',
                         f'{(4*1024*1024//4)/t.sample_rate_hz:.6f}',
                         '待采集' if recognizer else '未生成',
                         '待识别' if recognizer else '未识别', '未保存'] for t in targets]
                frame += 1
                state = dict(frame=frame, axis=axis, center=self.center, nn_state=nn_state,
                             model_error=model_error,
                             source_start_sample=start,
                             db=10*np.log10(np.maximum(power, 1e-20)),
                             # Mark every detected occupied interval in the
                             # spectrum, including one that cannot yet be
                             # sampled at 3x its width.  Scheduling remains
                             # limited to ``targets`` below.
                             display_targets=detected_targets,
                             targets=targets, rows=rows,
                             status=f'{self.info.path.name} | 第 {reader.cycle} 轮 | '
                                    f'{start/self.info.sample_rate:.3f} / '
                                    f'{self.info.samples/self.info.sample_rate:.3f} s | '
                                    f'{detected_count} 个候选 / {len(assignments)} 个符合条件 / 预划分{len(batches)}批 | '
                                    f'{len(rejected)} 路采样率不足 | {model_status}')
                self.publish({**state, 'rows': [r.copy() for r in rows]})
                if (recognizer and not self._round_busy and not self._recognition_error
                        and batches and not self.cancel.is_set()):
                    frozen = [[(t, self.tracker.identifier_for(t)) for t in batch]
                              for batch in batches]
                    self._round_busy = True
                    self._scheduled_ids = {identifier: index
                        for index, batch in enumerate(frozen, 1) for _, identifier in batch}
                    self._round_ids = tuple(self._scheduled_ids)
                    self._round_display_targets = tuple(detected_targets)
                    self._active_batch = 0
                    self._jobs.put_nowait(dict(batches=frozen, start=start, frame=frame,
                        cycle=reader.cycle, filter=frame_filter, multiplier=frame_multiplier))
                    # 立即发布 t0 快照，替换刚才的实时检测帧，供 GUI 固定显示。
                    self.publish({**state, 'rows': [r.copy() for r in rows]}, observe=False)
                # Every sample is visited; processing-speed playback, not 320 MSps realtime.
                self.cancel.wait(max(0, .1 - (time.monotonic()-tick)))
        except Exception as exc:
            state = {**(state or {}), 'error': str(exc),
                     'nn_state': f'识别流程失败：{str(exc)[:240]}', 'model_error': str(exc)}
            self.publish(state, observe=False)
        finally:
            self.cancel.set()
            if reader:
                reader.close()
            if recognition_thread:
                # The parent remains alive until its consumer exits. Model switches
                # cannot start another consumer on the GPU before this one ends.
                recognition_thread.join()
            if state is not None:
                self.publish(state, observe=False)


class SpectrumBandsItem(pg.GraphicsObject):
    """Draw all read-only frequency marks in one item instead of hundreds."""
    def __init__(self):
        super().__init__()
        self._path = QtGui.QPainterPath()
        self._bounds = QtCore.QRectF()
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.setZValue(-5)

    def set_bands(self, bands):
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.FillRule.WindingFill)
        for low, high in bands:
            path.addRect(QtCore.QRectF(low, -1000, high-low, 2000))
        self.prepareGeometryChange()
        self._path, self._bounds = path, path.boundingRect()
        self.update()

    def boundingRect(self):
        return self._bounds

    def paint(self, painter, option, widget=None):
        # Marks are deliberately subtle: they locate a detected channel but
        # must not obscure weak detail in the spectrum underneath.
        painter.setPen(pg.mkPen(70, 180, 230, 120))
        painter.setBrush(pg.mkBrush(70, 180, 230, 10))
        painter.drawPath(self._path)


def smooth_spectrum(values, window: int) -> np.ndarray:
    """Return an edge-preserving moving average for display only."""
    values = np.asarray(values, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or values.size < 3:
        return values.copy()
    window = min(window | 1, values.size if values.size % 2 else values.size - 1)
    if window <= 1:
        return values.copy()
    half = window // 2
    padded = np.pad(values, (half, half), mode='edge')
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return np.convolve(padded, kernel, mode='valid').astype(np.float32, copy=False)


class FileReplayPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('receiverPage')
        self.worker = None
        self.info = None
        remembered = remembered_checkpoint()
        self.checkpoint = remembered or discover_sigrec_checkpoint()
        self._restart_after_model_change = False
        self._frame = None
        self._last_state = None
        self._visible_targets = None
        self._regions_last_update = 0.0
        self._tracker = SignalTracker()
        self._table_last_update = 0.0
        self._table_ids = []
        self._trace = TraceProcessor()
        self._trace_axis = np.empty(0, dtype=np.float64)
        self._trace_raw = np.empty(0, dtype=np.float32)
        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton('选择 IQ 文件…')
        self.play_button = QtWidgets.QPushButton('开始循环')
        self.pause_button = QtWidgets.QPushButton('暂停')
        self.pause_button.setCheckable(True)
        self.stop_button = QtWidgets.QPushButton('停止')
        self.model_button = QtWidgets.QPushButton('加载识别模型…')
        self.model_button.setToolTip('播放时也可选择模型；选中后结束当前处理，重新开始文件回放并加载模型')
        self.history_button = QtWidgets.QPushButton('识别记录…')
        self.images_button = QtWidgets.QPushButton('查看时频图…')
        self.images_button.setToolTip('选择信号查看实际神经网络输入图像；没有模型时可从真实IQ单独生成预览')
        self.auto_adjust_button = QtWidgets.QPushButton('自动调整')
        self.auto_adjust_button.setToolTip('恢复文件频率范围，并按当前频谱调整功率刻度和瀑布图色阶')
        for button in (self.open_button, self.play_button, self.pause_button, self.stop_button, self.model_button):
            bar.addWidget(button)
        bar.addWidget(self.history_button)
        bar.addWidget(self.images_button)
        bar.addStretch()
        self.show_waterfall = QtWidgets.QCheckBox('显示瀑布图')
        bar.addWidget(self.show_waterfall)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel('频谱迹线'))
        self.trace_mode = QtWidgets.QComboBox()
        self.trace_mode.addItems(('实时', '最小保持', '最大保持'))
        self.trace_mode.setToolTip('只影响上方频谱显示；不会影响信号检测、采集或瀑布图。')
        bar.addWidget(self.trace_mode)
        bar.addWidget(QtWidgets.QLabel('平滑'))
        self.trace_smoothing = QtWidgets.QSpinBox()
        self.trace_smoothing.setRange(1, 101)
        self.trace_smoothing.setSingleStep(2)
        self.trace_smoothing.setValue(1)
        self.trace_smoothing.setSuffix(' 点')
        self.trace_smoothing.setToolTip('频谱显示的移动平均窗口；1 点表示关闭平滑。')
        bar.addWidget(self.trace_smoothing)
        self.clear_trace_button = QtWidgets.QPushButton('清除保持')
        self.clear_trace_button.setToolTip('清空最小/最大保持的历史，并从下一帧重新累计。')
        bar.addWidget(self.clear_trace_button)
        bar.addWidget(self.auto_adjust_button)
        layout.addLayout(bar)
        # Keep the parameter block independently resizable.  The controls used
        # to consume a fixed amount of height above the spectrum, particularly
        # on wide displays; a vertical splitter lets it collapse when viewing
        # fine spectrum detail is the priority.
        self.replay_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.replay_splitter.setChildrenCollapsible(True)
        self.replay_settings_panel = QtWidgets.QWidget()
        settings_panel_layout = QtWidgets.QVBoxLayout(self.replay_settings_panel)
        settings_panel_layout.setContentsMargins(0, 0, 0, 0)
        settings = QtWidgets.QHBoxLayout()
        self.center = QtWidgets.QDoubleSpinBox()
        self.center.setRange(0, 100000)
        self.center.setDecimals(6)
        self.center.setSuffix(' MHz')
        self.band = QtWidgets.QDoubleSpinBox()
        self.band.setRange(.001, 10000)
        self.band.setDecimals(3)
        self.band.setValue(300)
        self.band.setSuffix(' MHz')
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(.1, 60)
        self.threshold.setValue(4)
        self.threshold.setSuffix(' dB')
        self.image_window = QtWidgets.QDoubleSpinBox()
        self.image_window.setRange(.1, 100)
        self.image_window.setDecimals(1)
        self.image_window.setValue(10)
        self.image_window.setSuffix(' ms')
        self.image_window.setToolTip('自动模式下为最小窗口，默认10 ms；窗口按整包时长延长，全部IQ参与投票')
        for label, widget in (('中心频率', self.center), ('有效检测带宽', self.band),
                              ('检测信噪比门限', self.threshold), ('识别图像窗口', self.image_window)):
            settings.addWidget(QtWidgets.QLabel(label))
            settings.addWidget(widget)
        self.auto_image_window = QtWidgets.QCheckBox('自动图像窗口')
        self.auto_image_window.setChecked(True)
        self.max_images = QtWidgets.QSpinBox()
        self.max_images.setRange(1, 512)
        self.max_images.setValue(64)
        self.max_images.setSuffix(' 张/包')
        self.max_images.setToolTip('每包最多生成此数量的真实时频图；自动扩大窗口并覆盖全部IQ')
        self.auto_image_window.toggled.connect(self.max_images.setEnabled)
        settings.addWidget(self.auto_image_window)
        settings.addWidget(self.max_images)
        settings.addStretch()
        settings_panel_layout.addLayout(settings)
        self.nn_status = QtWidgets.QLabel('神经网络未启用：尚未选择训练权重。请点击“加载识别模型”。')
        self.nn_status.setWordWrap(True)
        self.nn_status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.nn_status.setStyleSheet('color: #ffd166; font-weight: 600;')
        settings_panel_layout.addWidget(self.nn_status)
        if self.checkpoint:
            self.model_button.setText(Path(self.checkpoint).name)
            prefix = '已记住权重' if remembered else '已自动找到sigRec模型'
            self.nn_status.setText(f'{prefix}：{self.checkpoint}；开始回放后加载神经网络。')
            self.model_button.setToolTip(f'当前权重：{self.checkpoint}；点击可切换模型')
        self.recognition_bandwidth = RecognitionBandwidth()
        self.recognition_bandwidth.changed.connect(self._recognition_filter_changed)
        self.recognition_bandwidth.rateMultiplierChanged.connect(self._rate_multiplier_changed)
        settings_panel_layout.addWidget(self.recognition_bandwidth)
        self.description = QtWidgets.QLabel('请选择 RF64/WAVE 文件。每帧读取 10 ms IQ，按处理速度循环回放。')
        self.description.setWordWrap(True)
        settings_panel_layout.addWidget(self.description)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.spectrum = pg.PlotWidget()
        self.spectrum.enableAutoRange(x=False, y=False)
        self.spectrum.setYRange(-120, 0, padding=0)
        self.spectrum.getViewBox().setMouseEnabled(x=True, y=False)
        self.spectrum.setMenuEnabled(False)
        self.spectrum.hideButtons()
        self.spectrum.setLabel('bottom', '频率', units='Hz')
        self.spectrum.setLabel('left', '功率', units='dBFS/bin')
        self.spectrum.showGrid(x=True, y=True, alpha=.25)
        self.curve = self.spectrum.plot(pen=pg.mkPen('#31d855', width=1))
        self._band_item = SpectrumBandsItem()
        self.spectrum.addItem(self._band_item, ignoreBounds=True)
        self.spectrum.setDownsampling(auto=True, mode='peak')
        self.waterfall = WaterfallPlot(history_rows=200, display_bins=2048)
        self.waterfall.setXLink(self.spectrum)
        self.table = QtWidgets.QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(['中心频率 (MHz)', '检测带宽 (MHz)', '信号类别', '模型置信度', '通道类型', '轮询批次', 'IQ采样率 (MSps)', '4 MiB采集时长 (s)', '识别图像数', '多数票占比', '数据库记录', '检测状态'])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setToolTip('连续检出的稳定信号按中心频率升序；新出现或短暂漏检的信号置于末尾。短暂漏检保留10个检测帧。')
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setResizeContentsPrecision(32)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for widget in (self.spectrum, self.waterfall, self.table):
            split.addWidget(widget)
        split.setSizes([400, 230, 190])
        self.waterfall.hide()
        self.show_waterfall.toggled.connect(self.waterfall.setVisible)
        self.replay_splitter.addWidget(self.replay_settings_panel)
        self.replay_splitter.addWidget(split)
        self.replay_splitter.setStretchFactor(0, 0)
        self.replay_splitter.setStretchFactor(1, 1)
        self.replay_splitter.setSizes([170, 830])
        layout.addWidget(self.replay_splitter, 1)
        self.status = QtWidgets.QLabel('未加载模型：可先查看频谱和检测频段；分类需要兼容的 ResNet 权重及 CUDA。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.open_button.clicked.connect(self.choose_file)
        self.model_button.clicked.connect(self.choose_model)
        self.history_button.clicked.connect(self.show_history)
        self.images_button.clicked.connect(self.show_images)
        self.play_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.pause_button.toggled.connect(self.pause)
        self.auto_adjust_button.clicked.connect(self.auto_adjust)
        self.trace_mode.currentTextChanged.connect(self._trace_mode_changed)
        self.trace_smoothing.valueChanged.connect(self._trace_smoothing_changed)
        self.clear_trace_button.clicked.connect(self._clear_trace_hold)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self._set_running(False)

    def _trace_mode_changed(self, mode):
        self._trace.configure(mode)
        self._redraw_trace()

    def _trace_smoothing_changed(self, _value):
        self._redraw_trace(reseed=True)

    def _clear_trace_hold(self):
        self._trace.reset()
        self._redraw_trace(reseed=True)

    def _redraw_trace(self, reseed=False):
        if not self._trace_raw.size:
            return
        if reseed:
            self._trace.reset()
        values = smooth_spectrum(self._trace_raw, self.trace_smoothing.value())
        self.curve.setData(self._trace_axis, self._trace.apply(values))

    def _update_trace(self, axis, values):
        axis = np.asarray(axis, dtype=np.float64)
        values = np.asarray(values, dtype=np.float32)
        if not np.array_equal(axis, self._trace_axis):
            self._trace.reset()
        self._trace_axis, self._trace_raw = axis.copy(), values.copy()
        self._redraw_trace()

    def _set_running(self, running):
        for control in (self.open_button, self.center, self.band, self.threshold, self.image_window,
                        self.auto_image_window):
            control.setEnabled(not running)
        self.max_images.setEnabled(not running and self.auto_image_window.isChecked())
        self.model_button.setEnabled(self.worker is None or not self.worker.cancel.is_set())
        self.play_button.setEnabled(not running and self.info is not None)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择单个 IQ 文件循环回放', str(Path(__file__).resolve().parents[1] / 'data'),
            'RF64 / WAVE (*.wav *.rf64);;所有文件 (*)')
        if not path:
            return
        try:
            info = inspect_iq_file(path)
            center = info.center_hz
            if center is None:
                mhz, ok = QtWidgets.QInputDialog.getDouble(self, '中心频率',
                    '文件未提供射频中心频率，请输入 MHz：', self.center.value(), 0, 100000, 6)
                if not ok:
                    return
                center = mhz * 1e6
            self.info = info
            self.center.setValue(center / 1e6)
            self.band.setMaximum(info.sample_rate / 1e6)
            self.band.setValue(min(300, info.sample_rate / 1e6))
            self.description.setText(f'{info.path.name} | I/Q int16 | {info.sample_rate/1e6:g} MSps | '
                                     f'{info.samples:,} 复数点 | {info.samples/info.sample_rate:g} s。'
                                     '每帧 10 ms，逐块处理；单文件循环，非实时速度。')
            self.start()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, '无法读取 IQ 文件', str(exc))

    def choose_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择兼容的 ResNet 权重', '',
                                                       '模型权重 (*.pth *.pt)')
        if path:
            self.checkpoint = path
            try:
                remember_checkpoint(path)
            except OSError as exc:
                self.status.setText(f'模型已选择，但无法记住路径：{exc}')
            self.model_button.setText(Path(path).name)
            self.nn_status.setText(f'已选择权重：{path}；等待重新开始并加载神经网络。')
            if self.worker is not None:
                self.stop()
                self._restart_after_model_change = True
                self.model_button.setEnabled(False)
                self.stop_button.setEnabled(True)  # Allows cancelling the queued restart.
                self.status.setText('已选择模型，正在结束当前处理；随后重新开始文件回放并加载模型。')
            else:
                self.status.setText('已选择模型，点击“开始循环”加载并识别；自动读取5类或含未知信号的6类输出。')

    def show_history(self):
        from .recognition_history import RecognitionHistoryDialog
        dialog = RecognitionHistoryDialog(DEFAULT_DATABASE, self)
        dialog.exec()

    def show_images(self):
        row = self.table.currentRow()
        if self.info is None or self._last_state is None or not 0 <= row < len(self._table_ids):
            self.status.setText('请先在信号列表中选择一行，再点击“查看时频图”。')
            return
        identifier = self._table_ids[row]
        track = self._tracker.tracks.get(identifier)
        cached = self._last_state.get('previews', {}).get(identifier)
        if track is None or (track.missed and cached is None):
            self.status.setText('该信号暂未检出且没有缓存图像，请选择当前检出的信号。')
            return
        from .iq_image_preview import IQImagePreviewDialog
        worker = self.worker
        already_paused = worker.paused.is_set() if worker else False
        if worker:
            worker.paused.set()
        try:
            dialog = IQImagePreviewDialog(self.info,
                self._last_state.get('source_start_sample', 0), track.target,
                self._last_state['center']+track.target.center_hz,
                self.image_window.value(), self, cached=cached,
                max_images=self.max_images.value() if self.auto_image_window.isChecked() else None)
            dialog.exec()
        finally:
            if worker and worker is self.worker and not already_paused:
                worker.paused.clear()

    def start(self):
        if self.info is None or self.worker is not None:
            return
        try:
            selected_filter = self.recognition_bandwidth.value()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, '识别带宽无效', str(exc))
            return
        self._frame = None
        self._last_state = None
        self._tracker = SignalTracker()
        self._table_last_update = 0.0
        self._table_ids = []
        self.recognition_bandwidth.current = selected_filter
        self.curve.setData([], [])
        self._trace.reset()
        self._trace_axis = np.empty(0, dtype=np.float64)
        self._trace_raw = np.empty(0, dtype=np.float32)
        self.table.setRowCount(0)
        self._clear_regions()
        self.waterfall.clear_history()
        span = min(self.band.value()*1e6, self.info.sample_rate)
        low = self.center.value()*1e6 - span/2
        high = low + span
        self.spectrum.setLimits(xMin=low, xMax=high,
                                minXRange=max(1, span/16384), maxXRange=span)
        self.spectrum.setXRange(low, high, padding=0)
        # The image still represents all sampled bins; clip its viewport only.
        sampled_low = self.center.value()*1e6 - self.info.sample_rate/2
        self.waterfall.set_freq_range(sampled_low, sampled_low+self.info.sample_rate)
        self.waterfall.setLimits(xMin=low, xMax=high,
                                 minXRange=max(1, span/16384), maxXRange=span)
        self.waterfall.setXRange(low, high, padding=0)
        self.worker = ReplayWorker(self.info, self.center.value()*1e6,
                                   self.band.value()*1e6, self.threshold.value(), self.checkpoint)
        self.worker.bandwidth_filter = selected_filter
        self.worker.rate_multiplier = float(self.recognition_bandwidth.rate_multiplier.currentData())
        self.worker.image_window_ms = self.image_window.value()
        self.worker.max_images_per_packet = self.max_images.value() if self.auto_image_window.isChecked() else None
        self.nn_status.setText('正在加载神经网络权重…' if self.checkpoint else
            '神经网络未启用：尚未选择训练权重。可以选择信号查看时频图预览。')
        self._set_running(True)
        self.worker.thread.start()

    def _recognition_filter_changed(self, selected_filter):
        if self.worker:
            self.worker.bandwidth_filter = selected_filter
        if self._last_state is not None:
            self._tracker.apply_filter(selected_filter)
            self._render_targets(self._last_state, force_table=True)

    def _rate_multiplier_changed(self, multiplier):
        if self.worker:
            self.worker.rate_multiplier = multiplier

    def pause(self, paused):
        if self.worker:
            self.worker.paused.set() if paused else self.worker.paused.clear()
        if paused:
            self._update_signal_table(force=True)
        self.pause_button.setText('继续' if paused else '暂停')

    def auto_adjust(self):
        low, high = self.spectrum.getViewBox().state['limits']['xLimits']
        if low is None or high is None:
            return
        self.spectrum.setXRange(low, high, padding=0)
        self.waterfall.setXRange(low, high, padding=0)
        frequencies, values = self.curve.getData()
        if values is None or frequencies is None:
            return
        values = values[np.isfinite(values) & (frequencies >= low) & (frequencies <= high)]
        if not values.size:
            return
        floor = float(np.percentile(values, 10))
        peak = float(np.max(values))
        margin = max(3.0, (peak-floor)*.1)
        self.spectrum.setYRange(floor-margin, peak+margin, padding=0)
        self.waterfall.set_levels(floor-margin, peak+margin)

    def stop(self):
        self._restart_after_model_change = False
        if self.worker:
            self.worker.cancel.set()
            self.status.setText('正在停止回放…')
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.model_button.setEnabled(False)

    def shutdown(self):
        self._restart_after_model_change = False
        self.timer.stop()
        if self.worker:
            self.worker.cancel.set()
            self.worker.thread.join(timeout=2)

    def _clear_regions(self):
        self._band_item.set_bands([])
        self._visible_targets = None
        self._regions_last_update = 0.0

    def _render_targets(self, state, force_table=False):
        visible = [(target, row) for target, row in zip(state['targets'], state['rows'])
                   if self.recognition_bandwidth.current.allows(target)]
        display_targets = tuple(target for target in state.get('display_targets', state['targets'])
                                if self.recognition_bandwidth.current.allows(target))
        now = time.monotonic()
        if display_targets != self._visible_targets and (force_table or now-self._regions_last_update >= .25):
            bands = []
            for target in display_targets:
                center = state['center'] + target.center_hz
                # These are the measured occupied edges, not a scaled display
                # approximation.  The fill is intentionally almost invisible
                # so exact-width marks do not hide fine spectral detail.
                bands.append((center-target.bandwidth_hz/2, center+target.bandwidth_hz/2))
            self._band_item.set_bands(bands)
            self._visible_targets = display_targets
            self._regions_last_update = now
        axis = state['axis']
        bin_hz = abs(float(axis[1]-axis[0])) if len(axis) > 1 else 1.0
        if 'tracks' in state:
            self._tracker.tracks = {t.identifier: replace(t, row=list(t.row))
                                    for t in state['tracks']
                                    if self.recognition_bandwidth.current.allows(t.target)}
        else:
            self._tracker.observe(state['frame'], [t for t, _ in visible],
                                  [row for _, row in visible], bin_hz)
        self._update_signal_table(force=force_table)

    def _update_signal_table(self, force=False):
        if self._last_state is None:
            return
        now = time.monotonic()
        if not force and now-self._table_last_update < 1.0:
            return
        self._table_last_update = now
        tracks = self._tracker.visible(self.recognition_bandwidth.current)
        selected_row = self.table.currentRow()
        selected_id = self._table_ids[selected_row] if 0 <= selected_row < len(self._table_ids) else None
        scroll = self.table.verticalScrollBar().value()
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(tracks))
            self._table_ids = [track.identifier for track in tracks]
            for row, track in enumerate(tracks):
                row_values = list(track.row)
                row_values.extend(['—'] * (11-len(row_values)))
                values = [*row_values,
                          '当前检出' if not track.missed else f'暂未检出（{track.missed}帧）']
                # kHz precision is readable without implying sub-bin accuracy.
                values[0:2] = [f'{float(value):.3f}' for value in values[0:2]]
                if values[5].isdigit():
                    values[5] = f'第 {values[5]} 批'
                if track.missed:
                    values[5] = '暂未检出'
                for col, value in enumerate(values):
                    item = self.table.item(row, col)
                    if item is None:
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
                    elif item.text() != value:
                        item.setText(value)
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
            if selected_id in self._table_ids:
                self.table.selectRow(self._table_ids.index(selected_id))
            self.table.verticalScrollBar().setValue(scroll)
        finally:
            self.table.setUpdatesEnabled(True)

    def poll(self):
        worker = self.worker
        if worker is None:
            return
        if worker.paused.is_set() and worker.thread.is_alive() and not worker.cancel.is_set():
            return
        try:
            state = worker.mailbox.get_nowait()
        except Empty:
            state = None
        if state and not worker.cancel.is_set() and not worker.paused.is_set():
            if 'nn_state' in state:
                self.nn_status.setText(state['nn_state'])
                self.nn_status.setToolTip(state.get('model_error') or state['nn_state'])
                self.nn_status.setStyleSheet('font-weight: 600; color: '+
                    ('#ff7474' if '失败' in state['nn_state'] else
                     '#31d855' if '已启用' in state['nn_state'] else '#ffd166')+';')
            self.status.setText(state.get('error', state.get('status', '')))
            if 'frame' in state:
                if self._frame != state['frame']:
                    self._update_trace(state['axis'], state['db'])
                    if self._frame is None:
                        # Seed once per playback; subsequent frames keep the
                        # scale until the user explicitly requests adjustment.
                        self.auto_adjust()
                        self.waterfall.clear_history()
                    if self.show_waterfall.isChecked():
                        self.waterfall.push_row(state['db'])
                    self._frame = state['frame']
                self._last_state = state
                self._render_targets(state)
        if not worker.thread.is_alive():
            self._update_signal_table(force=True)
            self.worker = None
            self.pause_button.setChecked(False)
            self._set_running(False)
            if self._restart_after_model_change:
                self._restart_after_model_change = False
                self.start()
            elif worker.cancel.is_set():
                self.status.setText('回放已停止，可选择其他文件或加载识别模型。')
        else:
            self._update_signal_table()
