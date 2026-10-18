"""Read-only view of persisted packet classifications and individual votes."""
from datetime import datetime
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from iq_file_source import spectrum_power


class RecordedSpectrumDialog(QtWidgets.QDialog):
    """Replay the stored post-DDC IQ spectrum for one recognition record."""
    def __init__(self, record, iq_root, parent=None):
        super().__init__(parent)
        self.record = record
        self.iq_root = Path(iq_root)
        self.raw_iq = None
        self.sample_rate = 0
        self.frame_samples = 0
        self.frame_count = 0
        self.setWindowTitle('记录频谱回放')
        self.resize(1200, 620)
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0e1116')
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setLabel('bottom', '频率', units='Hz')
        self.plot.setLabel('left', '功率', units='dBFS/bin')
        self.plot.showGrid(x=True, y=True, alpha=.25)
        self.curve = self.plot.plot(pen=pg.mkPen('#31d855', width=1))
        self.band = pg.LinearRegionItem(movable=False,
            pen=pg.mkPen('#46b4e6', width=1), brush=pg.mkBrush(70, 180, 230, 18))
        self.plot.addItem(self.band)
        layout.addWidget(self.plot, 1)
        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton('上一帧')
        self.play_button = QtWidgets.QPushButton('播放')
        self.play_button.setCheckable(True)
        self.next_button = QtWidgets.QPushButton('下一帧')
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_label = QtWidgets.QLabel('未加载')
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.play_button)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.frame_slider, 1)
        navigation.addWidget(self.frame_label)
        layout.addLayout(navigation)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        refresh = buttons.addButton('重新读取频谱', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        refresh.clicked.connect(self.load_spectrum)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.frame_slider.valueChanged.connect(self._show_frame)
        self.previous_button.clicked.connect(lambda: self._step_frame(-1))
        self.next_button.clicked.connect(lambda: self._step_frame(1))
        self.play_button.toggled.connect(self._set_playing)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(lambda: self._step_frame(1))
        self.load_spectrum()

    def load_spectrum(self):
        relative = self.record.get('iq_path')
        if not relative:
            self.status.setText('该记录没有保存重采样 IQ，无法频谱回放。旧格式记录已不支持回放。')
            self.curve.setData([], [])
            self.band.hide()
            self.raw_iq = None
            return
        saved_iq = self.iq_root / relative
        if not saved_iq.is_file():
            self.status.setText(f'保存的重采样 IQ 文件不存在。\n{saved_iq}')
            self.curve.setData([], [])
            self.band.hide()
            self.raw_iq = None
            return
        try:
            with np.load(saved_iq, allow_pickle=False) as saved:
                self.raw_iq = np.asarray(saved['iq'], dtype=np.int16)
                self.sample_rate = int(saved['sample_rate_hz'])
            self.frame_samples = max(1, round(self.sample_rate*.01))
            self.frame_count = int(np.ceil(len(self.raw_iq) / self.frame_samples))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, self.frame_count-1))
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            half = self.record['bandwidth_hz']/2
            self.band.setRegion((self.record['center_hz']-half, self.record['center_hz']+half))
            self.band.show()
            self._show_frame(0, reset_range=True)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f'无法回放该记录的频谱：{exc}')
            self.curve.setData([], [])
            self.band.hide()
            self.raw_iq = None

    def _show_frame(self, index, reset_range=False):
        if self.raw_iq is None or not self.frame_count:
            return
        index = max(0, min(int(index), self.frame_count-1))
        start = index*self.frame_samples
        raw = self.raw_iq[start:start+self.frame_samples]
        power = spectrum_power(raw)
        axis = self.record['center_hz'] + np.fft.fftshift(
            np.fft.fftfreq(len(power), 1/self.sample_rate))
        self.curve.setData(axis, 10*np.log10(np.maximum(power, 1e-20)))
        end = min(start+len(raw), len(self.raw_iq))
        self.frame_label.setText(f'第 {index+1}/{self.frame_count} 帧')
        self.status.setText(
            f'连续重采样 IQ 回放：{start/self.sample_rate*1000:.3f}–'
            f'{end/self.sample_rate*1000:.3f} ms / 共 {len(self.raw_iq)/self.sample_rate*1000:.3f} ms | '
            f'{self.sample_rate/1e6:.6f} MSps。蓝色区域为检测带宽。')
        if reset_range:
            # A replay is one continuous DDC packet.  Lock both axes from the
            # first frame so a changing peak/noise floor cannot make the plot
            # appear to jump while the user steps through it.
            values = 10*np.log10(np.maximum(power, 1e-20))
            floor = float(np.percentile(values, 5))
            peak = float(np.percentile(values, 99.5))
            margin = max(3.0, (peak-floor)*.08)
            self.plot.setXRange(float(axis[0]), float(axis[-1]), padding=0)
            self.plot.setYRange(floor-margin, peak+margin, padding=0)
            self.plot.enableAutoRange(x=False, y=False)

    def _step_frame(self, step):
        if not self.frame_count:
            return
        next_index = (self.frame_slider.value()+step) % self.frame_count
        self.frame_slider.setValue(next_index)

    def _set_playing(self, playing):
        self.play_button.setText('暂停' if playing else '播放')
        self.timer.start() if playing and self.frame_count else self.timer.stop()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


class RecognitionHistoryDialog(QtWidgets.QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.records = []
        self.setWindowTitle('识别记录')
        self.resize(1400, 750)
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel(str(self.path))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QtWidgets.QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(['记录', '时间', '信号编号', '中心频率 (MHz)',
            '带宽 (MHz)', 'IQ采样率 (MSps)', '类别', '模型置信度', '多数票占比', '图像数'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table, 2)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('选择记录查看每张图的识别结果；编号在各次回放会话内有效。')
        layout.addWidget(self.details, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        refresh = buttons.addButton('刷新记录', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.replay_button = buttons.addButton('频谱回放', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        refresh.clicked.connect(self.refresh)
        self.replay_button.clicked.connect(self.replay_spectrum)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.itemSelectionChanged.connect(self.show_details)
        self.refresh()

    def query(self, sql, parameters=()):
        connection = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            return [dict(r) for r in connection.execute(sql, parameters)]
        finally:
            connection.close()

    def refresh(self):
        if not self.path.exists():
            self.status.setText(f'尚无识别记录。加载模型并完成识别后保存至：{self.path}')
            return
        try:
            records = self.query('''SELECT p.*, s.source, s.settings_json FROM packets p
                JOIN sessions s ON s.id=p.session_id ORDER BY p.id DESC LIMIT 200''')
            self.table.clearSelection()
            self.details.clear()
            self.records = records
            self.table.setRowCount(len(records))
            for index, record in enumerate(records):
                values = [str(record['id']),
                    datetime.fromisoformat(record['created_utc']).astimezone().strftime('%m-%d %H:%M:%S'),
                    f"S{record['signal_id']:04d}" if record['signal_id'] is not None else '—',
                    f"{record['center_hz']/1e6:.3f}", f"{record['bandwidth_hz']/1e6:.3f}",
                    f"{record['sample_rate_hz']/1e6:.6f}", record['class_name'],
                    f"{record['confidence']:.1%}", f"{record['vote_fraction']:.1%}"+
                    ('（平票）' if record['tied'] else ''), str(record['image_count'])]
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.table.setItem(index, column, item)
            self.status.setText(f'最近 {len(records)} 条记录（每个4 MiB包一条，最新在前） | {self.path}')
        except (sqlite3.Error, ValueError) as exc:
            self.status.setText(f'读取记录失败：{exc}')

    def show_details(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.records):
            return
        record = self.records[row]
        try:
            windows = self.query('SELECT * FROM windows WHERE packet_id=? ORDER BY image_index', (record['id'],))
            votes = json.loads(record['votes_json'])
            text = [f"会话：{record['session_id']}  | 来源：{record['source']}",
                f"投票：{votes}  | {record['window_ms']:g} ms/图  | FFT：{record['nfft']}",
                '模型置信度为最终类别在所有图上的平均得分；多数票占比为该类别的票数/图像数。']
            rate = record['sample_rate_hz']
            for window in windows:
                start = window['start_sample']/rate*1000
                end = (window['start_sample']+window['sample_count'])/rate*1000
                text.append(f"图 {window['image_index']} | {start:.6f}–{end:.6f} ms | "
                    f"{window['class_name']} {window['confidence']:.1%} | "
                    f"{window['probabilities_json']}")
            self.details.setPlainText('\n'.join(text))
        except (sqlite3.Error, ValueError) as exc:
            self.details.setPlainText(f'读取投票明细失败：{exc}')

    def replay_spectrum(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.records):
            self.status.setText('请先选择一条识别记录，再进行频谱回放。')
            return
        RecordedSpectrumDialog(self.records[row], self.path.parent, self).exec()
