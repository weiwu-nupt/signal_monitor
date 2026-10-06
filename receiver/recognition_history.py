"""Read-only view of persisted packet classifications and individual votes."""
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from PyQt6 import QtWidgets


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
        self.table = QtWidgets.QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(['记录', '时间', '信号编号', '中心频率 (MHz)',
            '带宽 (MHz)', 'IQ采样率 (MSps)', '类别', '模型置信度', '多数票占比', '图像数', '文件/来源'])
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
        refresh.clicked.connect(self.refresh)
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
            records = self.query('''SELECT p.*, s.source FROM packets p
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
                    ('（平票）' if record['tied'] else ''), str(record['image_count']),
                    Path(record['source']).name]
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(value if column != 10 else record['source'])
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
