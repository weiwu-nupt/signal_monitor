"""Cancelable, asynchronous preview without requiring trained network weights."""
from queue import Queue, Empty
import threading
import time

from PyQt6 import QtCore, QtGui, QtWidgets

from iq_file_source import read_channel_packet, collect_channel_batch
from iq_images import preview_images


class IQImagePreviewDialog(QtWidgets.QDialog):
    def __init__(self, info, start, target, center_hz, window_ms, parent=None, cached=None,
                 max_images=None):
        super().__init__(parent)
        self.setWindowTitle('信号时频图')
        self.resize(1030, 680)
        self.cancel = threading.Event()
        self.mailbox = Queue(maxsize=1)
        self._image_labels = []
        layout = QtWidgets.QVBoxLayout(self)
        description = (
            f'中心频率 {center_hz/1e6:.6f} MHz | 带宽 {target.bandwidth_hz/1e6:.6f} MHz | '
            f'IQ采样率 {target.sample_rate_hz/1e6:.6f} MSps | {window_ms:g} ms/图\n'
            '从文件采集真实4 MiB IQ后预览最多8张图。此预览不调用神经网络，不产生分类或数据库记录。')
        if cached is not None:
            description = (f"神经网络实际输入图像 | 中心频率 {cached['center_hz']/1e6:.6f} MHz | "
                f"IQ采样率 {cached['sample_rate_hz']/1e6:.6f} MSps | 检测帧 {cached['frame']}\n"
                '显示最近该路推理的前8张图及逐图类别；全部图像参与最终多数投票。')
            if 'window_ms' in cached:
                description += f" 实际窗口 {cached['window_ms']:.3f} ms。"
        elif max_images is not None:
            description += f' 自动窗口，整包最多 {max_images} 张，按需延长窗口。'
        self.description = QtWidgets.QLabel(description)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.status = QtWidgets.QLabel('正在采集4 MiB IQ…')
        layout.addWidget(self.status)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.thread = None
        self.finished.connect(lambda _: self.cancel.set())
        self.finished.connect(lambda _: self.timer.stop())
        if cached is not None:
            self.status.setText('以下图像已送入神经网络；缓存属于上面注明的检测帧。')
            self.set_images(cached['images'], cached['sample_rate_hz'])
            self.timer.stop()
        else:
            self.thread = threading.Thread(target=self.run, args=(info, start, target, window_ms, max_images), daemon=True)
            self.thread.start()

    def publish(self, value):
        try:
            self.mailbox.get_nowait()
        except Empty:
            pass
        self.mailbox.put_nowait(value)

    def run(self, info, start, target, window_ms, max_images):
        try:
            last_update = 0
            def progress(done, total):
                nonlocal last_update
                now = time.monotonic()
                if done == total or now-last_update >= .2:
                    self.publish({'status': f'采集4 MiB IQ：{done/total:.1%}；软件下采样按处理速度进行。'})
                    last_update = now
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False
            if cuda_available:
                packets = []
                collect_channel_batch(info, start, [(target, 1)],
                    lambda t, identifier, iq, rate: packets.append((iq.cpu().numpy(), rate)),
                    self.cancel.is_set, lambda identifier, done, total: progress(done, total), device='cuda:0')
                packet = packets[0] if packets else None
            else:
                packet = read_channel_packet(info, start, target, self.cancel.is_set, progress)
            if packet is None:
                return
            self.publish({'status': '4 MiB已收齐，正在生成时频图…'})
            iq, rate = packet
            images, total, nfft = preview_images(iq, rate, window_ms, cancelled=self.cancel.is_set,
                                                max_images=max_images)
            self.publish({'images': images, 'sample_rate_hz': rate,
                'status': f'4 MiB可分为 {total} 张图，当前预览 {len(images)} 张；FFT {nfft} 点。未运行神经网络。'})
        except InterruptedError:
            pass
        except Exception as exc:
            self.publish({'status': f'图像生成失败：{exc}'})

    def poll(self):
        try:
            value = self.mailbox.get_nowait()
        except Empty:
            return
        self.status.setText(value['status'])
        if 'images' in value:
            self.set_images(value['images'], value['sample_rate_hz'])

    def set_images(self, images, sample_rate_hz):
        for position, item in enumerate(images):
            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            rgb = item['rgb']
            qimage = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                QtGui.QImage.Format.Format_RGB888).copy()
            image_label = QtWidgets.QLabel()
            image_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
            image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._image_labels.append(image_label)
            layout.addWidget(image_label)
            start = item['start_sample']/sample_rate_hz*1000
            end = (item['start_sample']+item['sample_count'])/sample_rate_hz*1000
            caption = QtWidgets.QLabel(f"图 {item['image_index']}：{start:.3f}–{end:.3f} ms")
            if 'class_name' in item:
                caption.setText(caption.text()+f"\n{item['class_name']}：{item['confidence']:.1%}")
            caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(caption)
            self.grid.addWidget(container, position//4, position%4)
