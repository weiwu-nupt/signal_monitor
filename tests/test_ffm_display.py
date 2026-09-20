import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from receiver.controller import FFMConfig, RxMode
from receiver.ffm_spectrum import TraceProcessor
from receiver.integrate import _StandaloneWindow
from receiver.rx_protocol import SpectrumData


class TraceTests(unittest.TestCase):
    def test_maximum_and_minimum(self):
        for mode, expected in (("最大保持", [30, 40]), ("最小保持", [10, 20])):
            processor = TraceProcessor()
            processor.configure(mode)
            processor.apply([10, 40])
            np.testing.assert_equal(processor.apply([30, 20]), expected)

    def test_average_is_finite_window_linear_power(self):
        processor = TraceProcessor()
        processor.configure("平均", 2)
        processor.apply([0, np.nan])
        result = processor.apply([10, 20])
        np.testing.assert_allclose(result, [10 * np.log10(5.5), 20], atol=1e-5)
        np.testing.assert_allclose(processor.apply([10, 30]), [10, 10 * np.log10(550)], atol=1e-5)
        processor.configure("平均", 3)
        np.testing.assert_allclose(processor.apply([40, 50]), [40, 50])


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = _StandaloneWindow()
        self.page = self.window.mgr.pages[RxMode.FFM]
        self.window.ensurePolished()
        self.window.grab()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_default_config_and_units_preserved(self):
        self.assertEqual(asdict(self.page.get_config()), asdict(FFMConfig()))
        labels = [label.text() for label in self.window.findChildren(QtWidgets.QLabel)]
        self.assertFalse(any("固定频点" in text or "FFM" in text or "dBm" in text for text in labels))
        self.assertFalse(any("清除保持" in b.text() for b in self.window.findChildren(QtWidgets.QAbstractButton)))
        self.assertEqual([self.page.trace_mode.itemText(i) for i in range(4)], list(TraceProcessor.MODES))
        self.assertEqual(self.page.spectrum.getAxis("left").labelUnits, "dBµV")

    def test_plain_editors_and_frequency_units(self):
        self.assertEqual(self.page.panel.findChildren(QtWidgets.QAbstractSpinBox), [])
        self.page.panel.center.editor.setText("100.123456")
        self.page.panel.center.unit.setCurrentText("MHz")
        self.assertAlmostEqual(self.page.get_config().center_mhz, 100.123456)
        self.page.panel.center.editor.clear()
        with self.assertRaises(ValueError):
            self.page.get_config()
        self.page.panel.center.setText(9e9)
        with self.assertRaises(ValueError):
            self.page.get_config()

    def test_gain_controls_and_rf_mode(self):
        panel = self.page.panel
        panel.attauto.setCurrentIndex(panel.attauto.findData(True))
        self.assertFalse(panel.rfatt.isEnabled())
        panel.rfmode.setCurrentIndex(panel.rfmode.findData("LOWNoise"))
        commands = []
        with patch.object(self.page.controller.tcp, "send_many", commands.extend):
            self.page.controller._apply_ffm(panel.get_config())
        self.assertTrue(any("LOWNoise" in command for command in commands))

    def test_document_parameters_and_measurement_commands(self):
        panel = self.page.panel
        labels = [label.text() for label in panel.findChildren(QtWidgets.QLabel)]
        self.assertIn("电平跨度（dB/格）", labels)
        self.assertIn("XdB 带宽（dB）", labels)
        self.assertFalse(any("间隔数量" in text or "标称间隔" in text for text in labels))
        self.assertEqual(panel.display_scale()[2], 10)
        self.assertEqual(panel.span.value_hz(), 409_600_000)
        self.assertGreaterEqual(panel.bw.findData(120.0), 0)
        self.assertEqual(panel.bw.findData(100.0), -1)
        panel.xdb_band.setText("40")
        panel.beta_band.setText("2.5")
        panel.meas_time.setText("50")
        commands = []
        with patch.object(self.page.controller.tcp, "send_many", commands.extend):
            self.page.controller._apply_ffm(panel.get_config())
        self.assertIn("MEASure:BANDwidth:XDB 40.0", commands)
        self.assertIn("MEASure:BANDwidth:BETA 2.5", commands)
        self.assertIn("MEASure:TIME:AUTomation OFF", commands)
        self.assertIn("MEASure:TIME 50.0 ms", commands)
        self.assertFalse(any("STEP" in c for c in commands))

    def test_document_bounds_and_bandwidth_relationship(self):
        panel = self.page.panel
        for editor, bad in ((panel.xdb_band, "101"), (panel.beta_band, "-1"),
                            (panel.meas_time, "10001")):
            original = editor.text()
            editor.setText(bad)
            with self.assertRaises(ValueError):
                panel.get_config()
            editor.setText(original)
        panel.center.setText(-1)
        with self.assertRaises(ValueError):
            panel.get_config()
        panel.center.setText(2_000_000_000)
        panel.span.setText(200_000)
        with self.assertRaisesRegex(ValueError, "解调带宽不能大于频谱带宽"):
            panel.get_config()
        panel.bw.setCurrentIndex(panel.bw.findData(120.0))
        self.assertEqual(panel.get_config().bandwidth_khz, 120)

    def test_frequency_step_is_local_tuning(self):
        panel = self.page.panel
        panel.step.setCurrentIndex(panel.step.findData(25.0))
        event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Up,
                               QtCore.Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(panel.center.editor, event)
        self.assertEqual(panel.center.value_hz(), 2_000_025_000)

    def test_manual_span_and_requested_limits(self):
        panel = self.page.panel
        self.assertEqual([panel.step.itemData(i) for i in range(panel.step.count())],
                         [12.5, 25, 50, 100, 200])
        self.assertEqual(panel.get_config().step_khz, 12.5)
        self.assertEqual(panel.span.editor.text(), "409.6")
        self.assertEqual(panel.span.unit.currentText(), "MHz")
        for hz in (0, 8_000_000_000):
            panel.center.setText(hz)
            panel.span.setText(hz)
            config = panel.get_config()
            self.page._apply_display_axes(config)
            self.assertEqual(config.center_mhz, hz / 1e6)
            self.assertEqual(config.span_khz, hz / 1e3)
        for hz in (-1, 8_000_000_001):
            panel.span.setText(hz)
            with self.assertRaises(ValueError):
                panel.get_config()
        panel.span.setText(409_600_000)
        commands = []
        with patch.object(self.page.controller.tcp, "send_many", commands.extend):
            self.page.controller._apply_ffm(panel.get_config())
        self.assertIn("FREQuency:SPAN 409600.0 kHz", commands)

    def test_user_parameter_list_and_zero_defaults(self):
        panel = self.page.panel
        labels = [label.text() for label in panel.findChildren(QtWidgets.QLabel)]
        for removed in ("参考时钟", "电平触发", "触发电平", "静噪门限", "射频模式 2",
                        "高性能模式", "回放速度", "红色占有率", "频率偏移", "射频输出模式"):
            self.assertFalse(any(removed in label for label in labels), removed)
        self.assertEqual(panel.get_config().beta_band_pct, 0)
        self.assertTrue(panel.parameter_tabs.widget(2).isAncestorOf(panel.bw))
        commands = []
        with patch.object(self.page.controller.tcp, "send_many", commands.extend):
            self.page.controller._apply_ffm(panel.get_config())
        self.assertIn("MEASure:BANDwidth:BETA 0.0", commands)
        self.assertFalse(any("THReshold" in command or "OFFSet" in command for command in commands))

    def test_hold_includes_frames_between_repaints_and_reset(self):
        self.page.controller._mode = RxMode.FFM
        self.page.trace_mode.setCurrentText("最大保持")
        self.page._last_ffm_update_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        for values in ([10, 20, 30], [50, 60, 70], [5, 6, 7]):
            self.page._on_spectrum(SpectrumData(2_000_000_000, 409_600_000, 0, 3, np.array(values)))
        np.testing.assert_allclose(self.page.spectrum.display_values, [50, 60, 70])
        with patch.object(self.page.controller, "apply_mode"):
            self.assertTrue(self.page._on_apply())
        self.assertEqual(self.page.spectrum.values.size, 0)
        self.assertTrue(np.all(self.page.waterfall._img == -80))
        self.assertIsNotNone(self.page.waterfall._item.scene())

    def test_frequency_change_resets_hold(self):
        plot = self.page.spectrum
        plot.set_trace_mode("最大保持")
        plot.update_spectrum([1, 2, 3], [100, 100, 100])
        plot.update_spectrum([4, 5, 6], [10, 10, 10])
        np.testing.assert_equal(plot.display_values, [10, 10, 10])

    def test_horizontal_zoom_level_controls_and_auto_adjust(self):
        plot = self.page.spectrum
        self.assertEqual(list(plot.getViewBox().state["mouseEnabled"]), [True, False])
        x_before, y_before = plot.viewRange()
        position = plot.mapFromScene(plot.getViewBox().sceneBoundingRect().center())
        wheel = QtGui.QWheelEvent(QtCore.QPointF(position), QtCore.QPointF(position),
                                 QtCore.QPoint(), QtCore.QPoint(0, 120),
                                 QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                                 QtCore.Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(plot.viewport(), wheel)
        x_after, y_after = plot.viewRange()
        self.assertLess(x_after[1]-x_after[0], x_before[1]-x_before[0])
        np.testing.assert_allclose(y_after, y_before)
        self.app.processEvents()
        np.testing.assert_allclose(self.page.waterfall.viewRange()[0], x_after, atol=1)
        plot.update_spectrum(np.linspace(*x_before, 101), np.r_[np.full(100, 20), 60])
        self.page._auto_adjust()
        np.testing.assert_allclose(plot.viewRange()[0], x_before)
        self.assertGreater(plot.viewRange()[1][1], 60)
        old_top = self.page.panel.ref.value()
        self.page._shift_reference(1)
        self.assertEqual(self.page.panel.ref.value(), old_top + self.page.panel.lvspan.value())
        self.assertEqual(plot.viewRange()[1][1], self.page.panel.ref.value())

    def test_invalid_input_does_not_start_receiver(self):
        self.page.panel.center.editor.clear()
        with patch.object(QtWidgets.QMessageBox, "warning"), patch.object(self.page.controller, "start") as start:
            self.page._on_start()
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
