import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

from PyQt6 import QtWidgets
from receiver.integrate import _StandaloneWindow
from receiver.controller import FFMConfig, RxMode


class FrequencyBoundsTests(unittest.TestCase):
    def test_task_band_limits_both_linked_plots_after_reconfiguration(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = _StandaloneWindow()
        window.show()
        page = window.mgr.pages[RxMode.FFM]
        try:
            for center, span in ((700, 320000), (750, 20000)):
                page._apply_display_axes(FFMConfig(center_mhz=center, span_khz=span))
                low, high = center*1e6-span*500, center*1e6+span*500
                for plot in (page.spectrum, page.waterfall):
                    view = plot.getViewBox()
                    view.setXRange(low-1e9, high+1e9, padding=0)
                    view.scaleBy((10, 1))
                    for distance in (-1e9, 1e9):
                        view.translateBy(x=distance)
                        app.processEvents()
                        for linked in (page.spectrum, page.waterfall):
                            start, stop = linked.viewRange()[0]
                            self.assertGreaterEqual(start, low-.01)
                            self.assertLessEqual(stop, high+.01)
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
