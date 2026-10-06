import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from queue import Queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyQt6 import QtWidgets
from receiver.file_replay import FileReplayPage


class ReplayModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.page = FileReplayPage()
        self.alive = True
        self.page.worker = SimpleNamespace(cancel=threading.Event(), paused=threading.Event(),
            mailbox=Queue(), thread=SimpleNamespace(is_alive=lambda: self.alive,
                                                   join=lambda **kwargs: None))
        self.page._set_running(True)

    def tearDown(self):
        self.page.shutdown()
        self.page.close()

    def select_model(self):
        with patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                          return_value=('C:/models/selected.pth', '')):
            self.page.choose_model()

    def test_selection_available_while_playing_and_restart_waits_for_old_worker(self):
        self.assertTrue(self.page.model_button.isEnabled())
        self.select_model()
        self.assertEqual(self.page.checkpoint, 'C:/models/selected.pth')
        self.assertTrue(self.page.worker.cancel.is_set())
        self.assertTrue(self.page._restart_after_model_change)
        with patch.object(self.page, 'start') as start:
            self.page.poll()
            start.assert_not_called()
            self.alive = False
            self.page.poll()
            start.assert_called_once()
        self.assertFalse(self.page._restart_after_model_change)
        self.assertTrue(self.page.model_button.isEnabled())

    def test_cancelled_dialog_keeps_paused_playback(self):
        self.page.worker.paused.set()
        self.assertTrue(self.page.model_button.isEnabled())
        with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=('', '')):
            self.page.choose_model()
        self.assertTrue(self.page.worker.paused.is_set())
        self.assertFalse(self.page.worker.cancel.is_set())
        self.assertFalse(self.page._restart_after_model_change)

    def test_explicit_stop_cancels_model_restart(self):
        self.select_model()
        self.assertTrue(self.page.stop_button.isEnabled())
        self.page.stop()
        self.alive = False
        with patch.object(self.page, 'start') as start:
            self.page.poll()
            start.assert_not_called()
        self.assertTrue(self.page.model_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
