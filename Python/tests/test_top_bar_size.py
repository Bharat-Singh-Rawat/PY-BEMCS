import unittest
from PyQt5.QtWidgets import QApplication, QSizePolicy
from gui.app.main_window import DigitalTwinApp


class TestTopBarSize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_top_bar_size_policy_and_stretch(self):
        win = DigitalTwinApp()
        top_bar = win.findChild(object, "topBar")
        self.assertIsNotNone(top_bar, "topBar should exist in DigitalTwinApp")
        
        # Verify vertical size policy is Fixed
        self.assertEqual(top_bar.sizePolicy().verticalPolicy(), QSizePolicy.Fixed)
        
        # Verify initial vs resized height does not expand
        win.show()
        self.app.processEvents()
        initial_h = top_bar.height()
        
        win.resize(1920, 1080)
        self.app.processEvents()
        resized_h = top_bar.height()
        
        self.assertEqual(resized_h, initial_h)
        win.close()


if __name__ == "__main__":
    unittest.main()
