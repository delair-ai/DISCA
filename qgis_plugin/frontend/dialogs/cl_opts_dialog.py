import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/cl_options.ui'))


class CLDialog(QtWidgets.QDialog, FORM_CLASS):
    """Window to select layers based on the current active layers in Qgis (get_layers)"""
    def __init__(self):
        super(CLDialog, self).__init__()
        self.setupUi(self)
