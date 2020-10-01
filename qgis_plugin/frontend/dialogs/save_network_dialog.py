import os
from os.path import dirname

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/save_network.ui'))


class SaveNetworkDialog(QtWidgets.QDialog, FORM_CLASS):
    """Window to select layers based on the current active layers in Qgis (get_layers)"""
    def __init__(self, default_filename):
        super(SaveNetworkDialog, self).__init__()
        self.setupUi(self)
        self.default_filename = default_filename
        self.lineEdit.setText(default_filename)
        self.pushButton.clicked.connect(self.return_net)

    def return_net(self):
        filename = QFileDialog.getSaveFileName(directory=dirname(self.default_filename), filter="*.pt")[0]
        if filename:
            self.lineEdit.setText(filename)
