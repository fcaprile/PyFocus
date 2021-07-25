# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mask_selection2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import config
import numpy as np

class Ui_MaskWindow(object):
    def setupUi(self, MaskWindow):
        MaskWindow.setObjectName("MaskWindow")
        MaskWindow.resize(331, 277)
        self.centralwidget = QtWidgets.QWidget(MaskWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 130, 161, 16))
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 170, 301, 16))
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 190, 181, 16))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 230, 141, 16))
        self.label_6.setObjectName("label_6")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(120, 60, 91, 31))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 150, 271, 16))
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 210, 241, 16))
        self.label_5.setObjectName("label_5")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(20, 100, 291, 21))
        self.textEdit.setObjectName("textEdit")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 10, 151, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        MaskWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MaskWindow)
        self.statusbar.setObjectName("statusbar")
        MaskWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.change_mask_function)
        self.pushButton_2.clicked.connect(self.load_mask_function)

        self.retranslateUi(MaskWindow)
        QtCore.QMetaObject.connectSlotsByName(MaskWindow)

    def change_mask_function(self):
        config.x=self.textEdit.toPlainText()
        config.y=True #internal parameter to set whether or the mask function is a function or a txt
        print('Function defined')

    def load_mask_function(self):
        config.x=np.loadtxt(QtWidgets.QFileDialog.getOpenFileName(None,'Select mask function File')[0],dtype=complex)
        config.y=False #internal parameter to set whether or the mask function is a function or a txt
        print('File loaded')

    def retranslateUi(self, MaskWindow):
        _translate = QtCore.QCoreApplication.translate
        MaskWindow.setWindowTitle(_translate("MaskWindow", "Define phase mask"))
        self.label_3.setText(_translate("MaskWindow", "Available parameters:"))
        self.label.setText(_translate("MaskWindow", "phi (azimutal coordinate, from 0 to 2pi)"))
        self.label_4.setText(_translate("MaskWindow", "w0 (gaussian beam radius, mm)"))
        self.label_6.setText(_translate("MaskWindow", "k (wavenumber, 1/mm)"))
        self.pushButton.setText(_translate("MaskWindow", "Define mask"))
        self.label_2.setText(_translate("MaskWindow", " rho (radial coordinate, from 0 to phase mask radius)"))
        self.label_5.setText(_translate("MaskWindow", "f (focal distance of objective lens, mm)"))
        self.textEdit.setHtml(_translate("MaskWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">np.exp(-(rho/w0)**2+1j*phi)</p></body></html>"))
        self.pushButton_2.setText(_translate("MaskWindow", "Load mask from .txt file"))
        try:
            self.textEdit.setText(config.x)#if a phase mask has already been given
        except:
            pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MaskWindow = QtWidgets.QMainWindow()
    ui = Ui_MaskWindow()
    ui.setupUi(MaskWindow)
    MaskWindow.show()
    sys.exit(app.exec_())

