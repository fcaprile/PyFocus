# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mask_selection.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import config
import numpy as np

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(315, 285)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 150, 271, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 130, 161, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(10, 190, 181, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 181, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(10, 230, 101, 16))
        self.label_6.setObjectName("label_6")
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(10, 100, 291, 21))
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(110, 60, 91, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(80, 10, 151, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 170, 301, 16))
        self.label.setObjectName("label")

        self.pushButton.clicked.connect(self.change_mask_function)
        self.pushButton_2.clicked.connect(self.load_mask_function)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def change_mask_function(self):
        config.x=self.textEdit.toPlainText()
        config.y=True #internal parameter to set whether or the mask function is a function or a txt

    def load_mask_function(self):
        config.x=np.loadtxt(QtWidgets.QFileDialog.getOpenFileName(None,'Select mask function File')[0],dtype=complex)
        config.y=False #internal parameter to set whether or the mask function is a function or a txt


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Define mask"))
        self.label_2.setText(_translate("Dialog", " rho (radial coordinate, from 0 to phase mask  radius)"))
        self.label_3.setText(_translate("Dialog", "Available parameters:"))
        self.label_4.setText(_translate("Dialog", "w0 (gaussian beam radius)"))
        self.label_5.setText(_translate("Dialog", "f (focal distance of objective lens)"))
        self.label_6.setText(_translate("Dialog", "k (wavenumber)"))
        self.textEdit.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">np.exp(-(rho/w0)**2+1j*phi)</p></body></html>"))
        self.pushButton.setText(_translate("Dialog", "Define mask"))
        self.pushButton_2.setText(_translate("Dialog", "Load mask from .txt file"))
        self.label.setText(_translate("Dialog", "phi (azimutal coordinate, from 0 to 2pi)"))
        try:
            self.textEdit.setText(config.x)#if a phase mask has already been given
        except:
            pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

