import sys
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QPushButton, QApplication, \
    QLabel, QVBoxLayout, QHBoxLayout, QSpinBox, QCheckBox, QSizePolicy, QDialog, \
    QMessageBox
from PyQt6 import QtCore, QtGui
 

# resetValue = False

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(250, 200)
        self.setWindowTitle("myGUI")

        # Variables
        self.counter = 0
        self.value = 0
        self.resetValue = False

        # Widgets
        self.label1 = QLabel(f"Value: {self.value}", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.button1 = QPushButton(f"Increment")
        self.button1.clicked.connect(self.handleClickCountUp)

        self.button2 = QPushButton(f"Decrement")
        self.button2.clicked.connect(self.handleClickCountDown)

        self.button_reset = QPushButton(f"Reset")
        self.button_reset.clicked.connect(self.handleClickReset)
        self.button_reset.setEnabled(False)

        self.label2 = QLabel("Increment value:", alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        self.incrementBox = QSpinBox()
        self.incrementBox.setValue(1)
        self.incrementBox.setMaximum(20)
        self.incrementBox.setMinimum(-20)

        self.checkBox_use_pi = QCheckBox("Use pi as increment value")
        self.checkBox_use_pi.stateChanged.connect(self.handleCheckBox)

        self.label3 = QLabel(f"A Button was Clicked {self.counter} times.", alignment=QtCore.Qt.AlignmentFlag.AlignBottom)
        
        # Layouts
        self.mainLayout = QVBoxLayout()
        self.layout1 = QHBoxLayout()
        self.layout2 = QHBoxLayout()
        self.layout3 = QVBoxLayout()

        self.mainLayout.addWidget(self.label1)
        self.mainLayout.addLayout(self.layout1)
        self.mainLayout.addLayout(self.layout2)
        self.mainLayout.addWidget(self.label3)

        self.layout1.addWidget(self.button1)
        self.layout1.addWidget(self.button2)

        self.layout2.addLayout(self.layout3)
        self.layout2.addWidget(self.button_reset)

        self.layout3.addWidget(self.label2)
        self.layout3.addWidget(self.incrementBox)
        self.layout3.addWidget(self.checkBox_use_pi)

        # set an empty widget with the mainlayout as centralwidget, otherwise the layouts won't be displayed
        widget = QWidget()
        widget.setLayout(self.mainLayout)
        self.setCentralWidget(widget)


        # Other Windows and its functionalities
        self.resetWindow = resetWindow()
        self.resetWindow.buttonYes.clicked.connect(self.handleClickYes)
        self.resetWindow.buttonNo.clicked.connect(self.handleClickNo)

        self.show()    
 
    def handleClickCountUp(self):
        self.button_reset.setEnabled(True)
        self.value += self.getIncrementValue()
        self.counter += 1
        self.label3.setText(f"A Button was clicked {self.counter} times.")
        self.label1.setText(f"Value: {self.value}")
        self.label1.adjustSize()
        self.label3.adjustSize()
    
    def handleClickCountDown(self):
        self.button_reset.setEnabled(True)
        self.value -= self.getIncrementValue()
        self.counter += 1
        self.label3.setText(f"A Button was clicked {self.counter} times.")
        self.label1.setText(f"Value: {self.value}")
        self.label1.adjustSize()
        self.label3.adjustSize()

    def handleClickReset(self):
        self.resetWindow.exec()
        if self.resetValue:
            self.button_reset.setEnabled(False)
            self.value = 0
            self.counter += 1
            self.label3.setText(f"A Button was clicked {self.counter} times.")
            print(f"The value has been reset.")
            self.label1.setText(f"Value: {self.value}")
            self.label1.adjustSize()
            self.label3.adjustSize()
        
    def handleCheckBox(self):
        if self.checkBox_use_pi.isChecked():
            self.incrementBox.setEnabled(False)
        else:
            self.incrementBox.setEnabled(True)

    def getIncrementValue(self):
        if self.checkBox_use_pi.isChecked():
            return np.pi
        else:
            return self.incrementBox.value()

    def handleClickYes(self):
        self.resetValue = True
        self.resetWindow.close()

    def handleClickNo(self):
        self.resetValue = False
        self.resetWindow.close()

    def exitHandler(self):
        print("exit GUI...")

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        button = QMessageBox.question(
            self,
            "Closeing App",
            "Do you really want to close this app?",
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No,
        )

        if button == QMessageBox.StandardButton.Yes:
            a0.accept()
        else:
            a0.ignore()


class resetWindow(QDialog):
    def __init__(self):
        super().__init__()

        label = QLabel("Do you really want to reset the counter?")

        self.buttonYes = QPushButton("Yes", self)

        self.buttonNo = QPushButton("No", self)

        mainLayout = QVBoxLayout(self)
        buttonLayout = QHBoxLayout()

        mainLayout.addWidget(label)
        mainLayout.addLayout(buttonLayout)
        buttonLayout.addWidget(self.buttonYes)
        buttonLayout.addWidget(self.buttonNo)

 
app = QApplication(sys.argv)
window = Window()
app.aboutToQuit.connect(window.exitHandler)
sys.exit(app.exec())