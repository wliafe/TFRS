import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    x = []
    y = []

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Matplotlib in PyQt5')
        self.canvas = FigureCanvasQTAgg(plt.figure())
        self.setCentralWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title('Sample Plot')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.line, = self.ax.plot([], [])

    def plot(self, x, y):
        self.x.append([x])
        self.y.append([y])
        self.line.set_data(self.x, self.y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    mainWindow.plot(3, 5)
    mainWindow.plot(4, 6)
    mainWindow.plot(5, 7)
    sys.exit(app.exec_())
