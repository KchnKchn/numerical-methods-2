from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
import numpy as np
import impl

class CoefTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __create_cell(self, text: str):
        cell = QtWidgets.QTableWidgetItem(text)
        cell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return cell

    def update_table(self, x: np.array, a: np.array, b: np.array, c: np.array, d: np.array):
        self.clear()
        n = x.shape[0]
        self.setColumnCount(6)
        self.setRowCount(n-1)
        self.setHorizontalHeaderLabels(["x[i-1]", "x[i]", "a[i]", "b[i]", "c[i]", "d[i]"])
        self.setVerticalHeaderLabels([str(i+1) for i in range(0, n)])
        for i in range(1, n):
            self.setItem(i-1, 0, self.__create_cell("{0:5.3f}".format(x[i-1])))
            self.setItem(i-1, 1, self.__create_cell("{0:5.3f}".format(x[i])))
            self.setItem(i-1, 2, self.__create_cell("{0:5.3f}".format(a[i])))
            self.setItem(i-1, 3, self.__create_cell("{0:5.3f}".format(b[i])))
            self.setItem(i-1, 4, self.__create_cell("{0:5.3f}".format(c[i])))
            self.setItem(i-1, 5, self.__create_cell("{0:5.3f}".format(d[i])))
        self.resizeColumnsToContents()

class AccuracyTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __create_cell(self, text: str):
        cell = QtWidgets.QTableWidgetItem(text)
        cell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return cell

    def update_table(self, x: np.array, f: np.array, s: np.array, f1: np.array, s1: np.array, f2: np.array, s2: np.array):
        self.clear()
        n = x.shape[0]
        self.setColumnCount(10)
        self.setRowCount(n)
        self.setHorizontalHeaderLabels(["x[i]", "f[i]", "s[i]", "f[i]-s[i]", "f1[i]", "s1[i]", "f1[i]-s1[i]", "f2[i]", "s2[i]", "f2[i]-s2[i]"])
        self.setVerticalHeaderLabels([str(i+1) for i in range(0, n)])

        fs = f - s
        fs1 = f1 - s1
        fs2 = f2 - s2

        for i in range(n):
            self.setItem(i, 0, self.__create_cell("{0:5.3f}".format(x[i])))
            self.setItem(i, 1, self.__create_cell("{0:5.3f}".format(f[i])))
            self.setItem(i, 2, self.__create_cell("{0:5.3f}".format(s[i])))
            self.setItem(i, 3, self.__create_cell("{0:5.3f}".format(fs[i])))
            self.setItem(i, 4, self.__create_cell("{0:5.3f}".format(f1[i])))
            self.setItem(i, 5, self.__create_cell("{0:5.3f}".format(s1[i])))
            self.setItem(i, 6, self.__create_cell("{0:5.3f}".format(fs1[i])))
            self.setItem(i, 7, self.__create_cell("{0:5.3f}".format(f2[i])))
            self.setItem(i, 8, self.__create_cell("{0:5.3f}".format(s2[i])))
            self.setItem(i, 9, self.__create_cell("{0:5.3f}".format(fs2[i])))
        self.resizeColumnsToContents()

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.__tasks = {
                "Тестовая задача": impl.test_task(), 
                "Основная задача": impl.b_task(), 
                "Осциллирующая задача": impl.c_task()
            }
        self.__curr_task = self.__tasks["Тестовая задача"]

        self.__grid = QtWidgets.QGridLayout(self)
        self.__init_methods_list()
        self.__init_n()
        self.__init_n2()
        self.__start_button()
        self.__eps()
        self.__eps1()
        self.__eps2()
        self.__coef_table()
        self.__accuracy_table()

    def __init_methods_list(self):
        methods_list = QtWidgets.QComboBox()
        methods_list.addItems(self.__tasks.keys())
        methods_list.activated[str].connect(self.__connect_change_method)
        self.__grid.addWidget(methods_list, 0, 0, 1, 4)

    def __connect_change_method(self, method_name: str):
        self.__curr_task = self.__tasks[method_name]

    def __init_n(self):
        label = QtWidgets.QLabel()
        label.setText("Размерность сетки")
        self.__n_input = QtWidgets.QLineEdit()
        self.__n_input.setText("4")
        self.__grid.addWidget(label, 1, 0)
        self.__grid.addWidget(self.__n_input, 1, 1)

    def __init_n2(self):
        label = QtWidgets.QLabel()
        label.setText("Размерность дополнительной сетки")
        self.__n2_input = QtWidgets.QLineEdit()
        self.__n2_input.setText("8")
        self.__grid.addWidget(label, 1, 2)
        self.__grid.addWidget(self.__n2_input, 1, 3)

    def __start_button(self):
        self.__button = QtWidgets.QPushButton()
        self.__button.setText("Решить")
        self.__button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(self.__button, 2, 0, 1, 4)

    def __eps(self):
        eps_label = QtWidgets.QLabel()
        eps_label.setText("Погрешность сплайна на контрольной сетке")
        self.__eps_input = QtWidgets.QLineEdit()
        self.__eps_input.setReadOnly(True)
        self.__eps_input.setText("")
        self.__grid.addWidget(eps_label, 3, 0)
        self.__grid.addWidget(self.__eps_input, 3, 1)

        x_label = QtWidgets.QLabel()
        x_label.setText("при x равном")
        self.__x_input = QtWidgets.QLineEdit()
        self.__x_input.setReadOnly(True)
        self.__x_input.setText("")
        self.__grid.addWidget(x_label, 3, 2)
        self.__grid.addWidget(self.__x_input, 3, 3)

    def __eps1(self):
        eps_label = QtWidgets.QLabel()
        eps_label.setText("Погрешность первой производной сплайна на контрольной сетке")
        self.__eps1_input = QtWidgets.QLineEdit()
        self.__eps1_input.setReadOnly(True)
        self.__eps1_input.setText("")
        self.__grid.addWidget(eps_label, 4, 0)
        self.__grid.addWidget(self.__eps1_input, 4, 1)

        x_label = QtWidgets.QLabel()
        x_label.setText("при x равном")
        self.__x1_input = QtWidgets.QLineEdit()
        self.__x1_input.setReadOnly(True)
        self.__x1_input.setText("")
        self.__grid.addWidget(x_label, 4, 2)
        self.__grid.addWidget(self.__x1_input, 4, 3)

    def __eps2(self):
        eps_label = QtWidgets.QLabel()
        eps_label.setText("Погрешность второй производной сплайна на контрольной сетке")
        self.__eps2_input = QtWidgets.QLineEdit()
        self.__eps2_input.setReadOnly(True)
        self.__eps2_input.setText("")
        self.__grid.addWidget(eps_label, 5, 0)
        self.__grid.addWidget(self.__eps2_input, 5, 1)

        x_label = QtWidgets.QLabel()
        x_label.setText("при x равном")
        self.__x2_input = QtWidgets.QLineEdit()
        self.__x2_input.setReadOnly(True)
        self.__x2_input.setText("")
        self.__grid.addWidget(x_label, 5, 2)
        self.__grid.addWidget(self.__x2_input, 5, 3)

    def __coef_table(self):
        self.__coef_table = CoefTable()
        self.__grid.addWidget(self.__coef_table, 6, 0, 3, 2)

    def __accuracy_table(self):
        self.__accuracy_table = AccuracyTable()
        self.__grid.addWidget(self.__accuracy_table, 6, 2, 3, 2)

    def __button_clicked(self):
        plt.close()
        n = int(self.__n_input.text())
        n2 = int(self.__n2_input.text())

        xi, a, b, c, d = self.__curr_task.get_coef(n)
        xi2, f, f1, f2 = self.__curr_task.get_F(n2)
        s, s1, s2 = self.__curr_task.get_spline(n, n2)

        self.__coef_table.update_table(xi, a, b, c, d)
        self.__accuracy_table.update_table(xi2, f, s, f1, s1, f2, s2)

        fs = np.abs(f - s)
        fs1 = np.abs(f1 - s1)
        fs2 = np.abs(f2 - s2)

        max, max1, max2 = np.argmax(fs), np.argmax(fs1), np.argmax(fs2)

        self.__eps_input.setText(str(fs[max]))
        self.__x_input.setText(str(max))
        self.__eps1_input.setText(str(fs1[max1]))
        self.__x1_input.setText(str(max1))
        self.__eps2_input.setText(str(fs2[max2]))
        self.__x2_input.setText(str(max2))

        fig, axs = plt.subplots(3, 1, num="Графики")

        axs[0].set_title("График функции")
        axs[0].grid(True)
        axs[0].plot(xi2, f, label="График функции f(x)", color="b")
        axs[0].plot(xi2, s, label="Кубический сплайн s(x)", color="g")
        axs[0].plot(xi2, fs, label="Ошибка", color="r")
        axs[0].legend(title="Легенда")

        axs[1].set_title("Первая производная")
        axs[1].grid(True)
        axs[1].plot(xi2, f1, label="График первой производной функции f(x)", color="b")
        axs[1].plot(xi2, s1, label="Первая производная кубического сплайна s(x)", color="g")
        axs[1].plot(xi2, fs1, label="Ошибка", color="r")
        axs[1].legend(title="Легенда")

        axs[2].set_title("Вторая производная")
        axs[2].grid(True)
        axs[2].plot(xi2, f2, label="График первой производной функции f(x)", color="b")
        axs[2].plot(xi2, s2, label="Первая производная кубического сплайна s(x)", color="g")
        axs[2].plot(xi2, fs2, label="Ошибка", color="r")
        axs[2].legend(title="Легенда")

        plt.show()

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 720)
        self.setWindowTitle("Лабораторная работа по численным методам")
        self.__central = MainWidget(self)
        self.setCentralWidget(self.__central)
