from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import impl

class TestTaskParametersGroup(QtWidgets.QGroupBox):

    n = None
    m = None
    w = None
    eps = None
    n_max = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Параметры для численного решения")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__n_parameter()
        self.__m_parameter()
        self.__w_parameter()
        self.__eps_parameter()
        self.__n_max_parameter()
        self.__apply_button()
        self.__parse_values()

    def __n_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Размерность сетки по x")
        self.__n_input = QtWidgets.QLineEdit()
        self.__n_input.setText("10")
        self.__grid.addWidget(label, 0, 0)
        self.__grid.addWidget(self.__n_input, 0, 1)

    def __m_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Размерность сетки по y")
        self.__m_input = QtWidgets.QLineEdit()
        self.__m_input.setText("10")
        self.__grid.addWidget(label, 1, 0)
        self.__grid.addWidget(self.__m_input, 1, 1)

    def __w_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Параметр метода w")
        self.__w_input = QtWidgets.QLineEdit()
        self.__w_input.setText("1")
        self.__grid.addWidget(label, 2, 0)
        self.__grid.addWidget(self.__w_input, 2, 1)

    def __eps_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Минимальная точность")
        self.__eps_input = QtWidgets.QLineEdit()
        self.__eps_input.setText("0.0000005")
        self.__grid.addWidget(label, 3, 0)
        self.__grid.addWidget(self.__eps_input, 3, 1)

    def __n_max_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Максимальное количество итераций")
        self.__n_max_input = QtWidgets.QLineEdit()
        self.__n_max_input.setText("500")
        self.__grid.addWidget(label, 4, 0)
        self.__grid.addWidget(self.__n_max_input, 4, 1)

    def __apply_button(self):
        button = QtWidgets.QPushButton()
        button.setText("Ввести параметры")
        button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(button, 5, 0, 1, 2)

    def __parse_values(self):
        self.n = int(self.__n_input.text())
        self.m = int(self.__m_input.text())
        self.w = self.__w_input.text()
        if self.w:
            self.w = float(self.w)
        else:
            l = 2 * ((np.arcsin(np.pi / (2 * self.n))) ** 2)
            self.w = 2 / (1 + (l * (2 - l)) ** 0.5)
        self.eps = float(self.__eps_input.text())
        self.n_max = int(self.__n_max_input.text())
        
    def __check_correct(self):
        if self.n <= 0:
            raise ValueError("Параметр n должен быть больше 0")
        if self.m <= 0:
            raise ValueError("Параметр m должен быть больше 0")
        if not (0 < self.w < 2):
            raise ValueError("Параметр w должен быть больше 0 и меньше 2")
        if self.eps <= 0:
            raise ValueError("Параметр eps должен быть больше 0")
        if self.n_max <= 0:
            raise ValueError("Параметр N_max должен быть больше 0")

    def __button_clicked(self):
        try:
            self.__parse_values()
            self.__check_correct()
        except Exception as error:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage(str(Exception))

class MainTaskParametersGroup(QtWidgets.QGroupBox):

    n = None
    m = None
    w = None
    w2 = None
    eps = None
    eps2 = None
    n_max = None
    n_max2 = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Параметры для численного решения")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__n_parameter()
        self.__m_parameter()
        self.__w_parameter()
        self.__eps_parameter()
        self.__n_max_parameter()
        self.__w2_parameter()
        self.__eps2_parameter()
        self.__n_max2_parameter()
        self.__apply_button()
        self.__parse_values()

    def __n_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Размерность сетки по x")
        self.__n_input = QtWidgets.QLineEdit()
        self.__n_input.setText("10")
        self.__grid.addWidget(label, 0, 0)
        self.__grid.addWidget(self.__n_input, 0, 1)

    def __m_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Размерность сетки по y")
        self.__m_input = QtWidgets.QLineEdit()
        self.__m_input.setText("10")
        self.__grid.addWidget(label, 1, 0)
        self.__grid.addWidget(self.__m_input, 1, 1)

    def __w_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Параметр метода w")
        self.__w_input = QtWidgets.QLineEdit()
        self.__w_input.setText("1")
        self.__grid.addWidget(label, 2, 0)
        self.__grid.addWidget(self.__w_input, 2, 1)

    def __eps_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Минимальная точность")
        self.__eps_input = QtWidgets.QLineEdit()
        self.__eps_input.setText("0.0000005")
        self.__grid.addWidget(label, 3, 0)
        self.__grid.addWidget(self.__eps_input, 3, 1)

    def __n_max_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Максимальное количество итераций")
        self.__n_max_input = QtWidgets.QLineEdit()
        self.__n_max_input.setText("500")
        self.__grid.addWidget(label, 4, 0)
        self.__grid.addWidget(self.__n_max_input, 4, 1)

    def __w2_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Параметр метода w (для сгущенной сетки)")
        self.__w2_input = QtWidgets.QLineEdit()
        self.__w2_input.setText("1")
        self.__grid.addWidget(label, 5, 0)
        self.__grid.addWidget(self.__w2_input, 5, 1)

    def __eps2_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Минимальная точность (для сгущенной сетки)")
        self.__eps2_input = QtWidgets.QLineEdit()
        self.__eps2_input.setText("0.0000005")
        self.__grid.addWidget(label, 6, 0)
        self.__grid.addWidget(self.__eps2_input, 6, 1)

    def __n_max2_parameter(self):
        label = QtWidgets.QLabel()
        label.setText("Максимальное количество итераций (для сгущенной сетки)")
        self.__n_max2_input = QtWidgets.QLineEdit()
        self.__n_max2_input.setText("500")
        self.__grid.addWidget(label, 7, 0)
        self.__grid.addWidget(self.__n_max2_input, 7, 1)

    def __apply_button(self):
        button = QtWidgets.QPushButton()
        button.setText("Ввести параметры")
        button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(button, 8, 0, 1, 2)

    def __parse_values(self):
        self.n = int(self.__n_input.text())
        self.m = int(self.__m_input.text())
        self.w = self.__w_input.text()
        self.w2 = self.__w2_input.text()
        if self.w:
            self.w = float(self.w)
        else:
            l = 2 * ((np.arcsin(np.pi / (2 * self.n))) ** 2)
            self.w = 2 / (1 + (l * (2 - l)) ** 0.5)
        if self.w2:
            self.w2 = float(self.w2)
        else:
            l = 2 * ((np.arcsin(np.pi / (4 * self.n))) ** 2)
            self.w = 2 / (1 + (l * (2 - l)) ** 0.5)
        self.eps = float(self.__eps_input.text())
        self.eps2 = float(self.__eps2_input.text())
        self.n_max = int(self.__n_max_input.text())
        self.n_max2 = int(self.__n_max2_input.text())

    def __check_correct(self):
        if self.n <= 0:
            raise ValueError("Параметр n должен быть больше 0")
        if self.m <= 0:
            raise ValueError("Параметр m должен быть больше 0")
        if not (0 < self.w < 2) or not (0 < self.w2 < 2):
            raise ValueError("Параметр w должен быть больше 0 и меньше 2")
        if self.eps <= 0 or self.eps2 <= 0:
            raise ValueError("Параметр eps должен быть больше 0")
        if self.n_max <= 0 or self.n_max2 <= 0:
            raise ValueError("Параметр N_max должен быть больше 0")

    def __button_clicked(self):
        try:
            self.__parse_values()
            self.__check_correct()
        except Exception as error:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage(str(Exception))

class ResultTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __create_cell(self, text: str):
        cell = QtWidgets.QTableWidgetItem(text)
        cell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return cell

    def update_table(self, result: np.array):
        self.clear()
        n, m = result.shape
        self.setColumnCount(m)
        self.setRowCount(n)
        self.setHorizontalHeaderLabels(["x{}".format(i) for i in range(n)])
        self.setVerticalHeaderLabels(["y{}".format(j) for j in range(m-1, -1, -1)])
        for i in range(n):
            for j in range(m):
                self.setItem(m - j - 1, i, self.__create_cell("{0:5.3f}".format(result[i][j])))
        self.resizeColumnsToContents()

class ResultTablesGroup(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Результирующие таблицы")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__first_table()
        self.__second_table()
        self.__third_table()

    def __first_table(self):
        self.__first_label = QtWidgets.QLabel()
        self.__first_label.setText("Первая таблица")
        self.__first_table = ResultTable()
        self.__grid.addWidget(self.__first_label, 0, 0, 1, 4)
        self.__grid.addWidget(self.__first_table, 1, 0, 4, 4)

    def __second_table(self):
        self.__second_label = QtWidgets.QLabel()
        self.__second_label.setText("Вторая таблица")
        self.__second_table = ResultTable()
        self.__grid.addWidget(self.__second_label, 0, 5, 1, 4)
        self.__grid.addWidget(self.__second_table, 1, 5, 4, 4)

    def __third_table(self):
        self.__third_label = QtWidgets.QLabel()
        self.__third_label.setText("Третья таблица")
        self.__third_table = ResultTable()
        self.__grid.addWidget(self.__third_label, 0, 9, 1, 4)
        self.__grid.addWidget(self.__third_table, 1, 9, 4, 4)

    def set_labels(self, first, second, third):
        self.__first_label.setText(first)
        self.__second_label.setText(second)
        self.__third_label.setText(third)

    def update_tables(self, first, second, third):
        self.__first_table.update_table(first)
        self.__second_table.update_table(second)
        self.__third_table.update_table(third)

class TestInfo(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Информация о тестовой задаче")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__task()
        self.info()

    def __task(self):
        self.__task = QtWidgets.QTextEdit()
        self.__task.setReadOnly(True)
        self.__task.setText(
        """
        ПОСТАНОВКА ЗАДАЧИ
        
        Δu(x,y) = –f*(x, y), где x пренадлежит (a, b), y пренадлежит (c, d).
        u(a,y) = μ*1(y), u(b,y) = μ*2(y), где y пренадлежит [c, d].
        u(x,c) = μ*3(x), u(x,d) = μ*4(x), где x пренадлежит [a, b].

        Функции:
        u(x,y) = exp(sin(pi*x*y)^2)
        f*(x,y) = 

        ГУ:
        μ1*(y) = 1, μ2*(y) = exp(sin(pi*y)^2)
        μ3*(x) = 1, μ4*(x) = exp(sin(pi*x)^2)

        Область:
        a = 0, b = 1, c = 0, d = 1.
        """
        )
        self.__grid.addWidget(self.__task, 0, 0, 1, 1)

    def info(self, n="", m="", w="", min_eps="", n_max="", niter="", eps1="", eps="", x="", y="", rn=""):
        self.__info = QtWidgets.QTextEdit()
        self.__info.setReadOnly(True)
        self.__info.setText(
        """
        СПРАВКА

        Для решения тестовой задачи использованы сетка с числом разбиений по x
        n = «{}» и числом разбиений по y m = «{}», метод верхней релаксации
        с параметром ω = «{}», применены критерии остановки по точности ε = «{}»
        и по числу итераций Nmax =«{}».

        На решение схемы (СЛАУ) затрачено итераций N =«{}» и достигнута точность
        итерационного метода ε = «{}».

        Схема (СЛАУ) решена с невязкой || R(N)|| = «{}» для невязки СЛАУ 
        использована норма бесконечности («max»).

        Задача решена с погрешностью ε =«{}»

        Максимальное отклонение точного и численного решений наблюдается в узле
        x=«{}» y=«{}».

        В качестве начального приближения использована интерполяция вдоль х.
        """.format(n, m, w, min_eps, n_max, niter, eps1, rn, eps, x, y)
        )
        self.__grid.addWidget(self.__info, 1, 0, 1, 1)

class MainInfo(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Информация об основной задаче")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__task()
        self.info()

    def __task(self):
        self.__task = QtWidgets.QTextEdit()
        self.__task.setReadOnly(True)
        self.__task.setText(
        """
        ПОСТАНОВКА ЗАДАЧИ

        Δu(x,y) = –f(x,y), где x пренадлежит (a, b), y пренадлежит (c, d).
        u(a,y) = μ1(y), u(b,y) = μ2(y), где y пренадлежит [c, d].
        u(x,c) = μ3(x), u(x,d) = μ4(x), где x пренадлежит [a, b].

        Функции:
        f(x,y) = sin(pi*x*y)^2

        ГУ:
        μ1(y) = sin(pi*y), μ2(y) = sin(pi*y)
        μ3(x) = x-x^2, μ4(x) = x-x^2

        Область:
        a = 0, b = 1, c = 0, d = 1.
        """
        )
        self.__grid.addWidget(self.__task, 0, 0, 1, 1)

    def info(self, n="", m="", w1="", w2="", min_eps1="", min_eps2="", 
             n_max1="", n_max2="", niter1="", niter2="", eps1="", eps2="", 
             eps="", x="", y="", rn1="", rn2=""):
        self.__info = QtWidgets.QTextEdit()
        self.__info.setReadOnly(True)
        self.__info.setText(
        """
        СПРАВКА

        Для решения основной задачи использована сетка с числом разбиений по x
        n = «{}» и числом разбиений по y m = «{}», метод верхней релаксации 
        с параметром ω = «{}», применены критерии остановки по точности ε = «{}»
        и по числу итераций Nmax =«{}».

        На решение схемы (СЛАУ) затрачено итераций N = «{}» и достигнута точность
        итерационного метода ε = «{}».

        Схема (СЛАУ) решена с невязкой || R(N)|| = «{}», использована 
        норма бесконечности («max»).

        Для контроля точности решения использована сетка с половинным шагом,
        метод верхней релаксации с параметром ω = «{}», применены критерии
        остановки по точности ε = «{}» и по числу итераций Nmax =«{}».

        На решение схемы (СЛАУ) с половинным шагом затрачено итераций 
        N = «{}» и достигнута точность итерационного метода ε = «{}».

        Схема (СЛАУ) на сетке с половинным шагом решена с невязкой
        || R(N2) || = «{}» для невязки СЛАУ использована норма бесконечности («max»).

        Задача решена с точностью ε = «{}».

        Максимальное отклонение численных решений на основной сетке и сетке с
        половинным шагом наблюдается в узле x=«{}»; y=«{}».

        В качестве начального приближения на основной сетке использована
        интерполяция вдоль х, на сетке с половинным шагом использована
        интерполяция вдоль х.
        """.format(n, m, w1, min_eps1, n_max1, niter1, eps1, rn1, w2, min_eps2, 
                 n_max2, niter2, eps2, rn2, eps, x, y)
        )
        self.__grid.addWidget(self.__info, 1, 0, 1, 1)

class TestTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__grid = QtWidgets.QGridLayout(self)
        self.__parameters()
        self.__start_button()
        self.__result_tables()
        self.__info()

    def __parameters(self):
        self.__task_parameters = TestTaskParametersGroup()
        self.__grid.addWidget(self.__task_parameters, 0, 0, 1, 1)

    def __start_button(self):
        self.__start_button = QtWidgets.QPushButton()
        self.__start_button.setText("Решить задачу")
        self.__start_button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(self.__start_button, 1, 0, 1, 2)

    def __result_tables(self):
        self.__result_tables = ResultTablesGroup()
        self.__result_tables.set_labels("Точное решение", "Численное решение", "Разница решений")
        self.__grid.addWidget(self.__result_tables, 2, 0, 3, 2)

    def __info(self):
        self.__info = TestInfo()
        self.__grid.addWidget(self.__info, 0, 1, 1, 1)

    def __button_clicked(self):
        n, m = self.__task_parameters.n, self.__task_parameters.m
        w = self.__task_parameters.w
        eps = self.__task_parameters.eps
        n_max = self.__task_parameters.n_max

        #init
        test_task = impl.test_task(n=n, m=m)
        h, k = test_task.get_h_k()
        v = test_task.init_v()
        f = test_task.init_f()
        u = test_task.init_u()

        #solve
        solver = impl.solver()
        solver.n, solver.m = n, m
        solver.h, solver.k = h, k
        solver.w = w
        solver.eps = eps
        solver.n_max = n_max

        v1, eps1, iter1 = solver.solve(v.copy(), f)
        rn1 = solver.calculate_rn(v1, f)

        uv = np.abs(u-v1)

        maxi, maxj = 0, 0
        for j in range(m+1):
            for i in range(n+1):
                if uv[maxi][maxj] < uv[i][j]:
                    maxi, maxj = i, j
        

        self.__info.info(n=n, m=m, w=w, min_eps=eps, n_max=n_max, 
                         niter=iter1, eps1=eps1, eps=uv[maxi][maxj],
                         x=maxi, y=maxj, rn=rn1)

        self.__result_tables.update_tables(u, v1, uv)

class MainTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__grid = QtWidgets.QGridLayout(self)
        self.__parameters()
        self.__start_button()
        self.__result_tables()
        self.__info()

    def __parameters(self):
        self.__task_parameters = MainTaskParametersGroup()
        self.__grid.addWidget(self.__task_parameters, 0, 0, 1, 1)

    def __start_button(self):
        self.__start_button = QtWidgets.QPushButton()
        self.__start_button.setText("Решить задачу")
        self.__start_button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(self.__start_button, 1, 0, 1, 2)

    def __result_tables(self):
        self.__result_tables = ResultTablesGroup()
        self.__result_tables.set_labels("Численное решение", "Численное решение на сетке с половинным шагом", "Разница решений")
        self.__grid.addWidget(self.__result_tables, 2, 0, 3, 2)

    def __info(self):
        self.__info = MainInfo()
        self.__grid.addWidget(self.__info, 0, 1, 1, 1)

    def __button_clicked(self):
        n, m = self.__task_parameters.n, self.__task_parameters.m
        w = self.__task_parameters.w
        w2 = self.__task_parameters.w2
        min_eps = self.__task_parameters.eps
        min_eps2 = self.__task_parameters.eps2
        n_max = self.__task_parameters.n_max
        n_max2 = self.__task_parameters.n_max2


        #init
        main_task = impl.main_task(n=n, m=m)
        h, k = main_task.get_h_k()
        v = main_task.init_v()
        f = main_task.init_f()

        #solve
        solver = impl.solver()
        solver.n, solver.m = n, m
        solver.h, solver.k = h, k
        solver.w = w
        solver.eps = min_eps
        solver.n_max = n_max

        v1, eps1, iter1 = solver.solve(v.copy(), f)
        rn1 = solver.calculate_rn(v1, f)

        #init
        main_task2 = impl.main_task(2*n, 2*m)
        h, k = main_task2.get_h_k()
        v = main_task2.init_v()
        f = main_task2.init_f()

        #solve
        solver = impl.solver()
        solver.n, solver.m = 2*n, 2*m
        solver.h, solver.k = h, k
        solver.w = w2
        solver.eps = min_eps2
        solver.n_max = n_max2

        v2, eps2, iter2 = solver.solve(v.copy(), f)
        rn2 = solver.calculate_rn(v2, f)

        v3 = np.zeros(shape=(n+1,m+1), dtype=np.float)
        maxi, maxj = 0, 0
        for j in range(m+1):
            for i in range(n+1):
                v3[i][j] = abs(v1[i][j] - v2[2*i][2*j])
                if v3[maxi][maxj] < v3[i][j]:
                    maxi, maxj = i, j

        self.__info.info(n=n, m=m, w1=w, w2=w2, min_eps1=min_eps, min_eps2=min_eps2, 
                         n_max1=n_max, n_max2=n_max2, niter1=iter1, niter2=iter2, 
                         eps1=eps1, eps2=eps2, eps=v3[maxi][maxj], x=maxi, y=maxj,
                         rn1=rn1, rn2=rn2)

        self.__result_tables.update_tables(v1, v2, v3)

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__grid = QtWidgets.QGridLayout(self)
        self.__tabs = QtWidgets.QTabWidget()
        self.__tabs.addTab(TestTab(), "Тестовая задача")
        self.__tabs.addTab(MainTab(), "Основная задача")
        self.__grid.addWidget(self.__tabs)

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 720)
        self.setWindowTitle("Лабораторная работа по численным методам")
        self.__central = MainWidget(self)
        self.setCentralWidget(self.__central)
