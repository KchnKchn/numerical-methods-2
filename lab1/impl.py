import abc
import numpy as np

class task(metaclass=abc.ABCMeta):

    a, b, c, d = 0, 1, 0, 1

    def __init__(self, n: int, m: int):
        self._n = n
        self._m = m

    def _f_func(self, x, y):
        return (np.sin(np.pi * x * y)) ** 2

    def _mu1(self, y):
        return np.sin(np.pi * y)

    def _mu2(self, y):
        return np.sin(np.pi * y)

    def _mu3(self, x):
        return x - (x ** 2)

    def _mu4(self, x):
        return x - (x ** 2)

    def get_h_k(self):
        h = (self.b - self.a) / self._n
        k = (self.d - self.c) / self._m
        return h, k

    def init_v(self):
        h = (self.b - self.a) / self._n
        k = (self.d - self.c) / self._m
        v = np.zeros(shape=(self._n+1, self._m+1), dtype=float)
        for i in range(self._n+1):
            v[i][0] = self._mu3(i*h)
            v[i][self._m] = self._mu4(i*h)
        for j in range(self._m+1):
            v[0][j] = self._mu1(j*k)
            v[self._n][j] = self._mu2(j*k)
        for j in range(1, self._m):
            for i in range(1 , self._n):
                v[i][j] = v[0][j] + (v[self._n][j] - v[0][j]) * (i*h - 0) / (self._n*h - 0)
        return v

    def init_f(self):
        h = (self.b - self.a) / self._n
        k = (self.d - self.c) / self._m
        f = np.zeros(shape=(self._n+1, self._m+1), dtype=float)
        for j in range(self._m+1):
            for i in range(self._n+1):
                f[i][j] = self._f_func(i*h, j*k)
        return f

class main_task(task):
    def __init__(self, n: int, m: int):
        super().__init__(n, m)

class test_task(task):
    def __init__(self, n: int, m: int):
        super().__init__(n, m)

    def _f_func(self, x, y):
        return -2*(np.pi**2)*np.exp((np.sin(np.pi*x*y))**2)*((2*(np.sin(np.pi*x*y)**2)+1)*(np.cos(np.pi*x*y)**2)-(np.sin(np.pi*x*y)**2))*((x**2)+(y**2))

    def _mu1(self, y):
        return 1

    def _mu2(self, y):
        return np.exp((np.sin(np.pi * y)) ** 2)

    def _mu3(self, x):
        return 1

    def _mu4(self, x):
        return np.exp((np.sin(np.pi * x)) ** 2)

    def _u_func(self, x, y):
        return np.exp((np.sin(np.pi * x * y)) ** 2)

    def init_u(self):
        h = (self.b - self.a) / self._n
        k = (self.d - self.c) / self._m
        u = np.zeros(shape=(self._n+1, self._m+1), dtype=float)
        for j in range(self._m+1):
            for i in range(self._n+1):
                u[i][j] = self._u_func(i*h, j*k)
        return u

class solver:

    n, m = None, None
    h, k = None, None
    w = None
    eps = None
    n_max = None

    def solve(self, v: np.ndarray, f: np.ndarray):
        h2 = (1 / self.h) ** 2
        k2 = (1 / self.k) ** 2
        a2 = -2 * (h2 + k2)

        finish = False
        iter = 0
        eps = 0

        while(not finish):
            eps = 0
            for j in range(1, self.m):
                for i in range(1, self.n):
                    v_prev = v[i][j]
                    v_curr = self.w * (-f[i][j] - h2 * (v[i+1][j] + v[i-1][j]) - k2 * (v[i][j+1] + v[i][j-1])) / a2
                    v_curr += (1 - self.w) * v[i][j]
                    v[i][j] = v_curr
                    eps = max(eps, abs(v_prev - v_curr))
            iter += 1
            if (eps < self.eps) or (self.n_max <= iter):
                finish = True
        return v, eps, iter

    def calculate_rn(self, v: np.ndarray, f: np.ndarray):
        h2 = (1 / self.h) ** 2
        k2 = (1 / self.k) ** 2
        a2 = -2 * (h2 + k2)

        rn = 0
        for j in range(1, self.m):
            for i in range(1, self.n):
                av = a2 * v[i][j] + h2 * (v[i+1][j] + v[i-1][j]) + k2 * (v[i][j+1] + v[i][j-1])
                rn = max(rn, abs(av + f[i][j]))
        return rn
