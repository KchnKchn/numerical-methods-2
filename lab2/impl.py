import abc
import numpy as np

class task(metaclass=abc.ABCMeta):

    a, b = None, None
    sa, sb = None, None

    @abc.abstractmethod
    def _F(self, x: float):
        pass

    @abc.abstractmethod
    def _F1(self, x: float):
        pass

    @abc.abstractmethod
    def _F2(self, x: float):
        pass

    def __sweep_method(self, n: int):
        h = (self.b - self.a) / n

        alf = np.zeros(shape=(n+1), dtype=float)
        bet = np.zeros(shape=(n+1), dtype=float)

        result = np.zeros(shape=(n+1), dtype=float)

        for i in range(1, n):
            fi = -6*(self._F(self.a+h*(i+1))-2*self._F(self.a+h*i)+self._F(self.a+h*(i-1))) / h
            A, B, C = h, h, -4*h
            alf[i + 1] = B / (C - A * alf[i])
            bet[i + 1] = (fi + A * bet[i]) / (C - A * alf[i])

        for i in range(n-1, 0, -1):
            result[i] = alf[i+1] * result[i+1] + bet[i+1]
        return result

    def get_coef(self, n: int):
        h = (self.b - self.a) / n

        a = np.zeros(shape=(n+1), dtype=float)
        b = np.zeros(shape=(n+1), dtype=float)
        d = np.zeros(shape=(n+1), dtype=float)

        c = self.__sweep_method(n)

        x = np.zeros(shape=(n+1), dtype=float)
        x[0] = self.a

        for i in range(1, n+1):
            x[i] = self.a+h*i
            a[i] = self._F(self.a+h*i)
            b[i] = (self._F(self.a+h*i)-self._F(self.a+h*(i-1)))/h + c[i]*h/3 + c[i-1]*h/6
            d[i] = (c[i]-c[i-1])/h

        return x, a, b, c, d

    def get_F(self, n: int):
        h = (self.b - self.a) / n

        f = np.zeros(shape=(n+1), dtype=float)
        f1 = np.zeros(shape=(n+1), dtype=float)
        f2 = np.zeros(shape=(n+1), dtype=float)

        x = np.zeros(shape=(n+1), dtype=float)

        for i in range(n+1):
            x[i] = self.a+i*h
            f[i] = self._F(self.a+i*h)
            f1[i] = self._F1(self.a+i*h)
            f2[i] = self._F2(self.a+i*h)
        return x, f, f1, f2

    def get_spline(self, n: int, n2: int):
        coef = int(n2 / n)
        h = (self.b - self.a) / n
        h2 = (self.b - self.a) / n2

        _, a, b, c, d = self.get_coef(n)

        s = np.zeros(shape=(n2+1), dtype=float)
        s1 = np.zeros(shape=(n2+1), dtype=float)
        s2 = np.zeros(shape=(n2+1), dtype=float)

        j = 1
        for i in range(n2+1):
            x = self.a + i*h2
            xk = self.a + j*h

            if (xk < x):
                j += 1
                xk = self.a + j*h

            s[i] = a[j] + b[j]*(x-xk) + c[j]/2*(x-xk)**2 + d[j]/6*(x-xk)**3
            s1[i] = b[j] + c[j]*(x-xk) + d[j]/2*(x-xk)**2
            s2[i] = c[j] + d[j]*(x-xk)

        return s, s1, s2

class test_task(task):

    a, b = -1, 1
    sa, sb = 0, 0

    def _F(self, x: float):
        result = 0.0
        if self.a <= x < 0:
            result = x**3 + 3 * x**2
        elif 0 <= x <= self.b:
            result = -x**3 + 3 * x**2
        return result

    def _F1(self, x: float):
        result = 0.0
        if self.a <= x < 0:
            result = 3*x**2 + 6*x
        elif 0 <= x <= self.b:
            result = -3*x**2 + 6*x
        return result

    def _F2(self, x: float):
        result = 0.0
        if self.a <= x < 0:
            result = 6*x + 6
        elif 0 <= x <= self.b:
            result = -6*x + 6
        return result

class b_task(task):

    a, b = 0, np.pi / 2
    sa, sb = 0, 0

    def _F(self, x: float):
        result = np.cos(x) / (1 + x**2)
        return result

    def _F1(self, x: float):
        result = -2*x*np.cos(x)/((1 + x**2)**2)-np.sin(x)/(1 + x**2)
        return result

    def _F2(self, x: float):
        result = 8*(x**2)*np.cos(x)/((1 + x**2)**3)+4*x*np.sin(x)/((1 + x**2)**2)-np.cos(x) / (1 + x**2)-2*np.cos(x)/((1 + x**2)**2)
        return result

class c_task(task):

    a, b = 0, np.pi / 2
    sa, sb = 0, 0

    def _F(self, x: float):
        result = np.cos(x) / (1 + x**2) + np.cos(10*x)
        return result

    def _F1(self, x: float):
        result = -2*x*np.cos(x)/((1 + x**2)**2)-np.sin(x)/(1 + x**2) - 10*np.sin(10*x)
        return result

    def _F2(self, x: float):
        result = 8*(x**2)*np.cos(x)/((1 + x**2)**3)+4*x*np.sin(x)/((1 + x**2)**2)-np.cos(x) / (1 + x**2)-2*np.cos(x)/((1 + x**2)**2) - 100*np.cos(10*x)
        return result
