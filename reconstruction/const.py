import numpy as np

DT: float = 0.05  # шаг интегрирования
N_STEPS: int = 250  # длина наблюдаемой траектории (количество кадров с вершинами)
NPRE: int = 50  # сколько шагов до начала наблюдений нужно восстановить

# начальное приближение для Ньютона по центру масс
INITIAL_R0 = np.array([5.0, 10.0, 0.5], dtype=float)
INITIAL_L0 = np.array([5.0, -0.5, 1.1], dtype=float)
P_INITIAL = np.hstack([INITIAL_R0, INITIAL_L0])

# параметры Гаусса–Ньютона
MAX_ITER = 15
TOL = 1e-15
DAMPING = 1.0

# параметры для синтетического теста
TRUE_SIZE = 2.0
TRUE_R0 = np.array(
    [0.0, 15.0, 0.0], dtype=float
)  # центр масс в момент начала (t = -NPRE*DT)
TRUE_L0 = np.array([10.0, 10.0, 5.5], dtype=float)  # линейный импульс
TRUE_Q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # начальная ориентация
TRUE_OMEGA = np.array([0.0, 2.0, 1.0], dtype=float)  # угловая скорость (рад/с)
NOISE_STD_VERTICES = 0.001
