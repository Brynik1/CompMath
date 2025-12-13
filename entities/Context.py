import numpy as np

SIZE = 2.0

class Context:
    def __init__(self):
        self.M_inv = 1.0 / 1.0
        I_body = (1.0 / 1.0) * ((2 * SIZE) ** 2) * np.diag([1 / 6.0, 1 / 6.0, 1 / 6.0])
        self.I_inv = np.linalg.inv(I_body)
