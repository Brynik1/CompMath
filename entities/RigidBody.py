import numpy as np


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


class RigidBody:
    def __init__(self):
        self.r = np.array([0.0, 15.0, 0.0])
        self.l = np.array([0.0, 0.0, 0.0])
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Начальная ориентация
        self.L = np.array([0.0, 0.0, 0.0])

    def vertices(self, size):
        vertices_local = np.array(
            [
                [size, size, -size],
                [-size, size, -size],
                [-size, size, size],
                [size, size, size],
                [size, -size, size],
                [-size, -size, size],
                [-size, -size, -size],
                [size, -size, -size],
            ]
        )
        R = quat_to_matrix(self.q)
        return [self.r + R @ v for v in vertices_local]
