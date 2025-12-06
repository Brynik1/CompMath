import numpy as np
from generate_synthetic_test import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def draw_grid(size=100.0, step=1.0, y=0.0):
    glLineWidth(1.0)
    glColor3f(0.2, 0.2, 0.2)
    glBegin(GL_LINES)

    z = -size
    while z <= size:
        glVertex3f(-size, y, z)
        glVertex3f(size, y, z)
        z += step

    x = -size
    while x <= size:
        glVertex3f(x, y, -size)
        glVertex3f(x, y, size)
        x += step

    glEnd()

    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(0.6, 0.1, 0.1)
    glVertex3f(-size, y, 0.0)
    glVertex3f(size, y, 0.0)

    glColor3f(0.1, 0.1, 0.6)
    glVertex3f(0.0, y, -size)
    glVertex3f(0.0, y, size)
    glEnd()
    glLineWidth(1.0)


def draw_trajectory(traj, color, width=2.0):
    if traj is None or traj.shape[0] < 2:
        return
    glColor3f(*color)
    glLineWidth(width)
    glBegin(GL_LINE_STRIP)
    for p in traj:
        glVertex3f(p[0], p[1], p[2])
    glEnd()
    glLineWidth(1.0)


def draw_points(traj, color, size=4.0):
    if traj is None:
        return
    glPointSize(size)
    glColor3f(*color)
    glBegin(GL_POINTS)
    for p in traj:
        glVertex3f(p[0], p[1], p[2])
    glEnd()
    glPointSize(1.0)


def draw_text_2d(x, y, text, font=GLUT_BITMAP_9_BY_15):
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(font, ord(ch))


def draw_legend(width, height, view_mode, flying_cubes):
    box_w, box_h = 380, 150
    margin = 12
    x0 = margin
    y0 = height - box_h - margin

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    glColor4f(0.0, 0.0, 0.0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(x0, y0)
    glVertex2f(x0 + box_w, y0)
    glVertex2f(x0 + box_w, y0 + box_h)
    glVertex2f(x0, y0 + box_h)
    glEnd()

    glColor3f(1.0, 1.0, 1.0)
    y_text = y0 + box_h - 18
    draw_text_2d(x0 + 8, y_text, "1: obs; 2: CM; 3: verts; 4: empty; k: launch cube")

    y_text -= 24
    if view_mode == 1:
        draw_text_2d(x0 + 8, y_text, "Mode 1: observation vertex points (blue)")
    elif view_mode == 2:
        draw_text_2d(x0 + 8, y_text, "Mode 2: centre of mass trajectories")
    elif view_mode == 3:
        draw_text_2d(x0 + 8, y_text, "Mode 3: vertex trajectories")
    elif view_mode == 4:
        draw_text_2d(x0 + 8, y_text, "Mode 4: empty field")

    y_text -= 24
    draw_text_2d(x0 + 8, y_text, f"Active cubes: {len(flying_cubes)}")

    # наблюдения (синие точки)
    y_text -= 20
    glColor3f(0.0, 0.5, 1.0)
    glBegin(GL_QUADS)
    glVertex2f(x0 + 8, y_text)
    glVertex2f(x0 + 26, y_text)
    glVertex2f(x0 + 26, y_text + 10)
    glVertex2f(x0 + 8, y_text + 10)
    glEnd()
    glColor3f(1.0, 1.0, 1.0)
    draw_text_2d(x0 + 32, y_text, "Observation vertex / CM points")

    # истинные траектории (зелёные)
    y_text -= 14
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(x0 + 8, y_text)
    glVertex2f(x0 + 26, y_text)
    glVertex2f(x0 + 26, y_text + 10)
    glVertex2f(x0 + 8, y_text + 10)
    glEnd()
    glColor3f(1.0, 1.0, 1.0)
    draw_text_2d(x0 + 32, y_text, "True CM / vertex trajectories")

    # восстановленные траектории (рыжие)
    y_text -= 14
    glColor3f(1.0, 0.4, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(x0 + 8, y_text)
    glVertex2f(x0 + 26, y_text)
    glVertex2f(x0 + 26, y_text + 10)
    glVertex2f(x0 + 8, y_text + 10)
    glEnd()
    glColor3f(1.0, 1.0, 1.0)
    draw_text_2d(x0 + 32, y_text, "Estimated CM / vertex trajectories")

    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_cube_from_vertices(verts: np.ndarray, R_color: np.ndarray | None = None):
    """
    verts: (8,3) мировые координаты вершин
    R_color: 3x3 ортонормальная матрица, задающая ориентацию «цветового» куба
    """
    # индексы вершин, как раньше
    faces = [
        (0, 1, 2, 3),  # top
        (4, 5, 6, 7),  # bottom
        (3, 2, 5, 4),  # front
        (6, 1, 0, 7),  # back
        (1, 6, 5, 2),  # left
        (0, 3, 4, 7),  # right
    ]
    colors = [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.5),
        (0.2, 0.2, 1.0),
        (0.4, 0.9, 0.4),
        (0.9, 0.3, 0.3),
        (0.9, 0.5, 0.2),
    ]

    if R_color is None:
        R_color = np.eye(3, dtype=float)

    # Центр куба в мировых координатах
    center = verts.mean(axis=0)

    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3f(*colors[i])
        for idx in face:
            v = verts[idx]
            # локальный вектор относительно центра
            local = v - center
            # повернуть локальный вектор матрицей цвета
            rotated_local = R_color @ local
            v_color = center + rotated_local
            glVertex3f(v_color[0], v_color[1], v_color[2])
    glEnd()


def random_orthonormal_matrix() -> np.ndarray:
    """
    Генерация случайной ортонормальной 3x3 матрицы (равномерно по SO(3)).
    """
    # случайная матрица Гаусса
    A = np.random.normal(size=(3, 3))
    # QR-разложение
    Q, R = np.linalg.qr(A)
    # гарантируем det(Q) = +1
    if np.linalg.det(Q) < 0.0:
        Q[:, 0] = -Q[:, 0]
    return Q


def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, w / float(h or 1), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
