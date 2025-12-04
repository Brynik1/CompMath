from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import numpy as np
import math


def calculate_plane_normal(angle_deg=0.0, rotation_axis='z'):
    """Рассчитывает нормаль плоскости по углу наклона и оси вращения.

    Параметры:
        angle_deg (float): Угол наклона в градусах (0-90)
        rotation_axis (str): Ось вращения ('x' или 'z')

    Возвращает:
        np.array: Нормализованный вектор нормали
    """
    angle_rad = np.radians(angle_deg)

    if rotation_axis.lower() == 'z':
        # Поворот вокруг оси Z (наклон в направлении X)
        normal = np.array([
            math.sin(angle_rad),  # X-компонента
            math.cos(angle_rad),  # Y-компонента
            0.0  # Z-компонента
        ])
    elif rotation_axis.lower() == 'x':
        # Поворот вокруг оси X (наклон в направлении Z)
        normal = np.array([
            0.0,  # X-компонента
            math.cos(angle_rad),  # Y-компонента
            math.sin(angle_rad)  # Z-компонента
        ])
    else:
        raise ValueError("Недопустимая ось вращения. Используйте 'x' или 'z'")

    return normal / np.linalg.norm(normal)


# Параметры камеры
CAMERA_TARGET = np.array([0.0, 3.0, 0.0])  # Точка, на которую смотрит камера
INIT_CAMERA_DISTANCE = 45.7  # Начальная дистанция камеры
ZOOM_SPEED = 1.0  # Скорость приближения/отдаления
MIN_DISTANCE = 5.0  # Минимальное расстояние камеры
MAX_DISTANCE = 70.0  # Максимальное расстояние камеры
camera_distance = INIT_CAMERA_DISTANCE
camera_direction = np.array([30.0, 17.0, 30.0])  # Вектор направления камеры
camera_direction /= np.linalg.norm(camera_direction)  # Нормализуем


# Параметры тела и физики
SIZE = 2.0
GRAVITY = np.array([0.0, -9.80665, 0.0])
RESTITUTION = 0.9
CONTACT_THRESHOLD = 0.1


# Параметры наклонной плоскости
PLANE_SIZE_X = 20.0
PLANE_SIZE_Z = 20.0
PLANE_NORMAL = calculate_plane_normal(angle_deg=0, rotation_axis='x')
PLANE_NORMAL /= np.linalg.norm(PLANE_NORMAL)  # Нормализация
PLANE_POINT = np.array([0.0, 0.0, 0.0])  # Точка на плоскости


class RigidBody:
    def __init__(self):
        self.r = np.array([0.0, 15.0, 0.0])
        self.l = np.array([0.0, 0.0, 0.0])
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Начальная ориентация
        self.L = np.array([0.0, 0.0, 0.0])

    def vertices(self, size):
        vertices_local = np.array([
            [size, size, -size],
            [-size, size, -size],
            [-size, size, size],
            [size, size, size],
            [size, -size, size],
            [-size, -size, size],
            [-size, -size, -size],
            [size, -size, -size]
        ])
        R = quat_to_matrix(self.q)
        return [self.r + R @ v for v in vertices_local]


class Context:
    def __init__(self):
        self.M_inv = 1.0 / 1.0
        I_body = (1.0 / 1.0) * ((2 * SIZE) ** 2) * np.diag([1 / 6.0, 1 / 6.0, 1 / 6.0])
        self.I_inv = np.linalg.inv(I_body)


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def compute_derivatives(rb, context):
    deriv = RigidBody()
    deriv.r = rb.l * context.M_inv

    R = quat_to_matrix(rb.q)
    I_inv_world = R @ context.I_inv @ R.T
    omega = I_inv_world @ rb.L

    omega_quat = np.array([0.0, *omega])
    deriv.q = 0.5 * quat_mult(omega_quat, rb.q)

    deriv.l = GRAVITY / context.M_inv
    deriv.L = np.zeros(3)
    return deriv


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])


def rk4_step(rb, context, dt):
    k1 = compute_derivatives(rb, context)

    k2_body = RigidBody()
    k2_body.r = rb.r + 0.5 * dt * k1.r
    k2_body.l = rb.l + 0.5 * dt * k1.l
    k2_body.q = rb.q + 0.5 * dt * k1.q
    k2_body.L = rb.L + 0.5 * dt * k1.L
    k2 = compute_derivatives(k2_body, context)

    k3_body = RigidBody()
    k3_body.r = rb.r + 0.5 * dt * k2.r
    k3_body.l = rb.l + 0.5 * dt * k2.l
    k3_body.q = rb.q + 0.5 * dt * k2.q
    k3_body.L = rb.L + 0.5 * dt * k2.L
    k3 = compute_derivatives(k3_body, context)

    k4_body = RigidBody()
    k4_body.r = rb.r + dt * k3.r
    k4_body.l = rb.l + dt * k3.l
    k4_body.q = rb.q + dt * k3.q
    k4_body.L = rb.L + dt * k3.L
    k4 = compute_derivatives(k4_body, context)

    rb.r += dt / 6 * (k1.r + 2 * k2.r + 2 * k3.r + k4.r)
    rb.l += dt / 6 * (k1.l + 2 * k2.l + 2 * k3.l + k4.l)
    rb.q += dt / 6 * (k1.q + 2 * k2.q + 2 * k3.q + k4.q)
    rb.L += dt / 6 * (k1.L + 2 * k2.L + 2 * k3.L + k4.L)
    rb.q /= np.linalg.norm(rb.q)


def is_inside_platform(position):
    """Проверяет, находится ли точка в пределах платформы"""
    default_normal = np.array([0.0, 1.0, 0.0])
    new_normal = PLANE_NORMAL

    # Вычисляем ось и угол вращения
    axis = np.cross(default_normal, new_normal)
    cos_angle = np.dot(default_normal, new_normal)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    if np.linalg.norm(axis) < 1e-6:
        # Нет вращения или поворот на 180 градусов (не учитываем отражение)
        R_T = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        # Формула Родрига для матрицы вращения
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        R_T = R.T  # Обратное вращение

    # Перенос точки в локальные координаты платформы
    translated_point = np.array(position) - PLANE_POINT
    local_point = R_T @ translated_point

    # Проверяем границы по X и Z
    return abs(local_point[0]) <= PLANE_SIZE_X and abs(local_point[2]) <= PLANE_SIZE_Z


def check_collision(rb, context):

    if not is_inside_platform(rb.r):
        return

    vertices = rb.vertices(SIZE)
    contact_points = []


    for v in vertices:
        distance = np.dot(v - PLANE_POINT, PLANE_NORMAL)
        if distance < CONTACT_THRESHOLD:
            contact_points.append(v)

    if not contact_points:
        return

    avg_contact = np.mean(contact_points, axis=0)
    r = avg_contact - rb.r

    R = quat_to_matrix(rb.q)
    I_inv = R @ context.I_inv @ R.T
    v = rb.l * context.M_inv
    omega = I_inv @ rb.L
    v_contact = v + np.cross(omega, r)
    v_rel = np.dot(v_contact, PLANE_NORMAL)

    if v_rel < 0:
        numerator = -(1 + RESTITUTION) * v_rel
        term1 = context.M_inv
        term2 = np.dot(PLANE_NORMAL, np.cross(I_inv @ np.cross(r, PLANE_NORMAL), r))
        j = numerator / (term1 + term2)
        rb.l += j * PLANE_NORMAL
        rb.L += np.cross(r, j * PLANE_NORMAL)


def draw_cube():
    glBegin(GL_QUADS)

    # Верхняя грань (верх)
    glColor3f(1.0, 1.0, 1.0)  # Яркий, но мягкий белый
    glVertex3f(SIZE, SIZE, -SIZE)
    glVertex3f(-SIZE, SIZE, -SIZE)
    glVertex3f(-SIZE, SIZE, SIZE)
    glVertex3f(SIZE, SIZE, SIZE)

    # Нижняя грань (низ)
    glColor3f(1.0, 1.0, 0.5)  # Яркий, но мягкий желтый
    glVertex3f(SIZE, -SIZE, SIZE)
    glVertex3f(-SIZE, -SIZE, SIZE)
    glVertex3f(-SIZE, -SIZE, -SIZE)
    glVertex3f(SIZE, -SIZE, -SIZE)

    # Передняя грань
    glColor3f(0.2, 0.2, 1.0)  # Яркий, но мягкий синий
    glVertex3f(SIZE, SIZE, SIZE)
    glVertex3f(-SIZE, SIZE, SIZE)
    glVertex3f(-SIZE, -SIZE, SIZE)
    glVertex3f(SIZE, -SIZE, SIZE)

    # Задняя грань
    glColor3f(0.4, 0.9, 0.4)  # Яркий, но мягкий зеленый
    glVertex3f(SIZE, -SIZE, -SIZE)
    glVertex3f(-SIZE, -SIZE, -SIZE)
    glVertex3f(-SIZE, SIZE, -SIZE)
    glVertex3f(SIZE, SIZE, -SIZE)

    # Левая грань
    glColor3f(0.9, 0.3, 0.3)  # Яркий, но мягкий красный
    glVertex3f(-SIZE, SIZE, SIZE)
    glVertex3f(-SIZE, SIZE, -SIZE)
    glVertex3f(-SIZE, -SIZE, -SIZE)
    glVertex3f(-SIZE, -SIZE, SIZE)

    # Правая грань
    glColor3f(0.9, 0.5, 0.2)  # Яркий, но мягкий оранжевый
    glVertex3f(SIZE, SIZE, -SIZE)
    glVertex3f(SIZE, SIZE, SIZE)
    glVertex3f(SIZE, -SIZE, SIZE)
    glVertex3f(SIZE, -SIZE, -SIZE)

    glEnd()


def draw_plane():
    glPushMatrix()

    # Переносим плоскость в заданную точку
    glTranslatef(*PLANE_POINT)

    # Исходная нормаль (вертикальная)
    default_normal = np.array([0.0, 1.0, 0.0])
    new_normal = PLANE_NORMAL

    # Вычисляем ось и угол вращения
    axis = np.cross(default_normal, new_normal)
    angle = np.degrees(np.arccos(np.dot(default_normal, new_normal)))

    # Если ось не нулевая - выполняем вращение
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)  # Нормализуем ось
        glRotatef(angle, *axis)

    # Параметры платформы
    size_x = PLANE_SIZE_X
    size_z = PLANE_SIZE_Z
    thickness = 1  # Толщина платформы

    # Основная поверхность (верх)
    glColor3f(0.5, 0.5, 0.56)  # Светло-серый
    glBegin(GL_QUADS)
    glVertex3f(-size_x, 0, -size_z)
    glVertex3f(-size_x, 0, size_z)
    glVertex3f(size_x, 0, size_z)
    glVertex3f(size_x, 0, -size_z)
    glEnd()

    # Нижняя поверхность
    glColor3f(0.2, 0.2, 0.26)  # Темно-серый
    glBegin(GL_QUADS)
    glVertex3f(-size_x, -thickness, -size_z)
    glVertex3f(-size_x, -thickness, size_z)
    glVertex3f(size_x, -thickness, size_z)
    glVertex3f(size_x, -thickness, -size_z)
    glEnd()

    # Боковые грани
    glColor3f(0.35, 0.35, 0.41)  # Средне-серый

    # Передняя грань (Z+)
    glBegin(GL_QUADS)
    glVertex3f(-size_x, 0, size_z)
    glVertex3f(size_x, 0, size_z)
    glVertex3f(size_x, -thickness, size_z)
    glVertex3f(-size_x, -thickness, size_z)
    glEnd()

    # Задняя грань (Z-)
    glBegin(GL_QUADS)
    glVertex3f(-size_x, 0, -size_z)
    glVertex3f(-size_x, -thickness, -size_z)
    glVertex3f(size_x, -thickness, -size_z)
    glVertex3f(size_x, 0, -size_z)
    glEnd()

    # Левая грань (X-)
    glBegin(GL_QUADS)
    glVertex3f(-size_x, 0, -size_z)
    glVertex3f(-size_x, 0, size_z)
    glVertex3f(-size_x, -thickness, size_z)
    glVertex3f(-size_x, -thickness, -size_z)
    glEnd()

    # Правая грань (X+)
    glBegin(GL_QUADS)
    glVertex3f(size_x, 0, -size_z)
    glVertex3f(size_x, -thickness, -size_z)
    glVertex3f(size_x, -thickness, size_z)
    glVertex3f(size_x, 0, size_z)
    glEnd()

    glPopMatrix()


def compute_energy(rb, context):
    """Вычисляет отдельные компоненты энергии."""
    m = 1.0 / context.M_inv
    g = abs(GRAVITY[1])  # Берем абсолютное значение гравитации

    # Потенциальная энергия (корректный расчет)
    pe = m * g * rb.r[1]

    # Поступательная кинетическая
    translational_ke = 0.5 * np.dot(rb.l, rb.l) * context.M_inv

    # Вращательная кинетическая
    R = quat_to_matrix(rb.q)
    I_inv_world = R @ context.I_inv @ R.T
    omega = I_inv_world @ rb.L
    rotational_ke = 0.5 * np.dot(rb.L, omega)

    ke = translational_ke + rotational_ke
    total = pe + ke
    return pe, ke, total


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    camera_pos = CAMERA_TARGET + camera_direction * camera_distance

    gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],  # Позиция камеры
              CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2],  # Точка наблюдения
              0, 1, 0)  # Вектор "вверх"

    # Отрисовка земли
    draw_plane()

    # Отрисовка куба
    glPushMatrix()
    glTranslate(*rb.r)
    q = rb.q
    angle = 2 * np.arccos(q[0]) * 180 / np.pi
    axis = q[1:] if np.linalg.norm(q[1:]) > 0 else [1, 0, 0]
    glRotatef(angle, *axis)
    draw_cube()
    glPopMatrix()

    # Отображение энергии
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 800, 0, 600)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    pe, ke, total = compute_energy(rb, context)

    # Настройка текста
    glColor3f(1, 1, 1)
    y_pos = 580  # Начальная позиция по Y

    # Потенциальная энергия
    text_pe = f"Potential: {pe:.2f} J"
    glRasterPos2f(10, y_pos)
    for char in text_pe:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    # Кинетическая энергия
    y_pos -= 20
    text_ke = f"Kinetic: {ke:.2f} J"
    glRasterPos2f(10, y_pos)
    for char in text_ke:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    # Полная энергия
    y_pos -= 20
    text_total = f"Total: {total:.2f} J"
    glRasterPos2f(10, y_pos)
    for char in text_total:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glutSwapBuffers()


def timer(value):
    dt = 0.02
    rk4_step(rb, context, dt)
    check_collision(rb, context)
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Rigid Body Simulation")

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, 800 / 600, 0.1, 100)
    glMatrixMode(GL_MODELVIEW)

    global rb, context
    rb = RigidBody()
    context = Context()

    glutDisplayFunc(display)
    glutTimerFunc(0, timer, 0)
    glutMainLoop()


if __name__ == "__main__":
    main()
