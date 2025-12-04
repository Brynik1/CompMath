import numpy as np

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import experiment
from experiment import RigidBody, Context, rk4_step  # модель динамики


# ---------------- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА / ОБРАТНОЙ ЗАДАЧИ ----------------

DT: float = 0.05   # шаг интегрирования
N_STEPS: int = 250 # длина наблюдаемой траектории (количество кадров с вершинами)
NPRE: int = 50     # сколько шагов до начала наблюдений нужно восстановить

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
TRUE_R0 = np.array([0.0, 15.0, 0.0], dtype=float)     # центр масс в момент начала (t = -NPRE*DT)
TRUE_L0 = np.array([10.0, 10.0, 5.5], dtype=float)    # линейный импульс
TRUE_Q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) # начальная ориентация
TRUE_OMEGA = np.array([0.0, 2.0, 1.0], dtype=float)   # угловая скорость (рад/с)
NOISE_STD_VERTICES = 0.001                            # шум в наблюдаемых вершинах


# ---------------- УТИЛИТЫ ДЛЯ МОДЕЛИ ЦЕНТРА МАСС (Гаусс–Ньютон) ----------------

def unpack_params_cm(params: np.ndarray) -> RigidBody:
    rb = RigidBody()
    rb.r = np.array(params[0:3], dtype=float)
    rb.l = np.array(params[3:6], dtype=float)
    return rb


def simulate_observed_traj_cm(params: np.ndarray,
                              dt: float,
                              n_steps: int,
                              npre: int = 10) -> np.ndarray:
    rb = unpack_params_cm(params)
    context = Context()

    for _ in range(npre):
        rk4_step(rb, context, dt)

    traj = np.zeros((n_steps, 3), dtype=float)
    for k in range(n_steps):
        rk4_step(rb, context, dt)
        traj[k] = rb.r.copy()
    return traj


def residuals_cm(params: np.ndarray,
                 y_obs_cm: np.ndarray,
                 dt: float,
                 npre: int = 10) -> np.ndarray:
    y_model = simulate_observed_traj_cm(params, dt, y_obs_cm.shape[0], npre=npre)
    return (y_model - y_obs_cm).ravel()


def jacobian_fd_cm(params: np.ndarray,
                   y_obs_cm: np.ndarray,
                   dt: float,
                   npre: int = 10,
                   eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    r0 = residuals_cm(params, y_obs_cm, dt, npre=npre)
    m = r0.size
    n = params.size
    J = np.zeros((m, n), dtype=float)

    for j in range(n):
        dp = np.zeros_like(params)
        dp[j] = eps
        r1 = residuals_cm(params + dp, y_obs_cm, dt, npre=npre)
        J[:, j] = (r1 - r0) / eps

    return J, r0


def gauss_newton_cm(y_obs_cm: np.ndarray,
                    dt: float,
                    p0: np.ndarray,
                    npre: int = 10,
                    max_iter: int = 20,
                    tol: float = 1e-6,
                    damping: float = 1.0) -> np.ndarray:
    p = p0.copy()
    for k in range(max_iter):
        J, r = jacobian_fd_cm(p, y_obs_cm, dt, npre=npre)
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            delta = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            print("JTJ is singular at iter", k)
            break

        p_new = p - damping * delta

        print(f"GN iter {k}: ||r||={np.linalg.norm(r):.6e}, "
              f"||delta||={np.linalg.norm(delta):.6e}")

        if np.linalg.norm(delta) < tol:
            p = p_new
            break

        p = p_new
    return p


# ---------------- ГЕОМЕТРИЯ КУБА: РАЗМЕР И ОРИЕНТАЦИЯ ----------------

def compute_com_traj_from_vertices(V_obs: np.ndarray) -> np.ndarray:
    return V_obs.mean(axis=1)


def estimate_cube_size_from_vertices_single_frame(verts: np.ndarray) -> float:
    diff = verts[None, :, :] - verts[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    mask = ~np.eye(8, dtype=bool)
    dists = dists[mask]
    dists_sorted = np.sort(dists)
    edge_lengths = dists_sorted[:12]
    edge_len_mean = edge_lengths.mean()
    size = edge_len_mean / 2.0
    return float(size)


def canonical_body_vertices(size: float) -> np.ndarray:
    s = size
    return np.array([
        [ s,  s, -s],
        [-s,  s, -s],
        [-s,  s,  s],
        [ s,  s,  s],
        [ s, -s,  s],
        [-s, -s,  s],
        [-s, -s, -s],
        [ s, -s, -s],
    ], dtype=float)


def kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def estimate_orientation_sequence_from_vertices(V_obs: np.ndarray,
                                                size_est: float) -> np.ndarray:
    N_steps = V_obs.shape[0]
    v_body = canonical_body_vertices(size_est)
    v_body_centered = v_body - v_body.mean(axis=0, keepdims=True)

    R_seq = np.zeros((N_steps, 3, 3), dtype=float)
    for k in range(N_steps):
        verts = V_obs[k]
        com = verts.mean(axis=0, keepdims=True)
        verts_centered = verts - com
        R = kabsch_rotation(v_body_centered, verts_centered)
        R_seq[k] = R
    return R_seq


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)

    if q[0] < 0.0:
        q = -q

    return q


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def relative_rotation(R1: np.ndarray, R0: np.ndarray) -> np.ndarray:
    return R1 @ R0.T


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    tr = np.trace(R)
    angle = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0]), 0.0

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=float)
    axis /= (2.0 * np.sin(angle))
    return axis, angle


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis /= norm
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def estimate_initial_rotation_and_L0(V_obs: np.ndarray,
                                     size_est: float,
                                     dt: float) -> tuple[np.ndarray, np.ndarray]:
    R_seq = estimate_orientation_sequence_from_vertices(V_obs, size_est)
    N_steps = R_seq.shape[0]

    omegas = []
    for k in range(N_steps - 1):
        R0 = R_seq[k]
        R1 = R_seq[k + 1]
        R_rel = relative_rotation(R1, R0)
        axis, angle = rotation_matrix_to_axis_angle(R_rel)
        omega_k = axis * (angle / dt)
        omegas.append(omega_k)

    omegas = np.vstack(omegas)
    omega_est = omegas.mean(axis=0)

    q_obs0 = rotation_matrix_to_quat(R_seq[0])

    I_scalar = ((2.0 * size_est) ** 2) / 6.0
    L0_obs = I_scalar * omega_est

    return q_obs0, L0_obs


def shift_orientation_back_to_start(q_obs0: np.ndarray,
                                    omega: np.ndarray,
                                    npre: int,
                                    dt: float) -> np.ndarray:
    omega_norm = np.linalg.norm(omega)
    if np.isclose(omega_norm, 0.0):
        q = q_obs0.copy()
    else:
        axis = omega / omega_norm
        total_back_angle = -omega_norm * (npre * dt)
        q_back = axis_angle_to_quat(axis, total_back_angle)
        q = quat_mult(q_back, q_obs0)

    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


# ---------------- ВОССТАНОВЛЕНИЕ ПОЛНОЙ ТРАЕКТОРИИ КУБА ----------------

def reconstruct_full_cube_trajectory(p_est: np.ndarray,
                                     size_est: float,
                                     q_init: np.ndarray,
                                     L_init: np.ndarray,
                                     dt: float,
                                     n_steps: int,
                                     npre: int = 0) -> tuple[np.ndarray, np.ndarray]:
    total_steps = npre + n_steps

    experiment.SIZE = float(size_est)

    rb = RigidBody()
    rb.r = np.array(p_est[0:3], dtype=float)
    rb.l = np.array(p_est[3:6], dtype=float)
    rb.q = q_init.copy()
    rb.L = L_init.copy()

    context = Context()

    traj_cm = np.zeros((total_steps, 3), dtype=float)
    traj_vertices = np.zeros((total_steps, 8, 3), dtype=float)

    for k in range(total_steps):
        rk4_step(rb, context, dt)
        traj_cm[k] = rb.r.copy()
        verts_k = rb.vertices(size_est)
        traj_vertices[k] = np.stack(verts_k, axis=0)

    return traj_cm, traj_vertices


def inverse_from_vertex_trajectories(V_obs: np.ndarray) -> dict:
    y_obs_cm = compute_com_traj_from_vertices(V_obs)

    p_est = gauss_newton_cm(
        y_obs_cm=y_obs_cm,
        dt=DT,
        p0=P_INITIAL,
        npre=NPRE,
        max_iter=MAX_ITER,
        tol=TOL,
        damping=DAMPING
    )

    size_est = estimate_cube_size_from_vertices_single_frame(V_obs[0])

    q_obs0, L0_obs = estimate_initial_rotation_and_L0(V_obs, size_est, DT)

    I_scalar = ((2.0 * size_est) ** 2) / 6.0
    omega_est = L0_obs / I_scalar if I_scalar != 0.0 else np.zeros(3)

    q_init = shift_orientation_back_to_start(q_obs0, omega_est, NPRE, DT)

    traj_cm_est, traj_vertices_est = reconstruct_full_cube_trajectory(
        p_est=p_est,
        size_est=size_est,
        q_init=q_init,
        L_init=L0_obs,
        dt=DT,
        n_steps=N_STEPS,
        npre=NPRE
    )

    return {
        "p_est": p_est,
        "size_est": size_est,
        "q_init": q_init,
        "L_init": L0_obs,
        "traj_cm_est": traj_cm_est,
        "traj_vertices_est": traj_vertices_est,
    }


# ---------------- СИНТЕТИЧЕСКИЙ ТЕСТ (ГЕНЕРАЦИЯ V_obs) ----------------

def generate_synthetic_vertex_data(dt: float,
                                   n_steps: int,
                                   npre: int,
                                   noise_std_vertices: float = 0.0):
    experiment.SIZE = float(TRUE_SIZE)

    rb_true = RigidBody()
    rb_true.r = TRUE_R0.copy()
    rb_true.l = TRUE_L0.copy()
    rb_true.q = TRUE_Q0.copy()

    I_scalar_true = ((2.0 * TRUE_SIZE) ** 2) / 6.0
    L_true = I_scalar_true * TRUE_OMEGA
    rb_true.L = L_true.copy()

    context = Context()

    total_steps = npre + n_steps
    full_traj_cm_true = np.zeros((total_steps, 3), dtype=float)
    full_traj_vertices_true = np.zeros((total_steps, 8, 3), dtype=float)

    for k in range(total_steps):
        rk4_step(rb_true, context, dt)
        full_traj_cm_true[k] = rb_true.r.copy()
        verts_k = rb_true.vertices(TRUE_SIZE)
        full_traj_vertices_true[k] = np.stack(verts_k, axis=0)

    V_obs = full_traj_vertices_true[npre:npre + n_steps].copy()

    if noise_std_vertices > 0.0:
        noise = np.random.normal(scale=noise_std_vertices, size=V_obs.shape)
        V_obs += noise

    return full_traj_cm_true, full_traj_vertices_true, V_obs


# ---------------- OPENGL: ВИЗУАЛИЗАЦИЯ ----------------

# данные для визуализации (заполняются в main)
full_traj_cm_true = None           # (NPRE+N_STEPS, 3)
full_traj_vertices_true = None     # (NPRE+N_STEPS, 8, 3)
traj_cm_est = None                 # (NPRE+N_STEPS, 3)
traj_vertices_est = None           # (NPRE+N_STEPS, 8, 3)
cm_obs = None                      # (N_STEPS, 3) - центр масс наблюдений
obs_vertices = None                # (N_STEPS, 8, 3) - наблюдаемые вершины

total_steps = None
frame_idx = 0

# камера и режим просмотра
cam_distance = 60.0
cam_theta = 45.0
cam_phi = 30.0


# режимы:
# 1: точки наблюдений вершин (синие)
# 2: траектории центра масс (true/est) + CM наблюдений
# 3: траектории вершин (true/est)
# 4: пустое поле (только сетка)
# 5: траектории вершин (true/est), как 3
view_mode = 1
prev_utf8_byte = None  # для отслеживания русской буквы "л"/"Л"

# список «летящих» кубов, каждый хранит локальный индекс по траектории
# элемент: {"idx": int}
flying_cubes: list[dict] = []


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


def draw_legend(width, height):
    global view_mode, flying_cubes
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
    draw_text_2d(
        x0 + 8,
        y_text,
        "1: obs verts; 2: CM; 3: verts; 4: empty; 5: verts; k: launch cube"
    )

    y_text -= 16
    if view_mode == 1:
        draw_text_2d(x0 + 8, y_text, "Mode 1: observation vertex points (blue)")
    elif view_mode == 2:
        draw_text_2d(x0 + 8, y_text, "Mode 2: centre of mass trajectories")
    elif view_mode == 3:
        draw_text_2d(x0 + 8, y_text, "Mode 3: vertex trajectories")
    elif view_mode == 4:
        draw_text_2d(x0 + 8, y_text, "Mode 4: empty field (grid only)")

    y_text -= 16
    draw_text_2d(x0 + 8, y_text, f"Active cubes: {len(flying_cubes)}")

    # наблюдения (синие точки)
    y_text -= 18
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


def draw_cube_from_vertices(verts: np.ndarray):
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

    glBegin(GL_QUADS)
    for (i, face) in enumerate(faces):
        glColor3f(*colors[i])
        for idx in face:
            v = verts[idx]
            glVertex3f(v[0], v[1], v[2])
    glEnd()


def display():
    global frame_idx, total_steps
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    theta_rad = np.radians(cam_theta)
    phi_rad = np.radians(cam_phi)
    cx = cam_distance * np.cos(phi_rad) * np.cos(theta_rad)
    cy = cam_distance * np.sin(phi_rad)
    cz = cam_distance * np.cos(phi_rad) * np.sin(theta_rad)

    gluLookAt(cx, cy, cz,
              0.0, 10.0, 0.0,
              0.0, 1.0, 0.0)

    draw_grid(size=100.0, step=1.0, y=0.0)

    # 1: только точки наблюдений вершин (синие)
    if view_mode == 1:
        if obs_vertices is not None:
            for i in range(8):
                draw_points(obs_vertices[:, i, :],
                            color=(0.0, 0.5, 1.0),
                            size=4.0)

    # 2: траектории центра масс (истинная и восстановленная) + CM наблюдений
    elif view_mode == 2:
        if full_traj_cm_true is not None:
            draw_trajectory(full_traj_cm_true, color=(0.0, 1.0, 0.0), width=2.0)
        if traj_cm_est is not None:
            draw_trajectory(traj_cm_est, color=(1.0, 0.3, 0.0), width=2.0)
        if cm_obs is not None:
            draw_points(cm_obs, color=(0.0, 0.5, 1.0), size=4.0)

    # 3: траектории вершин (истинные и восстановленные)
    elif view_mode == 3:
        if full_traj_vertices_true is not None:
            for i in range(8):
                draw_trajectory(full_traj_vertices_true[:, i, :],
                                color=(0.0, 0.9, 0.0), width=1.0)
        if traj_vertices_est is not None:
            for i in range(8):
                draw_trajectory(traj_vertices_est[:, i, :],
                                color=(1.0, 0.4, 0.0), width=1.5)

    # 4: пустое поле (только сетка)
    elif view_mode == 4:
        pass

    # 5: траектории вершин (true/est) — как 3
    elif view_mode == 5:
        if full_traj_vertices_true is not None:
            for i in range(8):
                draw_trajectory(full_traj_vertices_true[:, i, :],
                                color=(0.0, 0.9, 0.0), width=1.0)
        if traj_vertices_est is not None:
            for i in range(8):
                draw_trajectory(traj_vertices_est[:, i, :],
                                color=(1.0, 0.4, 0.0), width=1.5)

    # Летящие кубы: каждый идёт по восстановленной траектории с собственного нулевого кадра
    if traj_vertices_est is not None and total_steps is not None:
        for cube in flying_cubes:
            k_local = cube["idx"]
            if 0 <= k_local < total_steps:
                verts_k = traj_vertices_est[k_local]
                draw_cube_from_vertices(verts_k)

    w = glutGet(GLUT_WINDOW_WIDTH)
    h = glutGet(GLUT_WINDOW_HEIGHT)
    draw_legend(w, h)

    glutSwapBuffers()


def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, w / float(h or 1), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)


def keyboard(key, x, y):
    global cam_distance, view_mode, flying_cubes, prev_utf8_byte

    if key in (b'-', b'_'):
        cam_distance += 2.0
    elif key in (b'=', b'+'):
        cam_distance = max(5.0, cam_distance - 2.0)
    elif key == b'1':
        view_mode = 1
    elif key == b'2':
        view_mode = 2
    elif key == b'3':
        view_mode = 3
    elif key == b'4':
        view_mode = 4
    elif key in (b'k', b'K'):
        # запускаем новый куб со старта восстановленной траектории
        if traj_vertices_est is not None and total_steps is not None and total_steps > 0:
            flying_cubes.append({"idx": 0})
    else:
        # Обработка русской "л"/"Л" в UTF‑8: D0 BB / D0 9B
        bval = key[0]
        if prev_utf8_byte == 0xD0 and bval in (0xBB, 0x9B):
            # распознали 'л' или 'Л'
            if traj_vertices_est is not None and total_steps is not None and total_steps > 0:
                flying_cubes.append({"idx": 0})
            prev_utf8_byte = None
        else:
            # запоминаем текущий байт как потенциальное начало многобайтового символа
            prev_utf8_byte = bval

    glutPostRedisplay()


def special_keys(key, x, y):
    global cam_theta, cam_phi
    if key == GLUT_KEY_LEFT:
        cam_theta += 5.0
    elif key == GLUT_KEY_RIGHT:
        cam_theta -= 5.0
    elif key == GLUT_KEY_UP:
        cam_phi = min(85.0, cam_phi + 3.0)
    elif key == GLUT_KEY_DOWN:
        cam_phi = max(-5.0, cam_phi - 3.0)
    glutPostRedisplay()


def timer(value):
    global frame_idx, total_steps, flying_cubes
    if total_steps is not None and total_steps > 0:
        frame_idx = (frame_idx + 1) % total_steps

        # обновляем локальный кадр для каждого куба
        new_list = []
        for cube in flying_cubes:
            cube["idx"] += 1
            if cube["idx"] < total_steps:
                new_list.append(cube)
            # если дошёл до конца траектории (после падения) — исчезает
        flying_cubes = new_list

    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


def init_glut():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(900, 700)
    glutCreateWindow(b"Inverse problem: full cube trajectory")
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.08, 1.0)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)
    glutTimerFunc(0, timer, 0)


# ---------------- main: синтетический тест + визуализация ----------------

def fmt_vec(v, prec=6):
    return "  ".join(f"{x:.{prec}f}" for x in np.asarray(v))


def main():
    global full_traj_cm_true, full_traj_vertices_true
    global traj_cm_est, traj_vertices_est, total_steps, cm_obs, obs_vertices, view_mode

    np.random.seed(0)

    print("Генерируем синтетические данные по истинной модели...")
    full_traj_cm_true, full_traj_vertices_true, V_obs = generate_synthetic_vertex_data(
        dt=DT,
        n_steps=N_STEPS,
        npre=NPRE,
        noise_std_vertices=NOISE_STD_VERTICES
    )

    # сохраняем наблюдаемые вершины и их центр масс
    obs_vertices = V_obs
    cm_obs = compute_com_traj_from_vertices(V_obs)

    total_steps = NPRE + N_STEPS

    print("Запускаем обратную задачу по траектории вершин...")
    result = inverse_from_vertex_trajectories(V_obs)

    p_est = result["p_est"]
    size_est = result["size_est"]
    q_init = result["q_init"]
    L_init = result["L_init"]
    traj_cm_est = result["traj_cm_est"]
    traj_vertices_est = result["traj_vertices_est"]

    I_true = ((2.0 * TRUE_SIZE) ** 2) / 6.0
    L_true = I_true * TRUE_OMEGA

    print("\nИстинные параметры тела в начале эксперимента:")
    print(f"  r0_true   = {fmt_vec(TRUE_R0)}")
    print(f"  l0_true   = {fmt_vec(TRUE_L0)}")
    print(f"  q0_true   = {fmt_vec(TRUE_Q0)}")
    print(f"  size_true = {TRUE_SIZE:.6f}")
    print(f"  L_true    = {fmt_vec(L_true)}")

    print("\nВосстановленные параметры тела:")
    print(f"  r0_est    = {fmt_vec(p_est[0:3])}")
    print(f"  l0_est    = {fmt_vec(p_est[3:6])}")
    print(f"  q0_est    = {fmt_vec(q_init)}")
    print(f"  size_est  = {size_est:.6f}")
    print(f"  L_est     = {fmt_vec(L_init)}")

    # по умолчанию показываем режим 5: траектории вершин; кубы запускаются на 'k'
    view_mode = 5

    init_glut()
    glutMainLoop()


if __name__ == "__main__":
    main()
