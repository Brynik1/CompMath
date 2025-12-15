from gauss_newton_utils import *


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
    return np.array(
        [
            [s, s, -s],
            [-s, s, -s],
            [-s, s, s],
            [s, s, s],
            [s, -s, s],
            [-s, -s, s],
            [-s, -s, -s],
            [s, -s, -s],
        ],
        dtype=float,
    )


def kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def estimate_orientation_sequence_from_vertices(
    V_obs: np.ndarray, size_est: float
) -> np.ndarray:
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
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


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
    axis /= 2.0 * np.sin(angle)
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


def estimate_initial_rotation_and_L0(
    V_obs: np.ndarray, size_est: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
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


def shift_orientation_back_to_start(
    q_obs0: np.ndarray, omega: np.ndarray, npre: int, dt: float
) -> np.ndarray:
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
