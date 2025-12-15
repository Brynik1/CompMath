from const import *
from cube_geometry import *

from entities.Context import Context
from entities.RigidBody import RigidBody
from experiment.experiment import *


def reconstruct_full_cube_trajectory(
    p_est: np.ndarray,
    size_est: float,
    q_init: np.ndarray,
    L_init: np.ndarray,
    dt: float,
    n_steps: int,
    npre: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    total_steps = npre + n_steps

    SIZE = float(size_est)

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

    p_est = levenberg_marquardt_cm(
        y_obs_cm=y_obs_cm,
        dt=DT,
        p0=P_INITIAL,
        npre=NPRE,
        max_iter=MAX_ITER,
        tol=TOL,
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
        npre=NPRE,
    )

    return {
        "p_est": p_est,
        "size_est": size_est,
        "q_init": q_init,
        "L_init": L0_obs,
        "traj_cm_est": traj_cm_est,
        "traj_vertices_est": traj_vertices_est,
    }
