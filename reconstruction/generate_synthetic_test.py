from const import *
from cube_geometry import *

from experiment import experiment


def generate_synthetic_vertex_data(
    dt: float, n_steps: int, npre: int, noise_std_vertices: float = 0.0
):
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

    V_obs = full_traj_vertices_true[npre : npre + n_steps].copy()

    if noise_std_vertices > 0.0:
        noise = np.random.normal(scale=noise_std_vertices, size=V_obs.shape)
        V_obs += noise

    return full_traj_cm_true, full_traj_vertices_true, V_obs
