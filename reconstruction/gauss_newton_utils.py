import numpy as np

from entities.Context import Context
from entities.RigidBody import RigidBody
from experiment.experiment import rk4_step  # модель динамики


def unpack_params_cm(params: np.ndarray) -> RigidBody:
    rb = RigidBody()
    rb.r = np.array(params[0:3], dtype=float)
    rb.l = np.array(params[3:6], dtype=float)
    return rb


def simulate_observed_traj_cm(
    params: np.ndarray, dt: float, n_steps: int, npre: int = 10
) -> np.ndarray:
    rb = unpack_params_cm(params)
    context = Context()

    for _ in range(npre):
        rk4_step(rb, context, dt)

    traj = np.zeros((n_steps, 3), dtype=float)
    for k in range(n_steps):
        rk4_step(rb, context, dt)
        traj[k] = rb.r.copy()
    return traj


def residuals_cm(
    params: np.ndarray, y_obs_cm: np.ndarray, dt: float, npre: int = 10
) -> np.ndarray:
    y_model = simulate_observed_traj_cm(params, dt, y_obs_cm.shape[0], npre=npre)
    return (y_model - y_obs_cm).ravel()


def jacobian_fd_cm(
    params: np.ndarray,
    y_obs_cm: np.ndarray,
    dt: float,
    npre: int = 10,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
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


def gauss_newton_cm(
    y_obs_cm: np.ndarray,
    dt: float,
    p0: np.ndarray,
    npre: int = 10,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 1.0,
) -> np.ndarray:
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

        print(
            f"GN iter {k}: ||r||={np.linalg.norm(r):.6e}, "
            f"||delta||={np.linalg.norm(delta):.6e}"
        )

        if np.linalg.norm(delta) < tol:
            p = p_new
            break

        p = p_new
    return p
