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


def levenberg_marquardt_cm(
        y_obs_cm, dt, p0, npre=10, max_iter=100, λ=1e-3, tol=1e-8
):
    p = p0.copy()

    for k in range(max_iter):
        J, r = jacobian_fd_cm(p, y_obs_cm, dt, npre)
        grad = J.T @ r
        H = J.T @ J
        H_reg = H + λ * np.diag(np.diag(H))

        try:
            Δp = np.linalg.solve(H_reg, -grad)
        except:
            λ *= 10
            continue

        # НОРМАЛИЗУЕМ шаг, если он слишком маленький
        Δp_norm = np.linalg.norm(Δp)
        if Δp_norm < 1e-10:
            print(f"Шаг слишком мал ({Δp_norm:.1e}), считаем сходимость")
            break

        p_new = p + Δp
        r_new = residuals_cm(p_new, y_obs_cm, dt, npre)

        cost_old = 0.5 * np.sum(r ** 2)
        cost_new = 0.5 * np.sum(r_new ** 2)

        # ЗАЩИТА от деления на ноль и огромных ρ
        if cost_old - cost_new > 0:
            # Шаг улучшает - принимаем
            p, r = p_new, r_new
            λ = max(1e-12, λ * 0.1)

            print(f"Iter {k}: cost={cost_new:.3e}, λ={λ:.1e}, "
                  f"Δcost={cost_old - cost_new:.1e} ✓")
        else:
            # Шог не улучшает - отвергаем
            λ = min(1e12, λ * 10.0)
            print(f"Iter {k}: шаг отвергнут, λ={λ:.1e}")

        # Критерии остановки
        if np.linalg.norm(grad) < tol:
            print(f"✓ Сошелся по градиенту")
            break
        if Δp_norm < tol * np.linalg.norm(p):
            print(f"✓ Сошелся по изменению параметров")
            break

    return p