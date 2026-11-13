"""Crank-Nicolson finite-difference solvers for  b_t = theta * b_xixi  on (0, 1)."""
import numpy as np
from scipy.linalg import solve_banded

from .reference import TAU_R, ramp_g


def fd_solve(N, nsteps, tau_end, variant="S", rannacher=False, return_history=False):
    """Problems S or A with N space intervals and nsteps time steps up to tau_end (theta = 1).

    Returns (xi, b at tau_end); with return_history also (xi, times, array of shape (n+1, N+1)).
    `rannacher=True` replaces the first two Crank-Nicolson steps by pairs of half-size backward-Euler steps.
    """
    h = 1.0 / N
    xi = np.linspace(0.0, 1.0, N + 1)
    left = 1.0 if variant == "A" else 0.0
    right = 0.0
    u = np.sin(np.pi * xi) if variant == "S" else np.zeros_like(xi)
    u[0], u[-1] = left, right
    m = N - 1

    def step(u, dt, th):
        r = dt / h ** 2
        ab = np.zeros((3, m))
        ab[0, 1:] = -th * r
        ab[1, :] = 1 + 2 * th * r
        ab[2, :-1] = -th * r
        ui = u[1:-1]
        rhs = ui + (1 - th) * r * (u[:-2] - 2 * ui + u[2:])
        rhs[0] += th * r * left
        rhs[-1] += th * r * right
        new = u.copy()
        new[1:-1] = solve_banded((1, 1), ab, rhs)
        new[0], new[-1] = left, right
        return new

    dt = tau_end / nsteps
    times, hist, t = [0.0], [u.copy()], 0.0
    n_start = 2 if (rannacher and nsteps >= 4) else 0
    for _ in range(n_start):
        for _ in range(2):
            u = step(u, dt / 2, 1.0)
            t += dt / 2
            times.append(t); hist.append(u.copy())
    for _ in range(nsteps - n_start):
        u = step(u, dt, 0.5)
        t += dt
        times.append(t); hist.append(u.copy())
    if return_history:
        return xi, np.array(times), np.array(hist)
    return xi, u


def fd_ramp(N, nsteps, t_end, theta=1.0, tau_r=TAU_R):
    """Problem R (smooth ramp drive at xi = 0), Crank-Nicolson with the new-time boundary value.

    Returns (xi, times, history) with history of shape (nsteps + 1, N + 1).
    """
    h = 1.0 / N
    dt = t_end / nsteps
    r = theta * dt / h ** 2
    n = N - 1
    ab = np.zeros((3, n))
    ab[0, 1:] = -r / 2
    ab[1] = 1 + r
    ab[2, :-1] = -r / 2
    b = np.zeros(N + 1)
    hist, times = [b.copy()], [0.0]
    for k in range(nsteps):
        t1 = (k + 1) * dt
        g1 = float(ramp_g(t1, tau_r))
        rhs = (1 - r) * b[1:N] + r / 2 * (b[0:N - 1] + b[2:N + 1])
        rhs[0] += r / 2 * g1
        b[0] = g1
        b[1:N] = solve_banded((1, 1), ab, rhs)
        hist.append(b.copy()); times.append(t1)
    return np.linspace(0.0, 1.0, N + 1), np.array(times), np.array(hist)
