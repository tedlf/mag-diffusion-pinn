"""Reference solutions of  b_t = theta * b_xixi  on 0 < xi < 1  (nondimensional; theta = 1 unless stated).

Problems
--------
S : b(xi, 0) = sin(pi xi), b(0, t) = b(1, t) = 0.            b = sin(pi xi) exp(-pi^2 t)
A : b(xi, 0) = 0, b(0, t) = 1, b(1, t) = 0   (step drive).   series and method-of-images forms
R : b(xi, 0) = 0, b(0, t) = g(t), b(1, t) = 0  with g a quintic smoothstep ramp over tau_r.
    Computed by Duhamel's principle from the step solution A.
"""
import numpy as np
from scipy.special import erfc

TAU_MAX = 0.5
TAU_R = 0.05


def exact_S(xi, tau):
    xi = np.asarray(xi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return np.sin(np.pi * xi) * np.exp(-np.pi ** 2 * tau)


def exact_A_series(xi, tau, nmax=4000):
    """Fourier series for the step problem; slow to converge at small tau."""
    xi = np.asarray(xi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n = np.arange(1, nmax + 1)
    xi_b, tau_b = np.broadcast_arrays(xi, tau)
    flat_xi, flat_tau = xi_b.ravel(), tau_b.ravel()
    out = np.empty_like(flat_xi)
    for start in range(0, flat_xi.size, 2000):  # chunked to bound memory
        sl = slice(start, start + 2000)
        terms = (2.0 / (n * np.pi)) * np.sin(np.outer(flat_xi[sl], n) * np.pi) \
            * np.exp(-np.outer(flat_tau[sl], n ** 2) * np.pi ** 2)
        out[sl] = 1.0 - flat_xi[sl] - terms.sum(axis=1)
    return out.reshape(xi_b.shape)


def exact_A_images(xi, tau, kmax=60):
    """Method of images for the step problem; accurate at small tau."""
    xi = np.asarray(xi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    s = 2.0 * np.sqrt(np.maximum(tau, 1e-300))
    xi_b, s_b = np.broadcast_arrays(xi, s)
    total = np.zeros_like(xi_b, dtype=float)
    for k in range(kmax + 1):
        total += erfc((2 * k + xi_b) / s_b) - erfc((2 * k + 2 - xi_b) / s_b)
    return total


def ramp_g(t, tau_r=TAU_R):
    """Quintic smoothstep: 0 at t = 0, 1 for t >= tau_r, with zero first and second derivative at both ends."""
    x = np.clip(np.asarray(t, dtype=float) / tau_r, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


def ramp_gprime(t, tau_r=TAU_R):
    t = np.asarray(t, dtype=float)
    x = np.clip(t / tau_r, 0.0, 1.0)
    return 30 * x ** 2 * (1 - x) ** 2 / tau_r * ((t > 0) & (t < tau_r))


def ramp_response(xi, t, theta=1.0, tau_r=TAU_R, n_gl=300):
    """b(xi, t) for problem R with diffusivity ratio theta, by Duhamel's principle:

        b(xi, t) = integral_0^{min(t, tau_r)} g'(s) A(xi, theta (t - s)) ds

    where A is the step solution. The ramp g is fixed in the time variable t (so it does not
    depend on theta). Gauss-Legendre quadrature with n_gl nodes. `xi` and `t` broadcast.
    """
    xi, t = np.broadcast_arrays(np.asarray(xi, dtype=float), np.asarray(t, dtype=float))
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)
    out = np.zeros(t.shape)
    t_flat, xi_flat, out_flat = t.ravel(), xi.ravel(), out.ravel()
    for tv in np.unique(t_flat):
        idx = np.where(t_flat == tv)[0]
        T = min(tv, tau_r)
        if T <= 0:
            continue
        s = 0.5 * T * (x_gl + 1)
        ws = 0.5 * T * w_gl * ramp_gprime(s, tau_r)
        u = np.maximum(theta * (tv - s), 1e-300)
        vals = exact_A_images(xi_flat[idx][:, None], u[None, :], kmax=30)
        out_flat[idx] = vals @ ws
    return out_flat.reshape(t.shape)


def ramp_reference_grid(xi_grid, tau_grid, theta=1.0, tau_r=TAU_R, n_gl=600):
    """Problem-R reference on a grid; result has shape (len(tau_grid), len(xi_grid))."""
    XI, T = np.meshgrid(np.asarray(xi_grid, float), np.asarray(tau_grid, float))
    return ramp_response(XI, T, theta=theta, tau_r=tau_r, n_gl=n_gl)
