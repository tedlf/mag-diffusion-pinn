"""Inverse problem: recover the diffusivity ratio theta in  b_t = theta * b_xixi  from sparse, noisy probe data.

Time t is a fixed reference time (t = eta_ref t_phys / L^2) so theta = eta / eta_ref. The boundary drive g(t) (problem R)
is fixed in these units, so the physical ramp does not change with theta. Three tools:
  * `ls_fit`      nonlinear least squares on the exact forward model (the classical baseline)
  * `crb_theta`   Cramer-Rao standard deviation of theta (the best any unbiased method can do)
  * `InverseRun`  PINN with learnable log(theta) and hard-constrained output
"""
import time
import numpy as np
import torch
from scipy.optimize import least_squares

from .pinn import MLP, field, pde_residual, TAU_MAX
from .reference import ramp_response

T_MAX = TAU_MAX
DEFAULT_SCHEDULE = dict(n_adam=3000, lam_adam=(1000.0, 100.0),
                        lbfgs_stages=((700, 1000.0), (700, 300.0), (700, 100.0)), lr_phi=2e-3)


def forward(xi, t, theta, tau_r=0.05):
    """Exact probe signal b(xi, t; theta) for problem R (no neural network)."""
    return ramp_response(xi, t, theta=theta, tau_r=tau_r, n_gl=300)


def make_probes(xi_probes, n_times=20, t_max=T_MAX):
    """All (xi, t) pairs for the given probe positions at n_times equally spaced times in (0, t_max]."""
    t = t_max * np.arange(1, n_times + 1) / n_times
    XI, T = np.meshgrid(np.asarray(xi_probes, float), t)
    return XI.ravel(), T.ravel()


def synth(xi_d, t_d, theta_true, sigma, rng):
    """Returns (clean signal, signal + Gaussian noise of standard deviation sigma)."""
    clean = forward(xi_d, t_d, theta_true)
    return clean, clean + sigma * rng.standard_normal(clean.shape)


def ls_fit(xi_d, t_d, y, theta0=1.0):
    """Nonlinear least squares in phi = log(theta). Returns (theta_hat, Gauss-Newton standard error of theta)."""
    f = lambda p: forward(xi_d, t_d, np.exp(p[0])) - y
    sol = least_squares(f, [np.log(theta0)], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    th = float(np.exp(sol.x[0]))
    J = sol.jac[:, 0]
    s2 = float(sol.fun @ sol.fun) / max(len(y) - 1, 1)
    return th, th * np.sqrt(s2 / float(J @ J))


def crb_theta(xi_d, t_d, theta_true, sigma):
    """Cramer-Rao standard deviation of theta for i.i.d. Gaussian noise sigma at this probe layout."""
    h = 1e-5
    J = (forward(xi_d, t_d, theta_true * (1 + h)) - forward(xi_d, t_d, theta_true * (1 - h))) / (2 * h * theta_true)
    return sigma / np.sqrt(float(J @ J))


class InverseRun:
    """Joint training of the network and log(theta) from probe data: an Adam stage, then staged L-BFGS.

    The data weight lam multiplies the mean squared data misfit; the PDE residual has weight 1.
    """

    def __init__(self, xi_d, t_d, y, seed=0, dtype=torch.float32, phi0=0.0, n_f=2048, n_f_lbfgs=8192):
        self.dtype, self.n_f, self.n_f_lbfgs = dtype, n_f, n_f_lbfgs
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.model = MLP().to(dtype)
        self.phi = torch.tensor([phi0], dtype=dtype, requires_grad=True)
        self.xd = torch.tensor(np.asarray(xi_d).reshape(-1, 1), dtype=dtype)
        self.td = torch.tensor(np.asarray(t_d).reshape(-1, 1), dtype=dtype)
        self.yd = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=dtype)
        self.hist = dict(step=[], theta=[], pde=[], data=[], phase=[], lam=[])
        self.step = 0
        self.t0 = time.perf_counter()

    @property
    def theta(self):
        return float(torch.exp(self.phi).detach())

    def pts(self, n):
        return (torch.rand(n, 1, generator=self.gen, dtype=self.dtype),
                T_MAX * torch.rand(n, 1, generator=self.gen, dtype=self.dtype))

    def terms(self, xf, tf):
        r = pde_residual(self.model, xf, tf, "R", True, theta=torch.exp(self.phi))
        data = ((field(self.model, self.xd, self.td, "R", True) - self.yd) ** 2).mean()
        return (r ** 2).mean(), data

    def _log(self, pde, data, phase, lam):
        h = self.hist
        h["step"].append(self.step); h["theta"].append(self.theta)
        h["pde"].append(float(pde)); h["data"].append(float(data)); h["phase"].append(phase); h["lam"].append(lam)

    def adam(self, n_steps, lam0, lam1, lr=1e-3, lr_phi=2e-3, log_every=100):
        opt = torch.optim.Adam([dict(params=self.model.parameters(), lr=lr), dict(params=[self.phi], lr=lr_phi)])
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1.0 / max(n_steps, 1)))
        for k in range(n_steps):
            xf, tf = self.pts(self.n_f)
            pde, data = self.terms(xf, tf)
            lam = lam0 * (lam1 / lam0) ** (k / max(n_steps - 1, 1))
            loss = pde + lam * data
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); self.step += 1
            if k % log_every == 0:
                self._log(pde.detach(), data.detach(), "adam", lam)

    def lbfgs(self, n_iter, lam, theta_free=True):
        xf, tf = self.pts(self.n_f_lbfgs)
        params = list(self.model.parameters()) + ([self.phi] if theta_free else [])
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=25, history_size=50, tolerance_grad=1e-14,
                               tolerance_change=1e-16, line_search_fn="strong_wolfe")
        last = {}

        def closure():
            lb.zero_grad()
            pde, data = self.terms(xf, tf)
            loss = pde + lam * data
            loss.backward(); last["t"] = (float(pde.detach()), float(data.detach()))
            return loss

        for _ in range(max(n_iter // 25, 1)):
            lb.step(closure); self.step += 25
            self._log(last["t"][0], last["t"][1], "lbfgs", lam)

    def result(self):
        h = self.hist
        return dict(theta=self.theta, wall_s=time.perf_counter() - self.t0, hist=h, pde=h["pde"][-1], data=h["data"][-1])


def run_job(xi_d, t_d, y, seed, schedule=DEFAULT_SCHEDULE):
    """One inverse run with the schedule used in the notebook (about 5 minutes on one CPU core)."""
    run = InverseRun(xi_d, t_d, y, seed=seed)
    run.adam(schedule["n_adam"], *schedule["lam_adam"], lr_phi=schedule["lr_phi"])
    for n_it, lam in schedule["lbfgs_stages"]:
        run.lbfgs(n_it, lam)
    return run.result()
