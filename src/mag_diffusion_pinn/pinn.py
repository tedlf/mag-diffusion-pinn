"""Forward PINN for  b_t = theta * b_xixi  (theta = 1 for the forward problems).

Problems: S (sin initial profile, zero boundaries), A (step at xi = 0), R (smooth ramp at xi = 0).
Options (a dict, see `DEFAULT_CFG`):
  hard      output transformation so boundary (and for R, initial) conditions hold exactly (A and R only)
  weights   'equal' | 'manual' (1, 10, 10) | 'adaptive' (gradient-ratio annealing; `lam_max` caps the weights)
  sampling  'uniform' | 'rad' (residual-adaptive: p ~ |r|/mean + 1, pool refreshed every `rad_every` steps)
  causal    epsilon of causal weighting over 16 time bins (0 = off)
"""
import time
import numpy as np
import torch
import torch.nn as nn

TAU_MAX = 0.5
TAU_R = 0.05
EARLY = 0.05           # "early-time window" tau <= EARLY used for diagnostics
CAUSAL_BINS = 16

DEFAULT_CFG = dict(hard=False, weights="equal", sampling="uniform", causal=0.0, lam_max=float("inf"))


def g_t(t, tau_r=TAU_R):
    x = torch.clamp(t / tau_r, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


PROBLEMS = {
    "S": dict(bc_left=lambda tau: torch.zeros_like(tau), ic=lambda xi: torch.sin(np.pi * xi)),
    "A": dict(bc_left=lambda tau: torch.ones_like(tau), ic=lambda xi: torch.zeros_like(xi)),
    "R": dict(bc_left=g_t, ic=lambda xi: torch.zeros_like(xi)),
}


class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xi, tau):
        return self.net(torch.cat([xi, tau], dim=1))


def field(model, xi, tau, prob, hard):
    """b(xi, tau). With hard=True the boundary (and, for R, initial) conditions hold by construction."""
    n = model(xi, tau)
    if not hard:
        return n
    if prob == "A":      # b(0)=1, b(1)=0 exactly; the initial condition stays a soft loss term
        return (1 - xi) + xi * (1 - xi) * n
    if prob == "R":      # b(0)=g(tau), b(1)=0 and b(xi, 0)=0 exactly
        return g_t(tau) * (1 - xi) + xi * (1 - xi) * tau * n
    raise ValueError("hard constraints are implemented for problems A and R only")


def pde_residual(model, xi, tau, prob="R", hard=True, theta=1.0):
    """r = b_tau - theta * b_xixi by automatic differentiation (theta may be a tensor)."""
    xi = xi.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    b = field(model, xi, tau, prob, hard)
    ones = torch.ones_like(b)
    b_tau = torch.autograd.grad(b, tau, ones, create_graph=True)[0]
    b_xi = torch.autograd.grad(b, xi, ones, create_graph=True)[0]
    b_xixi = torch.autograd.grad(b_xi, xi, torch.ones_like(b_xi), create_graph=True)[0]
    return b_tau - theta * b_xixi


def causal_pde_loss(r, tau, eps, dtype):
    """Causal weighting of the residual over time bins (denser at early time)."""
    edges = TAU_MAX * (torch.arange(CAUSAL_BINS + 1, dtype=dtype) / CAUSAL_BINS) ** 2
    idx = torch.clamp(torch.bucketize(tau.squeeze(1).detach(), edges[1:-1]), 0, CAUSAL_BINS - 1)
    sums = torch.zeros(CAUSAL_BINS, dtype=dtype).index_add(0, idx, (r ** 2).squeeze(1))
    cnt = torch.zeros(CAUSAL_BINS, dtype=dtype).index_add(0, idx, torch.ones(idx.shape[0], dtype=dtype))
    L = sums / cnt.clamp(min=1)
    w = torch.exp(-eps * (torch.cumsum(L, 0) - L)).detach()
    return (w * L).mean()


def loss_terms(model, prob, cfg, pts, dtype):
    """Returns (pde loss, boundary loss, initial loss, plain PDE mean-square residual)."""
    hard = cfg["hard"]
    p = PROBLEMS[prob]
    r = pde_residual(model, pts["xi_f"], pts["tau_f"], prob, hard)
    pde_plain = (r ** 2).mean()
    l_pde = causal_pde_loss(r, pts["tau_f"], cfg["causal"], dtype) if cfg["causal"] > 0 else pde_plain
    zeros = torch.zeros_like(pts["tau_b"]); ones = torch.ones_like(pts["tau_b"])
    l_bc = ((field(model, zeros, pts["tau_b"], prob, hard) - p["bc_left"](pts["tau_b"])) ** 2).mean() \
        + ((field(model, ones, pts["tau_b"], prob, hard)) ** 2).mean()
    l_ic = ((field(model, pts["xi_i"], torch.zeros_like(pts["xi_i"]), prob, hard) - p["ic"](pts["xi_i"])) ** 2).mean()
    return l_pde, l_bc, l_ic, pde_plain.detach()


def uniform_points(n, dtype, gen):
    return torch.rand(n, 1, generator=gen, dtype=dtype), TAU_MAX * torch.rand(n, 1, generator=gen, dtype=dtype)


class Sampler:
    """Uniform collocation points, or residual-adaptive (RAD: p ~ |r|^k / mean + c)."""

    def __init__(self, mode, dtype, gen, n_pool=20000, k=1.0, c=1.0):
        self.mode, self.dtype, self.gen, self.n_pool, self.k, self.c = mode, dtype, gen, n_pool, k, c
        self.pool = None

    def refresh(self, model, prob, hard):
        if self.mode != "rad":
            return
        xi, tau = uniform_points(self.n_pool, self.dtype, self.gen)
        with torch.enable_grad():
            r = pde_residual(model, xi, tau, prob, hard).detach().abs().squeeze(1)
        p = r ** self.k
        p = p / p.mean() + self.c
        self.pool = (xi, tau, (p / p.sum()))

    def interior(self, n):
        if self.mode != "rad" or self.pool is None:
            return uniform_points(n, self.dtype, self.gen)
        xi, tau, p = self.pool
        idx = torch.multinomial(p, n, replacement=True, generator=self.gen)
        return xi[idx], tau[idx]

    def points(self, n_f, n_b, n_i):
        xi_f, tau_f = self.interior(n_f)
        return dict(xi_f=xi_f, tau_f=tau_f,
                    tau_b=TAU_MAX * torch.rand(n_b, 1, generator=self.gen, dtype=self.dtype),
                    xi_i=torch.rand(n_i, 1, generator=self.gen, dtype=self.dtype))


@torch.no_grad()
def predict(model, prob, hard, xi_grid, tau_grid, dtype):
    """Network field on the grid; result has shape (len(tau_grid), len(xi_grid))."""
    XI, TAU = np.meshgrid(xi_grid, tau_grid)
    x = torch.tensor(XI.reshape(-1, 1), dtype=dtype); t = torch.tensor(TAU.reshape(-1, 1), dtype=dtype)
    return field(model, x, t, prob, hard).numpy().reshape(XI.shape).astype(np.float64)


def residual_mse(model, prob, hard, dtype, n=20000, seed=12345):
    gen = torch.Generator().manual_seed(seed)
    xi, tau = uniform_points(n, dtype, gen)
    r = pde_residual(model, xi, tau, prob, hard)
    return float((r.detach().double() ** 2).mean())


def summarize(pred, ref, xi_grid, tau_grid):
    err = pred - ref
    k = np.unravel_index(np.argmax(np.abs(err)), err.shape)
    early = tau_grid <= EARLY
    return dict(rel_l2=float(np.linalg.norm(err) / np.linalg.norm(ref)),
                early_l2=float(np.linalg.norm(err[early]) / np.linalg.norm(ref[early])),
                max_abs=float(np.abs(err).max()),
                max_at=(float(xi_grid[k[1]]), float(tau_grid[k[0]])),
                bc_err=float(np.abs(err[:, 0]).max()))


def train(prob, cfg, ref, xi_grid, tau_grid, seed=0, dtype=torch.float32,
          n_adam=2000, n_lbfgs=2000, lr=1e-3, n_f=2048, n_b=256, n_i=256,
          n_f_lbfgs=8192, log_every=100, rad_every=100, adapt_every=50, verbose=False):
    """Adam (fresh points each step) then L-BFGS (one fixed point set). Returns (model, hist, metrics, pred).

    `ref` is the reference field on the grid, shape (len(tau_grid), len(xi_grid)).
    """
    cfg = dict(DEFAULT_CFG, **cfg)
    hard = cfg["hard"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP().to(dtype)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1.0 / max(n_adam, 1)))
    smp = Sampler(cfg["sampling"], dtype, gen)
    lam = {"manual": [1.0, 10.0, 10.0]}.get(cfg["weights"], [1.0, 1.0, 1.0])
    hist = dict(step=[], pde=[], bc=[], ic=[], rel_l2=[], early_l2=[], lam_bc=[], lam_ic=[], phase=[])
    t0 = time.perf_counter()

    def log(step, terms, phase):
        pred = predict(model, prob, hard, xi_grid, tau_grid, dtype)
        s = summarize(pred, ref, xi_grid, tau_grid)
        hist["step"].append(step); hist["pde"].append(float(terms[3]))
        hist["bc"].append(float(terms[1])); hist["ic"].append(float(terms[2]))
        hist["rel_l2"].append(s["rel_l2"]); hist["early_l2"].append(s["early_l2"])
        hist["lam_bc"].append(lam[1]); hist["lam_ic"].append(lam[2]); hist["phase"].append(phase)

    def gnorm(term):
        if not term.requires_grad:
            return None
        gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        gs = [g.abs().flatten() for g in gs if g is not None]
        return torch.cat(gs) if gs else None

    for step in range(n_adam):
        if step % rad_every == 0:
            smp.refresh(model, prob, hard)
        pts = smp.points(n_f, n_b, n_i)
        terms = loss_terms(model, prob, cfg, pts, dtype)
        if cfg["weights"] == "adaptive" and step % adapt_every == 0 and step > 0:
            gp = gnorm(terms[0])
            for j in (1, 2):
                gj = gnorm(terms[j])
                if gp is not None and gj is not None and gj.mean() > 0:
                    lam[j] = min(cfg["lam_max"], 0.9 * lam[j] + 0.1 * float(gp.max() / gj.mean()))
        loss = lam[0] * terms[0] + lam[1] * terms[1] + lam[2] * terms[2]
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % log_every == 0:
            log(step, [t.detach() for t in terms], "adam")
            if verbose:
                print(step, [float(t) for t in terms], hist["rel_l2"][-1], flush=True)

    smp.refresh(model, prob, hard)
    fixed = smp.points(n_f_lbfgs, n_b * 4, n_i * 4)
    lb = torch.optim.LBFGS(params, lr=1.0, max_iter=25, history_size=50, tolerance_grad=1e-14,
                           tolerance_change=1e-16, line_search_fn="strong_wolfe")
    last = {}

    def closure():
        lb.zero_grad()
        terms = loss_terms(model, prob, cfg, fixed, dtype)
        loss = lam[0] * terms[0] + lam[1] * terms[1] + lam[2] * terms[2]
        loss.backward()
        last["terms"] = [t.detach() for t in terms]
        return loss

    for k in range(n_lbfgs // 25):
        lb.step(closure)
        log(n_adam + 25 * (k + 1), last["terms"], "lbfgs")
        if verbose and k % 8 == 0:
            print(n_adam + 25 * (k + 1), [float(t) for t in last["terms"]], hist["rel_l2"][-1], flush=True)

    wall = time.perf_counter() - t0
    pred = predict(model, prob, hard, xi_grid, tau_grid, dtype)
    m = summarize(pred, ref, xi_grid, tau_grid)
    m["res_mse"] = residual_mse(model, prob, hard, dtype)
    m["wall_s"] = wall
    m["steps"] = n_adam + n_lbfgs
    m["lam"] = list(lam)
    return model, hist, m, pred
