import numpy as np
import pytest
import torch

from mag_diffusion_pinn import pinn
from mag_diffusion_pinn.reference import exact_S, ramp_reference_grid


class ExactDecay(torch.nn.Module):
    """b = sin(pi xi) exp(-pi^2 theta t): an exact solution, used to test the autograd residual."""

    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def forward(self, xi, t):
        return torch.sin(np.pi * xi) * torch.exp(-np.pi ** 2 * self.theta * t)


def test_residual_operator_vanishes_on_exact_solution():
    gen = torch.Generator().manual_seed(0)
    xi, t = pinn.uniform_points(200, torch.float64, gen)
    for theta in (1.0, 0.6):
        r = pinn.pde_residual(ExactDecay(theta), xi, t, "S", False, theta=theta)
        assert r.abs().max() < 1e-10
    # and is large for the wrong theta
    r = pinn.pde_residual(ExactDecay(1.0), xi, t, "S", False, theta=0.6)
    assert r.abs().max() > 0.5


def test_hard_constraints_exact_for_random_network():
    torch.manual_seed(3)
    model = pinn.MLP()
    xi = torch.rand(50, 1); t = 0.5 * torch.rand(50, 1)
    zero, one = torch.zeros_like(xi), torch.ones_like(xi)
    # R: left value g(t), right 0, initial 0
    assert torch.allclose(pinn.field(model, zero, t, "R", True), pinn.g_t(t), atol=1e-6)
    assert torch.allclose(pinn.field(model, one, t, "R", True), zero, atol=1e-6)
    assert torch.allclose(pinn.field(model, xi, zero, "R", True), zero, atol=1e-6)
    # A: left 1, right 0
    assert torch.allclose(pinn.field(model, zero, t, "A", True), one, atol=1e-6)
    assert torch.allclose(pinn.field(model, one, t, "A", True), zero, atol=1e-6)
    with pytest.raises(ValueError):
        pinn.field(model, xi, t, "S", True)


def _grid():
    xi = np.linspace(0, 1, 21); tau = np.linspace(0.025, 0.5, 20)
    return xi, tau


@pytest.mark.slow
def test_smoke_train_S_reduces_error():
    xi, tau = _grid()
    ref = exact_S(xi[None, :], tau[:, None])
    model, hist, m, pred = pinn.train("S", {}, ref, xi, tau, seed=0, n_adam=150, n_lbfgs=50, n_f=256, n_b=32, n_i=32,
                                      n_f_lbfgs=512, log_every=50)
    assert pred.shape == ref.shape and np.isfinite(m["rel_l2"])
    assert hist["rel_l2"][-1] < hist["rel_l2"][0]
    assert m["rel_l2"] < 0.8 * hist["rel_l2"][0]          # a clear reduction, not a convergence claim


@pytest.mark.slow
def test_smoke_train_R_hard_options_run():
    xi, tau = _grid()
    ref = ramp_reference_grid(xi, tau, n_gl=100)
    for cfg in (dict(hard=True), dict(hard=True, weights="manual"), dict(weights="adaptive", lam_max=100.0),
                dict(sampling="rad"), dict(causal=10.0)):
        model, hist, m, pred = pinn.train("R", cfg, ref, xi, tau, seed=1, n_adam=60, n_lbfgs=25, n_f=128, n_b=16,
                                          n_i=16, n_f_lbfgs=256, log_every=30, rad_every=20, adapt_every=20)
        assert np.isfinite(m["rel_l2"]) and np.isfinite(m["res_mse"]), cfg
        if cfg.get("hard"):
            assert m["bc_err"] < 1e-5                      # boundary value exact by construction


def test_training_is_reproducible_for_fixed_seed():
    xi, tau = _grid()
    ref = exact_S(xi[None, :], tau[:, None])
    kw = dict(seed=5, n_adam=20, n_lbfgs=0, n_f=64, n_b=8, n_i=8, n_f_lbfgs=64, log_every=10)
    a = pinn.train("S", {}, ref, xi, tau, **kw)[3]
    b = pinn.train("S", {}, ref, xi, tau, **kw)[3]
    assert np.array_equal(a, b)
