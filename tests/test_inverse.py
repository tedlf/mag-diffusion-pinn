import numpy as np
import pytest

from mag_diffusion_pinn.inverse import (InverseRun, crb_theta, forward, ls_fit, make_probes, synth)


def test_forward_is_monotone_in_theta():
    # faster diffusion -> larger field at a fixed probe and time
    vals = [float(forward(0.5, 0.3, th)) for th in (0.3, 0.6, 1.0)]
    assert vals[0] < vals[1] < vals[2]


def test_make_probes_layout():
    xi, t = make_probes([0.2, 0.5], n_times=4, t_max=0.4)
    assert xi.shape == (8,) and set(np.round(xi, 3)) == {0.2, 0.5}
    assert np.allclose(np.unique(t), [0.1, 0.2, 0.3, 0.4])


def test_ls_recovers_theta_noise_free():
    xi, t = make_probes([0.2, 0.5, 0.8])
    clean = forward(xi, t, 0.6)
    th, se = ls_fit(xi, t, clean, theta0=1.0)
    assert abs(th - 0.6) < 1e-6


def test_ls_scatter_matches_cramer_rao():
    xi, t = make_probes([0.3, 0.7])
    th_true, sigma = 0.6, 0.01
    est = np.array([ls_fit(xi, t, synth(xi, t, th_true, sigma, np.random.default_rng(i))[1])[0] for i in range(60)])
    crb = crb_theta(xi, t, th_true, sigma)
    assert abs(est.mean() - th_true) < 3 * crb / np.sqrt(len(est)) + 1e-3     # unbiased within sampling error
    assert 0.7 < est.std(ddof=1) / crb < 1.35


def test_crb_scales_linearly_with_noise():
    xi, t = make_probes([0.5])
    assert abs(crb_theta(xi, t, 0.6, 0.02) / crb_theta(xi, t, 0.6, 0.01) - 2.0) < 1e-6


@pytest.mark.slow
def test_inverse_run_smoke():
    xi, t = make_probes([0.3, 0.6], n_times=8)
    y = forward(xi, t, 0.6)
    run = InverseRun(xi, t, y, seed=0, n_f=128, n_f_lbfgs=256)
    run.adam(40, 1000.0, 100.0, log_every=20)
    run.lbfgs(25, 100.0)
    r = run.result()
    assert np.isfinite(r["theta"]) and r["theta"] > 0
    assert len(r["hist"]["theta"]) >= 3
    assert r["hist"]["data"][-1] < r["hist"]["data"][0]
