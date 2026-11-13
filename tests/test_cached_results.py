"""Check that the headline numbers quoted in the README are what the cached training runs contain."""
import os
import pickle

import numpy as np
import pytest

from mag_diffusion_pinn.inverse import crb_theta, make_probes

RUN_KEYS = ["m3_A", "m3_R", "m3b_1", "m3b_2", "m3b_3", "m3b_4", "m3b_5", "m3b_6", "m3b_7"]


def _load(results_dir, names):
    runs = {}
    for n in names:
        p = os.path.join(results_dir, n + ".pkl")
        if not os.path.exists(p):
            pytest.skip(f"{p} not present")
        runs.update(pickle.load(open(p, "rb")))
    return runs


def _rel(runs, prob, cfg, seeds):
    return np.array([runs[(prob, cfg, s)]["metrics"]["rel_l2"] for s in seeds])


def test_forward_headlines(results_dir):
    runs = _load(results_dir, RUN_KEYS)
    m2 = _load(results_dir, ["m2_runs"])
    plain_A = [m2[("A", "float32", s)]["metrics"]["rel_l2"] for s in (0, 1, 2)] + \
              [runs[("A", "plain", s)]["metrics"]["rel_l2"] for s in (3, 4, 5)]
    assert 1.0e-2 < np.median(plain_A) < 1.8e-2
    hard_A = _rel(runs, "A", "hard", range(6))
    assert 2.5e-3 < np.median(hard_A) < 5e-3
    assert np.median(hard_A) < np.median(plain_A) / 3
    plain_R = _rel(runs, "R", "plain", range(6))
    hard_R = _rel(runs, "R", "hard", range(6))
    assert 8e-5 < np.median(hard_R) < 1.5e-4
    assert hard_R.max() < 3e-4 and plain_R.min() > hard_R.max()
    assert 20 < np.median(plain_R) / np.median(hard_R) < 45


def test_inverse_headlines(results_dir):
    runs = _load(results_dir, ["m4_0", "m4_1"])
    assert len(runs) == 28
    assert abs(runs[("p3", 0.0, 0)]["theta"] - 0.6) / 0.6 < 1e-3          # noise-free recovery
    th = np.array([runs[("p3", 0.01, s)]["theta"] for s in range(3)])
    assert np.all(np.abs(th - 0.6) / 0.6 < 0.015)
    xi, t = make_probes([0.2, 0.5, 0.8])
    assert abs(crb_theta(xi, t, 0.6, 0.01) - 0.0055) < 2e-4
    assert all(np.isfinite(v["theta"]) and 0.3 < v["theta"] < 1.2 for v in runs.values())   # none diverged
