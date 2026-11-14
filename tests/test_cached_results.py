"""Headline numbers of the cached training runs in results/.

Two kinds of test:

* `test_*_committed_numbers` pin the numbers quoted in the README to the committed cache. They fail, by design,
  if the cache is regenerated (training is not bitwise reproducible across machines). After regenerating, run
  with MDP_REGENERATED=1 to skip them.
* `test_*_qualitative` check the conclusions of the study, with wide margins. A regenerated cache should pass these.
"""
import os

import numpy as np
import pytest

from mag_diffusion_pinn.experiments import M2_JOBS, M3_JOBS, M4_JOBS, load_runs
from mag_diffusion_pinn.inverse import crb_theta, make_probes

regenerated = pytest.mark.skipif(os.environ.get("MDP_REGENERATED") == "1",
                                 reason="cache was regenerated; the committed numbers no longer apply")


def _load(name, results_dir):
    try:
        return load_runs(name, results_dir)
    except FileNotFoundError:
        pytest.skip(f"results/{name} cache not present")


def _rel(runs, prob, cfg, seeds):
    return np.array([runs[(prob, cfg, s)]["metrics"]["rel_l2"] for s in seeds])


@pytest.fixture(scope="module")
def fwd(results_dir):
    m2, m3 = _load("m2", results_dir), _load("m3", results_dir)
    plain_A = np.array([m2[("A", "float32", s)]["metrics"]["rel_l2"] for s in (0, 1, 2)]
                       + list(_rel(m3, "A", "plain", (3, 4, 5))))
    return dict(m3=m3, plain_A=plain_A, hard_A=_rel(m3, "A", "hard", range(6)),
                plain_R=_rel(m3, "R", "plain", range(6)), hard_R=_rel(m3, "R", "hard", range(6)))


def test_caches_are_complete(results_dir):
    assert set(_load("m2", results_dir)) >= set(M2_JOBS)
    assert set(_load("m3", results_dir)) >= set(M3_JOBS)
    assert set(_load("m4", results_dir)) >= set(M4_JOBS)


@regenerated
def test_forward_committed_numbers(fwd):
    assert 1.0e-2 < np.median(fwd["plain_A"]) < 1.8e-2
    assert 2.5e-3 < np.median(fwd["hard_A"]) < 5e-3
    assert 8e-5 < np.median(fwd["hard_R"]) < 1.5e-4
    assert fwd["hard_R"].max() < 3e-4 and fwd["plain_R"].min() > fwd["hard_R"].max()
    assert 20 < np.median(fwd["plain_R"]) / np.median(fwd["hard_R"]) < 45


def test_forward_qualitative(fwd):
    """Conclusions that should survive retraining: hard constraints help, hugely on the smooth ramp problem."""
    assert np.median(fwd["hard_A"]) < np.median(fwd["plain_A"]) / 2
    assert np.median(fwd["hard_R"]) < np.median(fwd["plain_R"]) / 10
    assert np.median(fwd["hard_R"]) < 1e-3 and np.median(fwd["hard_A"]) < 1e-2
    assert np.median(fwd["plain_A"]) < 5e-2 and np.median(fwd["plain_R"]) < 5e-2
    assert all(np.isfinite(v["metrics"]["rel_l2"]) for v in fwd["m3"].values())


@regenerated
def test_inverse_committed_numbers(results_dir):
    runs = _load("m4", results_dir)
    assert abs(runs[("p3", 0.0, 0)]["theta"] - 0.6) / 0.6 < 1e-3          # noise-free recovery
    th = np.array([runs[("p3", 0.01, s)]["theta"] for s in range(3)])
    assert np.all(np.abs(th - 0.6) / 0.6 < 0.015)


def test_inverse_qualitative(results_dir):
    runs = _load("m4", results_dir)
    assert len(runs) == 28
    assert abs(runs[("p3", 0.0, 0)]["theta"] - 0.6) / 0.6 < 0.01          # noise-free: within 1%
    xi, t = make_probes([0.2, 0.5, 0.8])
    crb = crb_theta(xi, t, 0.6, 0.01)
    assert abs(crb - 0.0055) < 2e-4                                       # analytic, independent of training
    th = np.array([runs[("p3", 0.01, s)]["theta"] for s in range(3)])
    assert np.all(np.abs(th - 0.6) < 5 * crb)                             # within a few CRB standard deviations
    assert all(np.isfinite(v["theta"]) and 0.3 < v["theta"] < 1.2 for v in runs.values())   # none diverged
