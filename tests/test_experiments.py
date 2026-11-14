"""The experiment definitions and the resumable runner (tiny budgets, seconds per run)."""
import os
import pickle

import numpy as np
import pytest

from mag_diffusion_pinn import experiments as E


# ---- job lists ------------------------------------------------------------------------------------------
def test_job_lists():
    assert len(E.M2_JOBS) == 8 and len(set(E.M2_JOBS)) == 8
    assert len(E.M3_JOBS) == 63 and len(set(E.M3_JOBS)) == 63
    assert len(E.M4_JOBS) == 28 and len(set(E.M4_JOBS)) == 28
    assert len(E.LS_CASES) == 9
    # plain-PINN runs of problem A with seeds 0-2 belong to the M2 set, not to M3
    assert not any(j[0] == "A" and j[1] == "plain" and j[2] < 3 for j in E.M3_JOBS)
    assert {j[1] for j in E.M3_JOBS} <= set(E.M3_CONFIGS)
    assert all(j[2] in (0, 1, 2, 3, 4, 5) for j in E.M3_JOBS)


def test_reference_fields_shapes_and_boundaries():
    for prob in "SAR":
        ref = E.reference_field(prob)
        assert ref.shape == (len(E.GRID_TAU), len(E.GRID_XI))
        assert np.all(np.isfinite(ref))
    assert np.allclose(E.reference_field("S")[:, [0, -1]], 0, atol=1e-12)
    assert np.allclose(E.reference_field("R")[:, -1], 0, atol=1e-12)


def test_data_for_is_reproducible_and_noise_scales():
    xd, td, clean, y = E.data_for("p3", 0.01, 0)
    _, _, _, y2 = E.data_for("p3", 0.01, 0)
    _, _, _, y3 = E.data_for("p3", 0.01, 1)
    assert np.array_equal(y, y2) and not np.array_equal(y, y3)
    assert abs(np.std(y - clean) - 0.01) < 0.003
    assert np.array_equal(E.data_for("p3", 0.0, 0)[2], E.data_for("p3", 0.0, 0)[3])


# ---- runner mechanics (trivial job function, no training) -------------------------------------------------
def _square(k):
    return {"k": k, "sq": k * k}


def _square_marked(k):                      # result differs from _square, to tell old from new
    return {"k": k, "sq": -k * k}


def _boom_on_3(k):
    if k == 3:
        raise RuntimeError("boom")
    return _square_marked(k)


def test_run_without_force_loads_and_fills_missing(tmp_path):
    cache = tmp_path / "c.pkl"
    pickle.dump({1: _square(1), 2: {"k": 2, "sq": "old"}}, open(cache, "wb"))
    out = E.run_jobs([1, 2, 3], _square_marked, cache, verbose=False)
    assert out[2]["sq"] == "old"                                  # existing result kept
    assert out[3]["sq"] == -9                                     # missing result computed
    assert pickle.load(open(cache, "rb")) == out


def test_force_rerun_replaces_everything_atomically(tmp_path):
    cache = tmp_path / "c.pkl"
    pickle.dump({k: _square(k) for k in (1, 2, 3)}, open(cache, "wb"))
    with pytest.raises(RuntimeError):
        E.run_jobs([1, 2, 3], _boom_on_3, cache, force=True, verbose=False)
    # interrupted: the old cache is untouched, partial results are in the .partial file
    assert pickle.load(open(cache, "rb"))[1]["sq"] == 1
    partial = tmp_path / "c.pkl.partial"
    assert set(pickle.load(open(partial, "rb"))) == {1, 2}
    # rerun resumes from the partial file (jobs 1 and 2 are not repeated), then replaces the cache
    calls = []
    def tracker(k):
        calls.append(k)
        return _square_marked(k)
    out = E.run_jobs([1, 2, 3], tracker, cache, force=True, verbose=False)
    assert calls == [3]
    assert not partial.exists()
    assert pickle.load(open(cache, "rb")) == out and all(v["sq"] < 0 for v in out.values())


def test_fresh_discards_partial(tmp_path):
    cache = tmp_path / "c.pkl"
    with pytest.raises(RuntimeError):
        E.run_jobs([1, 2, 3], _boom_on_3, cache, force=True, verbose=False)
    calls = []
    def tracker(k):
        calls.append(k)
        return _square(k)
    E.run_jobs([1, 2, 3], tracker, cache, force=True, fresh=True, verbose=False)
    assert calls == [1, 2, 3]


def test_workers_give_same_results_as_serial(tmp_path):
    a = E.run_jobs([1, 2, 3, 4], _square, tmp_path / "a.pkl", workers=1, verbose=False)
    b = E.run_jobs([1, 2, 3, 4], _square, tmp_path / "b.pkl", workers=2, verbose=False)
    assert a == b


# ---- real jobs with a tiny budget -----------------------------------------------------------------------
def test_quick_m2_m3_jobs_run_and_cache(tmp_path):
    jobs2 = E.M2_JOBS[:1]
    r = E.run_jobs(jobs2, lambda k: E.m2_job(k, quick=True), tmp_path / "m2.pkl", verbose=False)
    res = r[jobs2[0]]
    assert set(res) >= {"hist", "metrics", "pred", "cfg"}
    assert res["pred"].shape == (len(E.GRID_TAU), len(E.GRID_XI)) and res["pred"].dtype == np.float32
    assert np.isfinite(res["metrics"]["rel_l2"])
    m3 = E.m3_job(("R", "hard", 0), quick=True)
    assert np.isfinite(m3["metrics"]["rel_l2"]) and m3["pred"].shape == res["pred"].shape


def test_quick_m4_job_runs(tmp_path):
    res = E.m4_job(("p3", 0.01, 0), quick=True)
    assert np.isfinite(res["theta"]) and set(res) >= {"theta", "pde", "data", "wall_s", "hist", "cfg"}


def test_ls_job_close_to_truth():
    th = E.ls_job(("p3", 0.01), n=20)
    assert th.shape == (20,) and abs(th.mean() - E.THETA_TRUE) < 0.01


def test_run_wrappers_write_expected_files(tmp_path):
    E.run_m4(results_dir=tmp_path, quick=True, jobs=[("p3", 0.0, 0)])
    E.run_m3(results_dir=tmp_path, quick=True, jobs=[("R", "hard", 0)])
    assert {p.name for p in tmp_path.iterdir()} == {"m3_runs.pkl", "m4_runs.pkl"}
