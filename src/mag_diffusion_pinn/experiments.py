"""The experiment sets behind the cached results, and a resumable runner that regenerates them.

Each experiment set is a list of *jobs* (small hashable keys) and a function that runs one job:

    M2  plain PINN on S and A                      results/m2_runs.pkl     (8 runs)
    M3  forward-PINN improvements on A and R       results/m3_runs.pkl     (63 runs)
    M4  PINN inverse runs for the probe study      results/m4_runs.pkl     (28 runs)
        least-squares Monte Carlo baseline         results/m4_ls.pkl       (9 cases x 200 draws)

`run_m2`, `run_m3`, `run_m4`, `run_ls_baseline` load the cache if it is complete, run what is missing, and with
`force=True` rerun everything. Progress is saved after every finished job, so an interrupted run can be resumed.
"""
import functools
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .reference import exact_A_images, exact_S, ramp_reference_grid

THETA_TRUE = 0.6
GRID_XI = np.linspace(0.0, 1.0, 201)
GRID_TAU = np.linspace(0.0025, 0.5, 200)

# ------------------------------------------------------------------------------------------ job lists
M2_JOBS = [(p, "float32", s) for p in ("S", "A") for s in (0, 1, 2)] + [(p, "float64", 0) for p in ("S", "A")]

M3_CONFIGS = {
    "plain": {},
    "hard": dict(hard=True),
    "w_manual": dict(weights="manual"),
    "w_adapt": dict(weights="adaptive"),
    "w_adapt_cap": dict(weights="adaptive", lam_max=100.0),
    "rad": dict(sampling="rad"),
    "causal": dict(causal=10.0),
    "hard_w": dict(hard=True, weights="manual"),
}
# (problem, config, seeds). Plain A seeds 0-2 come from the M2 set, so they are not repeated here.
_M3_SPEC = [
    ("A", "plain", (3, 4, 5)), ("A", "hard", range(6)), ("A", "w_manual", range(6)), ("A", "w_adapt", range(3)),
    ("A", "w_adapt_cap", range(6)), ("A", "rad", range(3)), ("A", "causal", range(3)), ("A", "hard_w", range(6)),
    ("R", "plain", range(6)), ("R", "hard", range(6)), ("R", "w_manual", range(3)), ("R", "w_adapt", range(3)),
    ("R", "w_adapt_cap", range(3)), ("R", "rad", range(3)), ("R", "causal", range(3)),
]
M3_JOBS = [(p, c, s) for p, c, seeds in _M3_SPEC for s in seeds]

LAYOUTS = {"p3": [0.2, 0.5, 0.8], "p1_0.1": [0.1], "p1_0.3": [0.3], "p1_0.5": [0.5], "p1_0.8": [0.8],
           "p5": [0.1, 0.3, 0.5, 0.7, 0.9]}
M4_JOBS = [("p3", 0.0, 0)] + [("p3", s, r) for r in range(3) for s in (0.005, 0.01, 0.02, 0.05)] \
    + [(lay, 0.01, r) for r in range(3) for lay in ("p1_0.1", "p1_0.3", "p1_0.5", "p1_0.8", "p5")]
LS_CASES = [("p3", s) for s in (0.005, 0.01, 0.02, 0.05)] + [(l, 0.01) for l in ("p1_0.1", "p1_0.3", "p1_0.5", "p1_0.8", "p5")]

FULL = dict(n_adam=2000, n_lbfgs=2000)
QUICK = dict(n_adam=40, n_lbfgs=25, n_f=128, n_b=16, n_i=16, n_f_lbfgs=256)      # for tests only: finishes in seconds
M4_SCHEDULE_QUICK = dict(n_adam=30, lam_adam=(1000.0, 100.0), lbfgs_stages=((25, 1000.0), (25, 100.0)), lr_phi=2e-3)


def default_results_dir():
    """`results/` of the repository this package was installed from (editable install), or $MDP_RESULTS."""
    if os.environ.get("MDP_RESULTS"):
        return Path(os.environ["MDP_RESULTS"])
    return Path(__file__).resolve().parents[2] / "results"


# ------------------------------------------------------------------------------------------ reference fields
@functools.lru_cache(maxsize=None)
def reference_field(prob):
    """Reference solution on the (GRID_TAU x GRID_XI) grid; result shape (200, 201)."""
    if prob == "S":
        return exact_S(GRID_XI[None, :], GRID_TAU[:, None])
    if prob == "A":
        return exact_A_images(GRID_XI[None, :], GRID_TAU[:, None])
    if prob == "R":
        return ramp_reference_grid(GRID_XI, GRID_TAU, n_gl=600)
    raise ValueError(prob)


# ------------------------------------------------------------------------------------------ job functions
def m2_job(key, quick=False):
    import torch
    from . import pinn
    prob, dtype_name, seed = key
    budget = dict(QUICK if quick else FULL)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    _, hist, m, pred = pinn.train(prob, {}, reference_field(prob), GRID_XI, GRID_TAU, seed=seed, dtype=dtype, **budget)
    return dict(hist=hist, metrics=m, pred=pred.astype(np.float32), cfg=budget)


def m3_job(key, quick=False):
    from . import pinn
    prob, name, seed = key
    budget = dict(QUICK if quick else FULL)
    cfg = M3_CONFIGS[name]
    _, hist, m, pred = pinn.train(prob, cfg, reference_field(prob), GRID_XI, GRID_TAU, seed=seed, **budget)
    return dict(hist=hist, metrics=m, pred=pred.astype(np.float32), cfg=cfg)


def data_for(layout, sigma, r, theta_true=THETA_TRUE):
    """Probe data for one noise realization: returns (xi, t, clean signal, noisy signal)."""
    from .inverse import forward, make_probes
    xd, td = make_probes(LAYOUTS[layout])
    clean = forward(xd, td, theta_true)
    y = clean + sigma * np.random.default_rng(100 + r).standard_normal(clean.shape)
    return xd, td, clean, y


def m4_job(key, quick=False):
    from .inverse import DEFAULT_SCHEDULE, run_job
    layout, sigma, r = key
    xd, td, clean, y = data_for(layout, sigma, r)
    schedule = M4_SCHEDULE_QUICK if quick else DEFAULT_SCHEDULE
    res = run_job(xd, td, y, r, schedule=schedule)
    return dict(theta=res["theta"], pde=res["pde"], data=res["data"], wall_s=res["wall_s"], hist=res["hist"],
                cfg=schedule)


def ls_job(key, n=200):
    """Least-squares estimates of theta over n noise draws for one (layout, sigma) case."""
    from .inverse import forward, ls_fit, make_probes
    layout, sigma = key
    xd, td = make_probes(LAYOUTS[layout])
    clean = forward(xd, td, THETA_TRUE)
    return np.array([ls_fit(xd, td, clean + sigma * np.random.default_rng(1000 + r).standard_normal(clean.shape))[0]
                     for r in range(n)])


# ------------------------------------------------------------------------------------------ runner
def _save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _init_worker():
    import torch
    torch.set_num_threads(1)


def run_jobs(jobs, job_fn, cache_path, force=False, fresh=False, workers=1, verbose=True):
    """Run `job_fn(key)` for every key in `jobs` and return {key: result}.

    force=False  load `cache_path` if it exists and run only the missing jobs, saving after each one.
    force=True   ignore the existing cache and rerun every job. Results go to `cache_path + '.partial'` as they
                 finish and replace `cache_path` only when all jobs are done, so an interrupted rerun never
                 mixes old and new numbers. Calling again resumes the partial file; fresh=True discards it.
    workers>1    run jobs in that many processes (each limited to one torch thread).
    """
    cache_path = Path(cache_path)
    partial = cache_path.with_name(cache_path.name + ".partial")
    if force:
        if fresh and partial.exists():
            partial.unlink()
        store_path = partial
        results = _load(partial) if partial.exists() else {}
    else:
        store_path = cache_path
        results = _load(cache_path) if cache_path.exists() else {}
    missing = [j for j in jobs if j not in results]
    if verbose:
        print(f"{cache_path.name}: {len(jobs) - len(missing)} of {len(jobs)} jobs already done, {len(missing)} to run"
              + (" (force: rerunning all)" if force else ""), flush=True)

    def record(key, res, t0):
        results[key] = res
        _save(results, store_path)
        if verbose:
            print(f"  done {key} ({time.perf_counter() - t0:.0f} s) [{len(results)}/{len(jobs)}]", flush=True)

    if missing and workers > 1:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = {pool.submit(job_fn, k): k for k in missing}
            try:
                for fut in as_completed(futures):
                    record(futures[fut], fut.result(), t0)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    else:
        for k in missing:
            t0 = time.perf_counter()
            record(k, job_fn(k), t0)
    if force:
        if not missing and not partial.exists():
            _save(results, partial)
        os.replace(partial, cache_path)
    return results


def _paths(results_dir, name):
    return Path(results_dir if results_dir is not None else default_results_dir()) / name


def run_m2(force=False, workers=1, results_dir=None, quick=False, fresh=False):
    fn = functools.partial(m2_job, quick=quick)
    return run_jobs(M2_JOBS, fn, _paths(results_dir, "m2_runs.pkl"), force, fresh, workers)


def run_m3(force=False, workers=1, results_dir=None, quick=False, fresh=False, jobs=None):
    fn = functools.partial(m3_job, quick=quick)
    return run_jobs(jobs or M3_JOBS, fn, _paths(results_dir, "m3_runs.pkl"), force, fresh, workers)


def run_m4(force=False, workers=1, results_dir=None, quick=False, fresh=False, jobs=None):
    fn = functools.partial(m4_job, quick=quick)
    return run_jobs(jobs or M4_JOBS, fn, _paths(results_dir, "m4_runs.pkl"), force, fresh, workers)


def run_ls_baseline(force=False, workers=1, results_dir=None, n=200, fresh=False):
    fn = functools.partial(ls_job, n=n)
    return run_jobs(LS_CASES, fn, _paths(results_dir, "m4_ls.pkl"), force, fresh, workers)


def load_runs(name, results_dir=None):
    """Load a cache by name: 'm2', 'm3', 'm4' or 'm4_ls'."""
    return _load(_paths(results_dir, {"m2": "m2_runs.pkl", "m3": "m3_runs.pkl", "m4": "m4_runs.pkl",
                                      "m4_ls": "m4_ls.pkl"}[name]))
