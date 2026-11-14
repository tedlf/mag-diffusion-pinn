# mag-diffusion-pinn

Physics-informed neural networks (PINNs) for 1D resistive magnetic diffusion, b_t = θ b_ξξ, validated against analytic and finite-difference baselines. It covers the forward problem (three variants, several training improvements) and an inverse problem (recovering the diffusivity from sparse, noisy probe data). See [SPEC.md](SPEC.md) for the plan and milestones.

## Headline results

All numbers are from the cached runs in `results/`; `python scripts/reproduce_headlines.py` prints them and redraws the figures below. They come from one network size (4 × 64, tanh), one training budget (2000 Adam + 2000 L-BFGS steps) and float32, with six seeds for the main forward comparisons. They say what helped *in this setup*, not what is generally best.

**Forward problems** (relative L2 error against the exact solution, median over six seeds):

| Problem | Plain PINN | Best variant | Notes |
|---|---|---|---|
| R: smooth ramp at the boundary | 3.6e-3 | **1.1e-4** (hard constraints) | about 31 times better; every hard-constrained seed beats every plain seed |
| A: step at the boundary | 1.4e-2 | 3.5e-3 (hard), 3.9e-3 (manual weights) | about 4 times better; the seed-to-seed spread is as large as the gap between the best variants; not below 1e-3 |
| S: smooth initial profile | about 1e-3 | | float32 narrowly misses the 1e-3 target; see notebook 02 |

![forward errors](figures/forward_errors.png)

**Inverse problem** (true θ = 0.6, PINN starts at 1.0, three probes, 20 times each): with 1% noise the PINN recovers θ to within 0.4% to 1.0% over three noise draws; the statistical limit (Cramér–Rao) is 0.9%. Noise-free recovery is 0.06%. The PINN tracks classical least squares on the exact forward model (median difference 0.2 of the Cramér–Rao standard deviation) and is not better than it. See notebook 04 for the noise and probe-placement studies and their caveats.

![inverse noise](figures/inverse_noise.png)

## Install and test

    git clone <this repository> && cd mag-diffusion-pinn
    python -m venv .venv && source .venv/bin/activate      # or: pyenv activate <env>
    pip install -e ".[dev]"
    python -m pytest                                       # about a minute on a laptop CPU

The tests check the analytic solutions against the PDE and each other, the finite-difference solvers (second-order convergence, agreement with the reference), the autograd residual on a manufactured solution, the hard-constraint boundary conditions, short training runs for each PINN option, the inverse tools (least squares recovers θ, its scatter matches the Cramér–Rao bound), the experiment runner (resume after interruption, atomic replacement, parallel workers) and short versions of every experiment job, and that the cached results contain the numbers quoted above (exact numbers, and wider qualitative bounds that also hold after regenerating). `requirements-lock.txt` lists the exact package versions the tests and cached runs were produced with.

## Reproduce

    python scripts/reproduce_headlines.py     # prints the headline numbers, writes figures/*.png (no training)
    jupyter lab notebooks/                    # the full analysis, with caveats

By default nothing is retrained: the notebooks and scripts load the cached runs in `results/`.

## Regenerate the results

Every file in `results/` can be regenerated, either from a notebook or from the command line. Both use the same code (`src/mag_diffusion_pinn/experiments.py`, which holds the list of runs and the training budgets).

| Cache | Contents | Notebook | Script | Time on one CPU core |
|---|---|---|---|---|
| `data/reference_fields.npz` | analytic reference fields | `01_baselines` (recomputed on every run) | | seconds |
| `results/m2_runs.pkl` | 8 plain-PINN runs (S and A, float32 and float64) | `02_plain_pinn` | `run_experiments.py m2` | about 40 min |
| `results/m3_runs.pkl` | 63 runs: A and R with the M3 options | `03_improvements` | `run_experiments.py m3` | about 5 hours |
| `results/m4_runs.pkl`, `results/m4_ls.pkl` | 28 inverse PINN runs; least-squares Monte Carlo | `04_inverse` | `run_experiments.py m4` | about 2.5 hours (PINN), 7 minutes (least squares) |

In a notebook, set `FORCE_RERUN = True` in the first code cell (and `WORKERS` to the number of cores to use), then run all cells. From the command line:

    python scripts/run_experiments.py all --force --workers 4     # everything, in parallel
    python scripts/run_experiments.py m3 --workers 4              # only runs that are missing from results/m3_runs.pkl

How it behaves:

* Without `--force` (`FORCE_RERUN = False`), cached runs are kept and only missing ones are trained.
* With `--force`, every run is redone. New results are collected in `results/<name>.pkl.partial` and replace the old file only when all runs are finished, so an interrupted rerun never leaves a mixture of old and new numbers. Running the same command again resumes from the partial file; `--fresh` discards it and starts over.
* `--quick` uses a tiny training budget (seconds per run) and `--results-dir` points it at another directory; this is how the tests exercise the machinery.
* Notebook 03 also reads the plain-PINN runs of problem A with seeds 0 to 2 from `results/m2_runs.pkl`, so regenerate notebook 02 first if you want those to be new as well.

**Regenerated numbers will not match the committed ones exactly.** Training in float32 is not bitwise reproducible across machines, CPU architectures or torch builds: on the machine that made the cache, the M3 runs I retried reproduce to machine precision, while the M2 runs differ by 2% to 11% in relative error (for example 9.0e-4 against the cached 1.0e-3 for problem S, seed 0), most likely because they were made with a different thread setup. The conclusions (hard constraints help, by a large factor on problem R; the inverse problem is recovered to within a fraction of a percent at 1% noise) should hold; the tables in the notebooks and in this README will move in the second digit. `results/m2_runs_mac.pkl` is an earlier retraining of the M2 set on a different machine, kept as an example of this spread. On the same machine, regeneration is reproducible: a full regeneration from the command line (`run_experiments.py all --force --workers 10`) and a second one from the notebooks (`FORCE_RERUN = True`, `WORKERS = 10`), run on one Apple-silicon laptop as separate processes, gave bit-for-bit identical predictions and estimates for all 99 training runs, and identical least-squares Monte Carlo results. The worker count and the way the runs are launched do not change the numbers; the machine and library versions do: against the committed caches, which were made on a different machine, the laptop regeneration gave the same conclusions with most individual runs within a factor of two of the committed ones (one problem-A causal run was about five times worse; median relative L2 error on problem R with hard constraints 1.19e-4 against 1.14e-4 committed; hard versus plain 27 times instead of 31 times; inverse estimates within 0.2 percentage points of the committed ones for the three-probe and five-probe cases, and within 0.8 for the single-probe cases). After regenerating, set `MDP_REGENERATED=1` when running `pytest` so that the tests pinning the committed numbers are skipped; the qualitative tests still run.

Retraining a single run from Python, for example one forward run (about 4 minutes) or one inverse run (about 5 minutes):

    from mag_diffusion_pinn import experiments as E
    print(E.m3_job(("R", "hard", 0))["metrics"]["rel_l2"])
    print(E.m4_job(("p3", 0.01, 0))["theta"])

## Layout

    src/mag_diffusion_pinn/   reference.py (analytic + Duhamel references), fd.py, pinn.py, inverse.py,
                              experiments.py (run lists, budgets, resumable runner)
    tests/                    pytest suite
    scripts/                  reproduce_headlines.py, run_experiments.py
    notebooks/                01 baselines, 02 plain PINN, 03 improvements, 04 inverse problem
    results/, data/           cached training runs and reference fields used by the notebooks and tests
    figures/                  figures embedded above

**Notebooks and package.** The notebooks import the package and print the source of its key functions, so the code you read in a notebook is the code that ran, and there is one maintained copy. The reference field for problem R uses 600 Gauss-Legendre nodes; the M3 runs in the cache were scored against a 200-node version of it, which differs by less than 4e-7 and does not affect any number quoted here.

## Status

| Milestone | State |
|---|---|
| M1 baselines (`01_baselines.ipynb`) | done |
| M2 plain PINN on S and A (`02_plain_pinn.ipynb`) | done; S error 1.0e-3 to 1.4e-3 in float32 (target 1e-3), 6.6e-4 in float64 (one seed); A fails at the early-time corner |
| M3 improvements (`03_improvements.ipynb`) | done; adds smooth-ramp problem R; see the table above |
| M4 inverse problem (`04_inverse.ipynb`) | done; see above |
| M5 package, tests, reproducibility | done; `pytest` passes in a fresh environment |
