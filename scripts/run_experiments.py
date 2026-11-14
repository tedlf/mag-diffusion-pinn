"""Regenerate the cached training runs in results/ (hours of compute for m3 and m4).

    python scripts/run_experiments.py m2                  # 8 runs,  about 5 minutes each on one CPU core
    python scripts/run_experiments.py m3 --workers 4      # 63 runs, about 4 minutes each
    python scripts/run_experiments.py m4 --workers 4      # 28 runs, about 5 minutes each, plus m4_ls (about 7 minutes)
    python scripts/run_experiments.py all --workers 4 --force

Without --force, existing results are kept and only missing runs are done. With --force every run is redone; the new
results go to a `.partial` file and replace the old cache only when all runs have finished, and rerunning the same
command after an interruption resumes from the partial file (--fresh discards it). Results are not bitwise
reproducible across machines, so expect numbers close to, but not identical with, the committed cache.
--quick uses a tiny budget (seconds per run) and writes to --results-dir; it exists to test this machinery.
"""
import argparse
import sys

from mag_diffusion_pinn import experiments as E


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["m2", "m3", "m4", "all"])
    ap.add_argument("--force", action="store_true", help="rerun every run instead of only the missing ones")
    ap.add_argument("--fresh", action="store_true", help="with --force: discard an earlier partial rerun")
    ap.add_argument("--workers", type=int, default=1, help="number of processes (one torch thread each)")
    ap.add_argument("--results-dir", default=None, help="default: results/ of this repository")
    ap.add_argument("--quick", action="store_true", help="tiny budget, for testing the machinery only")
    a = ap.parse_args(argv)
    kw = dict(force=a.force, fresh=a.fresh, workers=a.workers, results_dir=a.results_dir)
    if a.which in ("m2", "all"):
        E.run_m2(quick=a.quick, **kw)
    if a.which in ("m3", "all"):
        E.run_m3(quick=a.quick, **kw)
    if a.which in ("m4", "all"):
        E.run_m4(quick=a.quick, **kw)
        if not a.quick:
            E.run_ls_baseline(**kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
