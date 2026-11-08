"""Reproduce the parameter sweep used to choose signal and noise_sd.

For each (signal, noise_sd) the normal dataset is generated with five data seeds
and evaluated with both splits. The other settings are the DataConfig defaults.
Takes a few minutes (4 worker processes).
"""
import os

# One BLAS thread per worker; otherwise each process starts its own thread pool
# and OpenBLAS can run out of memory.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakage_demo.data import DEFAULT_SEED, DataConfig, generate_dataset  # noqa: E402
from leakage_demo.evaluate import evaluate  # noqa: E402

SEEDS = [DEFAULT_SEED, 1, 2, 3, 4]
SETTINGS = [  # (signal, noise_sd)
    (0.25, 0.3),
    (0.35, 0.3),
    (0.40, 0.3),
    (0.40, 0.5),  # chosen
    (0.45, 0.3),
    (0.55, 0.3),
    (0.70, 0.3),
]


def run(job: tuple[float, float, int]) -> tuple[float, float, int, float, float]:
    signal, noise_sd, seed = job
    df = generate_dataset(replace(DataConfig(), signal=signal, noise_sd=noise_sd, seed=seed))
    return signal, noise_sd, seed, evaluate(df, "image")["auc_mean"], evaluate(df, "patient")["auc_mean"]


def main() -> None:
    jobs = [(s, n, seed) for s, n in SETTINGS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(run, jobs))

    print("Per seed (mean AUC over 5 folds)")
    print("| signal | noise_sd | seed | image-level AUC | patient-level AUC |")
    print("|---|---|---|---|---|")
    for s, n, seed, img, pat in rows:
        print(f"| {s:.2f} | {n:.1f} | {seed} | {img:.3f} | {pat:.3f} |")

    print()
    print(f"Mean over seeds {SEEDS}")
    print("| signal | noise_sd | image-level AUC | patient-level AUC (min - max) |")
    print("|---|---|---|---|")
    for s, n in SETTINGS:
        sel = [r for r in rows if r[0] == s and r[1] == n]
        img = sum(r[3] for r in sel) / len(sel)
        pats = [r[4] for r in sel]
        print(f"| {s:.2f} | {n:.1f} | {img:.3f} | {sum(pats) / len(pats):.3f} ({min(pats):.3f} - {max(pats):.3f}) |")


if __name__ == "__main__":
    main()
