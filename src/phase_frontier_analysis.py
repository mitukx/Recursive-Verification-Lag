"""Quantify which stale-shift coordinate best collapses the recursive failure frontier.

This script is deliberately dependency-light (numpy/pandas only).  It evaluates
candidate scalar coordinates on the archived 2D recursive phase experiment by
asking two questions:

1. How accurately can one scalar threshold separate safe/failing cells?
2. How stable is the implied threshold across per-step optimization strengths?

The second criterion is important: a coordinate that only classifies the pooled
sample can look good while failing to define a transferable phase law.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "recursive_eta_refresh_population_phase.csv"


@dataclass
class ThresholdResult:
    metric: str
    threshold: float
    balanced_accuracy: float
    direction: str
    boundary_mean: float
    boundary_std: float
    boundary_cv: float
    n_eta_with_failure: int


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    pos = y_true
    neg = ~y_true
    tpr = (y_pred[pos].mean() if pos.any() else np.nan)
    tnr = ((~y_pred[neg]).mean() if neg.any() else np.nan)
    return float(np.nanmean([tpr, tnr]))


def best_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    finite = np.isfinite(x)
    x = x[finite]
    y = y[finite].astype(int)
    unique = np.unique(x)
    if len(unique) < 2:
        return float(unique[0]), 0.5, ">="
    mids = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.concatenate(([unique[0] - 1e-12], mids, [unique[-1] + 1e-12]))
    best = (-1.0, float(candidates[0]), ">=")
    for t in candidates:
        for direction in (">=", "<="):
            pred = x >= t if direction == ">=" else x <= t
            score = balanced_accuracy(y, pred)
            if score > best[0]:
                best = (score, float(t), direction)
    return best[1], best[0], best[2]


def per_eta_boundary(df: pd.DataFrame, metric: str) -> np.ndarray:
    boundaries: list[float] = []
    for _, g in df.groupby("eta_step"):
        g = g.sort_values("refresh_interval")
        failed = g[g["ever_below_initial"].astype(bool)]
        if failed.empty:
            continue
        first = failed.iloc[0]
        boundaries.append(float(first[metric]))
    return np.asarray(boundaries, dtype=float)


def evaluate_metric(df: pd.DataFrame, metric: str) -> ThresholdResult:
    x = df[metric].to_numpy(dtype=float)
    y = df["ever_below_initial"].to_numpy(dtype=int)
    threshold, score, direction = best_threshold(x, y)
    b = per_eta_boundary(df, metric)
    mean = float(np.mean(b)) if len(b) else np.nan
    std = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
    cv = float(std / abs(mean)) if len(b) > 1 and mean != 0 else np.nan
    return ThresholdResult(metric, threshold, score, direction, mean, std, cv, len(b))


def main() -> None:
    df = pd.read_csv(DATA)
    metrics = [
        "eta_times_L",
        "max_score_span_exposure",
        "max_block_KL",
    ]
    results = [evaluate_metric(df, m) for m in metrics if m in df.columns]
    out = pd.DataFrame([r.__dict__ for r in results])
    out = out.sort_values(["boundary_cv", "balanced_accuracy"], ascending=[True, False])
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False, float_format=lambda z: f"{z:.6g}"))
    print("\nDescriptive in-sample analysis only; this does not measure held-out transfer.")
    print("Lower boundary_cv means less dispersion in the observed grid boundaries;")
    print("balanced_accuracy measures pooled safe/failure separation by one threshold.")


if __name__ == "__main__":
    main()

