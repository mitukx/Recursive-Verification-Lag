#!/usr/bin/env python3
"""Print a compact numerical summary of the archived experiments."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def show(name, columns=None):
    df = pd.read_csv(DATA / name)
    if columns:
        df = df[columns]
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    show(
        "program_synthesis_threshold_summary.csv",
        ["n", "eta_50pct_wrong", "transition_width_10_90"],
    )
    show(
        "learned_dsl_misspecified_threshold_v2.csv",
        ["n", "eta50_wrong", "width10_90"],
    )
    show(
        "recursive_refresh_interval_mc_summary.csv",
        [
            "refresh_interval",
            "ever_below_initial_fraction",
            "ever_harmful_update_fraction",
            "mean_num_updates",
        ],
    )
    show(
        "recursive_eta_refresh_finite_boundary.csv",
        ["eta_step", "first_L_with_failure_prob_ge_0.5", "etaL_at_boundary"],
    )
