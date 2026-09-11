#!/usr/bin/env python3
"""Reproduce selected figures from archived experiment CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def save_program_threshold():
    df = pd.read_csv(DATA / "program_synthesis_threshold_summary.csv")
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["n"], df["eta_50pct_wrong"], marker="o")
    plt.xscale("log")
    plt.xlabel("Trusted verifier labels n")
    plt.ylabel("50% confident-wrong threshold eta")
    plt.title("Program synthesis: threshold center vs audit budget")
    plt.tight_layout()
    plt.savefig(OUT / "program_synthesis_threshold_vs_budget_reproduced.png", dpi=180)
    plt.close()


def save_recursive_heatmap():
    df = pd.read_csv(DATA / "recursive_eta_refresh_population_phase.csv")
    etas = np.sort(df["eta_step"].unique())
    intervals = np.sort(df["refresh_interval"].unique())
    baseline_proxy = df["minimum_true_reward"].max()  # only used for common offset visualization
    matrix = np.empty((len(intervals), len(etas)))
    for i, L in enumerate(intervals):
        for j, eta in enumerate(etas):
            row = df[(df.refresh_interval == L) & (df.eta_step == eta)].iloc[0]
            matrix[i, j] = row["ever_below_initial"]
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[etas.min(), etas.max(), intervals.min(), intervals.max()],
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, label="Baseline collapse indicator")
    plt.xlabel("Per-step optimization strength eta")
    plt.ylabel("Verifier refresh interval L")
    plt.title("Recursive verification-lag phase")
    plt.tight_layout()
    plt.savefig(OUT / "recursive_eta_refresh_phase_reproduced.png", dpi=180)
    plt.close()


def save_composed_hardness():
    df = pd.read_csv(DATA / "composed_block_hardness_learned_generator.csv")
    plt.figure(figsize=(7, 4.5))
    for col, label in [
        ("h_current", "current-policy audit"),
        ("h_balanced", "balanced p/q audit"),
        ("h_candidate", "candidate audit"),
    ]:
        plt.plot(df["lambda_stale"], np.minimum(df[col], 1e5), label=label)
    plt.yscale("log")
    plt.xlabel("Cumulative stale strength")
    plt.ylabel("V / Delta^2 (clipped at 1e5)")
    plt.title("Composed-block verification hardness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "composed_block_hardness_reproduced.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    save_program_threshold()
    save_recursive_heatmap()
    save_composed_hardness()
    print(f"Wrote reproduced figures to {OUT}")
