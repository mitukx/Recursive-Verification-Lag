"""Real-model candidate-bank experiment harness for Recursive Verification Lag.

This script assumes candidate programs have already been sampled from a pretrained
code model and stored in JSONL.  That design separates expensive model generation
from the recursive verification experiment and makes optimizer/verifier sweeps
reproducible.

Expected JSONL schema (one row per candidate):
{
  "task_id": "task_001",
  "candidate_id": "...",
  "public_score": 0.8,
  "trusted_score": 0.6,
  "features": {"public_score": 0.8, "length": 143, ...},
  "base_logprob": -4.2
}

The experiment repeatedly fits a linear verifier on a trusted subset sampled from
the current policy, then performs soft/exponential selection while holding the
verifier stale for a configurable number of rounds.  All true/trusted scores are
used only for auditing outcomes, not for selection between refreshes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Config:
    eta: float = 1.0
    refresh_interval: int = 4
    rounds: int = 24
    audit_per_refresh: int = 64
    seed: int = 0
    ridge: float = 1e-4


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    flat = []
    feature_names = set()
    for r in rows:
        feature_names.update((r.get("features") or {}).keys())
    for r in rows:
        out = {
            "task_id": r["task_id"],
            "candidate_id": r["candidate_id"],
            "trusted_score": float(r["trusted_score"]),
            "base_logprob": float(r.get("base_logprob", 0.0)),
        }
        feats = r.get("features") or {}
        for name in sorted(feature_names):
            out[f"f::{name}"] = float(feats.get(name, 0.0))
        if "public_score" in r and "f::public_score" not in out:
            out["f::public_score"] = float(r["public_score"])
        flat.append(out)
    df = pd.DataFrame(flat)
    if df.empty:
        raise ValueError("candidate bank is empty")
    return df


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def fit_verifier(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ y)


def task_policy(df: pd.DataFrame, logits: np.ndarray) -> np.ndarray:
    probs = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("task_id").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        probs[ii] = softmax(logits[ii])
    probs /= probs.sum()
    return probs


def expected_score(df: pd.DataFrame, probs: np.ndarray, col: str) -> float:
    return float(np.dot(probs, df[col].to_numpy(dtype=float)))


def run(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    feature_cols = [c for c in df.columns if c.startswith("f::")]
    if not feature_cols:
        raise ValueError("candidate bank needs at least one verifier feature")
    X = df[feature_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), X])
    true = df["trusted_score"].to_numpy(dtype=float)
    logits = df["base_logprob"].to_numpy(dtype=float).copy()
    p = task_policy(df, logits)
    initial_true = expected_score(df, p, "trusted_score")
    theta = np.zeros(X.shape[1])
    history = []
    for t in range(cfg.rounds):
        if t % cfg.refresh_interval == 0:
            n = min(cfg.audit_per_refresh, len(df))
            audit_idx = rng.choice(len(df), size=n, replace=True, p=p)
            theta = fit_verifier(X[audit_idx], true[audit_idx], cfg.ridge)
        proxy = X @ theta
        old_p = p.copy()
        logits = logits + cfg.eta * proxy
        p = task_policy(df, logits)
        kl = float(np.sum(np.where(p > 0, p * (np.log(p + 1e-300) - np.log(old_p + 1e-300)), 0.0)))
        true_mean = expected_score(df, p, "trusted_score")
        proxy_mean = float(np.dot(p, proxy))
        history.append({
            "round": t + 1,
            "true_reward": true_mean,
            "proxy_reward": proxy_mean,
            "below_initial": int(true_mean < initial_true),
            "kl_from_previous": kl,
            "refresh": int(t % cfg.refresh_interval == 0),
            "eta": cfg.eta,
            "refresh_interval": cfg.refresh_interval,
            "audit_per_refresh": cfg.audit_per_refresh,
            "seed": cfg.seed,
        })
    return pd.DataFrame(history)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("candidate_bank", type=Path)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--refresh-interval", type=int, default=4)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--audit-per-refresh", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("candidate_bank_run.csv"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = Config(a.eta, a.refresh_interval, a.rounds, a.audit_per_refresh, a.seed)
    df = load_jsonl(a.candidate_bank)
    out = run(df, cfg)
    out.to_csv(a.output, index=False)
    print(out.tail().to_string(index=False))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
