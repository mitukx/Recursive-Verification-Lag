#!/usr/bin/env python3
"""Standalone sanity-check for stale exponential-update composition.

This is a mathematical demo, not the archived learned-generator experiment.
"""
import numpy as np


def normalize(x):
    return x / x.sum()


def exp_update(p, score, eta):
    a = eta * score
    a -= a.max()
    return normalize(p * np.exp(a))


def composed_update(p, score, etas):
    q = p.copy()
    for eta in etas:
        q = exp_update(q, score, eta)
    return q


def direct_update(p, score, etas):
    return exp_update(p, score, sum(etas))


def chi2(q, p):
    return float(np.sum((q - p) ** 2 / p))


def balanced_full_class_V(q, p):
    mu = 0.5 * (p + q)
    return float(np.sum((q - p) ** 2 / mu))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = normalize(rng.random(20) + 0.1)
    score = rng.normal(size=20)
    etas = [0.4, 0.7, 0.2, 0.9]

    q_seq = composed_update(p, score, etas)
    q_direct = direct_update(p, score, etas)

    print("max |sequential - direct|:", np.max(np.abs(q_seq - q_direct)))
    print("chi2(q || p):", chi2(q_seq, p))
    print("balanced full-class V:", balanced_full_class_V(q_seq, p))
    print("balanced V <= 4:", balanced_full_class_V(q_seq, p) <= 4 + 1e-12)
