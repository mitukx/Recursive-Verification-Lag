from __future__ import annotations

import math
from dataclasses import asdict

from .agent import public_reward, trusted_reward
from .contracts import Verdict


def feature_cell(trajectory):
    # Features available without semantic labels: family, coefficient sign,
    # constant/nonconstant program, completion. Not a shift-only safety scalar.
    a, _ = trajectory.program
    return f"{trajectory.task.family}:{(a>0)-(a<0)}:{int(a==0)}:{int(trajectory.completed)}"


class VerifierEnsemble:
    """Executable public grader + label-trained Bayesian critic.

    The critic is deliberately small/misspecified so exploitation can be studied.
    Trusted hidden tests are called ONLY through audit(), not score().
    """
    def __init__(self):
        self.version = 0
        self.cells = {}
        self.critic_weight = 0.0
        self.residuals = {}
        self.audits = 0
        self.exploits = 0

    def score(self, trajectory):
        public = public_reward(trajectory.task, trajectory.program) * trajectory.completed
        cell = feature_cell(trajectory)
        successes, total = self.cells.get(cell, (1.0, 2.0))
        critic = successes/total
        uncertainty = min(1.0, 2 * math.sqrt(critic*(1-critic)/(total+1)))
        w = self.critic_weight
        return Verdict((1-w)*public+w*critic, self.version, (public, critic),
                       cell, uncertainty)

    def audit(self, trajectory):
        self.audits += 1
        y = trusted_reward(trajectory.task, trajectory.program, trajectory.completed)
        public = public_reward(trajectory.task, trajectory.program)*trajectory.completed
        if public > y:
            self.exploits += 1
        return Verdict(y, self.version, (public, y), feature_cell(trajectory), 0.0, True)

    def fit(self, labeled):
        if not labeled:
            return False
        cells, residuals = {}, {}
        for t, y in labeled:
            c = feature_cell(t)
            s, n = cells.get(c, (1.0, 2.0))
            cells[c] = (s+y, n+1)
            proxy = public_reward(t.task, t.program)*t.completed
            e, count = residuals.get(c, (0.0, 0))
            residuals[c] = (e+abs(proxy-y), count+1)
        self.cells = cells
        self.residuals = {c: e/n for c, (e,n) in residuals.items()}
        self.critic_weight = 0.9
        self.version += 1
        return True

    def geometry_risk(self, cell):
        # Unobserved cells are treated conservatively.
        return self.residuals.get(cell, 1.0)

    def coverage(self, cell):
        return max(0, self.cells.get(cell, (1,2))[1]-2)

    def state(self):
        return {"version": self.version, "cells": self.cells, "residuals": self.residuals,
                "critic_weight": self.critic_weight, "audits": self.audits, "exploits": self.exploits}

    def restore(self, state):
        for key, value in state.items():
            setattr(self, key, value)
