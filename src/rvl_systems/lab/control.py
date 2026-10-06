from __future__ import annotations

import math
from dataclasses import dataclass

from .agent import softmax
from .contracts import digest


@dataclass(frozen=True)
class ControlConfig:
    mode: str = "adaptive"
    cadence: int = 8
    audit_budget: int = 64
    audit_per_batch: int = 2
    refit_labels: int = 8
    risk_threshold: float = 0.45

    def __post_init__(self):
        if self.mode not in ("adaptive", "fixed", "never"):
            raise ValueError("unknown controller")
        if min(self.cadence, self.audit_per_batch, self.refit_labels) <= 0 or self.audit_budget < 0:
            raise ValueError("invalid controller budgets")


class RVLControlPlane:
    """Heuristic online intervention controller, NOT a statistical certificate.

    Uses restricted residual geometry, endpoint policy KL, disagreement,
    uncertainty, label coverage and policy/reward version lag.
    """
    def __init__(self, config):
        self.config = config
        self.spent = 0
        self.last_refit_labels = 0
        self.anchor = {}
        self.decisions = []

    def policy_shift(self, snapshot):
        if hasattr(snapshot, "estimated_shift"):
            return snapshot.estimated_shift
        values = []
        for k, logits in snapshot.logits.items():
            q = softmax(logits)
            p = softmax(self.anchor.get(k, (0.0,)*len(logits)))
            values.append(sum(qi*math.log(qi/pi) for qi,pi in zip(q,p)))
        return sum(values)/max(1,len(values))

    def priority(self, t, verdict, verifier, snapshot):
        shift = self.policy_shift(snapshot)
        coverage = 1/math.sqrt(1+verifier.coverage(verdict.cell))
        residual = verifier.geometry_risk(verdict.cell)
        stale = min(1.0, max(0,snapshot.version-t.policy_version)/8)
        reward_stale = min(1.0, max(0,verifier.version-verdict.verifier_version)/4)
        return (verdict.disagreement + verdict.uncertainty + coverage +
                residual*(1+math.sqrt(max(0,shift))) + stale + reward_stale)/6

    def select_audits(self, rows, verifier, snapshot, batch_index):
        cfg = self.config
        remaining = cfg.audit_budget-self.spent
        if remaining <= 0 or cfg.mode == "never":
            return []
        if cfg.mode == "fixed" and (batch_index+1)%cfg.cadence:
            return []
        ranked = sorted(rows, key=lambda tv: (-self.priority(*tv,verifier,snapshot),
                                             tv[0].trajectory_id))
        if cfg.mode == "adaptive":
            ranked = [tv for tv in ranked if self.priority(*tv,verifier,snapshot)>=cfg.risk_threshold]
        selected = ranked[:min(cfg.audit_per_batch,remaining)]
        self.spent += len(selected)
        self.decisions.append({"batch":batch_index,"audit_ids":[t.trajectory_id for t,_ in selected],
                               "policy_shift": self.policy_shift(snapshot),"budget_spent":self.spent})
        return [t for t,_ in selected]

    def should_refit(self, label_count):
        return label_count-self.last_refit_labels >= self.config.refit_labels

    def refitted(self, label_count, snapshot):
        self.last_refit_labels = label_count
        self.anchor = dict(snapshot.logits)

    def reevaluate_ids(self, rows, verifier, snapshot, limit=32):
        ranked = sorted(rows, key=lambda tv: (-self.priority(*tv,verifier,snapshot),
                                             tv[0].trajectory_id))
        return [t for t,v in ranked if not v.trusted and v.verifier_version < verifier.version][:limit]

    def state(self):
        return {"spent":self.spent,"last_refit_labels":self.last_refit_labels,"anchor":self.anchor,
                "decisions":self.decisions}

    def restore(self,state):
        for k,v in state.items():
            setattr(self,k,v)


class Curriculum:
    """Failure-prioritized task generation and explicit proxy-exploit self-play."""
    def __init__(self, families=3):
        self.families = families
        self.failures = [1.0]*families
        self.critic = [0.5]*families
        self.generated = 0
        self.attack_successes = 0
        self.attack_values = [0.5,0.5,0.5]
        self.attack_counts = [0,0,0]

    def generate(self, index, seed):
        import random
        from .contracts import Task
        rng = random.Random(seed)
        family = rng.choices(range(self.families),weights=self.failures)[0]
        coefficient = (family+1)*rng.choice((-1,1))
        bias = rng.randint(-3,3)
        self.generated += 1
        return Task(f"train-{index}",family,coefficient,bias)

    def observe(self, t, reward):
        f = t.task.family
        self.failures[f] = 0.9*self.failures[f]+0.1*(1-reward+0.1)
        self.critic[f] = 0.9*self.critic[f]+0.1*reward

    def attack(self, task, snapshot, index, verifier=None):
        # Bandit attacker explores constant, sign reversal and off-by-one edits.
        # It adapts to observed exploits and the current verifier's acceptance.
        from .contracts import Trajectory
        import math
        programs = [(0,task.bias),(-task.coefficient,task.bias),
                    (task.coefficient,task.bias+1)]
        candidates = [Trajectory(f"attack-{index}-{arm}",task,snapshot.version,(),
                                 program,True,index) for arm,program in enumerate(programs)]
        scores = []
        for arm,t in enumerate(candidates):
            acceptance = verifier.score(t).reward if verifier else 1.0
            exploration = math.sqrt(math.log(2+index)/(1+self.attack_counts[arm]))
            scores.append(acceptance*self.attack_values[arm]+.15*exploration)
        arm = max(range(3),key=lambda i:(scores[i],-i))
        self.last_attack_arm = arm
        return candidates[arm]

    def observe_attack(self,proxy,trusted):
        arm = self.last_attack_arm
        self.attack_counts[arm] += 1
        n = self.attack_counts[arm]
        self.attack_values[arm] += ((proxy-trusted)-self.attack_values[arm])/n
        self.attack_successes += int(proxy > trusted)

    def state(self):
        return vars(self).copy()

    def restore(self,state):
        self.__dict__.update(state)
