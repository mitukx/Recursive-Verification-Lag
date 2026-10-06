from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .candidates import CandidateGenerator
from .config import RSIConfig, digest
from .evaluation import EvaluationStack, bundle_from_payload
from .failure_analysis import FailureAnalyzer, TaskObservation
from .memory import ResearchMemory
from .metrics import MetricsWriter
from .math_rsi import MathematicalRSIGate
from .models import ChampionSnapshot, GenerationRecord
from .planner import ExperimentPlanner, PlannerContext
from .promotion import PromotionGate, detect_false_progress
from .sandbox import SandboxRunner
from .verifier_lag import RecursiveVerificationLagMonitor

DEFAULT_STATE: dict[str, Any] = {
    "theta":{"model":"frozen-baseline","version":0},
    "F":{"learning_rate":0.0,"batch_size":1,"adapter_rank":0,"clip_ratio":0.2},
    "V":{"version":0,"threshold":0.5,"ensemble_size":1,"refresh_cadence":2,"trusted_fraction":0.2},
    "H":{"reasoning_budget":2,"retry_limit":0,"context_window":2,"memory_slots":0,"tool_policy":"bounded","sampling_temperature":0.0,"curriculum_level":1,"format_guard":"strict"},
}

class RSIController:
    def __init__(self,config:RSIConfig):
        self.cfg=config; self.root=Path(config.output_dir); self.root.mkdir(parents=True,exist_ok=True)
        self.memory=ResearchMemory(self.root/"research_memory.sqlite"); self.metrics=MetricsWriter(self.root/"metrics")
        self.evaluation=EvaluationStack(seed=config.seed); self.planner=ExperimentPlanner(self.memory,config.resources)
        self.generator=CandidateGenerator(config.mutation,config.mode); self.sandbox=SandboxRunner()
        self.promotion=PromotionGate(config.promotion); self.lag=RecursiveVerificationLagMonitor(config.verifier_trust); self.failure_analyzer=FailureAnalyzer()
        self.mathematical_rsi=MathematicalRSIGate(config.mathematical_rsi,config.promotion)

    def close(self): self.memory.close()

    def _evaluate(self,state,policy_version,verifier_version,age,*,include_sealed=False):
        # Candidate/champion comparisons use the same fixed evaluation seed.
        # Proposal seeds govern candidate construction, never evaluator randomness.
        result=self.sandbox.run("src.rsi_controller.evaluation:synthetic_experiment_entrypoint",{
            "suite_seed":self.cfg.seed,"candidate_seed":self.cfg.seed,"candidate_state":state,
            "policy_version":policy_version,"verifier_version":verifier_version,
            "policy_verifier_age":age,"include_sealed":include_sealed},self.cfg.resources)
        if not result.ok or result.payload is None: raise RuntimeError(result.error or "sandbox experiment failed")
        return bundle_from_payload(result.payload),result

    def _ensure_initial_champion(self):
        try: return self.memory.current_champion()
        except RuntimeError:
            state=json.loads(json.dumps(DEFAULT_STATE)); c0=ChampionSnapshot("champion-0000",0,state,0,0,None); self.memory.add_champion(c0)
            baseline,_=self._evaluate(state,0,0,0)
            record=GenerationRecord(0,c0.champion_id,"baseline",baseline.development.trusted_score,baseline.promotion.trusted_score,baseline.development.reward,baseline.promotion.trusted_score,abs(baseline.development.reward-baseline.development.trusted_score),0,0,0,baseline.promotion.latency_p50,baseline.promotion.latency_p95,baseline.promotion.throughput,baseline.promotion.failure_rate,baseline.promotion.compute_cost,"BASELINE")
            self.metrics.append(record); self.memory.event("baseline_evaluated",suite_digests=baseline.suite_digests,metrics=asdict(record)); return c0

    def _development_failures(self,champion_eval):
        d=champion_eval.development; rows=[]
        for i in range(12):
            rows.append(TaskObservation(task_id=f"development-observation-{i}",success=(i/12)<d.trusted_score,reward=d.reward,trusted=d.trusted_score,verifier_score=d.trusted_score+(1-d.verifier_agreement),latency_s=d.latency_p95,selected_wrong_tool=(i==0 and d.failure_rate>.3),context_overflow=(i==1 and d.failure_rate>.3),stale_policy_age=champion_eval.policy_verifier_age,stale_verifier_age=champion_eval.policy_verifier_age))
        return self.failure_analyzer.analyze(rows)

    def run(self,generations=None):
        target=generations if generations is not None else self.cfg.generations
        if target<=0: raise ValueError("generations must be positive")
        champion=self._ensure_initial_champion()
        events=self.memory.recent_events(100000)
        existing=len([e for e in events if e["kind"]=="promotion_decision"])
        if any(e["kind"]=="sealed_final_audit" for e in events):
            raise RuntimeError("sealed final audit already opened; this experiment directory is terminal")
        start_generation=existing+1
        attempted=promoted=rejected=0
        for generation in range(start_generation,target+1):
            champion=self.memory.current_champion(); champ_age=max(0,champion.policy_version-champion.verifier_version)
            champion_eval,_=self._evaluate(dict(champion.state),champion.policy_version,champion.verifier_version,champ_age)
            ctx=PlannerContext(generation,champion.champion_id,dict(champion.state),self._development_failures(champion_eval),{"promotion":champion_eval.promotion.trusted_score},self.cfg.seed)
            proposal=self.planner.propose(ctx); self.memory.record_proposal(generation,proposal)
            candidate=self.generator.generate(proposal,champion.state); self.memory.record_candidate(candidate); attempted+=1
            candidate_policy_version=champion.policy_version+1; candidate_verifier_version=champion.verifier_version
            age=max(0,candidate_policy_version-candidate_verifier_version)
            candidate_eval,sandbox_result=self._evaluate(dict(candidate.full_state),candidate_policy_version,candidate_verifier_version,age)
            for split in ("evolution","development","promotion"):
                self.memory.record_evaluation(candidate.candidate_id,split,candidate_eval.suite_digests[split],asdict(getattr(candidate_eval,split)))
            self.memory.event("sandbox_completed",candidate_id=candidate.candidate_id,elapsed_s=sandbox_result.elapsed_s,stdout=sandbox_result.stdout,stderr=sandbox_result.stderr)
            hacking=detect_false_progress(candidate_eval,champion_eval); lag=self.lag.assess(candidate_eval,champion_eval)
            self.memory.event("rvl_assessment",candidate_id=candidate.candidate_id,**asdict(lag)); self.memory.event("false_progress_assessment",candidate_id=candidate.candidate_id,**asdict(hacking))
            math_rsi=self.mathematical_rsi.assess(candidate,candidate_eval,champion_eval,hacking,lag)
            self.memory.event("mathematical_rsi_assessment",candidate_id=candidate.candidate_id,**math_rsi.to_dict())
            decision=self.promotion.decide(candidate.candidate_id,candidate_eval,champion_eval,hacking,lag,math_rsi)
            if lag.trust_level=="uncertain" and decision.accepted:
                refreshed_version=candidate_policy_version; refreshed_state=json.loads(json.dumps(candidate.full_state)); refreshed_state.setdefault("V",{})["version"]=refreshed_version
                refreshed_eval,refresh_run=self._evaluate(refreshed_state,candidate_policy_version,refreshed_version,0)
                refreshed_lag=self.lag.assess(refreshed_eval,champion_eval); refreshed_hacking=detect_false_progress(refreshed_eval,champion_eval)
                refreshed_math_rsi=self.mathematical_rsi.assess(candidate,refreshed_eval,champion_eval,refreshed_hacking,refreshed_lag)
                self.memory.event("mathematical_rsi_refresh_assessment",candidate_id=candidate.candidate_id,**refreshed_math_rsi.to_dict())
                decision=self.promotion.decide(candidate.candidate_id,refreshed_eval,champion_eval,refreshed_hacking,refreshed_lag,refreshed_math_rsi)
                candidate_eval,lag,hacking,math_rsi=refreshed_eval,refreshed_lag,refreshed_hacking,refreshed_math_rsi
                self.memory.event("verifier_refresh_intervention",candidate_id=candidate.candidate_id,verifier_version=refreshed_version,reason="uncertain verifier trust",reevaluation_elapsed_s=refresh_run.elapsed_s)
                candidate_verifier_version=refreshed_version; candidate=replace(candidate,full_state=refreshed_state)
            if decision.accepted:
                new_id="champion-"+digest({"candidate":candidate.candidate_id,"generation":generation})[:12]; new_state=json.loads(json.dumps(candidate.full_state))
                new_state.setdefault("theta",{})["version"]=candidate_policy_version; new_state.setdefault("V",{})["version"]=candidate_verifier_version
                champion=ChampionSnapshot(new_id,generation,new_state,candidate_policy_version,candidate_verifier_version,champion.champion_id)
                promoted+=1; status="PROMOTED"; lesson=f"Supported: {proposal.hypothesis}"
                activated=champion
            else:
                rejected+=1; status="REJECTED"; lesson="Rejected: "+"; ".join(decision.reasons)
                activated=None
            self.memory.commit_generation(
                decision,generation,candidate.candidate_id,lesson,champion=activated
            )
            record=GenerationRecord(generation,champion.champion_id,candidate.candidate_id,candidate_eval.development.trusted_score,candidate_eval.promotion.trusted_score,candidate_eval.development.reward,candidate_eval.promotion.trusted_score,hacking.verification_gap,candidate_eval.verifier_version,candidate_eval.policy_version,candidate_eval.policy_verifier_age,candidate_eval.promotion.latency_p50,candidate_eval.promotion.latency_p95,candidate_eval.promotion.throughput,candidate_eval.promotion.failure_rate,candidate_eval.promotion.compute_cost,status)
            self.metrics.append(record); print(f"Generation {generation} | {candidate.candidate_id} | {status} | promotion={record.promotion_score:.4f} gap={record.verification_gap:.4f}")

        # The sealed suite is opened exactly once, after all promotion decisions.
        # Once opened, this experiment directory is terminal: continuing would
        # make subsequent candidate choices adaptive to the sealed result.
        baseline_champion=self.memory.get_champion("champion-0000")
        final_champion=self.memory.current_champion()
        baseline_final,_=self._evaluate(dict(baseline_champion.state),baseline_champion.policy_version,baseline_champion.verifier_version,0,include_sealed=True)
        final_age=max(0,final_champion.policy_version-final_champion.verifier_version)
        champion_final,_=self._evaluate(dict(final_champion.state),final_champion.policy_version,final_champion.verifier_version,final_age,include_sealed=True)
        if baseline_final.sealed is None or champion_final.sealed is None:
            raise RuntimeError("sealed final audit was not produced")
        sealed_audit={
            "baseline_champion":baseline_champion.champion_id,
            "final_champion":final_champion.champion_id,
            "baseline_trusted_score":baseline_final.sealed.trusted_score,
            "final_trusted_score":champion_final.sealed.trusted_score,
            "trusted_score_delta":champion_final.sealed.trusted_score-baseline_final.sealed.trusted_score,
            "suite_digest":champion_final.suite_digests["sealed"],
            "access_policy":"opened only after the final promotion decision; never used by planner or promotion gate",
        }
        self.memory.event("sealed_final_audit",**sealed_audit)
        rows=self.metrics.read(); plots=self.metrics.render_plots(rows)
        summary={"mode":self.cfg.mode.value,"attempted":attempted,"promoted":promoted,"rejected":rejected,"current_champion":final_champion.champion_id,"mathematical_rsi_enabled":self.cfg.mathematical_rsi.enabled,"research_memory_integrity":self.memory.integrity_check(),"evaluation":self.evaluation.public_description(),"final_sealed_audit":sealed_audit,"plots":[str(p) for p in plots],"bounded_claim":"This is a bounded experimental self-improvement system. It is not evidence of unrestricted or generally recursive intelligence improvement."}
        self.metrics.write_summary(summary); return summary
