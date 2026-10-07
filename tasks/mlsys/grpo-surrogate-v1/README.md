# GRPO surrogate optimization: development task

Optimize `torch_grpo_surrogate` in `solution.py`, preserving its API, all four
outputs and the gradient of the surrogate with respect to current logprobs.
The starting function is copied exactly from the pinned real RL implementation
in `src/rvl_systems/triton_grpo.py`; `task.json` records the commit and hashes.
The trainer in `src/rvl_systems/hf_trainer.py` calls this operator on response
tokens during causal-LM post-training.

Public diagnostics cover float32/float64, strided and empty inputs, positive,
negative and zero advantages, clipping boundaries, overflow protection and
invalid parameter/shape rejection. One CPU thread and the same image apply to
both candidates. Maintain dependency compatibility with NumPy and PyTorch;
do not add network dependencies or write output files from the solution.

Correctness gates all performance reward. Timing comes from the external
evaluator and includes numeric RPC and serialization. It is **not** a kernel
microbenchmark. Seven alternating baseline/candidate pairs are retained after
warmup, and every pair must exceed the locked 1.05 speedup threshold before a
performance reward is granted. Failures and negative measurements are retained.

This is a public **development** task with generated numerical probes, extracted
from a real repository. It is not a private test task, a task-generalization
result, a complete repository repair, an end-to-end training speedup, or evidence
of model/agent capability improvement. Private terminal probes require a spec
held by a separate evaluator, and still do not make this a held-out task bank.
