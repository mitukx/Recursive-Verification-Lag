# Progress XXIX — Pinned MBPP+ Python program bank, scoring gated

## Hypothesis and observed result

Can a pretrained 1.5B code model generate full Python functions on a public
standard coding benchmark rather than only restricted integer expressions?
The frozen eight-task, eight-occurrence pilot produced 64 unedited model
outputs. Static inspection finds 55/64 parse as Python and define the named
entry function, 59 distinct raw sources, and two outputs lacking an EOS before
the 256-token cap. Exactly 63 outputs begin with a code fence; a predetermined
fence-only extraction removes an exact complete fence. **These are unscored
programs.** There is no measurement of public pass rate, trusted pass rate,
correct-program support, recursive collapse, or refresh-controller transfer.

The pinned official Hugging Face `evalplus/mbppplus` dataset revision is
`b2d74c91837c3f2a20c1299ae98133cbe7cfa077`; the parquet SHA256 is
`dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`.
The pretrained Qwen2.5-Coder-1.5B-Instruct revision is
`2e1fd397ee46e1388853d2af2c993145b0f1098a`. Dataset selection salt,
task IDs [787, 754, 279, 606, 451, 573, 733, 4], seed 20260927,
temperature .8, top-p .95, eight samples per task and 256-token maximum were
recorded in `docs/mbppplus_real_program_protocol.md` and committed before
generation. The public test assertions were in the prompt; reference `code`
and the released, larger `test` program were not.

The completed bank is
`data/mbppplus_qwen15b_pilot_v1.jsonl`, SHA256
`1e659345d057a3e311d8f62e8bb0f7c3b213dbfccb34c6025a7dc559022eb1d7`.
Its manifest is complete and hash-verified. Generation took 743.5 seconds
on this executor's CPU. The actual user's Apple Silicon speed is not inferred
from this runtime. Per-task results are retained in
`results/mbppplus_qwen15b_static_quality.csv`.

## Evaluation gate and negative environment result

`src.score_mbppplus_docker` and `src.mbppplus_docker_worker` prepare a
disposable, digest-pinned Docker scorer with no network or host mounts and
resource limits. It requires every official reference program to pass both
public assertions and the released EvalPlus additional-test program before
any model output is scored. Public assertion fraction is cheap verifier; full
additional-test pass is binary trusted reward, explicitly evaluation-only
and not secret from pretraining. Errors and timeouts are retained as zero.

This executor has no Docker executable. Bubblewrap is installed but user
namespace setup is denied with EPERM. A scorer invocation failed closed at
Docker startup, **before any reference or model code executed**, and did not
produce a scored artifact. Thus it would be scientifically wrong to run RVL
with invented/assumed labels or call the 55 syntactically valid completions
correct. The Apple Silicon runbook builds a native arm64 image and pins its
local image ID; code and reference validation must be exercised there.

## Falsifiability and next decision

The hypothesis that a real pretrained LM supplies adequate correct-program
support remains open. Once Docker is available, score all 64 outputs, retain
all invalid completions, and report the exact task-level public/trusted
matrix. If zero or very few fully correct functions appear, record that
negative result; do not resample tasks or seeds to improve it. If support is
adequate, freeze Best-of-N/soft strengths, cadence, same-source audit stream
and adaptive controller before viewing loop outcomes. Benchmark problems and
tests are public and may have appeared in model pretraining; success is not
proof of out-of-distribution coding capability. A frozen candidate bank still
tests recursive reweighting, not parameter-updating self-improvement.
