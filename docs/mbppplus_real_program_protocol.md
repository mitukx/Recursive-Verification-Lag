# Locked MBPP+ real-program pilot

## Scientific status and hypothesis

This is the first standard-function program-generation pilot in the RVL
repository. The primary feasibility outcome is whether the pretrained model
can produce nontrivial complete Python functions from natural-language
problems and public examples. The full recursive public/extra-test experiment
requires an isolated evaluator and must not be claimed until it actually runs.
The eight sampled benchmark problems are a small feasibility pilot, not a
population-level performance estimate. They may appear in pretraining data.

## Frozen acquisition and generation

Official `evalplus/mbppplus` Hugging Face dataset, revision
`b2d74c91837c3f2a20c1299ae98133cbe7cfa077`, parquet SHA256
`dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`.
Only the public natural-language prompt and original `test_list` assertions
are exposed to generation; reference `code` and larger `test` are excluded.
Select the first eight SHA256(`rvl-mbppplus-pilot-v1:` + task_id) tasks among
those with exactly one reference function used in at least three public tests
and a problem prompt shorter than 600 characters. The selection was fixed
before viewing model output, and yields IDs 787, 754, 279, 606, 451, 573,
733, 4. The code stores source hashes of all test material in its manifest.

Qwen/Qwen2.5-Coder-1.5B-Instruct revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`: seed 20260927,
eight occurrences per task, temperature .8, top-p .95, up to 256 new tokens,
batch size 4. Retain all outputs and duplicates without editing or scoring.
The first observation is bank integrity, program shape, distinct-source count
and truncation. Avoid choosing a second seed or task list in response to
observed quality.

## Trusted evaluation gate

The official dataset's `test_list` gives cheap public checks; the larger
EvalPlus `test` is evaluation-only reward. These tests are *publicly released*,
so ``hidden'' means withheld from the optimization loop, not secret from model
pretraining. Program code must be run only in a pinned, disposable container
with no network, no host mounts, no privileges and strict resource limits. This
executor has neither an operable Docker daemon nor user namespaces (bubblewrap
fails with EPERM). Therefore no trusted labels or recursive-loop outcomes are
claimed from an unscored bank. A separate Mac M1 Pro/32GB runner can use Docker
Desktop to perform the next gate, after the evaluator is independently tested
on official reference solutions and adversarial completions.

The repository includes `docker/Dockerfile.mbppplus` (Python and NumPy) and
`src.score_mbppplus_docker`. On an Apple Silicon Docker Desktop machine,
install the dependencies from `requirements-mbppplus.txt`, build a local arm64
image, and obtain its immutable local ID:

```sh
python -m pip install -r requirements-mbppplus.txt
docker build --platform linux/arm64 -f docker/Dockerfile.mbppplus -t rvl-mbppplus:pilot .
RVL_IMAGE_ID="$(docker image inspect --format '{{.Id}}' rvl-mbppplus:pilot)"
python -m src.score_mbppplus_docker data/mbppplus_qwen15b_pilot_v1.jsonl --image "$RVL_IMAGE_ID" --output data/mbppplus_qwen15b_pilot_v1_scored.jsonl
```

This passes the exact local `sha256:...` image ID to the scorer. It first tests all eight
official reference implementations in
fresh containers. If any reference fails, it aborts before assigning model
scores. It then launches a fresh container per generated candidate with no
network or host mounts, read-only root, unprivileged UID, limited processes,
CPU, memory, output, and wall time. The original public assertions yield a
fractional cheap score; the additional released EvalPlus test program gives a
binary trusted score. Container failures/timeouts count as zero and remain in
the bank. The image ID, source hashes and timeouts are recorded. Test code
provenance remains pinned to the downloaded official parquet SHA256; it is
never treated as secret. This execution path has **not** been run or certified
on this executor because Docker is unavailable.

## Decision after generation

If outputs include plausible callable programs, implement and validate the
isolated scorer and only then preregister the controller comparisons. If the
model produces mostly invalid or truncated functions, retain the negative
result and revise generation capability in a new protocol. MBPP+ success
alone would still not establish genuine recursive model improvement: the
frozen-bank policy is a reweighting loop.
