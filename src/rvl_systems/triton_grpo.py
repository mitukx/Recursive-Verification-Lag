from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


VALID_OBJECTIVE_BACKENDS = {"torch", "triton"}


def validate_objective_backend(name: str) -> str:
    if name not in VALID_OBJECTIVE_BACKENDS:
        raise ValueError(
            f"unsupported objective backend: {name}; "
            f"expected one of {sorted(VALID_OBJECTIVE_BACKENDS)}"
        )
    return name


def triton_available() -> bool:
    return bool(
        torch is not None
        and triton is not None
        and torch.cuda.is_available()
    )


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for GRPO tensor objectives")
    return torch


def torch_grpo_surrogate(
    current_logps,
    old_logps,
    advantages,
    *,
    clip_eps: float = 0.2,
    max_abs_log_ratio: float = 20.0,
):
    """Reference tokenwise clipped GRPO/PPO surrogate."""
    t = _require_torch()
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps must be in (0, 1)")
    if max_abs_log_ratio <= 0:
        raise ValueError("max_abs_log_ratio must be positive")
    if current_logps.shape != old_logps.shape:
        raise ValueError("current and old log-probabilities must have equal shape")
    if current_logps.shape != advantages.shape:
        raise ValueError("advantages must match token log-probability shape")

    raw_log_ratio = current_logps - old_logps
    log_ratio = t.clamp(
        raw_log_ratio,
        min=-max_abs_log_ratio,
        max=max_abs_log_ratio,
    )
    ratio = t.exp(log_ratio)
    clipped = t.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    )
    surrogate = t.minimum(
        ratio * advantages,
        clipped * advantages,
    )
    clip_mask = (
        (ratio < 1.0 - clip_eps)
        | (ratio > 1.0 + clip_eps)
    ).to(current_logps.dtype)
    abs_log_ratio = log_ratio.abs()
    behavior_kl = (ratio - 1.0) - log_ratio
    return surrogate, clip_mask, abs_log_ratio, behavior_kl


if triton is not None and torch is not None:

    @triton.jit
    def _grpo_forward_kernel(
        current_ptr,
        old_ptr,
        advantage_ptr,
        surrogate_ptr,
        local_grad_ptr,
        clip_mask_ptr,
        abs_log_ratio_ptr,
        kl_ptr,
        n_elements,
        CLIP_EPS: tl.constexpr,
        MAX_ABS_LOG_RATIO: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(
            0,
            BLOCK_SIZE,
        )
        mask = offsets < n_elements
        current = tl.load(current_ptr + offsets, mask=mask).to(tl.float32)
        old = tl.load(old_ptr + offsets, mask=mask).to(tl.float32)
        advantage = tl.load(
            advantage_ptr + offsets,
            mask=mask,
        ).to(tl.float32)

        raw_log_ratio = current - old
        log_ratio = tl.maximum(
            -MAX_ABS_LOG_RATIO,
            tl.minimum(MAX_ABS_LOG_RATIO, raw_log_ratio),
        )
        ratio = tl.exp(log_ratio)
        low = 1.0 - CLIP_EPS
        high = 1.0 + CLIP_EPS
        clipped = tl.maximum(low, tl.minimum(high, ratio))

        unclipped_objective = ratio * advantage
        clipped_objective = clipped * advantage
        surrogate = tl.minimum(
            unclipped_objective,
            clipped_objective,
        )

        inside_log_clamp = (
            (raw_log_ratio >= -MAX_ABS_LOG_RATIO)
            & (raw_log_ratio <= MAX_ABS_LOG_RATIO)
        )
        active_surrogate = (
            ((advantage >= 0.0) & (ratio <= high))
            | ((advantage < 0.0) & (ratio >= low))
        )
        local_grad = tl.where(
            inside_log_clamp & active_surrogate,
            ratio * advantage,
            0.0,
        )
        clipped_flag = (ratio < low) | (ratio > high)
        behavior_kl = (ratio - 1.0) - log_ratio

        tl.store(surrogate_ptr + offsets, surrogate, mask=mask)
        tl.store(local_grad_ptr + offsets, local_grad, mask=mask)
        tl.store(
            clip_mask_ptr + offsets,
            tl.where(clipped_flag, 1.0, 0.0),
            mask=mask,
        )
        tl.store(
            abs_log_ratio_ptr + offsets,
            tl.abs(log_ratio),
            mask=mask,
        )
        tl.store(kl_ptr + offsets, behavior_kl, mask=mask)


    @triton.jit
    def _grpo_backward_kernel(
        upstream_ptr,
        local_grad_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(
            0,
            BLOCK_SIZE,
        )
        mask = offsets < n_elements
        upstream = tl.load(upstream_ptr + offsets, mask=mask).to(tl.float32)
        local_grad = tl.load(
            local_grad_ptr + offsets,
            mask=mask,
        ).to(tl.float32)
        tl.store(
            output_ptr + offsets,
            upstream * local_grad,
            mask=mask,
        )


    class _TritonGRPOSurrogate(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            current_logps,
            old_logps,
            advantages,
            clip_eps: float,
            max_abs_log_ratio: float,
        ):
            if not current_logps.is_cuda:
                raise RuntimeError("Triton GRPO requires CUDA tensors")
            if current_logps.ndim != 1:
                raise ValueError("Triton GRPO expects flat 1D tensors")
            if (
                current_logps.shape != old_logps.shape
                or current_logps.shape != advantages.shape
            ):
                raise ValueError("GRPO tensor shapes must match")
            if not 0.0 < clip_eps < 1.0:
                raise ValueError("clip_eps must be in (0, 1)")
            if max_abs_log_ratio <= 0:
                raise ValueError("max_abs_log_ratio must be positive")

            current = current_logps.contiguous()
            old = old_logps.contiguous()
            advantage = advantages.contiguous()
            surrogate = torch.empty_like(current)
            local_grad = torch.empty_like(current)
            clip_mask = torch.empty_like(current)
            abs_log_ratio = torch.empty_like(current)
            behavior_kl = torch.empty_like(current)

            n_elements = current.numel()
            grid = lambda meta: (
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            _grpo_forward_kernel[grid](
                current,
                old,
                advantage,
                surrogate,
                local_grad,
                clip_mask,
                abs_log_ratio,
                behavior_kl,
                n_elements,
                CLIP_EPS=clip_eps,
                MAX_ABS_LOG_RATIO=max_abs_log_ratio,
                BLOCK_SIZE=256,
            )
            ctx.save_for_backward(local_grad)
            ctx.input_shape = current_logps.shape
            ctx.mark_non_differentiable(
                clip_mask,
                abs_log_ratio,
                behavior_kl,
            )
            return (
                surrogate,
                clip_mask,
                abs_log_ratio,
                behavior_kl,
            )

        @staticmethod
        def backward(
            ctx,
            grad_surrogate,
            _grad_clip_mask,
            _grad_abs_log_ratio,
            _grad_behavior_kl,
        ):
            (local_grad,) = ctx.saved_tensors
            upstream = grad_surrogate.contiguous()
            output = torch.empty_like(local_grad)
            n_elements = local_grad.numel()
            grid = lambda meta: (
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            _grpo_backward_kernel[grid](
                upstream,
                local_grad,
                output,
                n_elements,
                BLOCK_SIZE=256,
            )
            return (
                output.reshape(ctx.input_shape),
                None,
                None,
                None,
                None,
            )


def triton_grpo_surrogate(
    current_logps,
    old_logps,
    advantages,
    *,
    clip_eps: float = 0.2,
    max_abs_log_ratio: float = 20.0,
):
    if not triton_available():
        raise RuntimeError(
            "Triton GRPO requires PyTorch, Triton, and a CUDA device"
        )
    return _TritonGRPOSurrogate.apply(
        current_logps,
        old_logps,
        advantages,
        float(clip_eps),
        float(max_abs_log_ratio),
    )


def grpo_surrogate(
    current_logps,
    old_logps,
    advantages,
    *,
    clip_eps: float = 0.2,
    max_abs_log_ratio: float = 20.0,
    backend: str = "torch",
):
    backend = validate_objective_backend(backend)
    if backend == "triton":
        return triton_grpo_surrogate(
            current_logps,
            old_logps,
            advantages,
            clip_eps=clip_eps,
            max_abs_log_ratio=max_abs_log_ratio,
        )
    return torch_grpo_surrogate(
        current_logps,
        old_logps,
        advantages,
        clip_eps=clip_eps,
        max_abs_log_ratio=max_abs_log_ratio,
    )
