from __future__ import annotations


VALID_PRECISIONS = {"auto", "fp32", "fp16", "bf16"}


def resolve_precision_name(
    device: str,
    requested: str,
    *,
    bf16_supported: bool = False,
) -> str:
    if requested not in VALID_PRECISIONS:
        raise ValueError(f"unsupported precision: {requested}")
    if requested != "auto":
        if requested == "bf16" and device == "cuda" and not bf16_supported:
            raise ValueError("bf16 requested but CUDA device does not report bf16 support")
        return requested
    if device == "cuda":
        return "bf16" if bf16_supported else "fp16"
    if device == "mps":
        return "fp32"
    return "fp32"


def torch_dtype_for_name(torch, precision: str):
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"precision must be resolved before dtype conversion: {precision}") from exc
