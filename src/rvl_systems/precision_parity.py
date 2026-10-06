from __future__ import annotations

import math
import struct
from dataclasses import dataclass


def quantize_fp16(value: float) -> float:
    return struct.unpack(">e", struct.pack(">e", float(value)))[0]


def quantize_bf16(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    bf16 = rounded & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bf16))[0]


@dataclass(frozen=True)
class ParityDiagnostics:
    max_abs_value_error: float
    max_relative_ratio_error: float
    mean_relative_ratio_error: float


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def evaluate_logprob_parity(
    values: list[tuple[float, float]],
    *,
    precision: str,
    max_abs_log_ratio: float = 8.0,
) -> ParityDiagnostics:
    if precision == "fp16":
        quantize = quantize_fp16
    elif precision == "bf16":
        quantize = quantize_bf16
    else:
        raise ValueError("precision must be fp16 or bf16")

    value_errors: list[float] = []
    ratio_errors: list[float] = []
    for new, old in values:
        q_new = quantize(new)
        q_old = quantize(old)
        value_errors.extend([abs(q_new - new), abs(q_old - old)])
        reference_delta = max(
            -max_abs_log_ratio,
            min(max_abs_log_ratio, new - old),
        )
        quantized_delta = max(
            -max_abs_log_ratio,
            min(max_abs_log_ratio, q_new - q_old),
        )
        reference_ratio = math.exp(reference_delta)
        quantized_ratio = math.exp(quantized_delta)
        ratio_errors.append(
            _relative_error(quantized_ratio, reference_ratio)
        )

    return ParityDiagnostics(
        max_abs_value_error=max(value_errors, default=0.0),
        max_relative_ratio_error=max(ratio_errors, default=0.0),
        mean_relative_ratio_error=(
            sum(ratio_errors) / len(ratio_errors)
            if ratio_errors
            else 0.0
        ),
    )
