from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Int8RowQuantization:
    qvalues: tuple[tuple[int, ...], ...]
    scales: tuple[float, ...]

    @property
    def rows(self) -> int:
        return len(self.qvalues)

    @property
    def cols(self) -> int:
        return len(self.qvalues[0]) if self.qvalues else 0


@dataclass(frozen=True)
class QuantizationDiagnostics:
    max_abs_output_error: float
    mean_abs_output_error: float
    max_relative_output_error: float
    mean_relative_output_error: float
    theoretical_storage_ratio: float


def _validate_matrix(matrix: list[list[float]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    cols = len(matrix[0])
    for row in matrix:
        if len(row) != cols:
            raise ValueError("matrix must be rectangular")
        if not all(math.isfinite(float(x)) for x in row):
            raise ValueError("matrix contains non-finite values")
    return len(matrix), cols


def quantize_per_row_symmetric_int8(
    matrix: list[list[float]],
) -> Int8RowQuantization:
    """Reference per-output-channel symmetric INT8 weight quantization.

    This is a numerical contract, not an optimized kernel. Each row receives
    its own FP32 scale. Zero rows use scale=1 so dequantization remains exact.
    """
    _validate_matrix(matrix)
    qrows: list[tuple[int, ...]] = []
    scales: list[float] = []
    for row in matrix:
        bound = max(abs(float(x)) for x in row)
        scale = bound / 127.0 if bound > 0.0 else 1.0
        q = tuple(
            max(-127, min(127, int(round(float(x) / scale))))
            for x in row
        )
        qrows.append(q)
        scales.append(scale)
    return Int8RowQuantization(tuple(qrows), tuple(scales))


def dequantize_int8(q: Int8RowQuantization) -> list[list[float]]:
    if not q.qvalues or len(q.qvalues) != len(q.scales):
        raise ValueError("invalid quantized matrix")
    cols = len(q.qvalues[0])
    out: list[list[float]] = []
    for row, scale in zip(q.qvalues, q.scales):
        if len(row) != cols or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid quantized row")
        if any(value < -127 or value > 127 for value in row):
            raise ValueError("INT8 value out of symmetric range")
        out.append([value * scale for value in row])
    return out


def linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    _, cols = _validate_matrix(matrix)
    if len(vector) != cols or not all(math.isfinite(float(x)) for x in vector):
        raise ValueError("invalid input vector")
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def evaluate_int8_linear_parity(
    weights: list[list[float]],
    inputs: list[list[float]],
) -> QuantizationDiagnostics:
    rows, cols = _validate_matrix(weights)
    if not inputs:
        raise ValueError("inputs must be non-empty")
    quantized = quantize_per_row_symmetric_int8(weights)
    reconstructed = dequantize_int8(quantized)
    abs_errors: list[float] = []
    relative_errors: list[float] = []
    for vector in inputs:
        reference = linear(weights, vector)
        actual = linear(reconstructed, vector)
        for expected, observed in zip(reference, actual):
            error = abs(observed - expected)
            abs_errors.append(error)
            relative_errors.append(error / max(abs(expected), 1e-8))

    fp32_bytes = rows * cols * 4
    int8_bytes = rows * cols + rows * 4
    return QuantizationDiagnostics(
        max_abs_output_error=max(abs_errors, default=0.0),
        mean_abs_output_error=sum(abs_errors) / max(1, len(abs_errors)),
        max_relative_output_error=max(relative_errors, default=0.0),
        mean_relative_output_error=sum(relative_errors) / max(1, len(relative_errors)),
        theoretical_storage_ratio=fp32_bytes / int8_bytes,
    )
