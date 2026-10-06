"""Measured GPU telemetry and explicit-input MFU estimates."""
from __future__ import annotations

import asyncio
import csv
import io
import subprocess
import time


def mfu_estimate(*,training_tokens,elapsed_s,flops_per_token=None,peak_flops_per_s=None):
    if training_tokens < 0 or elapsed_s <= 0:
        raise ValueError("invalid token clock")
    if flops_per_token is None or peak_flops_per_s is None:
        return None
    if flops_per_token <= 0 or peak_flops_per_s <= 0:
        raise ValueError("FLOP model and device peak must be positive")
    # User must specify whether peak is aggregate across all participating GPUs.
    return training_tokens*flops_per_token/(elapsed_s*peak_flops_per_s)


def parse_gpu_csv(raw):
    fields = ("uuid","utilization_pct","memory_mib","power_w")
    result = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) != 4:
            raise ValueError("unexpected nvidia-smi output")
        values = [x.strip() for x in row]
        result.append(dict(zip(fields,[values[0]]+[float(x) if x not in ("[N/A]","N/A","") else None for x in values[1:]])))
    return result


class GPUProfiler:
    def __init__(self,interval_s=.5):
        if interval_s <= 0:
            raise ValueError("invalid sampling interval")
        self.interval_s = interval_s
        self.samples,self.errors = [],[]
        self.stop = asyncio.Event()

    def _sample(self):
        raw = subprocess.run(["nvidia-smi","--query-gpu=uuid,utilization.gpu,memory.used,power.draw",
                              "--format=csv,noheader,nounits"],capture_output=True,text=True,
                             check=True,timeout=5).stdout
        return parse_gpu_csv(raw)

    async def run(self):
        start = time.perf_counter()
        while not self.stop.is_set():
            try:
                rows = await asyncio.to_thread(self._sample)
                self.samples.append({"elapsed_s":time.perf_counter()-start,"devices":rows})
            except (OSError,subprocess.SubprocessError,ValueError) as exc:
                self.errors.append(type(exc).__name__)
                break
            try:
                await asyncio.wait_for(self.stop.wait(),timeout=self.interval_s)
            except TimeoutError:
                pass

    def report(self):
        return {"interval_s":self.interval_s,"samples":self.samples,"errors":self.errors,
                "available":bool(self.samples)}
