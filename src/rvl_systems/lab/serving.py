"""Episode-pinned blue/green serving built on the existing scheduler."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

from ..scheduler import LeastLoadedScheduler, WorkerSlot


class ServingLease:
    def __init__(self,version,scheduler):
        self.version,self.scheduler = version,scheduler
        self.closed = False

    async def generate(self,request):
        if self.closed:
            raise RuntimeError("episode lease already released")
        rows = await self.scheduler.run([request])
        return [replace(g,metadata={**g.metadata,"policy_version":self.version}) for g in rows]


class VersionedServingFleet:
    """Activate health-probed immutable pools; drain old episode leases.

    Each version must be served by separate backend instances, with immutable
    model names/weight artifacts. This is blue/green routing, not in-place
    vLLM weight reload. Capacity for two versions is an explicit requirement.
    """
    def __init__(self):
        self.pools,self.references = {},{}
        self.active = None
        self.lock = asyncio.Lock()

    async def prepare(self,version,workers,probe,**scheduler_options):
        if version in self.pools or not workers:
            raise ValueError("invalid or duplicate pool")
        if any(id(w.backend)==id(old.backend)
               for pool in self.pools.values() for old in pool.workers for w in workers):
            raise ValueError("backend reused across immutable policy versions")
        results = await asyncio.gather(*(probe(w,version) for w in workers),return_exceptions=True)
        if any(isinstance(r,BaseException) or r is not True for r in results):
            raise RuntimeError("candidate pool failed version/health probe")
        self.pools[version] = LeastLoadedScheduler(workers,**scheduler_options)
        self.references[version] = 0

    async def activate(self,version):
        async with self.lock:
            if version not in self.pools or (self.active is not None and version <= self.active):
                raise ValueError("activation must advance to a prepared version")
            self.active = version

    @asynccontextmanager
    async def episode(self):
        async with self.lock:
            if self.active is None:
                raise RuntimeError("no active serving pool")
            version = self.active
            self.references[version] += 1
            lease = ServingLease(version,self.pools[version])
        try:
            yield lease
        finally:
            lease.closed = True
            async with self.lock:
                self.references[version] -= 1

    async def retire(self,version):
        async with self.lock:
            if version == self.active or self.references.get(version,0):
                raise RuntimeError("pool is active or has in-flight episodes")
            self.pools.pop(version)
            self.references.pop(version)

    def status(self):
        return {"active":self.active,"episode_references":dict(self.references)}
