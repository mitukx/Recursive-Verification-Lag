from __future__ import annotations
import contextlib, importlib, io, multiprocessing as mp, os, time, traceback
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from .models import ResourceLimits

@dataclass(frozen=True)
class SandboxResult:
    ok: bool; payload: Mapping[str, Any] | None; stdout: str; stderr: str; elapsed_s: float; timed_out: bool=False; error: str|None=None

def _apply_limits(limits):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(int(limits["cpu_time_s"]),int(limits["cpu_time_s"])+1))
        memory=int(limits["memory_mb"])*1024*1024
        if hasattr(resource,"RLIMIT_AS"): resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
        if hasattr(resource,"RLIMIT_FSIZE"): resource.setrlimit(resource.RLIMIT_FSIZE,(int(limits["max_output_bytes"]),int(limits["max_output_bytes"])))
    except (ImportError,OSError,ValueError): pass

def _worker(conn,entrypoint,payload,limits):
    out,err=io.StringIO(),io.StringIO()
    try:
        _apply_limits(limits)
        keep={k:os.environ[k] for k in ("PATH","PYTHONPATH","LANG","LC_ALL") if k in os.environ}; os.environ.clear(); os.environ.update(keep)
        module_name,fn_name=entrypoint.split(":",1); fn=getattr(importlib.import_module(module_name),fn_name)
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err): result=fn(payload)
        conn.send({"ok":True,"payload":result,"stdout":out.getvalue(),"stderr":err.getvalue(),"error":None})
    except BaseException:
        conn.send({"ok":False,"payload":None,"stdout":out.getvalue(),"stderr":err.getvalue(),"error":traceback.format_exc()})
    finally: conn.close()

class SandboxRunner:
    """Trusted-entrypoint-only bounded execution; never a general shell/code executor."""
    def __init__(self,allowed_entrypoints=("src.rsi_controller.evaluation:synthetic_experiment_entrypoint",)):
        self.allowed_entrypoints=set(allowed_entrypoints); self.killed=False
    def kill_switch(self): self.killed=True
    def run(self,entrypoint,payload,limits):
        if self.killed: raise RuntimeError("sandbox kill switch is active")
        if entrypoint not in self.allowed_entrypoints: raise PermissionError(f"untrusted experiment entrypoint: {entrypoint}")
        if limits.network_allowed: raise ValueError("bounded RSI sandbox does not permit network access")
        ctx=mp.get_context("spawn"); parent,child=ctx.Pipe(duplex=False); p=ctx.Process(target=_worker,args=(child,entrypoint,dict(payload),asdict(limits))); start=time.monotonic(); p.start(); child.close(); p.join(limits.wall_time_s); elapsed=time.monotonic()-start
        if p.is_alive():
            p.terminate(); p.join(2)
            if p.is_alive(): p.kill(); p.join(2)
            parent.close(); return SandboxResult(False,None,"","",elapsed,True,"experiment timeout")
        if parent.poll():
            try: msg=parent.recv()
            except EOFError: msg={"ok":False,"payload":None,"stdout":"","stderr":"","error":f"child exited {p.exitcode} without result"}
        else: msg={"ok":False,"payload":None,"stdout":"","stderr":"","error":f"child exited {p.exitcode}"}
        parent.close(); cap=limits.max_output_bytes
        return SandboxResult(bool(msg.get("ok")),msg.get("payload"),str(msg.get("stdout",""))[:cap],str(msg.get("stderr",""))[:cap],elapsed,error=msg.get("error"))

def timeout_test_entrypoint(payload):
    time.sleep(float(payload.get("sleep_s",0.0))); return {"ok":True}
