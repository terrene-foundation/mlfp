"""TEMPORARY audit runner (untracked; delete after the audit). Executes every
solution script for the given modules and prints one JSON line per script plus
the failing tail. Exit 0 only if all pass."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mods = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"mlfp0{i}" for i in range(1, 7)]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
files = sorted(
    p for m in mods for p in (ROOT / f"modules/{m}/solutions").rglob("*.py")
    if "__pycache__" not in p.parts and not p.name.startswith("_")
)
env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")

# Files exempt from the warnings-as-errors gate (documented upstream cause):
# - ex_0/00_destination_first.py: km.train fans families out to spawned worker
#   processes; spawn inherits -W flags but not in-process warning filters, so
#   Lightning's "GPU available but not used" nag (PossibleUserWarning) is fatal
#   there. See decisions.md P6.
STRICT_SKIP = ("mlfp05/solutions/ex_0/00_destination_first.py",)
bad = 0
for p in files:
    rel = str(p.relative_to(ROOT))
    t = time.time()
    try:
        cmd = [str(ROOT / ".venv/bin/python")]
        if not any(rel.endswith(skip) for skip in STRICT_SKIP):
            cmd += ["-W", "error::UserWarning"]
        cmd.append(str(p))
        r = subprocess.run(cmd, cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = "TIMEOUT", str(e.stdout or "")[-1500:], str(e.stderr or "")[-3000:]
    noisy = "_connection_worker_thread" in err
    rec = {"file": rel, "code": code, "secs": round(time.time() - t), "shutdown_noise": noisy}
    print(json.dumps(rec), flush=True)
    if code != 0:
        bad += 1
        print(f"----- tail stdout {rel} -----\n{out[-1500:]}\n----- tail stderr -----\n{err[-3000:]}", flush=True)
print(f"SUMMARY {len(files) - bad}/{len(files)} passed", flush=True)
sys.exit(1 if bad else 0)
