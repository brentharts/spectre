"""Independent build of Tsiokos's Lean development (ioannist/six-birds-tiles @ d53d224) on Modal.

Ephemeral: one container (8 cores, 16 GiB), no volumes, no secrets passed in.  Returns the logs and the axiom report.
"""
import modal

COMMIT = "d53d2245bc91c01cf81b6490a1cd6732e7f73bd3"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd", "ca-certificates", "build-essential")
    .run_commands(
        "curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /tmp/elan-init.sh",
        "sh /tmp/elan-init.sh -y --default-toolchain none",
        "/root/.elan/bin/elan toolchain install leanprover/lean4:v4.31.0",
    )
)
app = modal.App("r44-lean-independent-build", image=image)


@app.function(cpu=8.0, memory=16384, timeout=5400)
def build():
    import os
    import subprocess
    import time
    env = dict(os.environ, PATH="/root/.elan/bin:" + os.environ["PATH"])
    out = {}

    def run(name, cmd, cwd=None, timeout=4800):
        t = time.time()
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        out[name] = {"rc": p.returncode, "secs": round(time.time() - t, 1),
                     "stdout_tail": p.stdout[-6000:], "stderr_tail": p.stderr[-4000:]}
        return p

    run("clone", f"git clone -q https://github.com/ioannist/six-birds-tiles.git /work && cd /work && git checkout -q {COMMIT} && git log -1 --format=%H")
    r44 = "/work/lean/R44"
    run("toolchain", "cat lean-toolchain && lean --version && lake --version", cwd=r44)
    run("cache_get", "lake exe cache get", cwd=r44)
    run("build", "lake build 2>&1 | tail -60", cwd=r44)
    # fresh axiom dependency lists, and the log as it stands after the build (to be compared with the committed file)
    run("save_committed_log", "cp build_axioms.log /tmp/committed_build_axioms.log", cwd=r44)
    ax = run("axioms", "lake env lean R44/Axioms.lean > /tmp/fresh_axioms.txt 2>&1; echo rc=$?; wc -l /tmp/fresh_axioms.txt", cwd=r44)
    out["fresh_axioms"] = open("/tmp/fresh_axioms.txt").read()[-60000:]
    out["committed_axioms_log"] = open("/tmp/committed_build_axioms.log").read()[-60000:]
    run("controls", "bash scripts/controls.sh 2>&1 | tail -80", cwd=r44, timeout=2400)
    return out


@app.local_entrypoint()
def main():
    import json
    res = build.remote()
    with open("r44_build_result.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, "rc=%s" % v["rc"], "%ss" % v["secs"])
    print("wrote r44_build_result.json")
