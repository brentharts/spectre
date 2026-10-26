"""Run Tsiokos's discharge diff audit at d53d224 with the delivery ZIPs restored from the reader commit 838bca5
(whose Lean sources are identical).  Tiny ephemeral container, no volumes."""
import modal

app = modal.App("r44-diff-audit", image=modal.Image.debian_slim(python_version="3.12").apt_install("git"))


@app.function(cpu=0.25, memory=1024, timeout=900)
def audit():
    import subprocess
    cmd = ("git clone -q https://github.com/ioannist/six-birds-tiles.git /w && cd /w && "
           "git checkout -q d53d2245bc91c01cf81b6490a1cd6732e7f73bd3 && git fetch -q origin 838bca514679b2531d31f4e0dfd66641269e2a8c && "
           "git diff --quiet d53d2245bc91c01cf81b6490a1cd6732e7f73bd3 838bca514679b2531d31f4e0dfd66641269e2a8c -- lean/R44/R44 && echo 'Lean sources identical at both commits' && "
           "git checkout -q 838bca514679b2531d31f4e0dfd66641269e2a8c -- proof/external_lean && "
           "cd lean/R44 && bash scripts/discharge_diff_audit.sh; echo rc=$?")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout[-6000:] + p.stderr[-3000:]


@app.local_entrypoint()
def main():
    print(audit.remote())
