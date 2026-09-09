"""Score the deterministic portfolio script offline, with no LLM in the loop.

Runs skills/robust-tabular/scripts/run_portfolio.py inside the sandbox image on
each practice dataset, then reproduces what the harness would actually award:
the auto-selection takes the top 2 candidates by PUBLIC score, and the final
standing is the best PRIVATE score among those two.

Usage:
  python bench_portfolio.py <agent_dir> [dataset ...]
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGE = "aap-sandbox:local"


def bench(agent_dir: Path, ds: str):
    d = DATA / ds
    work = Path(tempfile.mkdtemp(prefix=f"bench_{ds}_"))
    for f in ("train.csv", "test.csv", "sample_submission.csv"):
        shutil.copy(d / f, work / f)
    scripts = agent_dir / "skills/robust-tabular/scripts"
    shutil.copytree(scripts, work / "scripts")

    proc = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{work}:/work", "-e", "ROBUST_TABULAR_WORKDIR=/work",
         "-w", "/work", IMAGE, "python", "scripts/run_portfolio.py"],
        capture_output=True, text=True, timeout=3600,
    )
    manifest = None
    for line in proc.stdout.splitlines():
        if "PORTFOLIO_MANIFEST" in line:
            manifest = json.loads(line.split("PORTFOLIO_MANIFEST=", 1)[1].strip())
    if manifest is None:
        return {"dataset": ds, "error": (proc.stderr or proc.stdout)[-400:]}

    sol = pd.read_csv(d / "solution.csv")
    rows = []
    for cand in manifest["candidates"]:
        p = work / Path(cand if isinstance(cand, str) else cand["path"]).name
        if not p.exists():
            continue
        sub = pd.read_csv(p)
        j = sol.merge(sub, on=sol.columns[0], suffixes=("_sol", "_sub"))
        tgt = [c for c in j.columns if c.endswith("_sub")][0]
        pub = roc_auc_score(j[j.Usage == "Public"]["target_sol"], j[j.Usage == "Public"][tgt])
        pri = roc_auc_score(j[j.Usage == "Private"]["target_sol"], j[j.Usage == "Private"][tgt])
        rows.append({"name": p.stem, "public": pub, "private": pri})

    rows.sort(key=lambda r: r["public"], reverse=True)
    selected = rows[:2]
    shutil.rmtree(work, ignore_errors=True)
    return {
        "dataset": ds,
        "candidates": rows,
        "awarded_private": max(r["private"] for r in selected) if selected else None,
        "best_possible_private": max(r["private"] for r in rows) if rows else None,
    }


if __name__ == "__main__":
    agent_dir = Path(sys.argv[1])
    names = sys.argv[2:] or sorted(p.name for p in DATA.iterdir() if p.is_dir())
    awarded = []
    for ds in names:
        r = bench(agent_dir, ds)
        if "error" in r:
            print(f"{ds}  FAILED  {r['error'][:200]}", flush=True)
            continue
        best = max(r["candidates"], key=lambda c: c["public"])
        print(f"{ds}  awarded_private={r['awarded_private']:.4f}  "
              f"best_possible={r['best_possible_private']:.4f}  "
              f"top_by_public={best['name']}", flush=True)
        awarded.append(r["awarded_private"])
    if awarded:
        print(f"\nMEAN awarded private = {sum(awarded)/len(awarded):.4f}  over {len(awarded)} datasets")
