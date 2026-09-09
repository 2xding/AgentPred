# AgentPred — Kaggle-in-Kaggle autonomous ML agent

An entry for a [Kaggle-in-Kaggle](https://www.kaggle.com/) style competition, where the
submission is not a set of predictions but an **autonomous agent**: a YAML-declared LLM
pipeline that, inside a network-isolated sandbox, explores a tabular dataset, trains models,
submits predictions, and picks its own best candidate — with no human in the loop.

The task in every dataset is binary classification scored by ROC AUC. The harness runs the
agent under hard budgets (wall clock, tool calls, submission count, USD spend), takes the
**top 2 candidates by public score**, and awards the better of their **private** scores.

## The core idea

LLM agents are unreliable at the *modeling* part of ML and unnecessary for it. So this design
pushes all the modeling into deterministic Python and uses the LLM only as a controller —
then spends whatever budget is left on a speculative optimizer that can only ever help.

The final submission is a three-phase `SequentialAgent`:

| Phase | Agent | Model | Job |
|---|---|---|---|
| 1 | `quick_baseline_controller` | `gemini-2.5-flash-lite` | Get *a* valid submission on the board immediately. Prior-probability fallback, then a fast CatBoost. |
| 2 | `portfolio_controller` | `gemini-2.5-flash-lite` | Run the 5-fold model portfolio, then submit each candidate from the manifest. |
| 3 | `gemini_pro_optimizer` | `gemini-3.1-pro-preview` | Freeroll: with the remaining ~45 min and ~$1.90, iterate on feature engineering to try to beat the portfolio. |

Phases 1 and 2 are the score. Phase 3 is a free option — by the time it runs, a good
submission is already banked, so it can only add upside.

### Why the cheap model runs the pipeline

The phase-1 and phase-2 prompts give the LLM almost nothing to decide. It calls `get_status()`,
runs one fixed skill script, parses a single `PORTFOLIO_MANIFEST` JSON line from stdout, and
submits exactly the paths in that line — never a path it invented. A flash-lite model does that
reliably and cheaply, which leaves the entire remaining budget for phase 3.

The prompts also encode the failure modes that actually kill runs in this environment:
never `cat` a CSV (it blows the context window), every response must contain a tool call, and
never read `solution.csv`. Time checks are defensive — phase 2 bails out if fewer than 8
minutes remain before modeling, and re-checks before submitting, treating a missing,
non-numeric, or non-finite `time_minutes_remaining` as a stop signal rather than a green light.

### The portfolio (`skills/robust-tabular`)

`run_portfolio.py` is the actual model. Five-fold stratified CV over four base learners —
CatBoost with native categoricals, LightGBM with native categoricals, ExtraTrees, and a
regularized logistic model — plus blends built on top of them: equal-weight rank averaging over
all models, rank averaging over the top 2, and an OOF-gated weighted rank blend. Every candidate
is written to `/work` and listed in CV order in the manifest.

That ordering is the part that matters. The harness selects on public score, so the portfolio's
job is to make sure the *candidates* are all good — the selection mechanism then does the rest.

## Results

Benchmarked offline with `bench_portfolio.py`, which runs the portfolio script in the sandbox
image across the 16 practice datasets and reproduces the harness's award rule exactly
(top 2 by public, best private of those two):

| Run | Mean awarded private AUC |
|---|---|
| Early portfolio (`bench_stock.log`) | 0.7174 |
| Multi-seed bagged (`bench_edge.log`) | 0.8042 |
| Final, single-seed (`bench_stock2.log`) | 0.8040 |

The jump from 0.717 to 0.804 came from fixing the degenerate cases: the early run collapsed to
0.5000 AUC on `train_06` and `train_08` and scraped 0.4988 on `train_13`. The final portfolio
has no dataset below 0.64.

**Awarded ≈ best possible.** On 15 of 16 datasets the awarded private score equals the best
private score available among all candidates, meaning public-score selection almost never picks
the wrong one. The gap where it exists is in the fourth decimal.

Multi-seed bagging (averaging OOF and test predictions over three fold seeds, and only on
datasets ≤20k rows where fold noise dominates) was worth +0.0002 mean AUC — inside the noise,
at roughly 3× the compute. It was cut from the final submission to buy runtime for the phase-3
freeroll. That decision is preserved as `04_edge` if it's ever worth revisiting.

## Layout

```
submissions/
  01_baseline/    single LLM agent, one prompt
  02_pro/         same shape, stronger model + prompt
  03_freeroll/    three-phase split: quick / portfolio / pro
  04_edge/        + multi-seed bagging in the portfolio
  05_final/       shipped: three-phase, single-seed portfolio
sample_submission/    immutable reference template from the harness
kaggle-kaggle-skill/  competitor-guide skill (submission mechanics, sandbox rules)
wheels/               local wheels: adk_submission, kaggle_kaggle
bench_portfolio.py    offline portfolio benchmark, no LLM in the loop
run_local_eval.py     local end-to-end evaluation harness
validate_submission.py  pre-submit archive validator
models.yaml           competition model registry
sandbox.Dockerfile    local sandbox image (aap-sandbox:local)
```

Each experiment directory is self-contained by design — the harness rules require it, and it
keeps every run reproducible after the fact.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY, or point at the Kaggle model proxy
docker build -t aap-sandbox:local -f sandbox.Dockerfile .
```

Then:

```bash
python validate_submission.py --agent-dir submissions/05_final/agent
python bench_portfolio.py submissions/05_final/agent   # offline, no LLM cost
python run_local_eval.py --submission-dir submissions/05_final/agent --dataset train_13
```

## Not in this repo

- `data/` — 16 practice datasets (~76 MB) from the competition. They include `solution.csv`
  label files, so they are not redistributed here. Pull them from the competition page.
- `.env` — API keys. `.env.example` shows the shape.
