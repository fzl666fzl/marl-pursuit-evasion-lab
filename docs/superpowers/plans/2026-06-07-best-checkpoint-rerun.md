# Best Checkpoint Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a batch evaluator for post-seedfix `best.pt` checkpoints, run longer best-checkpoint evaluations, and update the project explanation with final-vs-best evidence.

**Architecture:** Reuse existing checkpoint loading and evaluation logic in `pursuit_lab.evaluation`. Add one small CLI, `pursuit_lab.evaluate_runs`, that scans a runs directory for experiment subdirectories containing `best.pt`, writes per-experiment evaluation files under a fresh output directory, and writes a comparison CSV.

**Tech Stack:** Windows PowerShell, `.venv\Scripts\python`, pytest, PyTorch, PettingZoo/MPE2, git.

---

### Task 1: Inspect Context

**Files:**
- Read: `src/pursuit_lab/evaluate.py`
- Read: `src/pursuit_lab/evaluation.py`
- Read: `src/pursuit_lab/plot.py`
- Read: `runs/post_seedfix_full_20260606_164815/*`

- [x] **Step 1: Confirm clean repository**

Run:

```powershell
git status --short --branch
```

Expected: `main...origin/main`, no uncommitted tracked files.

- [x] **Step 2: Confirm prior full run artifacts**

Run:

```powershell
Get-ChildItem -LiteralPath 'runs\post_seedfix_full_20260606_164815' -Directory
Get-ChildItem -LiteralPath 'figures\post_seedfix_full_20260606_164815'
```

Expected: random, DQN, improved DQN, MAPPO run directories and comparison figures exist.

### Task 2: Implement Batch Best-Checkpoint Evaluation

**Files:**
- Create: `src/pursuit_lab/evaluate_runs.py`
- Create: `tests/test_evaluate_runs.py`

- [x] **Step 1: Write failing tests**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests\test_evaluate_runs.py -q
```

Expected before implementation: `ModuleNotFoundError: No module named 'pursuit_lab.evaluate_runs'`.

- [x] **Step 2: Implement minimal CLI and helpers**

Functions:

```python
find_checkpoint_runs(runs_dir)
evaluate_run_checkpoints(runs_dir, output_dir, episodes)
```

CLI:

```powershell
.\.venv\Scripts\python -m pursuit_lab.evaluate_runs --runs <runs_dir> --output <output_dir> --episodes 100
```

- [x] **Step 3: Verify tests and help**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests\test_evaluate_runs.py -q
.\.venv\Scripts\python -m pursuit_lab.evaluate_runs --help
```

Expected: tests pass and CLI help lists `--runs`, `--output`, `--episodes`.

### Task 3: Run Long Best-Checkpoint Evaluation

**Files:**
- Generate ignored artifacts under: `runs/post_seedfix_best_eval_<timestamp>/`

- [x] **Step 1: Run 100-episode best checkpoint evaluation**

Run:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = "runs\post_seedfix_best_eval_100_$stamp"
.\.venv\Scripts\python -m pursuit_lab.evaluate_runs --runs runs\post_seedfix_full_20260606_164815 --output $out --episodes 100
```

Expected: output contains DQN, improved DQN, MAPPO subdirectories and `eval_comparison.csv`.

Actual output: `runs\post_seedfix_best_eval_100_20260607_025117\eval_comparison.csv`

- [x] **Step 2: If needed, run 500-episode evaluation to exceed one hour**

Run:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = "runs\post_seedfix_best_eval_500_$stamp"
.\.venv\Scripts\python -m pursuit_lab.evaluate_runs --runs runs\post_seedfix_full_20260606_164815 --output $out --episodes 500
```

Expected: no existing artifacts are overwritten.

Actual output: `runs\post_seedfix_best_eval_500_20260607_030815\eval_comparison.csv`

### Task 4: Update Documentation

**Files:**
- Modify: `PROJECT_AUDIT_REPORT.md`
- Modify: `PROJECT_INTERVIEW_GUIDE.md`
- Modify: `README.md`

- [x] **Step 1: Add best checkpoint comparison**

Document final-network versus `best.pt` results for post-seedfix DQN, improved DQN, and MAPPO.

- [x] **Step 2: Explain interview takeaway**

Explain that final-network results can understate DQN because the project saves `best.pt` by eval score, and stronger claims should use longer best-checkpoint evaluation.

### Task 5: Verify and Commit

- [x] **Step 1: Run focused tests**

```powershell
.\.venv\Scripts\python -m pytest tests\test_evaluate_runs.py -q
```

Actual: `2 passed in 20.66s`.

- [x] **Step 2: Run full tests**

```powershell
.\.venv\Scripts\python -m pytest -q
```

Actual: `23 passed in 235.24s`.

- [x] **Step 3: Commit and push**

```powershell
git add src/pursuit_lab/evaluate_runs.py tests/test_evaluate_runs.py README.md PROJECT_AUDIT_REPORT.md PROJECT_INTERVIEW_GUIDE.md docs/superpowers/plans/2026-06-07-best-checkpoint-rerun.md
git commit -m "feat: add best checkpoint batch evaluation"
git push
```

Actual commit: `a387ae6 feat: add best checkpoint batch evaluation`, pushed to `origin/main`.
