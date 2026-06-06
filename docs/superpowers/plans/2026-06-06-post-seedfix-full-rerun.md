# Post-Seedfix Full Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the pursuit-evasion benchmark after the random action-space seeding fix, extract fresh metrics, update Chinese project documentation, verify, and push the result.

**Architecture:** Keep experiment artifacts in timestamped `runs/` and `figures/` directories so previous results are not overwritten. Commit only durable source/documentation updates; generated training artifacts stay ignored unless explicitly requested.

**Tech Stack:** Windows PowerShell, `.venv\Scripts\python`, PyTorch, PettingZoo/MPE2, pytest, git.

---

### Task 1: Confirm Repository and Experiment Entry

**Files:**
- Read: `configs/random_baseline.yaml`
- Read: `configs/dqn_baseline.yaml`
- Read: `configs/dqn_curriculum_team_reward.yaml`
- Read: `configs/mappo_baseline.yaml`
- Read: `src/pursuit_lab/run_experiments.py`

- [x] **Step 1: Check git state**

Run:

```powershell
git status --short --branch
```

Expected: current branch is `main`, aligned with `origin/main`, and no uncommitted tracked changes before the long rerun starts.

- [x] **Step 2: Confirm CLI entrypoint**

Run:

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --help
```

Expected: help text includes `--configs`, `--output-dir`, `--figures`, `--episodes`, and `--seed`.

- [x] **Step 3: Confirm full-run configs**

Run:

```powershell
rg -n "total_episodes|max_cycles|eval_episodes|seeds" configs
```

Expected: trainable configs use `total_episodes: 3000`, `max_cycles: 50`, and evaluation seeds `[0, 1, 2, 3, 4]`.

### Task 2: Run Timestamped Post-Seedfix Benchmark

**Files:**
- Generate ignored artifacts under: `runs/post_seedfix_full_<timestamp>/`
- Generate ignored figures under: `figures/post_seedfix_full_<timestamp>/`
- Generate ignored transcript: `runs/post_seedfix_full_<timestamp>.log`

- [x] **Step 1: Start full experiment**

Run:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = "runs\post_seedfix_full_$stamp"
$fig = "figures\post_seedfix_full_$stamp"
$log = "runs\post_seedfix_full_$stamp.log"
"OUT=$out" | Tee-Object -FilePath $log
"FIG=$fig" | Tee-Object -FilePath $log -Append
"START=$(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
.\.venv\Scripts\python -m pursuit_lab.run_experiments `
  --configs configs/random_baseline.yaml configs/dqn_baseline.yaml configs/dqn_curriculum_team_reward.yaml configs/mappo_baseline.yaml `
  --output-dir $out `
  --figures $fig 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
"END=$(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
"EXIT=$code" | Tee-Object -FilePath $log -Append
exit $code
```

Expected: generated output directories are timestamped and no existing `runs/`, `figures/`, or `videos/` files are deleted.

- [x] **Step 2: Monitor without interrupting**

Run:

```powershell
Get-Process | Where-Object { $_.ProcessName -like '*python*' } | Select-Object Id,ProcessName,CPU,StartTime,Path
Get-Content -LiteralPath 'runs\post_seedfix_full_20260606_164815\dqn_baseline\metrics.csv' -Tail 5
```

Expected: Python process CPU time increases while metrics files grow. If the command times out but the process remains alive, treat it as a still-running job rather than a completed run.

### Task 3: Extract Fresh Metrics

**Files:**
- Read: `figures/post_seedfix_full_<timestamp>/eval_comparison.csv`
- Read: `runs/post_seedfix_full_<timestamp>/*/eval_summary.json`
- Optional read: `runs/post_seedfix_full_<timestamp>/*/metrics.csv`

- [x] **Step 1: List final summaries**

Run:

```powershell
Get-ChildItem -LiteralPath 'runs\post_seedfix_full_20260606_164815' -Directory |
  ForEach-Object {
    $summary = Join-Path $_.FullName 'eval_summary.json'
    if (Test-Path $summary) {
      "$($_.Name):"
      Get-Content -LiteralPath $summary -Raw
    }
  }
```

Expected: each completed experiment has `capture_rate`, `episodes`, `mean_episode_reward`, `mean_steps_to_capture`, and `success_episode_count`.

- [x] **Step 2: Inspect comparison CSV**

Run:

```powershell
Get-Content -LiteralPath 'figures\post_seedfix_full_20260606_164815\eval_comparison.csv' -Raw
```

Expected: comparison table contains random baseline, DQN baseline, improved DQN, and MAPPO rows. If not all rows exist, document which experiment was incomplete and do not fabricate results.

### Task 4: Update Chinese Project Documentation

**Files:**
- Modify: `PROJECT_AUDIT_REPORT.md`
- Modify: `PROJECT_INTERVIEW_GUIDE.md`

- [x] **Step 1: Update audit report with post-seedfix status**

Add a concise section to `PROJECT_AUDIT_REPORT.md` covering:

```markdown
## Post-Seedfix Long Rerun

- Command:
  `.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/dqn_baseline.yaml configs/dqn_curriculum_team_reward.yaml configs/mappo_baseline.yaml --output-dir runs\post_seedfix_full_20260606_164815 --figures figures\post_seedfix_full_20260606_164815`
- Scope: random baseline, DQN baseline, DQN + curriculum + team reward, MAPPO.
- Status: record whether every experiment completed.
- Fresh metrics: list only metrics actually produced by the run.
- Interpretation: explicitly note that metrics before commit `b60ca94` are stale because action-space seeding changed.
```

- [x] **Step 2: Update interview guide with resume-ready current numbers**

Add or adjust a short section in `PROJECT_INTERVIEW_GUIDE.md` covering:

```markdown
### 当前可引用的实验结果

说明这些结果来自 post-seedfix rerun；未完成的长跑不写成结论。随机基线、DQN、改进 DQN、MAPPO 的 capture_rate、mean_steps_to_capture 和 evaluation episode 数必须来自实际输出文件。
```

### Task 5: Verify and Commit

**Files:**
- Verify: all modified tracked files

- [ ] **Step 1: Run tests**

Run:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Expected: all tests pass. If any fail, inspect root cause before patching.

- [ ] **Step 2: Inspect diff**

Run:

```powershell
git status --short
git diff -- PROJECT_AUDIT_REPORT.md PROJECT_INTERVIEW_GUIDE.md docs/superpowers/plans/2026-06-06-post-seedfix-full-rerun.md
```

Expected: only the intended plan/docs changes are tracked. Generated `runs/` and `figures/` outputs remain ignored.

- [ ] **Step 3: Commit and push**

Run:

```powershell
git add PROJECT_AUDIT_REPORT.md PROJECT_INTERVIEW_GUIDE.md docs/superpowers/plans/2026-06-06-post-seedfix-full-rerun.md
git commit -m "docs: record post-seedfix rerun plan and results"
git push
```

Expected: commit succeeds and `main` pushes to `origin/main`.
