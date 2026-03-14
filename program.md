# CVXPY Canon Optimizer — Agent Protocol v2

## SETUP
```bash
source .venv/bin/activate && export ENSUE_API_KEY=$(cat .autoresearch-key)
```

## PHASE 1: RECALL (mandatory)
```python
from coordinator import Coordinator; coord = Coordinator()
coord.analyze()                          # summary of all work
coord.list_hypotheses(status="open")     # what to try
```
Do NOT load source files into context yet.

## PHASE 2: CHECK PRIOR ART (mandatory)
```python
coord.check_tried("description of planned experiment")
```
If similar experiment found with conclusive result → skip, pick another.

## PHASE 3: THINK
Pick highest-priority open hypothesis OR propose a radical new idea.
Read ONLY the files you will modify.

**Correctness oracles**: `get_problem_data(solver, canon_backend='SCIPY')` gives
ground-truth A, b, c matrices. ANY approach producing identical output is valid.

Think beyond incremental: new data structures, JIT, lazy eval, caching, fusion…

## PHASE 4: IMPLEMENT + TEST
Edit files. Run:
```bash
pytest cvxpy/tests/ -x -q
```

## PHASE 5: BENCHMARK + VERIFY (use /bench skill)
```bash
python canon_benchmark.py --quick --json --verify           # fast check
python canon_benchmark.py --all-backends --json --trace --verify  # full run
```

## PHASE 6: PUBLISH (mandatory, even for failures)
```python
coord.publish_result(description, bench_json, git_diff, "keep"|"discard"|"error")
coord.post_insight("what we learned")          # embed for future search
coord.publish_hypothesis("next idea", ...)     # seed future sessions
```
If `--trace` revealed hot functions:
```python
coord.publish_trace(problem_name, backend, trace_data)
```

## RULES
1. ONE change per experiment — isolate effects
2. ALWAYS `--verify` — never publish unverified results
3. ALWAYS publish — negative results prevent re-trying
4. Do NOT load files you aren't editing — use Ensue for context
5. Existing backends are oracles — matching A,b,c = correct
6. Think creatively — radical approaches welcome if oracle-verified
7. Traces go to Ensue — if `--trace` reveals a hot function, `coord.publish_trace()`
