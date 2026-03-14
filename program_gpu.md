# GPU-Native Canonicalization — Agent Protocol

## Goal
Build a GPU-native compiler that takes a **CVXPY Problem object** (the user's program —
variables, objective, constraints, atoms) and produces conic standard form `(A, b, c)`
**entirely on the GPU**, faster than CVXPY's CPU reduction chain.

The input is the CVXPY expression tree as the user wrote it. You are free to traverse,
inspect, or reinterpret it however you want — but you do NOT route through CVXPY's
internal reduction chain, LinOp backends, or ConeMatrixStuffing. You own the entire
compilation pipeline from CVXPY Problem → GPU-resident `(A, b, c)`.

Metric: geometric mean of `gpu_canon_benchmark.py` GPU times vs CPU baseline (speedup).

## Setup (run once per session)
```bash
# Use the venv python directly (no activate needed)
export PY=.venv/bin/python
export ENSUE_API_KEY=$(cat .autoresearch-key)
# Verify GPU
$PY -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPUs')"
```

Install with `uv`:
```bash
uv pip install --python .venv/bin/python -e . cupy-cuda12x scipy numpy requests
```

## Agent Loop

### 1. RECALL
```python
import sys; sys.path.insert(0, ".")
from coordinator import Coordinator
coord = Coordinator()  # Uses namespace "cvxpy-gpu-canon"
coord.analyze()
coord.ask("GPU canonicalization insights")
best = coord.pull_best()
hypotheses = coord.list_hypotheses(status="open")
```

### 2. THINK
- Pick the highest-priority open hypothesis, OR
- Profile the current GPU pipeline and find a new bottleneck
- Focus on one change at a time

### 3. IMPLEMENT
- Edit files in `gpu_canon/` — that's your playground
- The input is always a `cvxpy.Problem` object
- The output is `(A, b, c)` as GPU arrays (CuPy sparse/dense)
- You can read the CVXPY expression tree (atoms, args, variables, parameters) but
  do NOT call CVXPY's canonicalization — build your own
- Run quick correctness check: compare your GPU output against CVXPY's CPU output

### 4. BENCHMARK
```bash
# Quick check
python gpu_canon_benchmark.py --quick --json

# Full benchmark (before publishing)
python gpu_canon_benchmark.py --json > /tmp/gpu_bench_result.json
```

Verify:
- All problems produce correct `(A, b, c)` (match CPU baseline within tolerance)
- Check GPU vs CPU speedup

### 5. PUBLISH
```python
import json, subprocess

diff = subprocess.check_output(["git", "diff", "--", "gpu_canon/"]).decode()
with open("/tmp/gpu_bench_result.json") as f:
    bench = json.load(f)

status = "keep" if bench.get("gpu_speedup", 0) > 1.0 else "discard"

coord.publish_result(
    "Description of what was changed",
    bench, diff, status
)
coord.post_insight("What we learned")
coord.publish_hypothesis("Next thing to try", "Details", priority=2)
```

### 6. REPEAT
Go back to step 1.

## Research Axes

These are the directions to explore. You don't have to follow this order — let
benchmark results guide you.

### Axis 1: Expression tree traversal on GPU
Walk the CVXPY expression tree (Problem → objective/constraints → atoms → args → leaves)
and translate it into GPU operations. Key question: which parts of the tree walk stay on
CPU (cheap, serial) vs which parts produce GPU work (matrix construction, batching)?

### Axis 2: Sparse matrix construction on GPU
Build the `A` matrix directly on GPU using CuPy CSR/CSC, or custom COO kernels.
Explore whether it's faster to accumulate triplets on CPU then transfer, or build
directly on GPU.

### Axis 3: Batched constraint processing
Process all constraints in one (or few) GPU kernel launches instead of one-at-a-time.
Group constraints by type (NonNeg, Zero, SOC) and batch their matrix contributions.

### Axis 4: LinOp chain → fused GPU ops
CVXPY atoms decompose into chains of linear operators (reshape, mul, sum, index, etc.).
Instead of executing these one at a time, fuse chains into single GPU calls.

### Axis 5: Memory layout — keep data GPU-resident
Avoid host↔device copies. Build coefficient data on GPU and keep it there.
The solver (e.g., cuOpt, GPU-native SCS) will consume it directly.

### Axis 6: Parameter updates (DPP-style)
When only Parameter values change, recompute only the parameter-dependent blocks on GPU.
Cache the static structure, update the dynamic parts.

## Key Files

| File | Role |
|------|------|
| `gpu_canon/__init__.py` | Package init, public API |
| `gpu_canon/tree.py` | CVXPY expression tree traversal + intermediate representation |
| `gpu_canon/backend.py` | GPU compilation: IR → `(A, b, c)` on GPU via CuPy |
| `gpu_canon/baseline.py` | CPU baseline: same problems via CVXPY's own canonicalization |
| `gpu_canon_benchmark.py` | Benchmark harness comparing GPU vs CPU |
| `coordinator.py` | Ensue coordinator (reused, namespace `cvxpy-gpu-canon`) |

## Rules
- The input is always a `cvxpy.Problem` — the user's program, not raw math
- Do NOT call CVXPY's reduction chain in the GPU path — build your own compiler
- CVXPY is used only for: (a) constructing test problems, (b) CPU baseline comparison
- One optimization per experiment — isolate effects
- Always verify correctness (GPU output matches CPU baseline) before publishing
- Record negative results too (status="discard")
- Keep git working tree clean between experiments
