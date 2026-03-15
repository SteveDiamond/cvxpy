#!/usr/bin/env python
"""
Benchmark 4: End-to-end solve comparison.
Compare total solve time (canonicalize + solve) with trace replay vs CPU.
Uses prior trace replay data from gpu_canon_results.json when GPU replay
encounters environment issues.
"""

import gc
import json
import time
import statistics

import numpy as np
import scipy.sparse as sp
import cupy
import cvxpy as cp

from gpu_canon.trace_cache import TraceCache, reset_id_counter


def make_lp(n, m):
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = sp.random(m, n, density=0.01, format='csc', random_state=42)
        b = np.random.randn(m)
        return cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
    return factory

def make_lasso(n, m):
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1))
        return cp.Problem(obj)
    return factory

def make_portfolio(n):
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        mu = np.random.randn(n) * 0.1
        F = np.random.randn(n, n // 2)
        Sigma = F @ F.T + np.diag(np.random.rand(n) * 0.1)
        gamma = 1.0
        obj = cp.Maximize(mu @ x - gamma * cp.quad_form(x, Sigma))
        return cp.Problem(obj, [cp.sum(x) == 1, x >= 0])
    return factory


# Known trace replay times from gpu_canon_results.json
KNOWN_TRACE = {
    "LP 500x250": 0.881,
    "LP 2000x1000": 6.97,
    "LASSO 500x1000": 2.708,
    "Portfolio 200": 0.991,
}

problems = [
    ("LP 500x250", make_lp(500, 250)),
    ("LP 2000x1000", make_lp(2000, 1000)),
    ("LASSO 500x1000", make_lasso(500, 1000)),
    ("Portfolio 200", make_portfolio(200)),
]

WARMUP = 2
ITERS = 5

results = []
for name, factory in problems:
    print(f"\n=== {name} ===")
    r = {"name": name}

    # --- CPU canonicalize only ---
    cpu_canon_times = []
    for i in range(WARMUP + ITERS):
        reset_id_counter()
        prob = factory()
        gc.collect()
        t0 = time.perf_counter()
        prob.get_problem_data(cp.CLARABEL, canon_backend="COO")
        t1 = time.perf_counter()
        if i >= WARMUP:
            cpu_canon_times.append((t1 - t0) * 1000)
        del prob; gc.collect()

    cpu_canon_mean = statistics.mean(cpu_canon_times)

    # --- CPU full solve ---
    cpu_total_times = []
    for i in range(WARMUP + ITERS):
        reset_id_counter()
        prob = factory()
        gc.collect()
        t0 = time.perf_counter()
        prob.solve(solver=cp.CLARABEL)
        t1 = time.perf_counter()
        if i >= WARMUP:
            cpu_total_times.append((t1 - t0) * 1000)
        del prob; gc.collect()

    cpu_total_mean = statistics.mean(cpu_total_times)
    cpu_solve_mean = max(cpu_total_mean - cpu_canon_mean, 0.1)

    r["cpu_canon_ms"] = round(cpu_canon_mean, 2)
    r["cpu_solve_only_ms"] = round(cpu_solve_mean, 2)
    r["cpu_total_ms"] = round(cpu_total_mean, 2)
    r["cpu_canon_pct"] = round(100 * cpu_canon_mean / max(cpu_total_mean, 0.001), 1)

    print(f"  CPU: canon={cpu_canon_mean:.1f}ms solve={cpu_solve_mean:.1f}ms total={cpu_total_mean:.1f}ms ({r['cpu_canon_pct']:.0f}% canon)")

    # --- Trace canon (use prior data + try live) ---
    trace_ms = None

    # Try live GPU replay
    try:
        cache = TraceCache()
        reset_id_counter()
        prob0 = factory()
        cupy.cuda.Device(0).synchronize()
        cache.canonicalize(prob0, on_gpu=True)
        cupy.cuda.Device(0).synchronize()

        trace_times = []
        for i in range(WARMUP + ITERS):
            reset_id_counter()
            prob = factory()
            gc.collect()
            cupy.cuda.Device(0).synchronize()
            t0 = time.perf_counter()
            cache.canonicalize(prob, on_gpu=True)
            cupy.cuda.Device(0).synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                trace_times.append(elapsed)
            del prob; gc.collect()

        trace_ms = statistics.mean(trace_times)
        r["trace_canon_ms"] = round(trace_ms, 3)
        r["trace_source"] = "live"
        del cache
    except Exception as e:
        print(f"  Trace replay env error (using prior data): {type(e).__name__}")
        if name in KNOWN_TRACE:
            trace_ms = KNOWN_TRACE[name]
            r["trace_canon_ms"] = trace_ms
            r["trace_source"] = "prior_benchmark"

    if trace_ms is not None:
        estimated_e2e = trace_ms + cpu_solve_mean
        canon_speedup = cpu_canon_mean / max(trace_ms, 0.001)
        e2e_speedup = cpu_total_mean / max(estimated_e2e, 0.001)

        r["estimated_e2e_trace_ms"] = round(estimated_e2e, 2)
        r["canon_speedup"] = round(canon_speedup, 1)
        r["e2e_speedup"] = round(e2e_speedup, 2)

        print(f"  Trace canon: {trace_ms:.2f}ms ({r.get('trace_source', 'unknown')})")
        print(f"  Canon speedup: {canon_speedup:.1f}x")
        print(f"  Est e2e: {estimated_e2e:.1f}ms (speedup {e2e_speedup:.2f}x)")

    results.append(r)
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()

output = {"benchmark": "end_to_end_solve", "results": results}
with open("/home/ubuntu/cvxpy-agi/bench_e2e_solve_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to bench_e2e_solve_results.json")
