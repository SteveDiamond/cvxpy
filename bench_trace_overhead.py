#!/usr/bin/env python
"""
Benchmark 1: Trace overhead on cold path.
Measures how much slower traced get_problem_data() is vs untraced.
"""

import gc
import json
import time
import statistics

import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from gpu_canon.tracer import NumpyTracer
from gpu_canon.trace_cache import reset_id_counter

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

def time_untraced(factory, warmup=3, iters=10):
    times = []
    for i in range(warmup + iters):
        reset_id_counter()
        prob = factory()
        gc.collect()
        start = time.perf_counter()
        prob.get_problem_data(cp.CLARABEL, canon_backend="COO")
        elapsed = (time.perf_counter() - start) * 1000
        if i >= warmup:
            times.append(elapsed)
        del prob
        gc.collect()
    return times

def time_traced(factory, warmup=3, iters=10):
    times = []
    for i in range(warmup + iters):
        reset_id_counter()
        prob = factory()
        gc.collect()
        start = time.perf_counter()
        with NumpyTracer() as tracer:
            prob.get_problem_data(cp.CLARABEL, canon_backend="COO")
        elapsed = (time.perf_counter() - start) * 1000
        if i >= warmup:
            times.append(elapsed)
        del prob
        gc.collect()
    return times

problems = [
    ("LP 500x250", make_lp(500, 250)),
    ("LASSO 500x1000", make_lasso(500, 1000)),
]

results = []
for name, factory in problems:
    print(f"Benchmarking {name}...")
    untraced = time_untraced(factory)
    traced = time_traced(factory)

    ut_mean = statistics.mean(untraced)
    ut_std = statistics.stdev(untraced)
    tr_mean = statistics.mean(traced)
    tr_std = statistics.stdev(traced)
    overhead_pct = ((tr_mean - ut_mean) / ut_mean) * 100

    r = {
        "name": name,
        "untraced_mean_ms": round(ut_mean, 3),
        "untraced_std_ms": round(ut_std, 3),
        "traced_mean_ms": round(tr_mean, 3),
        "traced_std_ms": round(tr_std, 3),
        "overhead_ms": round(tr_mean - ut_mean, 3),
        "overhead_pct": round(overhead_pct, 1),
    }
    results.append(r)
    print(f"  Untraced: {ut_mean:.2f} +/- {ut_std:.2f} ms")
    print(f"  Traced:   {tr_mean:.2f} +/- {tr_std:.2f} ms")
    print(f"  Overhead: {tr_mean - ut_mean:.2f} ms ({overhead_pct:.1f}%)")

output = {"benchmark": "trace_overhead_cold_path", "results": results}
with open("/home/ubuntu/cvxpy-agi/bench_trace_overhead_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to bench_trace_overhead_results.json")
