#!/usr/bin/env python
"""
Benchmark 2: GPU memory usage of cached constants.
Measures total bytes of cached constant arrays per problem type.
"""

import gc
import json

import numpy as np
import scipy.sparse as sp
import cupy
import cvxpy as cp

from gpu_canon.trace_cache import TraceCache, reset_id_counter
from gpu_canon.backend import reset_trace_cache

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

def make_param_lp(n, m):
    def factory():
        x = cp.Variable(n)
        A_param = cp.Parameter((m, n))
        b = np.random.randn(m) + 10
        c = np.random.randn(n)
        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b, x >= 0])
        np.random.seed(42)
        A_param.value = np.random.randn(m, n)
        return prob
    return factory

def measure_cache_memory(factory, name):
    """Trace a problem, measure memory of cached constants."""
    cache = TraceCache()
    reset_id_counter()
    prob = factory()
    cache.canonicalize(prob, on_gpu=True)

    # Get the cached trace
    fp = list(cache._cache.keys())[0]
    cached = cache._cache[fp]

    # Count cached constant GPU arrays
    n_constants = len(cached._cached_constants)
    total_bytes = 0
    for tid, arr in cached._cached_constants.items():
        if hasattr(arr, 'nbytes'):
            total_bytes += arr.nbytes

    # Count structural GPU arrays
    n_structural = len(cached._structural_gpu)
    structural_bytes = 0
    for tid, arr in cached._structural_gpu.items():
        if hasattr(arr, 'nbytes'):
            structural_bytes += arr.nbytes

    # A structural (indices, indptr)
    a_struct_bytes = 0
    if cached.A_structural is not None:
        if hasattr(cached, '_A_indices_gpu'):
            a_struct_bytes += cached._A_indices_gpu.nbytes
        if hasattr(cached, '_A_indptr_gpu'):
            a_struct_bytes += cached._A_indptr_gpu.nbytes

    # Trace stats
    total_ops = len(cached.trace)
    data_ops = len(cached._data_op_indices)
    constant_ops = total_ops - data_ops

    # Input arrays stored on CPU (in data_input_ids)
    data_input_bytes = sum(
        arr.nbytes for arr in cached.data_input_ids.values()
        if hasattr(arr, 'nbytes')
    )

    return {
        "name": name,
        "total_ops": total_ops,
        "constant_ops": constant_ops,
        "data_ops": data_ops,
        "constant_pct": round(100 * constant_ops / max(total_ops, 1), 1),
        "cached_constant_arrays": n_constants,
        "cached_constant_bytes": total_bytes,
        "cached_constant_MB": round(total_bytes / 1e6, 3),
        "structural_gpu_arrays": n_structural,
        "structural_gpu_bytes": structural_bytes,
        "structural_gpu_MB": round(structural_bytes / 1e6, 3),
        "A_structural_bytes": a_struct_bytes,
        "A_structural_MB": round(a_struct_bytes / 1e6, 3),
        "data_input_cpu_bytes": data_input_bytes,
        "data_input_cpu_MB": round(data_input_bytes / 1e6, 3),
        "total_gpu_cache_bytes": total_bytes + structural_bytes + a_struct_bytes,
        "total_gpu_cache_MB": round((total_bytes + structural_bytes + a_struct_bytes) / 1e6, 3),
    }

problems = [
    ("LP 500x250", make_lp(500, 250)),
    ("LP 2000x1000", make_lp(2000, 1000)),
    ("LP 5000x2500", make_lp(5000, 2500)),
    ("LASSO 500x1000", make_lasso(500, 1000)),
    ("LASSO 2000x4000", make_lasso(2000, 4000)),
    ("Portfolio 200", make_portfolio(200)),
    ("Param LP 500x250", make_param_lp(500, 250)),
    ("Param LP 2000x1000", make_param_lp(2000, 1000)),
]

results = []
for name, factory in problems:
    print(f"Measuring {name}...")
    try:
        r = measure_cache_memory(factory, name)
        results.append(r)
        print(f"  Total ops: {r['total_ops']}, Constant: {r['constant_ops']} ({r['constant_pct']}%)")
        print(f"  GPU cache: {r['total_gpu_cache_MB']:.3f} MB")
        print(f"    - Cached constants: {r['cached_constant_MB']:.3f} MB ({r['cached_constant_arrays']} arrays)")
        print(f"    - Structural GPU:   {r['structural_gpu_MB']:.3f} MB ({r['structural_gpu_arrays']} arrays)")
        print(f"    - A structural:     {r['A_structural_MB']:.3f} MB")
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"name": name, "error": str(e)})
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()

output = {"benchmark": "gpu_memory_cached_constants", "results": results}
with open("/home/ubuntu/cvxpy-agi/bench_memory_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to bench_memory_results.json")
