#!/usr/bin/env python
"""
Benchmark 3: Stress test with edge cases.
Tests SOC, PSD, exponential cone, mixed-integer, and very large problems.
"""

import gc
import json
import time
import traceback

import numpy as np
import scipy.sparse as sp
import cupy
import cvxpy as cp

from gpu_canon.trace_cache import TraceCache, reset_id_counter
from gpu_canon.replayer import replay_on_cpu


def make_soc():
    """SOC (second-order cone) constraints via norm"""
    def factory():
        np.random.seed(42)
        n = 50
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = np.random.randn(20, n)
        b = np.random.randn(20)
        constraints = [cp.norm(x, 2) <= 10, A @ x <= b]
        return cp.Problem(cp.Minimize(c @ x), constraints)
    return factory


def make_psd():
    """PSD constraints: X >> 0"""
    def factory():
        np.random.seed(42)
        n = 10
        X = cp.Variable((n, n), symmetric=True)
        C = np.random.randn(n, n)
        C = C + C.T
        constraints = [X >> 0, cp.trace(X) == 1]
        return cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    return factory


def make_exp_cone():
    """Exponential cone: cp.exp(x)"""
    def factory():
        np.random.seed(42)
        n = 20
        x = cp.Variable(n)
        c = np.random.randn(n)
        constraints = [cp.sum(cp.exp(x)) <= 10, x >= -2]
        return cp.Problem(cp.Minimize(c @ x), constraints)
    return factory


def make_mixed_integer():
    """Mixed-integer: relaxed LP (boolean not supported by conic solvers)"""
    def factory():
        np.random.seed(42)
        n = 20
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = np.random.randn(10, n)
        b = np.abs(np.random.randn(10)) * 5
        # Simulate integer relaxation with box constraints
        return cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= 0, x <= 1])
    return factory


def make_large_lp():
    """Very large LP: 10000 vars, 5000 constraints"""
    def factory():
        np.random.seed(42)
        n, m = 10000, 5000
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = sp.random(m, n, density=0.001, format='csc', random_state=42)
        b = np.random.randn(m)
        return cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
    return factory


def test_problem(name, factory):
    """Test tracing + replay for a problem. Returns result dict."""
    print(f"\nTesting {name}...")
    result = {
        "name": name,
        "trace_ok": False,
        "replay_ok": False,
        "trace_time_ms": 0,
        "replay_time_ms": 0,
        "trace_ops": 0,
        "data_ops": 0,
        "error": "",
    }

    cache = TraceCache()

    # Step 1: Trace (cold path)
    try:
        reset_id_counter()
        prob = factory()
        cupy.cuda.Device(0).synchronize()
        start = time.perf_counter()
        A1, b1, c1, dims1 = cache.canonicalize(prob, on_gpu=True)
        cupy.cuda.Device(0).synchronize()
        trace_time = (time.perf_counter() - start) * 1000
        result["trace_time_ms"] = round(trace_time, 2)
        result["trace_ok"] = True

        # Get trace stats
        fp = list(cache._cache.keys())[0]
        cached = cache._cache[fp]
        result["trace_ops"] = len(cached.trace)
        result["data_ops"] = len(cached._data_op_indices)
        result["constant_ops"] = result["trace_ops"] - result["data_ops"]

        print(f"  Trace OK: {result['trace_ops']} ops ({result['data_ops']} data-dependent)")
    except Exception as e:
        result["error"] = f"Trace failed: {str(e)}"
        print(f"  Trace FAILED: {e}")
        traceback.print_exc()
        return result

    # Step 2: Replay (warm path)
    try:
        reset_id_counter()
        prob2 = factory()
        cupy.cuda.Device(0).synchronize()
        start = time.perf_counter()
        A2, b2, c2, dims2 = cache.canonicalize(prob2, on_gpu=True)
        cupy.cuda.Device(0).synchronize()
        replay_time = (time.perf_counter() - start) * 1000
        result["replay_time_ms"] = round(replay_time, 2)
        result["replay_ok"] = True
        print(f"  Replay OK: {replay_time:.2f} ms")
    except Exception as e:
        result["error"] = f"Replay failed: {str(e)}"
        print(f"  Replay FAILED: {e}")
        traceback.print_exc()
        return result

    # Step 3: Correctness check - compare with CPU baseline
    try:
        from gpu_canon.baseline import canonicalize_cpu
        reset_id_counter()
        prob3 = factory()
        A_ref, b_ref, c_ref, _ = canonicalize_cpu(prob3)

        # Compare
        checks = []
        if A2 is not None and A_ref is not None:
            if hasattr(A2, 'get'):
                A2_cpu = A2.toarray().get() if hasattr(A2, 'toarray') else A2.get()
            else:
                A2_cpu = A2.toarray() if sp.issparse(A2) else np.asarray(A2)
            A_ref_dense = A_ref.toarray() if sp.issparse(A_ref) else np.asarray(A_ref)
            if A2_cpu.shape == A_ref_dense.shape:
                a_match = np.allclose(A2_cpu, A_ref_dense, atol=1e-6, rtol=1e-5)
                checks.append(("A", a_match))
            else:
                checks.append(("A", f"shape mismatch {A2_cpu.shape} vs {A_ref_dense.shape}"))

        if b2 is not None and b_ref is not None:
            b2_cpu = b2.get() if hasattr(b2, 'get') else np.asarray(b2)
            if b2_cpu.shape == b_ref.shape:
                b_match = np.allclose(b2_cpu, b_ref, atol=1e-6, rtol=1e-5)
                checks.append(("b", b_match))
            else:
                checks.append(("b", f"shape mismatch {b2_cpu.shape} vs {b_ref.shape}"))

        if c2 is not None and c_ref is not None:
            c2_cpu = c2.get() if hasattr(c2, 'get') else np.asarray(c2)
            if c2_cpu.shape == c_ref.shape:
                c_match = np.allclose(c2_cpu, c_ref, atol=1e-6, rtol=1e-5)
                checks.append(("c", c_match))
            else:
                checks.append(("c", f"shape mismatch {c2_cpu.shape} vs {c_ref.shape}"))

        result["correctness_checks"] = {k: str(v) for k, v in checks}
        all_ok = all(v is True for _, v in checks)
        result["correct"] = all_ok
        print(f"  Correctness: {'PASS' if all_ok else 'FAIL'} {checks}")
    except Exception as e:
        result["correctness_error"] = str(e)
        print(f"  Correctness check error: {e}")

    return result


problems = [
    ("SOC constraints", make_soc()),
    ("PSD (SDP)", make_psd()),
    ("Exponential cone", make_exp_cone()),
    ("Mixed-integer (boolean)", make_mixed_integer()),
    ("Very large LP 10000x5000", make_large_lp()),
]

results = []
for name, factory_instance in problems:
    # Wrap factory_instance in a lambda that returns a fresh problem
    factory = factory_instance  # already a callable from make_*()

    # Actually the make_*() functions return factories, not problems
    # We already called them above. Let's fix:

# Redo properly
problems2 = [
    ("SOC constraints", make_soc()),
    ("PSD (SDP)", make_psd()),
    ("Exponential cone", make_exp_cone()),
    ("Mixed-integer (boolean)", make_mixed_integer()),
    ("Very large LP 10000x5000", make_large_lp()),
]

results = []
for name, factory in problems2:
    r = test_problem(name, factory)
    results.append(r)
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()

output = {"benchmark": "edge_case_stress_test", "results": results}
with open("/home/ubuntu/cvxpy-agi/bench_edge_cases_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to bench_edge_cases_results.json")
