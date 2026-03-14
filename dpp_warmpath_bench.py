#!/usr/bin/env python
"""
DPP warm-path benchmark: GPU vs CPU re-canonicalization.

Both GPU and CPU paths have already compiled the problem structure.
We measure only the re-canonicalization step (apply new parameter values).

This is the fair comparison: warm-path GPU vs warm-path CPU.
"""

import gc
import json
import math
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp

import cvxpy as cp

try:
    import cupy as cup
    import cupyx.scipy.sparse as cusp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from gpu_canon.backend import CompiledProgram
from cvxpy.cvxcore.python import canonInterface


# ── Problem factories ────────────────────────────────────────────────────────

def make_dpp_lp(n_vars, n_constraints):
    """DPP LP with parametric A matrix."""
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    b_vec = np.random.randn(n_constraints) + 10
    c_vec = np.random.randn(n_vars)
    prob = cp.Problem(cp.Minimize(c_vec @ x), [A_param @ x <= b_vec, x >= 0])
    np.random.seed(42)
    A_param.value = np.random.randn(n_constraints, n_vars)
    return prob, [A_param]


def make_dpp_lasso(n, m):
    """DPP LASSO with parametric A and b."""
    x = cp.Variable(n)
    A_param = cp.Parameter((m, n))
    b_param = cp.Parameter(m)
    lam = cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(A_param @ x - b_param) + lam * cp.norm(x, 1))
    )
    np.random.seed(42)
    A_param.value = np.random.randn(m, n)
    b_param.value = np.random.randn(m)
    lam.value = 0.1
    return prob, [A_param, b_param, lam]


def make_dpp_portfolio(n):
    """DPP Markowitz portfolio with parametric expected returns."""
    x = cp.Variable(n)
    mu = cp.Parameter(n)
    np.random.seed(42)
    F = np.random.randn(n, n // 2)
    Sigma = F @ F.T + np.diag(np.random.rand(n) * 0.1)
    gamma = cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Maximize(mu @ x - gamma * cp.quad_form(x, Sigma)),
        [cp.sum(x) == 1, x >= 0],
    )
    mu.value = np.random.randn(n) * 0.1
    gamma.value = 1.0
    return prob, [mu, gamma]


def make_dpp_sparse_lp(n_vars, n_constraints, density=0.05):
    """DPP LP with sparse parametric A."""
    x = cp.Variable(n_vars)
    # Use dense Parameter but with sparse structure for realistic workload
    A_param = cp.Parameter((n_constraints, n_vars))
    b_vec = np.random.randn(n_constraints) + 10
    c_vec = np.random.randn(n_vars)
    prob = cp.Problem(cp.Minimize(c_vec @ x), [A_param @ x <= b_vec, x >= 0])
    np.random.seed(42)
    A_val = sp.random(n_constraints, n_vars, density=density,
                      format='csc', random_state=42).toarray()
    A_param.value = A_val
    return prob, [A_param]


# ── Problem suite ────────────────────────────────────────────────────────────

PROBLEMS = [
    # (name, factory_func, args, n_warmup, n_iters, n_resolves)
    # n_resolves = how many parameter updates per trial (simulates solve loop)

    # Small scale
    ("dpp_lp_100x50",       make_dpp_lp,        (100, 50),   3, 20, 10),
    ("dpp_lp_500x200",      make_dpp_lp,        (500, 200),  3, 15, 10),
    ("dpp_lasso_100x200",   make_dpp_lasso,     (100, 200),  3, 15, 10),
    ("dpp_portfolio_100",   make_dpp_portfolio,  (100,),      3, 15, 10),

    # Medium scale
    ("dpp_lp_1000x500",     make_dpp_lp,        (1000, 500),  2, 10, 10),
    ("dpp_lp_2000x1000",    make_dpp_lp,        (2000, 1000), 2, 8,  5),
    ("dpp_lasso_500x1000",  make_dpp_lasso,     (500, 1000),  2, 8,  5),
    ("dpp_portfolio_500",   make_dpp_portfolio,  (500,),       2, 8,  5),

    # Large scale
    ("dpp_lp_5000x2000",    make_dpp_lp,        (5000, 2000), 2, 5,  3),
    ("dpp_lp_10000x5000",   make_dpp_lp,        (10000, 5000), 1, 3, 3),
    ("dpp_lasso_1000x2000", make_dpp_lasso,     (1000, 2000),  2, 5, 3),
    ("dpp_sparse_lp_5000x2000", make_dpp_sparse_lp, (5000, 2000, 0.02), 2, 5, 3),
]


# ── CPU warm-path timing ────────────────────────────────────────────────────

def time_cpu_warm(prob, params, n_warmup, n_iters, n_resolves):
    """Time CVXPY's own DPP warm path (apply_parameters)."""
    # Get param_prob via get_problem_data
    data, chain, inv = prob.get_problem_data(cp.CLARABEL, canon_backend="COO")
    pp = data.get("param_prob")
    if pp is None:
        return [], "No param_prob (not DPP)"

    pp.reduced_A.cache(False)

    times = []
    for trial in range(n_warmup + n_iters):
        # Update parameter values
        for p in params:
            if p.size == 1:
                p.value = abs(np.random.randn()) + 0.01
            else:
                p.value = np.random.randn(*p.shape)

        gc.collect()
        start = time.perf_counter()

        for _ in range(n_resolves):
            # This is exactly what CVXPY does on re-solve with DPP
            pp.apply_parameters()

        elapsed = (time.perf_counter() - start) * 1000  # ms for n_resolves calls

        if trial >= n_warmup:
            times.append(elapsed / n_resolves)  # per-call ms

    return times, ""


# ── GPU warm-path timing ────────────────────────────────────────────────────

def time_gpu_warm(prob, params, n_warmup, n_iters, n_resolves):
    """Time GPU DPP warm path (CompiledProgram.canonicalize)."""
    if not HAS_GPU:
        return [], "No GPU"

    compiled = CompiledProgram(prob, "CLARABEL")
    if not compiled._dpp_ready:
        return [], "Not DPP-ready on GPU"

    times = []
    for trial in range(n_warmup + n_iters):
        # Update parameter values
        for p in params:
            if p.size == 1:
                p.value = abs(np.random.randn()) + 0.01
            else:
                p.value = np.random.randn(*p.shape)

        gc.collect()
        cup.cuda.Device(0).synchronize()
        start = time.perf_counter()

        for _ in range(n_resolves):
            compiled.canonicalize()  # builds param_vec from CVXPY params + GPU matmul

        cup.cuda.Device(0).synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        if trial >= n_warmup:
            times.append(elapsed / n_resolves)

    return times, ""


# ── GPU warm-path with pre-built param vector ───────────────────────────────

def time_gpu_warm_prebuilt(prob, params, n_warmup, n_iters, n_resolves):
    """GPU warm path where param_vec is already on GPU (best case)."""
    if not HAS_GPU:
        return [], "No GPU"

    compiled = CompiledProgram(prob, "CLARABEL")
    if not compiled._dpp_ready:
        return [], "Not DPP-ready on GPU"

    times = []
    for trial in range(n_warmup + n_iters):
        # Build param vector on GPU once (simulates GPU-resident pipeline)
        for p in params:
            if p.size == 1:
                p.value = abs(np.random.randn()) + 0.01
            else:
                p.value = np.random.randn(*p.shape)

        param_vec_gpu = compiled._build_param_vector_gpu()

        gc.collect()
        cup.cuda.Device(0).synchronize()
        start = time.perf_counter()

        for _ in range(n_resolves):
            compiled.canonicalize(param_vec_gpu)

        cup.cuda.Device(0).synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        if trial >= n_warmup:
            times.append(elapsed / n_resolves)

    return times, ""


# ── Correctness check ───────────────────────────────────────────────────────

def check_correctness(prob, params):
    """Verify GPU warm path matches CPU warm path."""
    if not HAS_GPU:
        return True, "No GPU"

    # Set params
    np.random.seed(123)
    for p in params:
        if p.size == 1:
            p.value = abs(np.random.randn()) + 0.01
        else:
            p.value = np.random.randn(*p.shape)

    # CPU
    data, chain, inv = prob.get_problem_data(cp.CLARABEL, canon_backend="COO")
    pp = data.get("param_prob")
    if pp is None:
        return True, "No param_prob"
    pp.reduced_A.cache(False)
    q_cpu, d_cpu, A_cpu, b_cpu = pp.apply_parameters()

    # GPU
    compiled = CompiledProgram(prob, "CLARABEL")
    A_gpu, b_gpu, c_gpu, _ = compiled.canonicalize()

    A_gpu_np = A_gpu.toarray().get() if hasattr(A_gpu.toarray(), 'get') else A_gpu.toarray()
    b_gpu_np = b_gpu.get() if hasattr(b_gpu, 'get') else np.asarray(b_gpu)

    a_ok = np.allclose(A_gpu_np, A_cpu.toarray(), atol=1e-10)
    b_ok = np.allclose(b_gpu_np, b_cpu, atol=1e-10)

    if not a_ok or not b_ok:
        return False, f"A_match={a_ok}, b_match={b_ok}"
    return True, "OK"


# ── Runner ───────────────────────────────────────────────────────────────────

def run_benchmark(problems=None, verbose=True):
    """Run the full DPP warm-path benchmark."""
    if problems is None:
        problems = PROBLEMS

    results = []
    cpu_means = []
    gpu_means = []
    gpu_pre_means = []

    for name, factory, args, n_warmup, n_iters, n_resolves in problems:
        if verbose:
            print(f"  {name:35s} ", end="", file=sys.stderr, flush=True)

        np.random.seed(42)
        prob, params = factory(*args)

        # Correctness check
        correct, msg = check_correctness(prob, params)

        # Reset seeds consistently
        np.random.seed(42)
        prob, params = factory(*args)

        # CPU warm path
        np.random.seed(7)
        cpu_times, cpu_err = time_cpu_warm(prob, params, n_warmup, n_iters, n_resolves)

        # GPU warm path (includes param_vec build)
        np.random.seed(7)
        gpu_times, gpu_err = time_gpu_warm(prob, params, n_warmup, n_iters, n_resolves)

        # GPU warm path (pre-built param_vec — pure GPU)
        np.random.seed(7)
        gpu_pre_times, gpu_pre_err = time_gpu_warm_prebuilt(
            prob, params, n_warmup, n_iters, n_resolves)

        r = {
            "name": name,
            "correct": correct,
        }

        if cpu_times:
            r["cpu_mean_ms"] = round(statistics.mean(cpu_times), 4)
            r["cpu_std_ms"] = round(statistics.stdev(cpu_times), 4) if len(cpu_times) > 1 else 0
            cpu_means.append(r["cpu_mean_ms"])
        if cpu_err:
            r["cpu_error"] = cpu_err

        if gpu_times:
            r["gpu_mean_ms"] = round(statistics.mean(gpu_times), 4)
            r["gpu_std_ms"] = round(statistics.stdev(gpu_times), 4) if len(gpu_times) > 1 else 0
            gpu_means.append(r["gpu_mean_ms"])
        if gpu_err:
            r["gpu_error"] = gpu_err

        if gpu_pre_times:
            r["gpu_prebuilt_mean_ms"] = round(statistics.mean(gpu_pre_times), 4)
            r["gpu_prebuilt_std_ms"] = round(
                statistics.stdev(gpu_pre_times), 4) if len(gpu_pre_times) > 1 else 0
            gpu_pre_means.append(r["gpu_prebuilt_mean_ms"])
        if gpu_pre_err:
            r["gpu_prebuilt_error"] = gpu_pre_err

        # Speedups
        if "cpu_mean_ms" in r and "gpu_mean_ms" in r and r["gpu_mean_ms"] > 0:
            r["speedup_gpu"] = round(r["cpu_mean_ms"] / r["gpu_mean_ms"], 2)
        if "cpu_mean_ms" in r and "gpu_prebuilt_mean_ms" in r and r["gpu_prebuilt_mean_ms"] > 0:
            r["speedup_gpu_prebuilt"] = round(
                r["cpu_mean_ms"] / r["gpu_prebuilt_mean_ms"], 2)

        results.append(r)

        if verbose:
            cpu_s = f"CPU={r.get('cpu_mean_ms', '?'):>8}"
            gpu_s = f"GPU={r.get('gpu_mean_ms', '?'):>8}"
            pre_s = f"GPU_pre={r.get('gpu_prebuilt_mean_ms', '?'):>8}"
            sp1 = f"x{r.get('speedup_gpu', '?')}" if 'speedup_gpu' in r else "?"
            sp2 = f"x{r.get('speedup_gpu_prebuilt', '?')}" if 'speedup_gpu_prebuilt' in r else "?"
            ok = "OK" if correct else "FAIL"
            print(f"{cpu_s}ms {gpu_s}ms {pre_s}ms  sp={sp1}/{sp2}  [{ok}]",
                  file=sys.stderr)

        del prob, params
        gc.collect()

    # Summary
    summary = {
        "problems": results,
        "n_problems": len(results),
    }
    if cpu_means:
        summary["geomean_cpu_ms"] = round(
            math.exp(sum(math.log(t) for t in cpu_means) / len(cpu_means)), 4)
    if gpu_means:
        summary["geomean_gpu_ms"] = round(
            math.exp(sum(math.log(t) for t in gpu_means) / len(gpu_means)), 4)
    if gpu_pre_means:
        summary["geomean_gpu_prebuilt_ms"] = round(
            math.exp(sum(math.log(t) for t in gpu_pre_means) / len(gpu_pre_means)), 4)
    if "geomean_cpu_ms" in summary and "geomean_gpu_ms" in summary:
        summary["geomean_speedup"] = round(
            summary["geomean_cpu_ms"] / summary["geomean_gpu_ms"], 2)
    if "geomean_cpu_ms" in summary and "geomean_gpu_prebuilt_ms" in summary:
        summary["geomean_speedup_prebuilt"] = round(
            summary["geomean_cpu_ms"] / summary["geomean_gpu_prebuilt_ms"], 2)

    # For coordinator compatibility
    summary["geomean_ms"] = summary.get("geomean_gpu_ms", 0)

    if verbose:
        print(f"\n  CPU geomean:         {summary.get('geomean_cpu_ms', '?')}ms",
              file=sys.stderr)
        print(f"  GPU geomean:         {summary.get('geomean_gpu_ms', '?')}ms",
              file=sys.stderr)
        print(f"  GPU prebuilt geomean:{summary.get('geomean_gpu_prebuilt_ms', '?')}ms",
              file=sys.stderr)
        print(f"  Speedup (GPU):       {summary.get('geomean_speedup', '?')}x",
              file=sys.stderr)
        print(f"  Speedup (prebuilt):  {summary.get('geomean_speedup_prebuilt', '?')}x",
              file=sys.stderr)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run only small problems")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quick:
        problems = [p for p in PROBLEMS if any(
            s in p[0] for s in ["100", "200", "500x200"])]
    else:
        problems = PROBLEMS

    verbose = not args.quiet
    if verbose:
        try:
            gpu_name = cup.cuda.runtime.getDeviceProperties(0)["name"].decode()
            print(f"GPU: {gpu_name}", file=sys.stderr)
        except Exception:
            print("GPU: unavailable", file=sys.stderr)
        print(f"Suite: {len(problems)} problems\n", file=sys.stderr)

    summary = run_benchmark(problems, verbose=verbose)

    if args.json:
        print(json.dumps(summary, indent=2))
