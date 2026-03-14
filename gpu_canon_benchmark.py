#!/usr/bin/env python
"""
GPU canonicalization benchmark.

Compares GPU-native canonicalization against CPU baseline (CVXPY's own
reduction chain) for a suite of problems.

The problems are constructed using CVXPY's user-facing API — the same way
a user would write them. The GPU compiler takes these Problem objects and
produces (A, b, c) on GPU.

Usage:
    python gpu_canon_benchmark.py --quick --json
    python gpu_canon_benchmark.py --json > /tmp/gpu_bench.json
    python gpu_canon_benchmark.py --quick --verbose
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp

import cvxpy as cp

# ── Check for GPU availability ───────────────────────────────────────────────

try:
    import cupy
    HAS_GPU = True
    GPU_NAME = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    GPU_MEM_GB = cupy.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9
except Exception:
    HAS_GPU = False
    GPU_NAME = "none"
    GPU_MEM_GB = 0


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ProblemResult:
    """Timing result for a single problem."""
    name: str
    cpu_mean_ms: float = 0.0
    cpu_std_ms: float = 0.0
    gpu_mean_ms: float = 0.0
    gpu_std_ms: float = 0.0
    gpu_transfer_ms: float = 0.0  # host→device transfer time (for v0)
    speedup: float = 0.0
    correct: bool = True
    error: str = ""


@dataclass
class BenchmarkSuite:
    """Full benchmark results."""
    timestamp: str = ""
    gpu_name: str = ""
    gpu_mem_gb: float = 0.0
    has_gpu: bool = False
    problems: list = field(default_factory=list)
    geomean_cpu_ms: float = 0.0
    geomean_gpu_ms: float = 0.0
    gpu_speedup: float = 0.0
    total_cpu_ms: float = 0.0
    total_gpu_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "gpu_name": self.gpu_name,
            "gpu_mem_gb": round(self.gpu_mem_gb, 1),
            "has_gpu": self.has_gpu,
            "problems": [asdict(p) for p in self.problems],
            "geomean_cpu_ms": round(self.geomean_cpu_ms, 3),
            "geomean_gpu_ms": round(self.geomean_gpu_ms, 3),
            "gpu_speedup": round(self.gpu_speedup, 3),
            "total_cpu_ms": round(self.total_cpu_ms, 1),
            "total_gpu_ms": round(self.total_gpu_ms, 1),
            # Coordinator uses this key
            "geomean_ms": round(self.geomean_gpu_ms, 3),
        }


# ── Problem factories ────────────────────────────────────────────────────────
# These build CVXPY Problem objects — the user's program.
# Both GPU and CPU paths receive the same Problem.

def make_sparse_lp(n: int, m: int = 0, density: float = 0.01) -> Callable:
    """Sparse LP."""
    if m == 0:
        m = n // 2
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = sp.random(m, n, density=density, format='csc', random_state=42)
        b = np.random.randn(m)
        return cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
    return factory


def make_dense_qp(n: int) -> Callable:
    """Dense QP with quad_form."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        Q = np.random.randn(n, n)
        Q = Q @ Q.T
        c = np.random.randn(n)
        A = np.random.randn(n // 2, n)
        b = np.random.randn(n // 2)
        obj = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
        return cp.Problem(obj, [A @ x <= b, x >= -1, x <= 1])
    return factory


def make_lasso(n: int, m: int) -> Callable:
    """LASSO: sum_squares + norm1."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1))
        return cp.Problem(obj)
    return factory


def make_many_constraints(n_vars: int, n_constraints: int) -> Callable:
    """Many small constraints."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n_vars)
        constraints = []
        for i in range(n_constraints):
            np.random.seed(42 + i)
            a = np.random.randn(n_vars)
            constraints.append(a @ x <= np.random.randn())
        return cp.Problem(cp.Minimize(cp.sum(x)), constraints)
    return factory


def make_portfolio(n: int) -> Callable:
    """Markowitz portfolio optimization: quad_form + constraints."""
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


def make_dpp_param_lp(n_vars: int, n_constraints: int) -> Callable:
    """DPP parametrized LP."""
    def factory():
        x = cp.Variable(n_vars)
        A_param = cp.Parameter((n_constraints, n_vars))
        b = np.random.randn(n_constraints) + 10
        c = np.random.randn(n_vars)
        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b, x >= 0])
        # Initialize parameter
        np.random.seed(42)
        A_param.value = np.random.randn(n_constraints, n_vars)
        return prob
    return factory


def make_dpp_lasso(n: int, m: int) -> Callable:
    """DPP LASSO with parametric data."""
    def factory():
        x = cp.Variable(n)
        A_param = cp.Parameter((m, n))
        b_param = cp.Parameter(m)
        obj = cp.Minimize(0.5 * cp.sum_squares(A_param @ x - b_param) + 0.1 * cp.norm(x, 1))
        prob = cp.Problem(obj)
        np.random.seed(42)
        A_param.value = np.random.randn(m, n)
        b_param.value = np.random.randn(m)
        return prob
    return factory


# ── Problem suites ───────────────────────────────────────────────────────────

QUICK_SUITE = [
    ("sparse_lp_100", make_sparse_lp(100), 5),
    ("dense_qp_50", make_dense_qp(50), 5),
    ("lasso_50x100", make_lasso(50, 100), 5),
    ("many_constraints_50x100", make_many_constraints(50, 100), 5),
    ("portfolio_50", make_portfolio(50), 5),
    ("dpp_param_lp_100x100", make_dpp_param_lp(100, 100), 5),
    ("dpp_lasso_50x100", make_dpp_lasso(50, 100), 5),
]

FULL_SUITE = QUICK_SUITE + [
    ("sparse_lp_500", make_sparse_lp(500), 3),
    ("sparse_lp_2000", make_sparse_lp(2000), 2),
    ("sparse_lp_5000", make_sparse_lp(5000), 2),
    ("dense_qp_200", make_dense_qp(200), 3),
    ("lasso_200x500", make_lasso(200, 500), 3),
    ("many_constraints_50x500", make_many_constraints(50, 500), 3),
    ("many_constraints_100x1000", make_many_constraints(100, 1000), 2),
    ("portfolio_200", make_portfolio(200), 3),
    ("dpp_param_lp_500x200", make_dpp_param_lp(500, 200), 3),
    ("dpp_param_lp_1000x500", make_dpp_param_lp(1000, 500), 2),
    ("dpp_lasso_200x500", make_dpp_lasso(200, 500), 3),
]


# ── Timing helpers ───────────────────────────────────────────────────────────

def _time_cpu(problem_factory: Callable, warmup: int, iterations: int) -> tuple[list, str]:
    """Time CPU cold canonicalization.

    Constructs problem OUTSIDE the timer, then times only canonicalization.
    This isolates the canonicalization performance from Problem construction.
    """
    from gpu_canon.baseline import canonicalize_cpu
    times = []

    for i in range(warmup + iterations):
        prob = problem_factory()
        gc.collect()
        start = time.perf_counter()
        try:
            canonicalize_cpu(prob)
        except Exception as e:
            return [], str(e)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= warmup:
            times.append(elapsed)
        del prob
        gc.collect()

    return times, ""


def _time_gpu(problem_factory: Callable, warmup: int, iterations: int) -> tuple[list, str]:
    """Time GPU cold canonicalization.

    Constructs problem OUTSIDE the timer, then times only the GPU compilation.
    """
    if not HAS_GPU:
        return [], "No GPU available"

    from gpu_canon.backend import canonicalize_gpu
    times = []

    for i in range(warmup + iterations):
        prob = problem_factory()
        gc.collect()
        cupy.cuda.Device(0).synchronize()
        start = time.perf_counter()
        try:
            canonicalize_gpu(prob)
        except Exception as e:
            return [], str(e)
        cupy.cuda.Device(0).synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if i >= warmup:
            times.append(elapsed)
        del prob
        gc.collect()

    return times, ""


# ── Runner ───────────────────────────────────────────────────────────────────

def run_suite(
    suite: list[tuple[str, Callable, int]],
    verbose: bool = True,
    warmup: int = 2,
) -> BenchmarkSuite:
    """Run the benchmark suite."""
    results = BenchmarkSuite(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        gpu_name=GPU_NAME,
        gpu_mem_gb=GPU_MEM_GB,
        has_gpu=HAS_GPU,
    )

    cpu_means = []
    gpu_means = []

    for name, factory, iters in suite:
        if verbose:
            print(f"  {name:35s} ", end="", file=sys.stderr, flush=True)

        # CPU timing
        cpu_times, cpu_err = _time_cpu(factory, warmup, iters)

        # GPU timing
        gpu_times, gpu_err = _time_gpu(factory, warmup, iters)

        result = ProblemResult(name=name)

        if cpu_err:
            result.error = f"CPU: {cpu_err}"
        elif gpu_err:
            # CPU worked, GPU didn't — record CPU time, note GPU error
            result.cpu_mean_ms = statistics.mean(cpu_times)
            result.cpu_std_ms = statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0
            result.error = f"GPU: {gpu_err}"
        else:
            result.cpu_mean_ms = statistics.mean(cpu_times)
            result.cpu_std_ms = statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0
            result.gpu_mean_ms = statistics.mean(gpu_times)
            result.gpu_std_ms = statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0
            if result.gpu_mean_ms > 0:
                result.speedup = result.cpu_mean_ms / result.gpu_mean_ms
            cpu_means.append(result.cpu_mean_ms)
            gpu_means.append(result.gpu_mean_ms)

        results.problems.append(result)

        if verbose:
            if result.error:
                print(f"ERROR: {result.error[:50]}", file=sys.stderr)
            else:
                print(
                    f"CPU={result.cpu_mean_ms:7.2f}ms  "
                    f"GPU={result.gpu_mean_ms:7.2f}ms  "
                    f"speedup={result.speedup:.2f}x",
                    file=sys.stderr,
                )

    # Compute geomeans
    if cpu_means:
        results.geomean_cpu_ms = math.exp(
            sum(math.log(t) for t in cpu_means) / len(cpu_means)
        )
        results.total_cpu_ms = sum(cpu_means)
    if gpu_means:
        results.geomean_gpu_ms = math.exp(
            sum(math.log(t) for t in gpu_means) / len(gpu_means)
        )
        results.total_gpu_ms = sum(gpu_means)
    if results.geomean_gpu_ms > 0:
        results.gpu_speedup = results.geomean_cpu_ms / results.geomean_gpu_ms

    if verbose:
        print(f"\n  CPU geomean: {results.geomean_cpu_ms:.2f}ms", file=sys.stderr)
        print(f"  GPU geomean: {results.geomean_gpu_ms:.2f}ms", file=sys.stderr)
        print(f"  Speedup:     {results.gpu_speedup:.2f}x", file=sys.stderr)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU canonicalization benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick run (small problems)")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--quiet", action="store_true", help="Suppress stderr")
    parser.add_argument("--verbose", action="store_true", help="Extra detail")
    args = parser.parse_args()

    suite = QUICK_SUITE if args.quick else FULL_SUITE
    verbose = not args.quiet

    if verbose:
        print(f"GPU: {GPU_NAME} ({GPU_MEM_GB:.0f}GB)" if HAS_GPU else "GPU: none",
              file=sys.stderr)
        print(f"Suite: {'quick' if args.quick else 'full'} "
              f"({len(suite)} problems)\n", file=sys.stderr)

    results = run_suite(suite, verbose=verbose)

    if args.json:
        print(json.dumps(results.to_dict(), indent=2, default=str))
    elif verbose:
        print(f"\nDone. GPU speedup: {results.gpu_speedup:.2f}x", file=sys.stderr)


if __name__ == "__main__":
    main()
