#!/usr/bin/env python
"""
Standardized CVXPY canonicalization benchmark with JSON output.

Produces deterministic, comparable results across backends (CPP, SCIPY, COO).
Used by the autoresearch loop to measure optimization impact.

Usage:
    python canon_benchmark.py --json                  # Default backend, JSON output
    python canon_benchmark.py --all-backends --json    # All backends
    python canon_benchmark.py --quick --json           # Quick run (small problems only)
    python canon_benchmark.py --backend COO            # Specific backend
"""

import argparse
import cProfile
import gc
import json
import math
import pstats
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

import cvxpy as cp

# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class TraceEntry:
    """Single function in an execution trace."""
    func: str
    cumtime_ms: float
    calls: int
    pct: float


@dataclass
class ProblemResult:
    """Timing result for a single problem."""
    name: str
    backend: str
    is_dpp: bool
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    times_ms: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results."""
    timestamp: str = ""
    backends: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    geomean_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "backends": self.backends,
            "problems": [asdict(p) for p in self.problems],
            "geomean_ms": self.geomean_ms,
            "total_ms": self.total_ms,
        }


# Paths we care about in traces (filter out noise from stdlib, numpy internals, etc.)
_TRACE_PATHS = [
    "lin_ops/backends/",
    "reductions/dcp2cone/",
    "reductions/solvers/",
    "cvxpy_rust",
    "utilities/coeff_extractor",
]


def _extract_trace(profiler: cProfile.Profile, total_ms: float, top_n: int = 15) -> list[dict]:
    """Extract top-N relevant functions from a cProfile run."""
    stats = pstats.Stats(profiler, stream=StringIO())
    entries = []

    for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in stats.stats.items():
        # Filter to CVXPY/backend code only
        if not any(p in filename for p in _TRACE_PATHS):
            continue
        # Shorten path: keep just the relevant part
        short = filename
        for p in _TRACE_PATHS:
            idx = filename.find(p)
            if idx >= 0:
                short = filename[idx:]
                break
        cumtime_ms = ct * 1000
        pct = (cumtime_ms / total_ms * 100) if total_ms > 0 else 0
        entries.append({
            "func": f"{short}:{funcname}",
            "cumtime_ms": round(cumtime_ms, 2),
            "tottime_ms": round(tt * 1000, 2),
            "calls": nc,
            "pct": round(pct, 1),
        })

    # Sort by tottime (self time) — most actionable for optimization
    entries.sort(key=lambda e: e["tottime_ms"], reverse=True)
    return entries[:top_n]


def time_canonicalization(
    problem_factory: Callable,
    backend: str,
    warmup: int = 2,
    iterations: int = 5,
    ignore_dpp: bool = False,
    trace: bool = False,
) -> ProblemResult:
    """Time the canonicalization phase of a problem."""
    times = []
    is_dpp = False
    trace_data = []

    # Warmup
    for _ in range(warmup):
        prob, init_params = problem_factory()
        if init_params is not None:
            is_dpp = True
            init_params()
        try:
            prob.get_problem_data(cp.CLARABEL, canon_backend=backend, ignore_dpp=ignore_dpp)
        except Exception as e:
            return ProblemResult(name="", backend=backend, is_dpp=is_dpp, error=str(e))
        del prob
        gc.collect()

    # Timed runs
    for i in range(iterations):
        prob, init_params = problem_factory()
        if init_params is not None:
            is_dpp = True
            init_params()
        gc.collect()

        # Clear cache to force re-canonicalization
        prob._cache = type(prob._cache)()

        # Profile the last iteration (representative, after JIT warmup)
        do_profile = trace and (i == iterations - 1)
        profiler = cProfile.Profile() if do_profile else None

        start = time.perf_counter()
        try:
            if do_profile:
                profiler.enable()
            prob.get_problem_data(cp.CLARABEL, canon_backend=backend, ignore_dpp=ignore_dpp)
            if do_profile:
                profiler.disable()
        except Exception as e:
            if do_profile:
                profiler.disable()
            return ProblemResult(name="", backend=backend, is_dpp=is_dpp, error=str(e))
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        if do_profile:
            trace_data = _extract_trace(profiler, elapsed_ms)

        del prob
        gc.collect()

    return ProblemResult(
        name="",
        backend=backend,
        is_dpp=is_dpp,
        times_ms=times,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_ms=min(times),
        trace=trace_data,
    )


# =============================================================================
# Problem Factories
# =============================================================================

def make_sparse_lp(n: int, m: int = 0, density: float = 0.01) -> Callable:
    """Sparse LP: hot path is build_matrix tree traversal."""
    if m == 0:
        m = n // 2
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = sp.random(m, n, density=density, format='csc', random_state=42)
        b = np.random.randn(m)
        return cp.Problem(cp.Minimize(c @ x), [A @ x <= b]), None
    return factory


def make_dense_qp(n: int) -> Callable:
    """Dense QP: hot path is quad_form extraction."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        Q = np.random.randn(n, n)
        Q = Q @ Q.T
        c = np.random.randn(n)
        A = np.random.randn(n // 2, n)
        b = np.random.randn(n // 2)
        obj = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
        return cp.Problem(obj, [A @ x <= b, x >= -1, x <= 1]), None
    return factory


def make_lasso(n: int, m: int) -> Callable:
    """LASSO: hot path is sum_squares + norm1."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1))
        return cp.Problem(obj), None
    return factory


def make_many_constraints(n_vars: int, n_constraints: int) -> Callable:
    """Many small constraints: hot path is LinOp iteration."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n_vars)
        constraints = []
        for i in range(n_constraints):
            np.random.seed(42 + i)
            a = np.random.randn(n_vars)
            constraints.append(a @ x <= np.random.randn())
        return cp.Problem(cp.Minimize(cp.sum(x)), constraints), None
    return factory


def make_scalar_squares(n: int) -> Callable:
    """N scalar Variable()s each squared and summed. Huge flat LinOp tree."""
    def factory():
        np.random.seed(42)
        xs = [cp.Variable() for _ in range(n)]
        obj = cp.Minimize(sum(cp.square(x) for x in xs))
        cons = [x >= -1 for x in xs] + [x <= 1 for x in xs]
        return cp.Problem(obj, cons), None
    return factory


def make_deep_chain(n: int) -> Callable:
    """Deeply nested addition chain: x + x*0.001 + x*0.001 + ... (depth=n)."""
    def factory():
        np.random.seed(42)
        x = cp.Variable()
        expr = x
        for _ in range(n):
            expr = expr + x * 0.001
        return cp.Problem(cp.Minimize(cp.square(expr)), [x >= -10, x <= 10]), None
    return factory


def make_scalar_grid(n: int) -> Callable:
    """N scalar vars with O(n) pairwise constraints. Wide + deep tree."""
    def factory():
        np.random.seed(42)
        xs = [cp.Variable() for _ in range(n)]
        cons = []
        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):
                cons.append(xs[i] + xs[j] <= 2)
                cons.append(xs[i] - xs[j] >= -2)
        obj = cp.Minimize(sum(cp.square(x) for x in xs))
        return cp.Problem(obj, cons), None
    return factory


def _fixed_var(shape=(), id_base=0, idx=0, **kwargs):
    """Create a Variable with a deterministic ID for reproducible benchmarks."""
    return cp.Variable(shape, var_id=id_base + idx, **kwargs)


def _fixed_param(shape=(), id_base=0, idx=0, **kwargs):
    """Create a Parameter with a deterministic ID for reproducible benchmarks."""
    return cp.Parameter(shape, id=id_base + 10000 + idx, **kwargs)


def make_dpp_scalar_lp(n: int) -> Callable:
    """N scalar vars with parametric scalar constraints. DPP scalarized tree."""
    ID = 100000
    def factory():
        xs = [_fixed_var(id_base=ID, idx=i) for i in range(n)]
        ps = [_fixed_param(nonneg=True, id_base=ID, idx=i) for i in range(n)]
        cons = [ps[i] * xs[i] <= 1 for i in range(n)] + [xs[i] >= -1 for i in range(n)]
        obj = cp.Minimize(sum(xs))
        prob = cp.Problem(obj, cons)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            for p in ps:
                p.value = abs(np.random.randn()) + 0.1

        return prob, init_params
    return factory


def make_dpp_param_lp(n_vars: int, n_constraints: int) -> Callable:
    """DPP parametrized LP: hot path is mul with parameter."""
    ID = 200000
    def factory():
        x = _fixed_var(n_vars, id_base=ID, idx=0)
        A_param = _fixed_param((n_constraints, n_vars), id_base=ID, idx=0)
        b = np.random.randn(n_constraints) + 10
        c = np.random.randn(n_vars)
        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b, x >= 0])

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A_param.value = np.random.randn(n_constraints, n_vars)

        return prob, init_params
    return factory


def make_dpp_lasso(n: int, m: int) -> Callable:
    """DPP LASSO: hot path is mul_elem parametric."""
    ID = 300000
    def factory():
        x = _fixed_var(n, id_base=ID, idx=0)
        A_param = _fixed_param((m, n), id_base=ID, idx=0)
        b_param = _fixed_param(m, id_base=ID, idx=1)
        obj = cp.Minimize(0.5 * cp.sum_squares(A_param @ x - b_param) + 0.1 * cp.norm(x, 1))
        prob = cp.Problem(obj)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A_param.value = np.random.randn(m, n)
            b_param.value = np.random.randn(m)

        return prob, init_params
    return factory


def make_dpp_multi_param(n_vars: int, n_constraints: int) -> Callable:
    """DPP multi-parameter: multiple mul ops."""
    ID = 400000
    def factory():
        x = _fixed_var(n_vars, id_base=ID, idx=0)
        A1 = _fixed_param((n_constraints // 2, n_vars), id_base=ID, idx=0)
        A2 = _fixed_param((n_constraints // 2, n_vars), id_base=ID, idx=1)
        b1 = np.random.randn(n_constraints // 2) + 10
        b2 = np.random.randn(n_constraints // 2) + 10
        c = np.random.randn(n_vars)
        prob = cp.Problem(
            cp.Minimize(c @ x),
            [A1 @ x <= b1, A2 @ x <= b2, x >= 0]
        )

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A1.value = np.random.randn(n_constraints // 2, n_vars)
            A2.value = np.random.randn(n_constraints // 2, n_vars)

        return prob, init_params
    return factory


def make_dpp_giant_lp(n_vars: int, n_constraints: int) -> Callable:
    """DPP with huge single parameter matrix. Tests O(nnz) vs O(param_size*rows)."""
    ID = 500000
    def factory():
        x = _fixed_var(n_vars, id_base=ID, idx=0)
        A_param = _fixed_param((n_constraints, n_vars), id_base=ID, idx=0)
        b = np.random.randn(n_constraints) + 10
        c = np.random.randn(n_vars)
        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b, x >= 0])

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A_param.value = np.random.randn(n_constraints, n_vars)

        return prob, init_params
    return factory


def make_dpp_3_matrices(n: int) -> Callable:
    """DPP with 3 large parameter matrices (3*n*n total params)."""
    ID = 600000
    def factory():
        x = _fixed_var(n, id_base=ID, idx=0)
        A1 = _fixed_param((n, n), id_base=ID, idx=0)
        A2 = _fixed_param((n, n), id_base=ID, idx=1)
        A3 = _fixed_param((n, n), id_base=ID, idx=2)
        b1 = np.random.randn(n) + 10
        b2 = np.random.randn(n) + 10
        b3 = np.random.randn(n) + 10
        c = np.random.randn(n)
        prob = cp.Problem(
            cp.Minimize(c @ x),
            [A1 @ x <= b1, A2 @ x <= b2, A3 @ x <= b3, x >= 0]
        )

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A1.value = np.random.randn(n, n)
            A2.value = np.random.randn(n, n)
            A3.value = np.random.randn(n, n)

        return prob, init_params
    return factory


def make_dpp_many_param_objs(n_params: int, n_vars: int = 100) -> Callable:
    """DPP with many separate Parameter objects (tests param object overhead)."""
    ID = 700000
    def factory():
        x = _fixed_var(n_vars, id_base=ID, idx=0)
        params = [_fixed_param(n_vars, id_base=ID, idx=i) for i in range(n_params)]
        cons = [p @ x <= 1 for p in params]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), cons)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            for p in params:
                p.value = np.random.randn(n_vars)

        return prob, init_params
    return factory


# =============================================================================
# Problem Suite Definition
# =============================================================================

QUICK_SUITE = [
    # Vectorized (non-DPP)
    ("sparse_lp_100", make_sparse_lp(100), 5),
    ("dense_qp_50", make_dense_qp(50), 5),
    ("lasso_50x100", make_lasso(50, 100), 5),
    ("many_constraints_50x100", make_many_constraints(50, 100), 5),
    # Scalarized trees
    ("scalar_squares_200", make_scalar_squares(200), 3),
    ("deep_chain_200", make_deep_chain(200), 3),
    ("scalar_grid_50", make_scalar_grid(50), 3),
    # DPP (small-medium params)
    ("dpp_param_lp_100x100", make_dpp_param_lp(100, 100), 5),
    ("dpp_lasso_50x100", make_dpp_lasso(50, 100), 5),
    ("dpp_multi_param_100x100", make_dpp_multi_param(100, 100), 5),
    ("dpp_scalar_lp_100", make_dpp_scalar_lp(100), 3),
    # DPP (huge params)
    ("dpp_giant_lp_1000x500", make_dpp_giant_lp(500, 1000), 2),
    ("dpp_3matrices_200", make_dpp_3_matrices(200), 3),
    ("dpp_many_param_objs_200", make_dpp_many_param_objs(200), 3),
]

FULL_SUITE = QUICK_SUITE + [
    # Vectorized (non-DPP) - larger
    ("sparse_lp_500", make_sparse_lp(500), 3),
    ("dense_qp_200", make_dense_qp(200), 3),
    ("lasso_200x500", make_lasso(200, 500), 3),
    ("many_constraints_50x500", make_many_constraints(50, 500), 3),
    # Scalarized trees - larger
    ("scalar_squares_1000", make_scalar_squares(1000), 2),
    ("deep_chain_500", make_deep_chain(500), 2),
    ("scalar_grid_100", make_scalar_grid(100), 2),
    # DPP (medium params)
    ("dpp_param_lp_500x200", make_dpp_param_lp(500, 200), 3),
    ("dpp_lasso_200x500", make_dpp_lasso(200, 500), 3),
    ("dpp_multi_param_200x500", make_dpp_multi_param(200, 500), 3),
    ("dpp_scalar_lp_500", make_dpp_scalar_lp(500), 2),
    # DPP (huge params) - stress test
    ("dpp_giant_lp_2000x1000", make_dpp_giant_lp(1000, 2000), 2),
    ("dpp_giant_lp_500x2000", make_dpp_giant_lp(2000, 500), 2),
    ("dpp_3matrices_500", make_dpp_3_matrices(500), 2),
    ("dpp_lasso_500x2000", make_dpp_lasso(500, 2000), 2),
    ("dpp_many_param_objs_500", make_dpp_many_param_objs(500), 2),
]


# =============================================================================
# Runner
# =============================================================================

_ORACLE_CACHE_DIR = Path(__file__).parent / ".canon_oracle_cache"
_oracle_cache: dict[str, dict[str, np.ndarray]] = {}


def _to_dense(val) -> np.ndarray:
    """Convert a possibly-sparse matrix to a flat float64 array."""
    if sp.issparse(val):
        val = val.toarray()
    return np.asarray(val, dtype=np.float64).ravel()


def _load_oracle(problem_name: str) -> dict[str, np.ndarray] | None:
    """Load cached oracle arrays from disk."""
    if problem_name in _oracle_cache:
        return _oracle_cache[problem_name]
    cache_file = _ORACLE_CACHE_DIR / f"{problem_name}.npz"
    if cache_file.exists():
        data = dict(np.load(cache_file))
        _oracle_cache[problem_name] = data
        return data
    return None


def _save_oracle(problem_name: str, arrays: dict[str, np.ndarray]) -> None:
    """Save oracle arrays to disk."""
    _ORACLE_CACHE_DIR.mkdir(exist_ok=True)
    _oracle_cache[problem_name] = arrays
    np.savez(_ORACLE_CACHE_DIR / f"{problem_name}.npz", **arrays)


def _extract_verify_arrays(data: dict, is_dpp: bool) -> dict[str, np.ndarray]:
    """Extract the arrays to verify from problem data.

    For non-DPP: A, b, c (the final matrices).
    For DPP: param_prob.A and param_prob.q (the reusable tensors).
    """
    arrays = {}
    if is_dpp and "param_prob" in data and data["param_prob"] is not None:
        pp = data["param_prob"]
        if pp.A is not None:
            arrays["pp_A"] = _to_dense(pp.A)
        if pp.q is not None:
            arrays["pp_q"] = _to_dense(pp.q)
    else:
        for key in ["A", "b", "c"]:
            val = data.get(key)
            if val is not None:
                arrays[key] = _to_dense(val)
    return arrays


def _get_oracle(
    problem_factory: Callable, problem_name: str, is_dpp: bool,
) -> dict[str, np.ndarray]:
    """Get SCIPY oracle arrays, computing and caching if needed."""
    cached = _load_oracle(problem_name)
    if cached is not None:
        return cached

    prob, init_params = problem_factory()
    if init_params is not None:
        np.random.seed(12345)
        init_params()
    data, _, _ = prob.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")

    arrays = _extract_verify_arrays(data, is_dpp)
    _save_oracle(problem_name, arrays)
    return arrays


def _verify_against_oracle(
    problem_factory: Callable,
    problem_name: str,
    test_backend: str,
    is_dpp: bool = False,
) -> dict:
    """Verify test backend output against cached SCIPY oracle arrays.

    On first call per problem, computes and caches SCIPY oracle arrays as .npz.
    Subsequent calls only run the test backend and compare with allclose.

    For non-DPP: compares A, b, c matrices.
    For DPP: compares param_prob.A and param_prob.q tensors (parameter-independent).

    Returns dict with 'pass', 'max_err', and 'mismatches' fields.
    """
    try:
        oracle = _get_oracle(problem_factory, problem_name, is_dpp)
    except Exception as e:
        return {"pass": False, "max_err": float("inf"), "mismatches": [f"oracle: {e}"]}

    try:
        prob, init_params = problem_factory()
        if init_params is not None:
            np.random.seed(12345)
            init_params()
        prob._cache = type(prob._cache)()
        test_data, _, _ = prob.get_problem_data(
            cp.CLARABEL, canon_backend=test_backend
        )
    except Exception as e:
        return {"pass": False, "max_err": float("inf"), "mismatches": [f"test: {e}"]}

    test_arrays = _extract_verify_arrays(test_data, is_dpp)

    mismatches = []
    max_err = 0.0

    all_keys = set(oracle.keys()) | set(test_arrays.keys())
    for key in sorted(all_keys):
        oracle_arr = oracle.get(key)
        test_arr = test_arrays.get(key)

        if oracle_arr is None and test_arr is None:
            continue
        if oracle_arr is None or test_arr is None:
            mismatches.append(
                f"{key}: missing in {'test' if test_arr is None else 'oracle'}"
            )
            continue

        if test_arr.shape != oracle_arr.shape:
            mismatches.append(
                f"{key}: shape mismatch {test_arr.shape} vs {oracle_arr.shape}"
            )
            continue

        if not np.allclose(test_arr, oracle_arr, atol=1e-8, rtol=1e-6):
            diff = np.abs(test_arr - oracle_arr)
            err = float(np.max(diff))
            max_err = max(max_err, err)
            mismatches.append(f"{key}: max_err={err:.2e}")
        else:
            diff = np.abs(test_arr - oracle_arr)
            if diff.size > 0:
                max_err = max(max_err, float(np.max(diff)))

    return {
        "pass": len(mismatches) == 0,
        "max_err": round(max_err, 12),
        "mismatches": mismatches,
    }


def run_suite(
    suite: list[tuple[str, Callable, int]],
    backend: str,
    verbose: bool = True,
    trace: bool = False,
    verify: bool = False,
) -> BenchmarkSuite:
    """Run the benchmark suite for a single backend."""
    results = BenchmarkSuite(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        backends=[backend],
    )

    valid_means = []

    for name, factory, iters in suite:
        result = time_canonicalization(factory, backend, iterations=iters, trace=trace)
        result.name = name

        if result.error:
            if verbose:
                print(f"  {name:35s} ERROR: {result.error[:50]}", file=sys.stderr)
        else:
            valid_means.append(result.mean_ms)

            # Verify against SCIPY oracle
            if verify and backend != "SCIPY":
                vresult = _verify_against_oracle(
                    factory, name, backend, is_dpp=result.is_dpp,
                )
                result.verification = vresult
                vtag = " PASS" if vresult["pass"] else " FAIL"
                if verbose:
                    print(f"  {name:35s} {result.mean_ms:8.2f}ms "
                          f"(±{result.std_ms:5.2f}){vtag}", file=sys.stderr)
                    if not vresult["pass"]:
                        for mm in vresult["mismatches"]:
                            print(f"    VERIFY FAIL: {mm}", file=sys.stderr)
                elif not vresult["pass"] and verbose:
                    for mm in vresult["mismatches"]:
                        print(f"    VERIFY FAIL: {mm}", file=sys.stderr)
            elif verbose:
                dpp_tag = " [DPP]" if result.is_dpp else ""
                print(f"  {name:35s} {result.mean_ms:8.2f}ms "
                      f"(±{result.std_ms:5.2f}){dpp_tag}", file=sys.stderr)

            if verbose and trace and result.trace:
                for t in result.trace[:5]:
                    print(f"    {t['pct']:5.1f}%  {t['tottime_ms']:7.1f}ms  "
                          f"{t['calls']:>6d}x  {t['func']}", file=sys.stderr)

        results.problems.append(result)

    if valid_means:
        results.geomean_ms = math.exp(sum(math.log(t) for t in valid_means) / len(valid_means))
        results.total_ms = sum(valid_means)

    if verbose:
        print(f"\n  Geomean: {results.geomean_ms:.2f}ms  |  Total: {results.total_ms:.1f}ms",
              file=sys.stderr)

    return results


def run_all_backends(
    suite: list[tuple[str, Callable, int]],
    backends: list[str],
    verbose: bool = True,
    trace: bool = False,
    verify: bool = False,
) -> dict[str, BenchmarkSuite]:
    """Run suite across all backends and return combined results."""
    all_results = {}
    for backend in backends:
        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"Backend: {backend}", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
        all_results[backend] = run_suite(
            suite, backend, verbose=verbose, trace=trace, verify=verify,
        )

    # Print comparison
    if verbose and len(backends) > 1:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("COMPARISON", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        for b, r in all_results.items():
            print(f"  {b:8s}: geomean={r.geomean_ms:.2f}ms  total={r.total_ms:.1f}ms",
                  file=sys.stderr)

    return all_results


def combine_results(all_results: dict[str, BenchmarkSuite]) -> dict:
    """Combine results from multiple backends into a single JSON-serializable dict."""
    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backends": {},
    }
    for backend, suite in all_results.items():
        combined["backends"][backend] = suite.to_dict()

    # Pick the default/best geomean
    geomeans = {b: s.geomean_ms for b, s in all_results.items() if s.geomean_ms > 0}
    if geomeans:
        best_backend = min(geomeans, key=geomeans.get)
        combined["best_backend"] = best_backend
        combined["geomean_ms"] = geomeans[best_backend]
    return combined


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CVXPY canonicalization benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick run (small problems only)")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--backend", type=str, default=None,
                        help="Specific backend (CPP, SCIPY, COO)")
    parser.add_argument("--all-backends", action="store_true",
                        help="Run all backends (CPP, SCIPY, COO)")
    parser.add_argument("--quiet", action="store_true", help="Suppress stderr output")
    parser.add_argument("--trace", action="store_true",
                        help="Capture execution trace (cProfile) for each problem")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output matrices against SCIPY oracle")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild SCIPY oracle cache (use after changing problem factories)")
    args = parser.parse_args()

    if args.rebuild_cache:
        import shutil
        if _ORACLE_CACHE_DIR.exists():
            shutil.rmtree(_ORACLE_CACHE_DIR)
            print("Oracle cache cleared.", file=sys.stderr)
        else:
            print("No oracle cache to clear.", file=sys.stderr)

    suite = QUICK_SUITE if args.quick else FULL_SUITE
    verbose = not args.quiet

    if args.all_backends:
        backends = ["CPP", "SCIPY", "COO"]
        all_results = run_all_backends(
            suite, backends, verbose=verbose, trace=args.trace, verify=args.verify,
        )
        output = combine_results(all_results)
    else:
        backend = args.backend or "CPP"
        result = run_suite(
            suite, backend, verbose=verbose, trace=args.trace, verify=args.verify,
        )
        output = result.to_dict()
        output["geomean_ms"] = result.geomean_ms

    if args.json:
        print(json.dumps(output, indent=2, default=str))
    elif not args.quiet:
        print(f"\nDone. geomean={output.get('geomean_ms', 0):.2f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
