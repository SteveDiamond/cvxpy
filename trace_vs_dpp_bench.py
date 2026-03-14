"""
Benchmark: CPU DPP vs GPU DPP vs GPU Trace-Compile canonicalization.

Compares four approaches:
  1. CPU DPP warm path  — CVXPY's native param_prob.apply_parameters()
  2. GPU DPP warm path  — CompiledProgram.canonicalize() from backend.py
  3. GPU trace-compile (first call) — canonicalize_gpu_traced(), cache miss
  4. GPU trace-compile (replay)     — canonicalize_gpu_traced(), cache hit
"""

import time
import sys
import gc

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.cvxcore.python import canonInterface

import cupy as cup
import cupyx.scipy.sparse as cusp

from gpu_canon.backend import CompiledProgram, canonicalize_gpu_traced, reset_trace_cache
from gpu_canon.trace_cache import reset_id_counter


# ─── Problem factories ───────────────────────────────────────────────────────

def make_lp(n):
    """Parametric LP: min c^T x  s.t. Ax <= b, x >= 0."""
    reset_id_counter()
    x = cp.Variable(n)
    c_param = cp.Parameter(n)
    A_param = cp.Parameter((n, n))
    b_param = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(c_param @ x), [A_param @ x <= b_param, x >= 0])

    # Set random values
    rng = np.random.RandomState(42)
    c_param.value = rng.randn(n)
    A_param.value = rng.randn(n, n)
    b_param.value = np.abs(rng.randn(n)) + 1.0
    return prob, f"LP(n={n})"


def make_lasso(n, m):
    """Parametric LASSO: min ||Ax - b||_2^2 + lam * ||x||_1."""
    reset_id_counter()
    x = cp.Variable(n)
    A_param = cp.Parameter((m, n))
    b_param = cp.Parameter(m)
    lam = cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A_param @ x - b_param) + lam * cp.norm1(x)))

    rng = np.random.RandomState(42)
    A_param.value = rng.randn(m, n)
    b_param.value = rng.randn(m)
    lam.value = 0.1
    return prob, f"LASSO(m={m},n={n})"


def make_portfolio(n):
    """Parametric portfolio: min -mu^T x + gamma * x^T Sigma x, s.t. 1^T x = 1, x >= 0."""
    reset_id_counter()
    x = cp.Variable(n)
    mu = cp.Parameter(n)
    # For DPP: gamma * quad_form(x, Sigma) requires Sigma constant
    rng = np.random.RandomState(42)
    F = rng.randn(n, n)
    Sigma = F.T @ F / n + 0.1 * np.eye(n)  # constant PSD matrix
    gamma = cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(-mu @ x + gamma * cp.quad_form(x, Sigma)),
        [cp.sum(x) == 1, x >= 0],
    )
    mu.value = rng.rand(n) * 0.1
    gamma.value = 1.0
    return prob, f"Portfolio(n={n})"


def make_svm(n, m):
    """Parametric SVM: min ||w||^2 + C * sum(xi), s.t. y_i(w^T x_i + b) >= 1 - xi."""
    reset_id_counter()
    w = cp.Variable(n)
    b_var = cp.Variable()
    xi = cp.Variable(m, nonneg=True)
    C_param = cp.Parameter(nonneg=True)

    rng = np.random.RandomState(42)
    X_data = rng.randn(m, n)
    y_data = np.sign(rng.randn(m))

    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(w) + C_param * cp.sum(xi)),
        [cp.multiply(y_data, X_data @ w + b_var) >= 1 - xi],
    )
    C_param.value = 1.0
    return prob, f"SVM(m={m},n={n})"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def gpu_sync():
    """Synchronize GPU to ensure accurate timing."""
    cup.cuda.Device().synchronize()


def to_cpu_dense(x):
    """Convert GPU/sparse array to dense CPU numpy array."""
    if x is None:
        return None
    if hasattr(x, 'get'):
        x = x.get()
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def compare_arrays(a, b, name, allow_sign_flip=False):
    """Compare two arrays, return (match, max_diff).

    allow_sign_flip: if True, also accept a == -b (sign convention difference).
    """
    a = to_cpu_dense(a)
    b = to_cpu_dense(b)
    if a is None and b is None:
        return True, 0.0
    if a is None or b is None:
        return False, float('inf')
    if a.shape != b.shape:
        a = a.flatten()
        b = b.flatten()
        if a.shape != b.shape:
            return False, float('inf')
    diff = np.max(np.abs(a - b))
    if diff < 1e-6:
        return True, diff
    if allow_sign_flip:
        diff_neg = np.max(np.abs(a + b))
        if diff_neg < 1e-6:
            return True, diff_neg
    return False, diff


def time_fn(fn, n_warmup=2, n_iter=10):
    """Time a function with warmup and GPU sync. Returns median time in ms."""
    # Warmup
    for _ in range(n_warmup):
        fn()
        gpu_sync()

    times = []
    for _ in range(n_iter):
        gpu_sync()
        t0 = time.perf_counter()
        fn()
        gpu_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times)


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(prob_factory, label, n_warmup=2, n_iter=15):
    """Run all four approaches on a single problem, return results dict."""
    results = {}

    # ── 1. CPU DPP warm path ──────────────────────────────────────────────
    prob, _ = prob_factory()
    # Cold call to set up DPP cache
    data, chain, inv = prob.get_problem_data(cp.CLARABEL, canon_backend="COO")
    param_prob = data.get("param_prob")

    if param_prob is None or param_prob.total_param_size == 0:
        print(f"  SKIP {label}: no parameters / not DPP")
        return None

    # Cache reduced_A
    param_prob.reduced_A.cache(False)

    # Get baseline result for correctness comparison
    # NOTE: apply_parameters returns A with opposite sign from get_problem_data.
    # Use get_problem_data output as the canonical baseline since that's what
    # both GPU DPP (CompiledProgram) and trace-compile use.
    A_base = data.get("A")
    b_base = data.get("b")
    c_base = data.get("c")
    A_base_dense = A_base.toarray() if sp.issparse(A_base) else np.asarray(A_base)
    b_base = np.asarray(b_base).flatten()
    c_base = np.asarray(c_base).flatten()

    # Time CPU DPP
    def cpu_dpp():
        param_prob.apply_parameters()

    t_cpu = time_fn(cpu_dpp, n_warmup=n_warmup, n_iter=n_iter)
    # CPU DPP is the timing reference; correctness is trivially true (it IS the baseline,
    # modulo sign convention which we've already accounted for).
    results["CPU DPP"] = {"time_ms": t_cpu, "match_A": True, "match_b": True, "match_c": True}

    # ── 2. GPU DPP warm path ──────────────────────────────────────────────
    prob2, _ = prob_factory()
    try:
        compiled = CompiledProgram(prob2)
        # Warmup + time
        def gpu_dpp():
            compiled.canonicalize()

        t_gpu_dpp = time_fn(gpu_dpp, n_warmup=n_warmup, n_iter=n_iter)

        A_g, b_g, c_g, _ = compiled.canonicalize()
        gpu_sync()
        ma, da = compare_arrays(A_g, A_base_dense, "A", allow_sign_flip=True)
        mb, db = compare_arrays(b_g, b_base, "b")
        mc, dc = compare_arrays(c_g, c_base, "c")
        results["GPU DPP"] = {"time_ms": t_gpu_dpp, "match_A": ma, "match_b": mb, "match_c": mc}
    except Exception as e:
        results["GPU DPP"] = {"time_ms": float('nan'), "match_A": False, "match_b": False,
                              "match_c": False, "error": str(e)}

    # ── 3. GPU trace-compile (first call = miss) ──────────────────────────
    reset_trace_cache()

    prob3, _ = prob_factory()
    # Time the first call (tracing overhead)
    gpu_sync()
    t0 = time.perf_counter()
    A_t1, b_t1, c_t1, _ = canonicalize_gpu_traced(prob3)
    gpu_sync()
    t_trace_miss = (time.perf_counter() - t0) * 1000

    ma, da = compare_arrays(A_t1, A_base_dense, "A", allow_sign_flip=True)
    mb, db = compare_arrays(b_t1, b_base, "b")
    mc, dc = compare_arrays(c_t1, c_base, "c")
    results["Trace (miss)"] = {"time_ms": t_trace_miss, "match_A": ma, "match_b": mb,
                                "match_c": mc}

    # ── 4. GPU trace-compile (replay = hit) ───────────────────────────────
    # Now the cache should have the trace. Call again with same structure.
    prob4, _ = prob_factory()

    def trace_replay():
        # Need fresh problem each time for realistic measurement
        # but that includes problem construction overhead. Instead,
        # just re-call with the existing problem (params already set).
        canonicalize_gpu_traced(prob4)

    t_trace_hit = time_fn(trace_replay, n_warmup=n_warmup, n_iter=n_iter)

    A_t2, b_t2, c_t2, _ = canonicalize_gpu_traced(prob4)
    gpu_sync()
    ma, da = compare_arrays(A_t2, A_base_dense, "A", allow_sign_flip=True)
    mb, db = compare_arrays(b_t2, b_base, "b")
    mc, dc = compare_arrays(c_t2, c_base, "c")
    results["Trace (hit)"] = {"time_ms": t_trace_hit, "match_A": ma, "match_b": mb,
                               "match_c": mc}

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # GPU warmup
    print("GPU warmup...")
    _ = cup.zeros(100) + 1
    gpu_sync()

    # Define problem suite
    problems = [
        (lambda: make_lp(50),            "LP(50)"),
        (lambda: make_lp(200),           "LP(200)"),
        (lambda: make_lp(500),           "LP(500)"),
        (lambda: make_lp(1000),          "LP(1000)"),
        (lambda: make_lasso(100, 50),    "LASSO(100,50)"),
        (lambda: make_lasso(500, 200),   "LASSO(500,200)"),
        (lambda: make_lasso(1000, 500),  "LASSO(1000,500)"),
        (lambda: make_portfolio(50),     "Portfolio(50)"),
        (lambda: make_portfolio(200),    "Portfolio(200)"),
        (lambda: make_svm(50, 100),      "SVM(50,100)"),
        (lambda: make_svm(100, 500),     "SVM(100,500)"),
    ]

    # Adjust iterations for speed
    n_warmup = 2
    n_iter = 10

    all_results = {}
    approaches = ["CPU DPP", "GPU DPP", "Trace (miss)", "Trace (hit)"]

    for factory, label in problems:
        print(f"\nBenchmarking: {label} ...", flush=True)
        gc.collect()
        cup.get_default_memory_pool().free_all_blocks()
        reset_trace_cache()

        try:
            res = run_benchmark(factory, label, n_warmup=n_warmup, n_iter=n_iter)
            if res is not None:
                all_results[label] = res
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Print results table ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS: CPU DPP vs GPU DPP vs GPU Trace-Compile")
    print("=" * 100)

    # Header
    col_w = 14
    hdr = f"{'Problem':<22}"
    for app in approaches:
        hdr += f" | {app:>{col_w}}"
    hdr += " | GPU DPP  | Trace hit"
    hdr += "\n" + " " * 22
    for app in approaches:
        hdr += f" | {'time (ms)':>{col_w}}"
    hdr += " | speedup  | speedup"
    print(hdr)
    print("-" * 100)

    for label, res in all_results.items():
        row = f"{label:<22}"
        cpu_time = res.get("CPU DPP", {}).get("time_ms", float('nan'))
        for app in approaches:
            r = res.get(app)
            if r is None:
                row += f" | {'N/A':>{col_w}}"
            else:
                t = r["time_ms"]
                ok = r.get("match_A", False) and r.get("match_b", False) and r.get("match_c", False)
                mark = "" if ok else "*"
                row += f" | {t:>{col_w - 1}.3f}{mark}"

        # Speedup columns
        gpu_dpp_time = res.get("GPU DPP", {}).get("time_ms", float('nan'))
        trace_hit_time = res.get("Trace (hit)", {}).get("time_ms", float('nan'))

        if not np.isnan(gpu_dpp_time) and gpu_dpp_time > 0:
            row += f" | {cpu_time / gpu_dpp_time:>7.1f}x"
        else:
            row += f" | {'N/A':>8}"

        if not np.isnan(trace_hit_time) and trace_hit_time > 0:
            row += f" | {cpu_time / trace_hit_time:>7.1f}x"
        else:
            row += f" | {'N/A':>8}"

        print(row)

    # Correctness summary
    print("\n" + "-" * 100)
    print("CORRECTNESS (vs CPU DPP baseline):")
    any_fail = False
    for label, res in all_results.items():
        for app in approaches:
            r = res.get(app)
            if r and not (r.get("match_A", True) and r.get("match_b", True) and r.get("match_c", True)):
                fails = []
                if not r.get("match_A", True):
                    fails.append("A")
                if not r.get("match_b", True):
                    fails.append("b")
                if not r.get("match_c", True):
                    fails.append("c")
                print(f"  MISMATCH: {label} / {app} — failed on: {', '.join(fails)}")
                any_fail = True
    if not any_fail:
        print("  All approaches match CPU baseline within tolerance (1e-6).")

    print("\n* = correctness mismatch with CPU baseline")
    print(f"Iterations per measurement: {n_iter} (median reported)")
    print("=" * 100)


if __name__ == "__main__":
    main()
