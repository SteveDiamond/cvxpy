"""
Comprehensive profiling of the trace-compile GPU replay system.

Goals:
1. Profile replay overhead for small problems
2. Measure op-level timing for medium problems
3. Compare trace replay vs DPP warm path
4. Test with larger problems
5. Identify optimization opportunities
"""

import time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

import cupy as cup
import cupyx.scipy.sparse as cusp

from gpu_canon.trace_cache import (
    TraceCache, _problem_fingerprint, _classify_inputs,
    reset_id_counter,
)
from gpu_canon.tracer import NumpyTracer
from gpu_canon.replayer import replay_on_gpu, replay_on_cpu, _DISPATCH, _is_valid_dtype
from gpu_canon.backend import CompiledProgram
from gpu_canon.baseline import canonicalize_cpu


def timeit(fn, warmup=2, repeat=10, sync_gpu=True):
    """Time a function, returning median time in seconds."""
    for _ in range(warmup):
        fn()
    if sync_gpu:
        cup.cuda.Stream.null.synchronize()
    times = []
    for _ in range(repeat):
        if sync_gpu:
            cup.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_gpu:
            cup.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.array(times)


# ── Problem factories ─────────────────────────────────────────────────────

def make_lp(m, n):
    """Standard LP: minimize c@x s.t. A@x <= b, x >= 0."""
    np.random.seed(42)
    A_data = np.random.randn(m, n)
    b_data = np.abs(np.random.randn(m)) * 10
    c_data = np.random.randn(n)

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(c_data @ x),
        [A_data @ x <= b_data, x >= 0]
    )
    return prob


def make_lasso(m, n, lam=1.0):
    """LASSO: minimize ||Ax - b||_2^2 + lam * ||x||_1."""
    np.random.seed(42)
    A_data = np.random.randn(m, n)
    b_data = np.random.randn(m)

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(A_data @ x - b_data) + lam * cp.norm1(x))
    )
    return prob


def make_parametric_lp(m, n):
    """Parametric LP with CVXPY Parameters (DPP-compatible)."""
    np.random.seed(42)
    A_param = cp.Parameter((m, n))
    b_param = cp.Parameter(m)
    c_param = cp.Parameter(n)

    A_param.value = np.random.randn(m, n)
    b_param.value = np.abs(np.random.randn(m)) * 10
    c_param.value = np.random.randn(n)

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(c_param @ x),
        [A_param @ x <= b_param, x >= 0]
    )
    return prob, A_param, b_param, c_param


def make_parametric_lasso(m, n, lam=1.0):
    """Parametric LASSO with CVXPY Parameters (DPP-compatible)."""
    np.random.seed(42)
    A_param = cp.Parameter((m, n))
    b_param = cp.Parameter(m)

    A_param.value = np.random.randn(m, n)
    b_param.value = np.random.randn(m)

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(A_param @ x - b_param) + lam * cp.norm1(x))
    )
    return prob, A_param, b_param


# ═══════════════════════════════════════════════════════════════════════════
# GOAL 1: Profile replay overhead for small problems
# ═══════════════════════════════════════════════════════════════════════════

def profile_small_problem_overhead():
    print("=" * 70)
    print("GOAL 1: Profile replay overhead breakdown (LP 100x50)")
    print("=" * 70)

    reset_id_counter()
    prob = make_lp(100, 50)

    # --- Trace the problem ---
    solver_cls = cp.CLARABEL
    with NumpyTracer() as tracer:
        data, chain, inv = prob.get_problem_data(solver_cls, canon_backend="COO")

    A_cpu = data.get("A")
    b_cpu = data.get("b")
    c_cpu = data.get("c")

    if A_cpu is not None and sp.issparse(A_cpu):
        A_csc = A_cpu.tocsc()
        tracer.register_output("A_data", A_csc.data)
    if b_cpu is not None:
        tracer.register_output("b", b_cpu)
    if c_cpu is not None:
        tracer.register_output("c", c_cpu)

    trace = tracer.get_trace()
    input_ids = tracer.get_input_ids()
    input_arrays = tracer.get_input_arrays()

    print(f"  Trace length: {len(trace)} ops")
    print(f"  Input arrays: {len(input_ids)}")
    total_input_bytes = sum(a.nbytes for a in input_arrays.values())
    print(f"  Total input bytes: {total_input_bytes:,}")

    # Classify
    data_inputs, structural_inputs = _classify_inputs(
        trace, input_ids, input_arrays, prob)
    print(f"  Data inputs: {len(data_inputs)} ({sum(a.nbytes for a in data_inputs.values()):,} bytes)")
    print(f"  Structural inputs: {len(structural_inputs)} ({sum(a.nbytes for a in structural_inputs.values()):,} bytes)")

    # --- Phase timing ---
    print("\n  Phase breakdown (median of 50 runs):")

    # 1. Fingerprint
    reset_id_counter()
    prob2 = make_lp(100, 50)

    t_fp, _ = timeit(lambda: _problem_fingerprint(prob2), warmup=5, repeat=50, sync_gpu=False)
    print(f"    Fingerprint computation: {t_fp*1000:.3f} ms")

    # 2. Input GPU transfer (data inputs only, structural already cached)
    def transfer_data_inputs():
        gpu_arrays = {}
        for tid, arr in data_inputs.items():
            gpu_arrays[tid] = cup.asarray(arr)
        return gpu_arrays

    t_transfer, _ = timeit(transfer_data_inputs, warmup=5, repeat=50)
    print(f"    Data input GPU transfer: {t_transfer*1000:.3f} ms")

    # 3. Build initial_arrays (structural from cache + fresh data)
    structural_gpu = {tid: cup.asarray(arr) for tid, arr in structural_inputs.items()}

    def build_initial():
        initial = dict(structural_gpu)
        for tid, arr in data_inputs.items():
            initial[tid] = cup.asarray(arr)
        return initial

    t_build, _ = timeit(build_initial, warmup=5, repeat=50)
    print(f"    Build initial_arrays: {t_build*1000:.3f} ms")

    # 4. Replay loop only
    initial_arrays = build_initial()

    def replay_only():
        return replay_on_gpu(trace, initial_arrays)

    t_replay, _ = timeit(replay_only, warmup=5, repeat=50)
    print(f"    Replay loop (GPU): {t_replay*1000:.3f} ms")

    # 5. CPU replay for comparison
    initial_cpu = {}
    initial_cpu.update(structural_inputs)
    initial_cpu.update(data_inputs)

    def replay_cpu():
        return replay_on_cpu(trace, initial_cpu)

    t_cpu_replay, _ = timeit(replay_cpu, warmup=5, repeat=50, sync_gpu=False)
    print(f"    Replay loop (CPU): {t_cpu_replay*1000:.3f} ms")

    # 6. Full TraceCache path
    cache = TraceCache()
    reset_id_counter()
    prob_trace1 = make_lp(100, 50)
    cache.canonicalize(prob_trace1)  # cold path

    reset_id_counter()
    prob_trace2 = make_lp(100, 50)

    def full_trace_replay():
        return cache.canonicalize(prob_trace2)

    t_full, _ = timeit(full_trace_replay, warmup=5, repeat=50)
    print(f"    Full TraceCache replay: {t_full*1000:.3f} ms")

    # 7. CPU baseline
    reset_id_counter()
    prob_base = make_lp(100, 50)

    def cpu_baseline():
        return canonicalize_cpu(prob_base)

    t_baseline, _ = timeit(cpu_baseline, warmup=5, repeat=50, sync_gpu=False)
    print(f"    CPU baseline (CVXPY): {t_baseline*1000:.3f} ms")

    print(f"\n  Speedup: {t_baseline / t_full:.2f}x (trace replay vs CPU)")
    print(f"  Replay/CPU ratio: {t_replay / t_cpu_replay:.2f}x (GPU replay loop vs CPU replay loop)")

    # Op type breakdown
    op_counts = {}
    const_ops = 0
    for op in trace:
        name = op.op
        op_counts[name] = op_counts.get(name, 0) + 1
        if op.constant_value is not None and not op.input_ids:
            const_ops += 1

    print(f"\n  Op type distribution ({len(trace)} total, {const_ops} constant):")
    for name, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name}: {count}")


# ═══════════════════════════════════════════════════════════════════════════
# GOAL 2: Op-level timing for medium problem (LP 500x250)
# ═══════════════════════════════════════════════════════════════════════════

def profile_medium_problem():
    print("\n" + "=" * 70)
    print("GOAL 2: Op-level timing (LP 500x250)")
    print("=" * 70)

    reset_id_counter()
    prob = make_lp(500, 250)

    # Trace
    with NumpyTracer() as tracer:
        data, chain, inv = prob.get_problem_data(cp.CLARABEL, canon_backend="COO")

    A_cpu = data["A"].tocsc() if sp.issparse(data["A"]) else data["A"]
    b_cpu = data["b"]
    c_cpu = data["c"]

    if sp.issparse(A_cpu):
        A_csc = A_cpu if isinstance(A_cpu, sp.csc_matrix) else A_cpu.tocsc()
        tracer.register_output("A_data", A_csc.data)
        A_structural = {
            "indices": A_csc.indices.copy(),
            "indptr": A_csc.indptr.copy(),
            "shape": A_csc.shape,
        }
    tracer.register_output("b", b_cpu)
    tracer.register_output("c", c_cpu)

    trace = tracer.get_trace()
    input_ids = tracer.get_input_ids()
    input_arrays = tracer.get_input_arrays()
    data_inputs, structural_inputs = _classify_inputs(
        trace, input_ids, input_arrays, prob)

    print(f"  Trace: {len(trace)} ops, {len(input_ids)} inputs")

    # --- Per-op timing on GPU ---
    structural_gpu = {tid: cup.asarray(arr) for tid, arr in structural_inputs.items()}

    initial_arrays = dict(structural_gpu)
    for tid, arr in data_inputs.items():
        initial_arrays[tid] = cup.asarray(arr)

    # Replay with per-op timing
    cup.cuda.Stream.null.synchronize()

    registry = dict(initial_arrays)
    op_times = []
    op_names = []
    op_is_const = []

    for op in trace:
        inputs = [registry[iid] for iid in op.input_ids if iid in registry]
        kwargs = dict(op.kwargs)
        kwargs["_output_dtype"] = op.output_dtype
        kwargs["_output_shape"] = op.output_shape

        cup.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        if op.constant_value is not None and not op.input_ids:
            cv = op.constant_value
            if hasattr(cv, 'dtype') and not _is_valid_dtype(str(cv.dtype)):
                cv = np.zeros(cv.shape, dtype=np.float64)
            result = cup.asarray(cv)
        else:
            handler = _DISPATCH.get(op.op)
            if handler is not None:
                result = handler(inputs, kwargs, cup)
            elif op.op.startswith("sp."):
                result = inputs[0].copy() if inputs else cup.array([], dtype="float64")
            else:
                raise ValueError(f"Unknown op: {op.op}")

        cup.cuda.Stream.null.synchronize()
        op_times.append(time.perf_counter() - t0)
        op_names.append(op.op)
        op_is_const.append(op.constant_value is not None and not op.input_ids)
        registry[op.output_id] = result

    total_op_time = sum(op_times)
    const_time = sum(t for t, c in zip(op_times, op_is_const) if c)
    compute_time = sum(t for t, c in zip(op_times, op_is_const) if not c)

    print(f"\n  Per-op replay total: {total_op_time*1000:.3f} ms")
    print(f"    Constant ops (cp.asarray): {const_time*1000:.3f} ms ({const_time/total_op_time*100:.1f}%)")
    print(f"    Compute ops: {compute_time*1000:.3f} ms ({compute_time/total_op_time*100:.1f}%)")

    # Group by op type
    op_type_times = {}
    op_type_counts = {}
    for name, t in zip(op_names, op_times):
        op_type_times[name] = op_type_times.get(name, 0) + t
        op_type_counts[name] = op_type_counts.get(name, 0) + 1

    print(f"\n  Time by op type (top 10):")
    for name, total_t in sorted(op_type_times.items(), key=lambda x: -x[1])[:10]:
        cnt = op_type_counts[name]
        print(f"    {name}: {total_t*1000:.3f} ms ({cnt} ops, {total_t/total_op_time*100:.1f}%)")

    # Phase timing
    print(f"\n  Full phase breakdown:")

    # Fingerprint
    reset_id_counter()
    prob2 = make_lp(500, 250)
    t_fp, _ = timeit(lambda: _problem_fingerprint(prob2), warmup=3, repeat=20, sync_gpu=False)
    print(f"    Fingerprint: {t_fp*1000:.3f} ms")

    # Transfer
    def transfer():
        r = {}
        for tid, arr in data_inputs.items():
            r[tid] = cup.asarray(arr)
        return r
    t_xfer, _ = timeit(transfer, warmup=3, repeat=20)
    print(f"    GPU transfer (data): {t_xfer*1000:.3f} ms")

    # Replay
    def replay():
        ia = dict(structural_gpu)
        for tid, arr in data_inputs.items():
            ia[tid] = cup.asarray(arr)
        return replay_on_gpu(trace, ia)
    t_replay, _ = timeit(replay, warmup=3, repeat=20)
    print(f"    Replay loop: {t_replay*1000:.3f} ms")

    # Output extraction (CSC assembly)
    reg = replay()
    def extract():
        if "A_data" in {k: v for k, v in [("A_data", None)]}:
            pass  # placeholder
        # Simulate CSC assembly
        cusp.csc_matrix(
            (reg[list(reg.keys())[-1]],
             cup.asarray(A_structural["indices"]),
             cup.asarray(A_structural["indptr"])),
            shape=A_structural["shape"],
        )
    # Simpler: time the full TraceCache path
    cache = TraceCache()
    reset_id_counter()
    p1 = make_lp(500, 250)
    cache.canonicalize(p1)
    reset_id_counter()
    p2 = make_lp(500, 250)
    t_full, _ = timeit(lambda: cache.canonicalize(p2), warmup=3, repeat=20)
    print(f"    Full TraceCache replay: {t_full*1000:.3f} ms")

    # CPU baseline
    reset_id_counter()
    p3 = make_lp(500, 250)
    t_cpu, _ = timeit(lambda: canonicalize_cpu(p3), warmup=3, repeat=20, sync_gpu=False)
    print(f"    CPU baseline: {t_cpu*1000:.3f} ms")
    print(f"    Speedup: {t_cpu/t_full:.2f}x")


# ═══════════════════════════════════════════════════════════════════════════
# GOAL 3: Trace replay vs DPP warm path
# ═══════════════════════════════════════════════════════════════════════════

def compare_trace_vs_dpp():
    print("\n" + "=" * 70)
    print("GOAL 3: Trace replay vs DPP warm path (CompiledProgram)")
    print("=" * 70)

    sizes = [(100, 50), (500, 250), (1000, 500)]

    for m, n in sizes:
        print(f"\n  --- Parametric LP {m}x{n} ---")

        # DPP path: CompiledProgram
        reset_id_counter()
        prob_dpp, A_p, b_p, c_p = make_parametric_lp(m, n)
        compiled = CompiledProgram(prob_dpp)
        # Warm up
        compiled.canonicalize()
        t_dpp, _ = timeit(lambda: compiled.canonicalize(), warmup=3, repeat=20)
        print(f"    DPP (CompiledProgram.canonicalize): {t_dpp*1000:.3f} ms")

        # DPP inplace
        t_dpp_ip, _ = timeit(lambda: compiled.canonicalize_inplace(), warmup=3, repeat=20)
        print(f"    DPP inplace: {t_dpp_ip*1000:.3f} ms")

        # Trace replay path
        try:
            cache = TraceCache()
            reset_id_counter()
            prob_t1, _, _, _ = make_parametric_lp(m, n)
            cache.canonicalize(prob_t1)  # cold: trace
            reset_id_counter()
            prob_t2, _, _, _ = make_parametric_lp(m, n)
            t_trace, _ = timeit(lambda: cache.canonicalize(prob_t2), warmup=3, repeat=20)
            print(f"    Trace replay: {t_trace*1000:.3f} ms")
        except Exception as e:
            print(f"    Trace replay failed: {e}")
            t_trace = None

        # CPU baseline
        reset_id_counter()
        prob_cpu, _, _, _ = make_parametric_lp(m, n)
        t_cpu, _ = timeit(lambda: canonicalize_cpu(prob_cpu), warmup=3, repeat=20, sync_gpu=False)
        print(f"    CPU baseline: {t_cpu*1000:.3f} ms")

        print(f"    DPP speedup vs CPU: {t_cpu/t_dpp:.2f}x")
        if t_trace is not None:
            print(f"    Trace speedup vs CPU: {t_cpu/t_trace:.2f}x")
            print(f"    DPP vs Trace: {t_trace/t_dpp:.2f}x (DPP is this many times faster)")
        else:
            print(f"    Trace replay not available for this problem type")

    # LASSO
    lasso_sizes = [(200, 100), (500, 250)]
    for m, n in lasso_sizes:
        print(f"\n  --- Parametric LASSO {m}x{n} ---")

        reset_id_counter()
        prob_dpp, A_p, b_p = make_parametric_lasso(m, n)
        try:
            compiled = CompiledProgram(prob_dpp)
            compiled.canonicalize()
            t_dpp, _ = timeit(lambda: compiled.canonicalize(), warmup=3, repeat=20)
            print(f"    DPP (CompiledProgram): {t_dpp*1000:.3f} ms")
        except Exception as e:
            print(f"    DPP failed: {e}")
            t_dpp = None

        try:
            cache = TraceCache()
            reset_id_counter()
            prob_t1, _, _ = make_parametric_lasso(m, n)
            cache.canonicalize(prob_t1)
            reset_id_counter()
            prob_t2, _, _ = make_parametric_lasso(m, n)
            t_trace, _ = timeit(lambda: cache.canonicalize(prob_t2), warmup=3, repeat=20)
            print(f"    Trace replay: {t_trace*1000:.3f} ms")
        except Exception as e:
            print(f"    Trace replay failed: {e}")
            t_trace = None

        reset_id_counter()
        prob_cpu, _, _ = make_parametric_lasso(m, n)
        t_cpu, _ = timeit(lambda: canonicalize_cpu(prob_cpu), warmup=3, repeat=20, sync_gpu=False)
        print(f"    CPU baseline: {t_cpu*1000:.3f} ms")

        if t_trace is not None:
            print(f"    Trace speedup vs CPU: {t_cpu/t_trace:.2f}x")
        if t_dpp is not None:
            print(f"    DPP speedup vs CPU: {t_cpu/t_dpp:.2f}x")


# ═══════════════════════════════════════════════════════════════════════════
# GOAL 4: Larger problems
# ═══════════════════════════════════════════════════════════════════════════

def test_larger_problems():
    print("\n" + "=" * 70)
    print("GOAL 4: Scaling to larger problems")
    print("=" * 70)

    configs = [
        ("LP", 100, 50),
        ("LP", 500, 250),
        ("LP", 1000, 500),
        ("LP", 2000, 1000),
        ("LP", 5000, 2500),
    ]

    results = []

    for ptype, m, n in configs:
        print(f"\n  --- {ptype} {m}x{n} ---")

        # CPU baseline
        reset_id_counter()
        prob_cpu = make_lp(m, n)
        t_cpu, _ = timeit(lambda: canonicalize_cpu(prob_cpu), warmup=2, repeat=10, sync_gpu=False)
        print(f"    CPU baseline: {t_cpu*1000:.2f} ms")

        # Trace replay
        cache = TraceCache()
        reset_id_counter()
        p1 = make_lp(m, n)
        cache.canonicalize(p1)  # cold

        # Count trace ops
        fp = _problem_fingerprint(p1)
        cached = cache._cache[fp]
        n_ops = len(cached.trace)
        n_const = sum(1 for op in cached.trace if op.constant_value is not None and not op.input_ids)

        reset_id_counter()
        p2 = make_lp(m, n)
        t_trace, _ = timeit(lambda: cache.canonicalize(p2), warmup=2, repeat=10)
        print(f"    Trace replay: {t_trace*1000:.2f} ms")
        print(f"    Speedup: {t_cpu/t_trace:.2f}x")
        print(f"    Trace ops: {n_ops} ({n_const} constant)")

        results.append((f"{ptype} {m}x{n}", t_cpu*1000, t_trace*1000, t_cpu/t_trace, n_ops, n_const))

    # LASSO
    lasso_configs = [
        ("LASSO", 500, 250),
        ("LASSO", 1000, 500),
        ("LASSO", 1000, 2000),
    ]

    for ptype, m, n in lasso_configs:
        print(f"\n  --- {ptype} {m}x{n} ---")

        reset_id_counter()
        prob_cpu = make_lasso(m, n)
        t_cpu, _ = timeit(lambda: canonicalize_cpu(prob_cpu), warmup=2, repeat=10, sync_gpu=False)
        print(f"    CPU baseline: {t_cpu*1000:.2f} ms")

        cache = TraceCache()
        reset_id_counter()
        p1 = make_lasso(m, n)
        cache.canonicalize(p1)

        fp = _problem_fingerprint(p1)
        cached = cache._cache[fp]
        n_ops = len(cached.trace)
        n_const = sum(1 for op in cached.trace if op.constant_value is not None and not op.input_ids)

        reset_id_counter()
        p2 = make_lasso(m, n)
        t_trace, _ = timeit(lambda: cache.canonicalize(p2), warmup=2, repeat=10)
        print(f"    Trace replay: {t_trace*1000:.2f} ms")
        print(f"    Speedup: {t_cpu/t_trace:.2f}x")
        print(f"    Trace ops: {n_ops} ({n_const} constant)")

        results.append((f"{ptype} {m}x{n}", t_cpu*1000, t_trace*1000, t_cpu/t_trace, n_ops, n_const))

    # Summary table
    print(f"\n  {'Problem':<20} {'CPU (ms)':>10} {'Trace (ms)':>12} {'Speedup':>10} {'Ops':>6} {'Const':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*6} {'-'*6}")
    for name, cpu_ms, trace_ms, speedup, ops, const in results:
        print(f"  {name:<20} {cpu_ms:>10.2f} {trace_ms:>12.2f} {speedup:>10.2f}x {ops:>6} {const:>6}")


# ═══════════════════════════════════════════════════════════════════════════
# GOAL 5: Optimization opportunity analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_optimization_opportunities():
    print("\n" + "=" * 70)
    print("GOAL 5: Optimization opportunity analysis")
    print("=" * 70)

    for label, make_fn, m, n in [
        ("LP 100x50", make_lp, 100, 50),
        ("LP 500x250", make_lp, 500, 250),
        ("LP 2000x1000", make_lp, 2000, 1000),
        ("LASSO 500x250", make_lasso, 500, 250),
    ]:
        print(f"\n  --- {label} ---")

        reset_id_counter()
        prob = make_fn(m, n)

        with NumpyTracer() as tracer:
            data, chain, inv = prob.get_problem_data(cp.CLARABEL, canon_backend="COO")

        A_cpu = data.get("A")
        if A_cpu is not None and sp.issparse(A_cpu):
            A_csc = A_cpu.tocsc()
            tracer.register_output("A_data", A_csc.data)
        if data.get("b") is not None:
            tracer.register_output("b", data["b"])
        if data.get("c") is not None:
            tracer.register_output("c", data["c"])

        trace = tracer.get_trace()
        input_ids = tracer.get_input_ids()
        input_arrays = tracer.get_input_arrays()

        # Analysis 1: Constant ops (produce same result every time)
        const_ops = [op for op in trace if op.constant_value is not None and not op.input_ids]
        const_bytes = sum(op.constant_value.nbytes for op in const_ops if hasattr(op.constant_value, 'nbytes'))
        print(f"    Constant ops: {len(const_ops)}/{len(trace)} ({len(const_ops)/len(trace)*100:.1f}%)")
        print(f"    Constant data: {const_bytes:,} bytes")
        print(f"    -> OPTIMIZATION: Pre-transfer constant results to GPU, skip replay")

        # Analysis 2: Structural ops (int/bool ops that don't depend on data)
        structural_ops = 0
        for op in trace:
            if op.output_dtype in ('int32', 'int64', 'bool'):
                structural_ops += 1
        print(f"    Structural ops (int/bool output): {structural_ops}/{len(trace)} ({structural_ops/len(trace)*100:.1f}%)")
        print(f"    -> OPTIMIZATION: Cache structural op results, skip on replay")

        # Analysis 3: Small array ops (overhead dominates)
        small_ops = 0
        tiny_ops = 0
        for op in trace:
            if op.output_shape is not None:
                size = 1
                for d in op.output_shape:
                    size *= d
                if size < 100:
                    small_ops += 1
                if size < 10:
                    tiny_ops += 1
        print(f"    Small ops (output < 100 elems): {small_ops}/{len(trace)} ({small_ops/len(trace)*100:.1f}%)")
        print(f"    Tiny ops (output < 10 elems): {tiny_ops}/{len(trace)} ({tiny_ops/len(trace)*100:.1f}%)")
        print(f"    -> OPTIMIZATION: Batch small ops or keep on CPU")

        # Analysis 4: Copy/identity ops
        copy_ops = sum(1 for op in trace if op.op in ('copy', 'np.copy'))
        print(f"    Copy ops: {copy_ops}/{len(trace)}")
        print(f"    -> OPTIMIZATION: Eliminate unnecessary copies")

        # Analysis 5: GPU transfer count
        n_transfers = len(input_arrays)
        total_bytes = sum(a.nbytes for a in input_arrays.values())
        print(f"    Input transfers: {n_transfers} arrays, {total_bytes:,} bytes")

        # Analysis 6: Op dispatch overhead estimate
        # Each Python dict lookup + function call ~ 0.5-1us
        est_dispatch_us = len(trace) * 0.8
        print(f"    Estimated dispatch overhead: {est_dispatch_us:.0f} us ({est_dispatch_us/1000:.2f} ms)")

        # Analysis 7: Skippable ops (constant + structural)
        skippable = len(const_ops) + structural_ops
        print(f"    Total skippable: {skippable}/{len(trace)} ({skippable/len(trace)*100:.1f}%)")
        remaining_dispatch_us = (len(trace) - skippable) * 0.8
        print(f"    Dispatch overhead after skip: {remaining_dispatch_us:.0f} us ({remaining_dispatch_us/1000:.2f} ms)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # GPU warmup
    print("Warming up GPU...")
    a = cup.ones(1000)
    b = cup.ones(1000)
    c = a + b
    cup.cuda.Stream.null.synchronize()
    print(f"GPU ready. Device: {cup.cuda.runtime.getDeviceProperties(0)['name'].decode()}\n")

    profile_small_problem_overhead()
    profile_medium_problem()
    compare_trace_vs_dpp()
    test_larger_problems()
    analyze_optimization_opportunities()

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
