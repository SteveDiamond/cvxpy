"""
Deep dive into the replay overhead: what specifically makes it slow on small problems.
"""

import time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

import cupy as cup

from gpu_canon.trace_cache import TraceCache, _problem_fingerprint, reset_id_counter
from gpu_canon.tracer import NumpyTracer
from gpu_canon.replayer import replay_on_gpu, _DISPATCH, _is_valid_dtype
from gpu_canon.baseline import canonicalize_cpu


def make_lp(m, n):
    np.random.seed(42)
    A_data = np.random.randn(m, n)
    b_data = np.abs(np.random.randn(m)) * 10
    c_data = np.random.randn(n)
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(c_data @ x), [A_data @ x <= b_data, x >= 0])


# ── Measure per-op dispatch overhead (no actual GPU work) ────────────────

print("=" * 70)
print("DEEP DIVE: Op-by-op overhead analysis for LP 500x250")
print("=" * 70)

reset_id_counter()
prob = make_lp(500, 250)

with NumpyTracer() as tracer:
    data, chain, inv = prob.get_problem_data(cp.CLARABEL, canon_backend="COO")

A_csc = data["A"].tocsc()
tracer.register_output("A_data", A_csc.data)
tracer.register_output("b", data["b"])
tracer.register_output("c", data["c"])

trace = tracer.get_trace()
input_ids = tracer.get_input_ids()
input_arrays = tracer.get_input_arrays()

# Classify ops
const_ops = []
structural_ops = []
data_ops = []

for i, op in enumerate(trace):
    is_const = op.constant_value is not None and not op.input_ids
    is_structural = op.output_dtype in ('int32', 'int64', 'bool')
    if is_const and is_structural:
        const_ops.append(i)
    elif is_const:
        const_ops.append(i)
    elif is_structural:
        structural_ops.append(i)
    else:
        data_ops.append(i)

print(f"\nTrace: {len(trace)} ops")
print(f"  Constant ops: {len(const_ops)} (produce fixed results, could be pre-cached)")
print(f"  Structural ops (non-const, int/bool): {len(structural_ops)} (structure-dependent)")
print(f"  Data ops: {len(data_ops)} (depend on parameter data)")

# Measure: Python dispatch overhead (empty loop with dict lookup)
print(f"\n--- Python dispatch overhead estimate ---")
n_iters = 10000
t0 = time.perf_counter()
for _ in range(n_iters):
    for op in trace:
        h = _DISPATCH.get(op.op)
t1 = time.perf_counter()
dispatch_per_call = (t1 - t0) / n_iters
print(f"  Dict lookup for {len(trace)} ops: {dispatch_per_call*1000:.3f} ms per full trace")
print(f"  Per-op: {dispatch_per_call/len(trace)*1e6:.2f} us")

# Measure: cp.asarray overhead for small arrays
sizes_to_test = [1, 10, 100, 1000, 10000, 100000]
print(f"\n--- cup.asarray overhead by size ---")
for sz in sizes_to_test:
    arr = np.random.randn(sz)
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        cup.asarray(arr)
    cup.cuda.Stream.null.synchronize()
    t_per = (time.perf_counter() - t0) / 200
    print(f"  {sz:>7} elements: {t_per*1e6:.1f} us per call")

# Measure: CuPy kernel launch overhead
print(f"\n--- CuPy op overhead (kernel launch) by array size ---")
for sz in sizes_to_test:
    a_gpu = cup.ones(sz)
    b_gpu = cup.ones(sz)
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(500):
        c = a_gpu + b_gpu
    cup.cuda.Stream.null.synchronize()
    t_per = (time.perf_counter() - t0) / 500
    print(f"  add {sz:>7} elements: {t_per*1e6:.1f} us per call")

for sz in sizes_to_test:
    a_gpu = cup.ones(sz)
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(500):
        c = cup.concatenate([a_gpu, a_gpu])
    cup.cuda.Stream.null.synchronize()
    t_per = (time.perf_counter() - t0) / 500
    print(f"  concat 2x{sz:>7}: {t_per*1e6:.1f} us per call")

# Measure: How much time is the constant op cp.asarray(constant_value) costing
print(f"\n--- Constant op transfer breakdown ---")
const_values = [trace[i].constant_value for i in const_ops]
const_nbytes = [cv.nbytes if hasattr(cv, 'nbytes') else 0 for cv in const_values]
const_sizes = [cv.size if hasattr(cv, 'size') else 0 for cv in const_values]

print(f"  Total constant ops: {len(const_ops)}")
print(f"  Total constant bytes: {sum(const_nbytes):,}")
print(f"  Mean size: {np.mean(const_sizes):.0f} elements")
print(f"  Median size: {np.median(const_sizes):.0f} elements")
print(f"  Max size: {max(const_sizes)} elements")

# Time transferring all constants at once vs one-by-one
valid_const = []
for cv in const_values:
    if cv is not None and hasattr(cv, 'dtype') and _is_valid_dtype(str(cv.dtype)):
        valid_const.append(cv)
    elif cv is not None and hasattr(cv, 'shape'):
        valid_const.append(np.zeros(cv.shape, dtype=np.float64))

cup.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    for cv in valid_const:
        cup.asarray(cv)
cup.cuda.Stream.null.synchronize()
t_one_by_one = (time.perf_counter() - t0) / 100
print(f"\n  Transfer constants one-by-one: {t_one_by_one*1000:.3f} ms")

# Batch: concatenate all constants, transfer once, split
flat_consts = np.concatenate([cv.ravel() for cv in valid_const])
cup.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    cup.asarray(flat_consts)
cup.cuda.Stream.null.synchronize()
t_batched = (time.perf_counter() - t0) / 100
print(f"  Transfer constants batched: {t_batched*1000:.3f} ms")
print(f"  Savings: {(t_one_by_one - t_batched)*1000:.3f} ms ({(1 - t_batched/t_one_by_one)*100:.0f}%)")

# ── Simulated optimized replay ──────────────────────────────────────────

print(f"\n{'='*70}")
print("SIMULATED OPTIMIZED REPLAY")
print(f"{'='*70}")

# Build initial arrays on GPU
from gpu_canon.trace_cache import _classify_inputs
data_inputs, structural_inputs = _classify_inputs(
    trace, input_ids, input_arrays, prob)

structural_gpu = {tid: cup.asarray(arr) for tid, arr in structural_inputs.items()}

# Pre-cache constant results on GPU
const_results_gpu = {}
for i, op in enumerate(trace):
    if op.constant_value is not None and not op.input_ids:
        cv = op.constant_value
        if hasattr(cv, 'dtype') and not _is_valid_dtype(str(cv.dtype)):
            cv = np.zeros(cv.shape, dtype=np.float64)
        const_results_gpu[op.output_id] = cup.asarray(cv)

# Pre-cache structural op results on GPU (run once)
initial_arrays = dict(structural_gpu)
for tid, arr in data_inputs.items():
    initial_arrays[tid] = cup.asarray(arr)

struct_results_gpu = {}
registry_full = replay_on_gpu(trace, initial_arrays)
for i in structural_ops:
    op = trace[i]
    struct_results_gpu[op.output_id] = registry_full[op.output_id]

print(f"\n  Pre-cached: {len(const_results_gpu)} constant results, {len(struct_results_gpu)} structural results")
print(f"  Data ops to replay: {len(data_ops)}")

# Simulated optimized replay: skip const + structural ops
def optimized_replay():
    registry = {}
    # Pre-cached
    registry.update(const_results_gpu)
    registry.update(struct_results_gpu)
    # Structural inputs already on GPU
    registry.update(structural_gpu)
    # Fresh data inputs
    for tid, arr in data_inputs.items():
        registry[tid] = cup.asarray(arr)
    # Only replay data ops
    for i in data_ops:
        op = trace[i]
        inputs = [registry[iid] for iid in op.input_ids if iid in registry]
        kwargs = dict(op.kwargs)
        kwargs["_output_dtype"] = op.output_dtype
        kwargs["_output_shape"] = op.output_shape
        handler = _DISPATCH.get(op.op)
        if handler is not None:
            result = handler(inputs, kwargs, cup)
        elif op.op.startswith("sp."):
            result = inputs[0].copy() if inputs else cup.array([], dtype="float64")
        else:
            raise ValueError(f"Unknown op: {op.op}")
        registry[op.output_id] = result
    return registry

# Warm up
optimized_replay()
cup.cuda.Stream.null.synchronize()

times_opt = []
for _ in range(50):
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    optimized_replay()
    cup.cuda.Stream.null.synchronize()
    times_opt.append(time.perf_counter() - t0)
t_opt = np.median(times_opt)

# Current replay
def current_replay():
    ia = dict(structural_gpu)
    for tid, arr in data_inputs.items():
        ia[tid] = cup.asarray(arr)
    return replay_on_gpu(trace, ia)

current_replay()
cup.cuda.Stream.null.synchronize()
times_cur = []
for _ in range(50):
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    current_replay()
    cup.cuda.Stream.null.synchronize()
    times_cur.append(time.perf_counter() - t0)
t_cur = np.median(times_cur)

# CPU baseline
reset_id_counter()
prob_cpu = make_lp(500, 250)
times_cpu = []
for _ in range(50):
    t0 = time.perf_counter()
    canonicalize_cpu(prob_cpu)
    times_cpu.append(time.perf_counter() - t0)
t_cpu = np.median(times_cpu)

print(f"\n  Current replay: {t_cur*1000:.3f} ms")
print(f"  Optimized replay (skip const+struct): {t_opt*1000:.3f} ms")
print(f"  CPU baseline: {t_cpu*1000:.3f} ms")
print(f"  Current speedup vs CPU: {t_cpu/t_cur:.2f}x")
print(f"  Optimized speedup vs CPU: {t_cpu/t_opt:.2f}x")
print(f"  Optimization gain: {t_cur/t_opt:.2f}x faster")

# ── Larger problem test ──────────────────────────────────────────────────

print(f"\n{'='*70}")
print("OPTIMIZED REPLAY vs CURRENT: LP 2000x1000")
print(f"{'='*70}")

reset_id_counter()
prob_big = make_lp(2000, 1000)

with NumpyTracer() as tracer:
    data, chain, inv = prob_big.get_problem_data(cp.CLARABEL, canon_backend="COO")

A_csc = data["A"].tocsc()
tracer.register_output("A_data", A_csc.data)
tracer.register_output("b", data["b"])
tracer.register_output("c", data["c"])

trace_big = tracer.get_trace()
input_ids_big = tracer.get_input_ids()
input_arrays_big = tracer.get_input_arrays()

data_inputs_big, structural_inputs_big = _classify_inputs(
    trace_big, input_ids_big, input_arrays_big, prob_big)

structural_gpu_big = {tid: cup.asarray(arr) for tid, arr in structural_inputs_big.items()}

# Classify ops
const_ops_big = []
structural_ops_big = []
data_ops_big = []
for i, op in enumerate(trace_big):
    is_const = op.constant_value is not None and not op.input_ids
    is_structural = op.output_dtype in ('int32', 'int64', 'bool')
    if is_const:
        const_ops_big.append(i)
    elif is_structural:
        structural_ops_big.append(i)
    else:
        data_ops_big.append(i)

# Pre-cache
const_results_big = {}
for i in const_ops_big:
    op = trace_big[i]
    cv = op.constant_value
    if hasattr(cv, 'dtype') and not _is_valid_dtype(str(cv.dtype)):
        cv = np.zeros(cv.shape, dtype=np.float64)
    const_results_big[op.output_id] = cup.asarray(cv)

initial_big = dict(structural_gpu_big)
for tid, arr in data_inputs_big.items():
    initial_big[tid] = cup.asarray(arr)
reg_full_big = replay_on_gpu(trace_big, initial_big)
struct_results_big = {}
for i in structural_ops_big:
    op = trace_big[i]
    struct_results_big[op.output_id] = reg_full_big[op.output_id]

print(f"  Trace: {len(trace_big)} ops ({len(const_ops_big)} const, {len(structural_ops_big)} struct, {len(data_ops_big)} data)")

def opt_replay_big():
    registry = {}
    registry.update(const_results_big)
    registry.update(struct_results_big)
    registry.update(structural_gpu_big)
    for tid, arr in data_inputs_big.items():
        registry[tid] = cup.asarray(arr)
    for i in data_ops_big:
        op = trace_big[i]
        inputs = [registry[iid] for iid in op.input_ids if iid in registry]
        kwargs = dict(op.kwargs)
        kwargs["_output_dtype"] = op.output_dtype
        kwargs["_output_shape"] = op.output_shape
        handler = _DISPATCH.get(op.op)
        if handler is not None:
            result = handler(inputs, kwargs, cup)
        elif op.op.startswith("sp."):
            result = inputs[0].copy() if inputs else cup.array([], dtype="float64")
        else:
            raise ValueError(f"Unknown op: {op.op}")
        registry[op.output_id] = result
    return registry

def cur_replay_big():
    ia = dict(structural_gpu_big)
    for tid, arr in data_inputs_big.items():
        ia[tid] = cup.asarray(arr)
    return replay_on_gpu(trace_big, ia)

# Warm up
opt_replay_big()
cur_replay_big()
cup.cuda.Stream.null.synchronize()

# Benchmark
times = []
for _ in range(30):
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    cur_replay_big()
    cup.cuda.Stream.null.synchronize()
    times.append(time.perf_counter() - t0)
t_cur_big = np.median(times)

times = []
for _ in range(30):
    cup.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    opt_replay_big()
    cup.cuda.Stream.null.synchronize()
    times.append(time.perf_counter() - t0)
t_opt_big = np.median(times)

reset_id_counter()
prob_cpu_big = make_lp(2000, 1000)
times = []
for _ in range(15):
    t0 = time.perf_counter()
    canonicalize_cpu(prob_cpu_big)
    times.append(time.perf_counter() - t0)
t_cpu_big = np.median(times)

print(f"  Current replay: {t_cur_big*1000:.3f} ms")
print(f"  Optimized replay: {t_opt_big*1000:.3f} ms")
print(f"  CPU baseline: {t_cpu_big*1000:.3f} ms")
print(f"  Current speedup: {t_cpu_big/t_cur_big:.2f}x")
print(f"  Optimized speedup: {t_cpu_big/t_opt_big:.2f}x")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
