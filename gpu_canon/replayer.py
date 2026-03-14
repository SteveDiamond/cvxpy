from __future__ import annotations
"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Replay a recorded numpy/scipy trace on GPU using CuPy.

Takes a trace (list of TraceOp) produced by NumpyTracer and replays every
operation using CuPy equivalents. The trace is a flat list — no control flow,
no branching — so replay is fast.
"""

from typing import Any

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from gpu_canon.tracer import TraceOp


# ── Op dispatch table ─────────────────────────────────────────────────────────

def _replay_np_concatenate(inputs, kwargs, xp):
    axis = kwargs.get("axis", 0)
    return xp.concatenate(inputs, axis=axis)


def _replay_np_stack(inputs, kwargs, xp):
    axis = kwargs.get("axis", 0)
    return xp.stack(inputs, axis=axis)


def _replay_np_vstack(inputs, kwargs, xp):
    return xp.vstack(inputs)


def _replay_np_hstack(inputs, kwargs, xp):
    return xp.hstack(inputs)


def _replay_np_column_stack(inputs, kwargs, xp):
    return xp.column_stack(inputs)


def _replay_np_argsort(inputs, kwargs, xp):
    kw = _strip_internal(kwargs)
    return xp.argsort(inputs[0], **kw)


def _replay_np_sort(inputs, kwargs, xp):
    kw = _strip_internal(kwargs)
    return xp.sort(inputs[0], **kw)


def _replay_np_unique(inputs, kwargs, xp):
    return xp.unique(inputs[0])


def _replay_np_cumsum(inputs, kwargs, xp):
    kw = _strip_internal(kwargs)
    return xp.cumsum(inputs[0], **kw)


def _replay_np_diff(inputs, kwargs, xp):
    kw = _strip_internal(kwargs)
    return xp.diff(inputs[0], **kw)


def _replay_np_ravel(inputs, kwargs, xp):
    return inputs[0].ravel()


def _replay_np_flatten(inputs, kwargs, xp):
    return inputs[0].flatten()


def _replay_np_squeeze(inputs, kwargs, xp):
    return xp.squeeze(inputs[0])


def _replay_np_copy(inputs, kwargs, xp):
    return inputs[0].copy()


def _replay_np_abs(inputs, kwargs, xp):
    return xp.abs(inputs[0])


def _replay_np_negative(inputs, kwargs, xp):
    return -inputs[0]


def _replay_np_reshape(inputs, kwargs, xp):
    if kwargs.get("shape_arg_is_input") and len(inputs) > 1:
        shape = tuple(int(x) for x in inputs[1].ravel())
    else:
        shape = kwargs.get("shape_arg", kwargs.get("newshape"))
    if shape is None:
        shape = kwargs.get("_output_shape")
    if isinstance(shape, list):
        shape = tuple(shape)
    order = kwargs.get("order", "C")
    return xp.reshape(inputs[0], shape, order=order)


def _replay_np_tile(inputs, kwargs, xp):
    if kwargs.get("shape_arg_is_input") and len(inputs) > 1:
        reps = inputs[1]
    else:
        reps = kwargs.get("shape_arg")
    return xp.tile(inputs[0], reps)


def _replay_np_repeat(inputs, kwargs, xp):
    if kwargs.get("shape_arg_is_input") and len(inputs) > 1:
        # repeats was traced as a second input array
        repeats = inputs[1]
        if hasattr(repeats, 'get'):
            repeats = repeats.get()  # CuPy -> numpy for int conversion
        if hasattr(repeats, 'tolist'):
            repeats = repeats.tolist()
    else:
        repeats = kwargs.get("shape_arg")
    # CuPy requires repeats to be int or sequence of ints
    if hasattr(repeats, 'tolist'):
        repeats = repeats.tolist()
    if isinstance(repeats, list) and len(repeats) == 1:
        repeats = repeats[0]
    axis = kwargs.get("axis")
    return xp.repeat(inputs[0], repeats, axis=axis)


def _replay_np_broadcast_to(inputs, kwargs, xp):
    shape = kwargs.get("shape_arg")
    return xp.broadcast_to(inputs[0], shape)


def _replay_np_searchsorted(inputs, kwargs, xp):
    return xp.searchsorted(inputs[0], inputs[1])


def _replay_np_where(inputs, kwargs, xp):
    n_args = kwargs.get("n_args", len(inputs))
    if n_args == 1 or len(inputs) == 1:
        return xp.where(inputs[0])
    elif len(inputs) >= 3:
        return xp.where(inputs[0], inputs[1], inputs[2])
    else:
        return xp.where(inputs[0])


def _replay_np_sum(inputs, kwargs, xp):
    kw = {k: v for k, v in kwargs.items() if k in ("axis", "keepdims")}
    return xp.sum(inputs[0], **kw)


def _replay_np_prod(inputs, kwargs, xp):
    kw = {k: v for k, v in kwargs.items() if k in ("axis", "keepdims")}
    return xp.prod(inputs[0], **kw)


def _replay_np_min(inputs, kwargs, xp):
    kw = {k: v for k, v in kwargs.items() if k in ("axis", "keepdims")}
    return xp.min(inputs[0], **kw) if inputs else xp.array(0)


def _replay_np_max(inputs, kwargs, xp):
    kw = {k: v for k, v in kwargs.items() if k in ("axis", "keepdims")}
    return xp.max(inputs[0], **kw) if inputs else xp.array(0)


def _replay_np_any(inputs, kwargs, xp):
    kw = {k: v for k, v in kwargs.items() if k in ("axis", "keepdims")}
    return xp.any(inputs[0], **kw) if inputs else xp.array(False)


def _replay_np_all(inputs, kwargs, xp):
    kw = {k: v for k, v in kwargs.items() if k in ("axis", "keepdims")}
    return xp.all(inputs[0], **kw) if inputs else xp.array(True)


# Constructors
def _strip_internal(kwargs):
    """Remove internal trace metadata keys before passing to numpy/cupy."""
    return {k: v for k, v in kwargs.items()
            if not k.startswith("_") and k != "extra_args"}


def _is_valid_dtype(d):
    """Check if dtype string is valid for numpy/cupy."""
    if d is None:
        return False
    if isinstance(d, str):
        if d.startswith("[") or d == "" or d == "object":
            return False
    return True


def _get_dtype(kwargs):
    """Get dtype, preferring _output_dtype from the TraceOp metadata."""
    out_dtype = kwargs.get("_output_dtype")
    d = kwargs.get("dtype")
    if _is_valid_dtype(d):
        return d
    if _is_valid_dtype(out_dtype):
        return out_dtype
    return "float64"


def _get_shape(kwargs):
    """Get shape from kwargs, falling back to _output_shape."""
    s = kwargs.get("shape_arg")
    if s is None:
        s = kwargs.get("like_shape") or kwargs.get("_output_shape")
    return s


def _replay_np_zeros(inputs, kwargs, xp):
    return xp.zeros(_get_shape(kwargs), dtype=_get_dtype(kwargs))


def _replay_np_ones(inputs, kwargs, xp):
    return xp.ones(_get_shape(kwargs), dtype=_get_dtype(kwargs))


def _replay_np_empty(inputs, kwargs, xp):
    return xp.empty(_get_shape(kwargs), dtype=_get_dtype(kwargs))


def _replay_np_full(inputs, kwargs, xp):
    return xp.full(_get_shape(kwargs), kwargs.get("fill_value", 0),
                   dtype=_get_dtype(kwargs))


def _replay_np_arange(inputs, kwargs, xp):
    shape_arg = kwargs.get("shape_arg")
    if shape_arg is not None:
        if isinstance(shape_arg, (list, tuple)):
            return xp.arange(*shape_arg)
        return xp.arange(shape_arg)
    return xp.arange(0)


def _replay_np_linspace(inputs, kwargs, xp):
    shape_arg = kwargs.get("shape_arg")
    if isinstance(shape_arg, (list, tuple)) and len(shape_arg) >= 2:
        return xp.linspace(shape_arg[0], shape_arg[1],
                           int(shape_arg[2]) if len(shape_arg) > 2 else 50)
    return xp.linspace(0, 1)


def _replay_np_zeros_like(inputs, kwargs, xp):
    if inputs:
        return xp.zeros_like(inputs[0])
    return xp.zeros(_get_shape(kwargs), dtype=_get_dtype(kwargs))


def _replay_np_ones_like(inputs, kwargs, xp):
    if inputs:
        return xp.ones_like(inputs[0])
    return xp.ones(_get_shape(kwargs), dtype=_get_dtype(kwargs))


def _replay_np_empty_like(inputs, kwargs, xp):
    if inputs:
        return xp.empty_like(inputs[0])
    return xp.empty(_get_shape(kwargs), dtype=_get_dtype(kwargs))


# Sparse ops — these produce numpy arrays on CPU replay, cupy on GPU
def _replay_sparse_constructor(inputs, kwargs, xp):
    """Replay a sparse constructor. Returns the data array."""
    # For sparse ops, we just pass through the data —
    # the sparse structure is recorded in kwargs
    if inputs:
        return inputs[0].copy()
    return xp.array([], dtype="float64")


def _replay_sparse_func(inputs, kwargs, xp):
    """Replay sparse functions like kron, block_diag."""
    if inputs:
        return inputs[0].copy()
    return xp.array([], dtype="float64")


# ── Dispatch table ────────────────────────────────────────────────────────────

_DISPATCH: dict[str, Any] = {
    "np.concatenate": _replay_np_concatenate,
    "np.stack": _replay_np_stack,
    "np.vstack": _replay_np_vstack,
    "np.hstack": _replay_np_hstack,
    "np.column_stack": _replay_np_column_stack,
    "np.argsort": _replay_np_argsort,
    "np.sort": _replay_np_sort,
    "np.unique": _replay_np_unique,
    "np.cumsum": _replay_np_cumsum,
    "np.diff": _replay_np_diff,
    "np.ravel": _replay_np_ravel,
    "np.flatten": _replay_np_flatten,
    "np.squeeze": _replay_np_squeeze,
    "np.copy": _replay_np_copy,
    "np.abs": _replay_np_abs,
    "np.negative": _replay_np_negative,
    "np.reshape": _replay_np_reshape,
    "np.tile": _replay_np_tile,
    "np.repeat": _replay_np_repeat,
    "np.broadcast_to": _replay_np_broadcast_to,
    "np.searchsorted": _replay_np_searchsorted,
    "np.where": _replay_np_where,
    "np.sum": _replay_np_sum,
    "np.prod": _replay_np_prod,
    "np.min": _replay_np_min,
    "np.max": _replay_np_max,
    "np.any": _replay_np_any,
    "np.all": _replay_np_all,
    "np.zeros": _replay_np_zeros,
    "np.ones": _replay_np_ones,
    "np.empty": _replay_np_empty,
    "np.full": _replay_np_full,
    "np.arange": _replay_np_arange,
    "np.linspace": _replay_np_linspace,
    "np.zeros_like": _replay_np_zeros_like,
    "np.ones_like": _replay_np_ones_like,
    "np.empty_like": _replay_np_empty_like,
    "copy": lambda inputs, kwargs, xp: inputs[0].copy(),
}


def replay_on_gpu(
    trace: list[TraceOp],
    initial_arrays: dict[int, Any],
) -> dict[int, Any]:
    """Replay a trace on GPU using CuPy.

    Args:
        trace: List of TraceOp from NumpyTracer.
        initial_arrays: Map of trace_id -> array for inputs.
            Arrays can be numpy (transferred to GPU) or CuPy.

    Returns:
        Registry of all trace_id -> CuPy array after replay.
        The final arrays (A, b, c data) are the last entries.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")

    xp = cp
    registry: dict[int, Any] = {}

    # Transfer initial arrays to GPU
    for tid, arr in initial_arrays.items():
        if isinstance(arr, np.ndarray):
            registry[tid] = cp.asarray(arr)
        else:
            registry[tid] = arr

    for op in trace:
        # Gather inputs
        inputs = []
        for iid in op.input_ids:
            if iid in registry:
                inputs.append(registry[iid])
            else:
                raise KeyError(
                    f"Trace replay: input {iid} not found for op {op.op}")

        # Inject authoritative dtype/shape from trace metadata
        kwargs = dict(op.kwargs)
        kwargs["_output_dtype"] = op.output_dtype
        kwargs["_output_shape"] = op.output_shape

        # Constructors with constant_value: use the baked-in value.
        # This captures the final array state including in-place mutations
        # (e.g., np.empty followed by a[:] = data).
        if op.constant_value is not None and not op.input_ids:
            cv = op.constant_value
            # CuPy doesn't support structured dtypes — convert to float64
            if hasattr(cv, 'dtype') and not _is_valid_dtype(str(cv.dtype)):
                cv = np.zeros(cv.shape, dtype=np.float64)
            result = cp.asarray(cv)
        else:
            # Dispatch
            handler = _DISPATCH.get(op.op)
            if handler is not None:
                result = handler(inputs, kwargs, xp)
            elif op.op.startswith("sp.") and op.op.split(".")[1] in (
                    "coo_matrix", "csr_matrix", "csc_matrix",
                    "coo_array", "csr_array", "csc_array"):
                result = _replay_sparse_constructor(inputs, kwargs, xp)
            elif op.op.startswith("sp."):
                result = _replay_sparse_func(inputs, kwargs, xp)
            else:
                raise ValueError(f"Unknown trace op: {op.op}")

        registry[op.output_id] = result

    return registry


def classify_trace_ops(
    trace: list[TraceOp],
    data_input_ids: set[int],
) -> tuple[set[int], list[int]]:
    """Classify trace ops as constant or data-dependent.

    An op is "constant" if:
    - It has constant_value and no inputs (constructor), OR
    - All its inputs are constant ops or structural inputs

    An op is "data-dependent" if any input (transitively) comes from
    a data input.

    Args:
        trace: The trace ops.
        data_input_ids: Set of trace IDs that are data inputs (change between calls).

    Returns:
        (constant_op_ids, data_op_indices):
        - constant_op_ids: set of output trace IDs that are constant
        - data_op_indices: list of indices into trace for data-dependent ops
    """
    # Start: all data input IDs are "tainted" (data-dependent)
    tainted = set(data_input_ids)
    constant_ids: set[int] = set()
    data_op_indices: list[int] = []

    for i, op in enumerate(trace):
        # Check if any input is tainted
        has_tainted_input = any(iid in tainted for iid in op.input_ids)

        if has_tainted_input:
            # This op depends on data — mark its output as tainted
            tainted.add(op.output_id)
            data_op_indices.append(i)
        else:
            # This op is constant — its result is the same every replay
            constant_ids.add(op.output_id)

    return constant_ids, data_op_indices


def replay_on_gpu_optimized(
    trace: list[TraceOp],
    initial_arrays: dict[int, Any],
    cached_constants: dict[int, Any] | None = None,
    data_op_indices: list[int] | None = None,
) -> dict[int, Any]:
    """Optimized GPU replay that skips constant ops.

    If cached_constants and data_op_indices are provided, only executes
    data-dependent ops. Constant results are seeded from the cache.

    Args:
        trace: Full trace.
        initial_arrays: Data inputs (transferred to GPU).
        cached_constants: Pre-computed constant op results on GPU.
        data_op_indices: Indices of data-dependent ops to execute.

    Returns:
        Registry of trace_id -> CuPy array.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")

    xp = cp
    registry: dict[int, Any] = {}

    # Seed with initial arrays
    for tid, arr in initial_arrays.items():
        if isinstance(arr, np.ndarray):
            registry[tid] = cp.asarray(arr)
        else:
            registry[tid] = arr

    # Seed with cached constants (no GPU transfer — already on GPU)
    if cached_constants is not None:
        registry.update(cached_constants)

    # If we have data_op_indices, only execute those ops
    if data_op_indices is not None:
        ops_to_run = [(i, trace[i]) for i in data_op_indices]
    else:
        ops_to_run = list(enumerate(trace))

    for _idx, op in ops_to_run:
        inputs = []
        for iid in op.input_ids:
            if iid in registry:
                inputs.append(registry[iid])
            else:
                raise KeyError(
                    f"Trace replay: input {iid} not found for op {op.op}")

        kwargs = dict(op.kwargs)
        kwargs["_output_dtype"] = op.output_dtype
        kwargs["_output_shape"] = op.output_shape

        if op.constant_value is not None and not op.input_ids:
            cv = op.constant_value
            if hasattr(cv, 'dtype') and not _is_valid_dtype(str(cv.dtype)):
                cv = np.zeros(cv.shape, dtype=np.float64)
            result = cp.asarray(cv)
        else:
            handler = _DISPATCH.get(op.op)
            if handler is not None:
                result = handler(inputs, kwargs, xp)
            elif op.op.startswith("sp.") and op.op.split(".")[1] in (
                    "coo_matrix", "csr_matrix", "csc_matrix",
                    "coo_array", "csr_array", "csc_array"):
                result = _replay_sparse_constructor(inputs, kwargs, xp)
            elif op.op.startswith("sp."):
                result = _replay_sparse_func(inputs, kwargs, xp)
            else:
                raise ValueError(f"Unknown trace op: {op.op}")

        registry[op.output_id] = result

    return registry


def replay_on_cpu(
    trace: list[TraceOp],
    initial_arrays: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Replay a trace on CPU using numpy (for verification).

    Same interface as replay_on_gpu but uses numpy throughout.
    """
    xp = np
    registry: dict[int, np.ndarray] = {}

    for tid, arr in initial_arrays.items():
        if not isinstance(arr, np.ndarray):
            registry[tid] = np.asarray(arr)
        else:
            registry[tid] = arr

    for op in trace:
        inputs = []
        for iid in op.input_ids:
            if iid in registry:
                inputs.append(registry[iid])
            else:
                raise KeyError(
                    f"Trace replay: input {iid} not found for op {op.op}")

        # Inject authoritative dtype/shape from trace metadata
        kwargs = dict(op.kwargs)
        kwargs["_output_dtype"] = op.output_dtype
        kwargs["_output_shape"] = op.output_shape

        if op.constant_value is not None and not op.input_ids:
            result = np.asarray(op.constant_value).copy()
        else:
            handler = _DISPATCH.get(op.op)
            if handler is not None:
                result = handler(inputs, kwargs, xp)
            elif op.op.startswith("sp."):
                result = inputs[0].copy() if inputs else np.array([])
            else:
                raise ValueError(f"Unknown trace op: {op.op}")

        registry[op.output_id] = result

    return registry
