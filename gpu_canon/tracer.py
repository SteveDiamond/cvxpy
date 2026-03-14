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
Trace-compile: record numpy/scipy operations during CVXPY canonicalization.

Monkey-patches numpy and scipy functions to record a linear trace of array
operations. The trace can then be replayed on GPU via CuPy (see replayer.py).

Key insight: CVXPY's canonicalization is 95% numpy/scipy array ops. Parameter
values affect the DATA flowing through ops, not WHICH ops are called. So we
trace once per problem structure and replay with different data forever.
"""

import contextlib
import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp


# ── Trace instruction ────────────────────────────────────────────────────────

@dataclass
class TraceOp:
    """A single recorded operation in the trace."""
    op: str                      # e.g. "np.concatenate", "add", "sp.coo_matrix"
    input_ids: list[int]         # trace IDs of input arrays
    output_id: int               # trace ID of output array
    output_shape: tuple          # shape of output
    output_dtype: str            # dtype name
    kwargs: dict = field(default_factory=dict)  # extra args (axis, etc.)
    constant_value: Any = None   # for structural constants baked into trace


# ── Array identity tracker ────────────────────────────────────────────────────

class ArrayRegistry:
    """Track numpy arrays by identity, assigning stable trace IDs."""

    def __init__(self):
        self._id_to_trace_id: dict[int, int] = {}
        self._next_id = 0
        # Keep refs to prevent GC reuse of id()
        self._refs: list[Any] = []

    def register(self, arr: np.ndarray, is_input: bool = False) -> int:
        """Register an array and return its trace ID."""
        obj_id = id(arr)
        if obj_id in self._id_to_trace_id:
            return self._id_to_trace_id[obj_id]
        tid = self._next_id
        self._next_id += 1
        self._id_to_trace_id[obj_id] = tid
        self._refs.append(arr)
        return tid

    def lookup(self, arr: np.ndarray) -> int | None:
        """Look up trace ID for an array. Returns None if unknown."""
        return self._id_to_trace_id.get(id(arr))

    def get_or_register(self, arr: np.ndarray) -> int:
        """Get existing trace ID or register a new one."""
        tid = self.lookup(arr)
        if tid is not None:
            return tid
        return self.register(arr)


# ── Numpy function wrappers ──────────────────────────────────────────────────

# Functions that take array inputs and produce array outputs.
# Grouped by calling convention.

# (arrays..., **kwargs) -> array
_NP_ARRAY_FUNCS = [
    "concatenate", "stack", "vstack", "hstack", "column_stack",
]

# (array, **kwargs) -> array
_NP_UNARY_FUNCS = [
    "argsort", "sort", "unique", "cumsum", "diff", "ravel", "flatten",
    "squeeze", "copy", "abs", "negative",
]

# (array, shape_or_int, **kwargs) -> array
_NP_SHAPE_FUNCS = [
    "reshape", "tile", "repeat", "broadcast_to",
]

# (**kwargs) -> array  (constructors)
_NP_CONSTRUCTOR_FUNCS = [
    "zeros", "ones", "empty", "full", "arange", "linspace",
    "zeros_like", "ones_like", "empty_like",
]

# (array, array, **kwargs) -> array
_NP_BINARY_FUNCS = [
    "searchsorted", "where",
]

# (array, ...) -> scalar-or-array
_NP_REDUCTION_FUNCS = [
    "sum", "prod", "min", "max", "any", "all",
]

# Ufunc names we intercept
_UFUNC_NAMES = [
    "add", "subtract", "multiply", "true_divide", "floor_divide",
    "remainder", "power", "negative", "absolute",
    "maximum", "minimum",
    "greater", "greater_equal", "less", "less_equal",
    "equal", "not_equal",
    "logical_and", "logical_or", "logical_not",
]


def _is_array_like(obj):
    """Check if obj is a numpy array or sparse matrix."""
    return isinstance(obj, (np.ndarray, np.generic))


def _to_array(obj):
    """Coerce scalars to 0-d arrays for tracing."""
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (int, float, complex, np.generic)):
        return np.asarray(obj)
    return None


class NumpyTracer:
    """Context manager that monkey-patches numpy/scipy to record a trace.

    Usage:
        with NumpyTracer() as tracer:
            data, chain, inv = problem.get_problem_data(cp.CLARABEL,
                                                         canon_backend="COO")
        trace = tracer.get_trace()
    """

    def __init__(self):
        self._ops: list[TraceOp] = []
        self._registry = ArrayRegistry()
        self._originals: dict[str, Any] = {}
        self._active = False
        # Track which trace IDs are "initial inputs" (existed before tracing)
        self._input_ids: set[int] = set()
        # Map trace_id -> original array (for inputs/constants)
        self._input_arrays: dict[int, np.ndarray] = {}

    def __enter__(self):
        self._install_patches()
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        self._uninstall_patches()
        return False

    def get_trace(self) -> list[TraceOp]:
        """Return the recorded trace."""
        return list(self._ops)

    def get_input_ids(self) -> set[int]:
        """Return trace IDs that are external inputs (not produced by ops)."""
        return set(self._input_ids)

    def get_input_arrays(self) -> dict[int, np.ndarray]:
        """Return map of trace_id -> numpy array for all inputs."""
        return dict(self._input_arrays)

    # ── Patching infrastructure ───────────────────────────────────────────

    def _install_patches(self):
        """Monkey-patch numpy and scipy functions."""
        # Patch numpy array functions
        for name in _NP_ARRAY_FUNCS:
            self._patch_np_func(name, self._wrap_array_func)

        for name in _NP_UNARY_FUNCS:
            if hasattr(np, name):
                self._patch_np_func(name, self._wrap_unary_func)

        for name in _NP_SHAPE_FUNCS:
            if hasattr(np, name):
                self._patch_np_func(name, self._wrap_shape_func)

        for name in _NP_CONSTRUCTOR_FUNCS:
            if hasattr(np, name):
                self._patch_np_func(name, self._wrap_constructor)

        for name in _NP_BINARY_FUNCS:
            if hasattr(np, name):
                self._patch_np_func(name, self._wrap_binary_func)

        for name in _NP_REDUCTION_FUNCS:
            if hasattr(np, name):
                self._patch_np_func(name, self._wrap_reduction_func)

        # Patch scipy sparse constructors
        self._patch_scipy()

    def _uninstall_patches(self):
        """Restore original numpy/scipy functions."""
        for key, original in self._originals.items():
            parts = key.split(".")
            if parts[0] == "np":
                setattr(np, parts[1], original)
            elif parts[0] == "sp":
                if len(parts) == 2:
                    setattr(sp, parts[1], original)

    def _patch_np_func(self, name: str, wrapper_factory):
        """Patch a numpy function."""
        original = getattr(np, name)
        self._originals[f"np.{name}"] = original
        wrapped = wrapper_factory(name, original)
        setattr(np, name, wrapped)

    def _patch_scipy(self):
        """Patch scipy sparse constructors and operations."""
        for cls_name in ["coo_matrix", "csr_matrix", "csc_matrix",
                         "coo_array", "csr_array", "csc_array"]:
            if hasattr(sp, cls_name):
                original = getattr(sp, cls_name)
                self._originals[f"sp.{cls_name}"] = original
                setattr(sp, cls_name, self._wrap_sparse_constructor(
                    cls_name, original))

        # Patch sp.kron, sp.block_diag, etc.
        for name in ["kron", "block_diag", "bmat"]:
            if hasattr(sp, name):
                original = getattr(sp, name)
                self._originals[f"sp.{name}"] = original
                setattr(sp, name, self._wrap_sparse_func(name, original))

    # ── Wrapper factories ─────────────────────────────────────────────────

    def _register_input(self, arr):
        """Register an array that existed before the traced op created it."""
        if not _is_array_like(arr):
            return
        tid = self._registry.lookup(arr)
        if tid is None:
            tid = self._registry.register(arr, is_input=True)
            self._input_ids.add(tid)
            self._input_arrays[tid] = arr

    def _register_inputs_recursive(self, obj):
        """Register all arrays found in a nested structure."""
        if _is_array_like(obj):
            self._register_input(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._register_inputs_recursive(item)

    def _record_op(self, op_name: str, inputs, output, **kwargs):
        """Record a single operation."""
        if not self._active:
            return

        # Register any previously-unseen input arrays
        if isinstance(inputs, (list, tuple)):
            input_ids = []
            for inp in inputs:
                if _is_array_like(inp):
                    self._register_input(inp)
                    input_ids.append(self._registry.get_or_register(inp))
            input_ids = input_ids
        elif _is_array_like(inputs):
            self._register_input(inputs)
            input_ids = [self._registry.get_or_register(inputs)]
        else:
            input_ids = []

        if _is_array_like(output):
            out_id = self._registry.register(output)
            out_shape = output.shape
            out_dtype = str(output.dtype)
        else:
            # Scalar result — wrap it
            arr = np.asarray(output)
            out_id = self._registry.register(arr)
            out_shape = arr.shape
            out_dtype = str(arr.dtype)

        self._ops.append(TraceOp(
            op=op_name,
            input_ids=input_ids,
            output_id=out_id,
            output_shape=out_shape,
            output_dtype=out_dtype,
            kwargs=kwargs,
        ))

    def _wrap_array_func(self, name: str, original):
        """Wrap functions like np.concatenate(arrays, axis=0)."""
        tracer = self

        @functools.wraps(original)
        def wrapper(arrays, *args, **kwargs):
            result = original(arrays, *args, **kwargs)
            if tracer._active:
                # Register all input arrays
                input_list = list(arrays) if not isinstance(
                    arrays, list) else arrays
                kw = {}
                if args:
                    kw["axis"] = args[0]
                elif "axis" in kwargs:
                    kw["axis"] = kwargs["axis"]
                tracer._record_op(f"np.{name}", input_list, result, **kw)
            return result
        return wrapper

    def _wrap_unary_func(self, name: str, original):
        """Wrap functions like np.argsort(a, axis=-1)."""
        tracer = self

        @functools.wraps(original)
        def wrapper(a, *args, **kwargs):
            result = original(a, *args, **kwargs)
            if tracer._active and _is_array_like(a):
                safe_kwargs = _safe_kwargs(kwargs, args, name)
                tracer._record_op(f"np.{name}", a, result, **safe_kwargs)
            return result
        return wrapper

    def _wrap_shape_func(self, name: str, original):
        """Wrap functions like np.reshape(a, newshape)."""
        tracer = self

        @functools.wraps(original)
        def wrapper(a, *args, **kwargs):
            result = original(a, *args, **kwargs)
            if tracer._active and _is_array_like(a):
                kw = dict(kwargs)
                if args:
                    kw["shape_arg"] = args[0]
                tracer._record_op(f"np.{name}", a, result, **kw)
            return result
        return wrapper

    def _wrap_constructor(self, name: str, original):
        """Wrap constructors like np.zeros(shape)."""
        tracer = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if tracer._active and _is_array_like(result):
                kw = dict(kwargs)
                if args:
                    kw["shape_arg"] = args[0]
                op = TraceOp(
                    op=f"np.{name}",
                    input_ids=[],
                    output_id=tracer._registry.register(result),
                    output_shape=result.shape,
                    output_dtype=str(result.dtype),
                    kwargs=kw,
                    constant_value=result.copy()
                    if result.size <= 10000 else None,
                )
                tracer._ops.append(op)
            return result
        return wrapper

    def _wrap_binary_func(self, name: str, original):
        """Wrap functions like np.where(cond, x, y)."""
        tracer = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if tracer._active:
                inputs = [a for a in args if _is_array_like(a)]
                tracer._record_op(f"np.{name}", inputs, result,
                                  n_args=len(args))
            return result
        return wrapper

    def _wrap_reduction_func(self, name: str, original):
        """Wrap reductions like np.sum(a, axis=0)."""
        tracer = self

        @functools.wraps(original)
        def wrapper(a=None, *args, **kwargs):
            result = original(a, *args, **kwargs)
            if tracer._active and a is not None and _is_array_like(a):
                safe_kw = _safe_kwargs(kwargs, args, name)
                tracer._record_op(f"np.{name}", a, result, **safe_kw)
            return result
        return wrapper

    def _wrap_sparse_constructor(self, cls_name: str, original_cls):
        """Wrap scipy sparse matrix constructors."""
        tracer = self

        class TracedSparse(original_cls):
            def __new__(cls, *args, **kwargs):
                # Use the original constructor
                obj = original_cls(*args, **kwargs)
                if tracer._active:
                    # Extract data arrays from the constructor args
                    inputs = []
                    if args:
                        arg0 = args[0]
                        if isinstance(arg0, tuple) and len(arg0) >= 2:
                            # (data, (row, col)) format
                            data_arr = arg0[0]
                            if _is_array_like(data_arr):
                                inputs.append(data_arr)
                            if isinstance(arg0[1], tuple):
                                for idx_arr in arg0[1]:
                                    if _is_array_like(idx_arr):
                                        inputs.append(idx_arr)
                        elif _is_array_like(arg0):
                            inputs.append(arg0)
                        elif sp.issparse(arg0):
                            # Conversion from another sparse format
                            if hasattr(arg0, 'data') and _is_array_like(
                                    arg0.data):
                                inputs.append(arg0.data)

                    kw = {}
                    if "shape" in kwargs:
                        kw["shape"] = kwargs["shape"]
                    elif len(args) > 1:
                        kw["shape"] = args[1] if isinstance(
                            args[1], tuple) else None

                    # Record the data array of the output
                    if hasattr(obj, 'data') and _is_array_like(obj.data):
                        tracer._record_op(
                            f"sp.{cls_name}",
                            inputs, obj.data,
                            sparse_shape=obj.shape,
                            sparse_format=cls_name,
                            **kw,
                        )

                return obj

        TracedSparse.__name__ = cls_name
        TracedSparse.__qualname__ = cls_name
        return TracedSparse

    def _wrap_sparse_func(self, name: str, original):
        """Wrap sparse functions like sp.kron."""
        tracer = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if tracer._active and sp.issparse(result):
                inputs = []
                for a in args:
                    if sp.issparse(a) and hasattr(a, 'data'):
                        inputs.append(a.data)
                    elif _is_array_like(a):
                        inputs.append(a)
                if hasattr(result, 'data'):
                    tracer._record_op(
                        f"sp.{name}", inputs, result.data,
                        sparse_shape=result.shape,
                        sparse_format=result.format
                        if hasattr(result, 'format') else 'unknown',
                    )
            return result
        return wrapper


def _safe_kwargs(kwargs, args, name):
    """Extract serializable kwargs for recording."""
    safe = {}
    for k, v in kwargs.items():
        if isinstance(v, (int, float, str, bool, tuple, list, type(None))):
            safe[k] = v
    if args:
        safe["extra_args"] = [
            a if isinstance(a, (int, float, str, bool, type(None))) else None
            for a in args
        ]
    return safe
