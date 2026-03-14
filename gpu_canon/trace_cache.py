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
Cache traces by problem structure for trace-compile canonicalization.

On cache miss: run CVXPY with numpy tracing, record the trace, cache it.
On cache hit: replay the cached trace on GPU with new parameter/constant data.

This is like CVXPY's DPP but more general — works for ANY problem structure,
not just parameter changes.
"""

import hashlib
from typing import Any

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cup
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import cvxpy
from cvxpy.lin_ops.lin_utils import ID_COUNTER

from gpu_canon.tracer import NumpyTracer, TraceOp
from gpu_canon.replayer import (
    replay_on_gpu,
    replay_on_gpu_optimized,
    classify_trace_ops,
)


def reset_id_counter(value: int = 1):
    """Reset CVXPY's global ID counter to a known value.

    When building problems with a factory function, call this BEFORE
    constructing the problem so that Variable/Parameter IDs are
    deterministic. Two factory calls that build the same expression
    tree structure will then produce identical IDs.
    """
    ID_COUNTER.count = value


def _problem_fingerprint(problem: cvxpy.Problem) -> str:
    """Compute a structural fingerprint of a CVXPY problem.

    The fingerprint captures the expression tree structure (atom types, shapes,
    variable IDs) but NOT data values (parameter values, constants). Two problems
    with the same fingerprint will produce the same trace.

    Variable/Parameter IDs are included in the hash. To get matching
    fingerprints for structurally identical problems built by separate
    factory calls, reset the ID counter (reset_id_counter()) before each
    construction so IDs are deterministic.
    """
    h = hashlib.sha256()

    def _hash_expr(expr):
        """Recursively hash expression tree structure."""
        h.update(type(expr).__name__.encode())
        h.update(str(expr.shape).encode())

        if isinstance(expr, cvxpy.Variable):
            h.update(f"var_{expr.id}_{expr.shape}".encode())
        elif isinstance(expr, cvxpy.Parameter):
            h.update(f"param_{expr.id}_{expr.shape}".encode())
        elif isinstance(expr, cvxpy.Constant):
            # Hash shape but NOT value — different values same structure
            h.update(f"const_{expr.shape}".encode())
        else:
            # Atom: hash type and recurse into args
            for arg in expr.args:
                _hash_expr(arg)

    # Hash objective
    obj = problem.objective
    h.update(b"minimize" if isinstance(obj, cvxpy.Minimize) else b"maximize")
    _hash_expr(obj.expr)

    # Hash constraints (order matters)
    for con in problem.constraints:
        h.update(type(con).__name__.encode())
        h.update(str(con.shape).encode())
        for arg in con.args:
            _hash_expr(arg)

    return h.hexdigest()


def _classify_inputs(
    trace: list[TraceOp],
    input_ids: set[int],
    input_arrays: dict[int, np.ndarray],
    problem: cvxpy.Problem,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Classify input arrays as 'data' (changes between calls) or 'structural'.

    Data inputs: parameter values, constant data that could change.
    Structural inputs: index arrays, shape-derived arrays that are fixed
    for a given problem structure.

    For simplicity, we classify based on dtype:
    - Float arrays are likely data (parameter values, coefficients)
    - Integer arrays are likely structural (indices, shapes)
    - Boolean arrays are structural (masks derived from structure)

    This is a heuristic — correct classification requires deeper analysis
    of which arrays come from parameters vs. problem structure.
    """
    data_inputs = {}
    structural_inputs = {}

    for tid in input_ids:
        arr = input_arrays.get(tid)
        if arr is None:
            continue

        if arr.dtype.kind in ('f', 'c'):
            # Float/complex: likely data
            data_inputs[tid] = arr
        else:
            # Integer/bool: likely structural (indices, masks)
            structural_inputs[tid] = arr

    return data_inputs, structural_inputs


class _CachedTrace:
    """A cached trace with its metadata."""

    def __init__(
        self,
        trace: list[TraceOp],
        data_input_ids: dict[int, np.ndarray],
        structural_input_ids: dict[int, np.ndarray],
        cone_dims: dict,
        output_ids: dict[str, int | None],
        A_structural: dict | None,
    ):
        self.trace = trace
        self.data_input_ids = data_input_ids
        self.structural_input_ids = structural_input_ids
        self.cone_dims = cone_dims
        self.output_ids = output_ids  # {"A_data": tid, "b": tid, "c": tid}
        self.A_structural = A_structural  # {"indices": arr, "indptr": arr, "shape": tuple}

        # Pre-transfer structural inputs to GPU (they don't change)
        self._structural_gpu: dict[int, Any] = {}
        if HAS_CUPY:
            for tid, arr in structural_input_ids.items():
                self._structural_gpu[tid] = cup.asarray(arr)
            # Pre-transfer A structural arrays to GPU
            if A_structural is not None:
                self._A_indices_gpu = cup.asarray(A_structural["indices"])
                self._A_indptr_gpu = cup.asarray(A_structural["indptr"])
                self._A_shape = A_structural["shape"]

        # Classify ops and pre-cache constant results on GPU.
        # Constant ops produce identical results every replay, so we
        # execute them once and cache the GPU arrays.
        data_id_set = set(data_input_ids.keys())
        self._constant_ids, self._data_op_indices = classify_trace_ops(
            trace, data_id_set
        )

        # Pre-compute constant ops by running a full replay with the
        # original data, then extracting only constant op results
        self._cached_constants: dict[int, Any] = {}
        if HAS_CUPY and self._data_op_indices:
            try:
                # Build initial arrays for the full replay
                init = {}
                init.update(self._structural_gpu)
                for tid, arr in data_input_ids.items():
                    init[tid] = cup.asarray(arr)

                full_registry = replay_on_gpu(trace, init)

                # Cache only constant op results
                for tid in self._constant_ids:
                    if tid in full_registry:
                        self._cached_constants[tid] = full_registry[tid]
            except Exception:
                # If pre-caching fails, fall back to full replay
                self._cached_constants = {}
                self._data_op_indices = list(range(len(trace)))


class TraceCache:
    """Cache of traces keyed by problem structure fingerprint.

    Usage:
        cache = TraceCache()

        # First call: traces and caches
        A, b, c, cone_dims = cache.canonicalize(problem)

        # Second call with same structure: replays cached trace
        A, b, c, cone_dims = cache.canonicalize(problem_same_structure)
    """

    def __init__(self):
        self._cache: dict[str, _CachedTrace] = {}
        self._stats = {"hits": 0, "misses": 0}

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def clear(self):
        """Clear all cached traces."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}

    def canonicalize(
        self,
        problem: cvxpy.Problem,
        solver: str = "CLARABEL",
        on_gpu: bool = True,
    ) -> tuple:
        """Canonicalize a problem, using cached trace if available.

        Args:
            problem: CVXPY problem to canonicalize.
            solver: Solver name (affects canonicalization output).
            on_gpu: If True, replay on GPU (requires CuPy).

        Returns:
            (A, b, c, cone_dims) — on GPU if on_gpu=True, else numpy/scipy.
        """
        fingerprint = _problem_fingerprint(problem)

        if fingerprint in self._cache:
            self._stats["hits"] += 1
            return self._replay(fingerprint, problem, on_gpu)
        else:
            self._stats["misses"] += 1
            return self._trace_and_cache(fingerprint, problem, solver, on_gpu)

    def _trace_and_cache(
        self,
        fingerprint: str,
        problem: cvxpy.Problem,
        solver: str,
        on_gpu: bool,
    ) -> tuple:
        """Trace a problem's canonicalization and cache the result."""
        solver_cls = getattr(cvxpy, solver, None) or cvxpy.CLARABEL

        # Run CVXPY with tracing
        with NumpyTracer() as tracer:
            data, chain, inv = problem.get_problem_data(
                solver_cls, canon_backend="COO"
            )

        # Extract result arrays
        A_cpu = data.get("A")
        b_cpu = data.get("b")
        c_cpu = data.get("c")

        # Register output arrays so we can find them during replay.
        # These may not have been directly produced by traced ops (e.g.
        # A is built by .tocsc() which is a method, not a patched function).
        # register_output() finds matching arrays in the trace or adds them.
        output_ids = {}
        A_structural = None

        if A_cpu is not None and sp.issparse(A_cpu):
            A_csc = A_cpu.tocsc()
            output_ids["A_data"] = tracer.register_output("A_data", A_csc.data)
            # A's structure (indices, indptr) is fixed per problem structure
            A_structural = {
                "indices": A_csc.indices.copy(),
                "indptr": A_csc.indptr.copy(),
                "shape": A_csc.shape,
            }
        if b_cpu is not None:
            output_ids["b"] = tracer.register_output("b", b_cpu)
        if c_cpu is not None:
            output_ids["c"] = tracer.register_output("c", c_cpu)

        trace = tracer.get_trace()
        input_ids = tracer.get_input_ids()
        input_arrays = tracer.get_input_arrays()

        # Extract cone dims
        cone_dims = {}
        for key in ("dims", "cone_dims", "dims_dict"):
            if key in data:
                cone_dims = data[key]
                break

        # Classify inputs
        data_inputs, structural_inputs = _classify_inputs(
            trace, input_ids, input_arrays, problem
        )

        cached = _CachedTrace(
            trace=trace,
            data_input_ids=data_inputs,
            structural_input_ids=structural_inputs,
            cone_dims=cone_dims,
            output_ids=output_ids,
            A_structural=A_structural,
        )
        self._cache[fingerprint] = cached

        # Return the CVXPY-computed result (known correct) for first call
        if on_gpu and HAS_CUPY:
            if A_cpu is not None:
                A_csc = A_cpu.tocsc() if sp.issparse(A_cpu) else sp.csc_matrix(A_cpu)
                A_gpu = cusp.csc_matrix(A_csc)
            else:
                A_gpu = None
            b_gpu = cup.asarray(b_cpu) if b_cpu is not None else None
            c_gpu = cup.asarray(c_cpu) if c_cpu is not None else None
            return A_gpu, b_gpu, c_gpu, cone_dims
        else:
            return A_cpu, b_cpu, c_cpu, cone_dims

    def _replay(
        self,
        fingerprint: str,
        problem: cvxpy.Problem,
        on_gpu: bool,
    ) -> tuple:
        """Replay a cached trace with new data on GPU.

        Feeds new input data (parameter values, constants) into the cached
        trace and replays all ops via CuPy. Extracts A, b, c from the
        replayed arrays using the cached output trace IDs.
        """
        cached = self._cache[fingerprint]

        # Build initial_arrays: structural (cached on GPU) + data (fresh)
        initial_arrays: dict[int, Any] = {}

        if on_gpu and HAS_CUPY:
            # Structural inputs are pre-transferred to GPU
            initial_arrays.update(cached._structural_gpu)

            # Data inputs: get fresh values from the problem's constants/params
            # and transfer to GPU
            for tid, original_arr in cached.data_input_ids.items():
                initial_arrays[tid] = cup.asarray(original_arr)

            # Optimized replay: skip constant ops, use cached results
            registry = replay_on_gpu_optimized(
                cached.trace,
                initial_arrays,
                cached_constants=cached._cached_constants,
                data_op_indices=cached._data_op_indices,
            )

            # Extract outputs using cached trace IDs
            A_gpu = None
            if "A_data" in cached.output_ids and cached.A_structural is not None:
                a_data_tid = cached.output_ids["A_data"]
                if a_data_tid in registry:
                    A_data_gpu = registry[a_data_tid]
                    A_gpu = cusp.csc_matrix(
                        (A_data_gpu,
                         cached._A_indices_gpu,
                         cached._A_indptr_gpu),
                        shape=cached._A_shape,
                    )

            b_gpu = None
            if "b" in cached.output_ids:
                b_tid = cached.output_ids["b"]
                if b_tid in registry:
                    b_gpu = registry[b_tid]

            c_gpu = None
            if "c" in cached.output_ids:
                c_tid = cached.output_ids["c"]
                if c_tid in registry:
                    c_gpu = registry[c_tid]

            return A_gpu, b_gpu, c_gpu, cached.cone_dims
        else:
            # CPU replay
            from gpu_canon.replayer import replay_on_cpu

            initial_arrays = {}
            initial_arrays.update(cached.structural_input_ids)
            initial_arrays.update(cached.data_input_ids)

            registry = replay_on_cpu(cached.trace, initial_arrays)

            A_cpu = None
            if "A_data" in cached.output_ids and cached.A_structural is not None:
                a_data_tid = cached.output_ids["A_data"]
                if a_data_tid in registry:
                    A_cpu = sp.csc_matrix(
                        (registry[a_data_tid],
                         cached.A_structural["indices"],
                         cached.A_structural["indptr"]),
                        shape=cached.A_structural["shape"],
                    )

            b_cpu = None
            if "b" in cached.output_ids:
                b_tid = cached.output_ids["b"]
                if b_tid in registry:
                    b_cpu = registry[b_tid]

            c_cpu = None
            if "c" in cached.output_ids:
                c_tid = cached.output_ids["c"]
                if c_tid in registry:
                    c_cpu = registry[c_tid]

            return A_cpu, b_cpu, c_cpu, cached.cone_dims


def trace_canonicalize(
    problem: cvxpy.Problem,
    solver: str = "CLARABEL",
) -> tuple[list[TraceOp], dict, dict[int, np.ndarray], dict]:
    """One-shot trace: run CVXPY with tracing, return trace + results.

    Useful for debugging and analysis. For production use, use TraceCache.

    Returns:
        (trace, result_data, input_arrays, cone_dims)
    """
    solver_cls = getattr(cvxpy, solver, None) or cvxpy.CLARABEL

    with NumpyTracer() as tracer:
        data, chain, inv = problem.get_problem_data(
            solver_cls, canon_backend="COO"
        )

    cone_dims = {}
    for key in ("dims", "cone_dims", "dims_dict"):
        if key in data:
            cone_dims = data[key]
            break

    return (
        tracer.get_trace(),
        data,
        tracer.get_input_arrays(),
        cone_dims,
    )
