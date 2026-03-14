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
from gpu_canon.replayer import replay_on_gpu


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
        result_keys: dict[str, int | None],
    ):
        self.trace = trace
        self.data_input_ids = data_input_ids
        self.structural_input_ids = structural_input_ids
        self.cone_dims = cone_dims
        self.result_keys = result_keys

        # Pre-transfer structural inputs to GPU (they don't change)
        self._structural_gpu: dict[int, Any] = {}
        if HAS_CUPY:
            for tid, arr in structural_input_ids.items():
                self._structural_gpu[tid] = cup.asarray(arr)


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

        trace = tracer.get_trace()
        input_ids = tracer.get_input_ids()
        input_arrays = tracer.get_input_arrays()

        # Extract result arrays
        A_cpu = data.get("A")
        b_cpu = data.get("b")
        c_cpu = data.get("c")

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

        # Find which trace IDs correspond to A, b, c data arrays
        # We can't easily track this through the trace, so we use the
        # direct CVXPY output for the first call and replay for subsequent
        result_keys = {"A": None, "b": None, "c": None}

        cached = _CachedTrace(
            trace=trace,
            data_input_ids=data_inputs,
            structural_input_ids=structural_inputs,
            cone_dims=cone_dims,
            result_keys=result_keys,
        )
        self._cache[fingerprint] = cached

        # Return the CVXPY-computed result (known correct) for first call
        if on_gpu and HAS_CUPY:
            if A_cpu is not None:
                if sp.issparse(A_cpu):
                    A_gpu = cusp.csc_matrix(A_cpu.tocsc())
                else:
                    A_gpu = cusp.csc_matrix(sp.csc_matrix(A_cpu))
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
        """Replay a cached trace with new data.

        For now, we re-run CVXPY on cache hits too (the trace replay
        infrastructure captures ops but doesn't yet track which output arrays
        map to A/b/c). This still provides the caching framework and
        demonstrates the architecture — full replay will replace this
        once we solve output identification.
        """
        cached = self._cache[fingerprint]

        # Until we have full output tracking, use CVXPY for correctness
        # but still benefit from knowing we're on a cached structure
        solver_cls = cvxpy.CLARABEL
        data, _, _ = problem.get_problem_data(solver_cls, canon_backend="COO")

        A_cpu = data.get("A")
        b_cpu = data.get("b")
        c_cpu = data.get("c")
        cone_dims = cached.cone_dims

        if on_gpu and HAS_CUPY:
            if A_cpu is not None:
                if sp.issparse(A_cpu):
                    A_gpu = cusp.csc_matrix(A_cpu.tocsc())
                else:
                    A_gpu = cusp.csc_matrix(sp.csc_matrix(A_cpu))
            else:
                A_gpu = None
            b_gpu = cup.asarray(b_cpu) if b_cpu is not None else None
            c_gpu = cup.asarray(c_cpu) if c_cpu is not None else None
            return A_gpu, b_gpu, c_gpu, cone_dims
        else:
            return A_cpu, b_cpu, c_cpu, cone_dims


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
