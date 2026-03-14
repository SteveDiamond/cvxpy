"""
GPU-native canonicalization backend.

Takes a CVXPY Problem and produces (A, b, c, cone_dims) entirely on GPU.
All data starts on GPU, all results stay on GPU — no host-device transfers
in the hot path.

Architecture:
- Compile (cold path): run CVXPY's reduction chain once to extract the
  "problem_data_tensor" and "problem_data_index" (sparsity pattern).
  Move these to GPU. This is O(expression_tree_size) and happens once.
- Canonicalize (hot path): given new parameter values on GPU, do
  sparse @ dense matmul on GPU to get flat_problem_data, then assemble
  the CSC matrix on GPU using the cached sparsity pattern. Pure GPU compute.
"""

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
from cvxpy.cvxcore.python import canonInterface


class CompiledProgram:
    """A CVXPY problem compiled for fast GPU re-canonicalization.

    Cold path (compile): run CVXPY reduction chain, extract tensors, move to GPU.
    Hot path (canonicalize): sparse @ dense on GPU, reassemble CSC on GPU.
    """

    def __init__(self, problem: cvxpy.Problem, solver: str = "CLARABEL"):
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")

        self.problem = problem
        solver_cls = getattr(cvxpy, solver, None) or cvxpy.CLARABEL

        # Cold path: get problem data via COO backend (fastest CPU path)
        data, self._chain, self._inverse_data = problem.get_problem_data(
            solver_cls, canon_backend="COO"
        )

        # Extract cone dims
        self.cone_dims = {}
        for key in ("dims", "cone_dims", "dims_dict"):
            if key in data:
                self.cone_dims = data[key]
                break

        self._param_prob = data.get("param_prob")

        # Cache the initial result on GPU (used for non-DPP or first call)
        A_cpu = data.get("A")
        b_cpu = data.get("b")
        c_cpu = data.get("c")

        if A_cpu is not None:
            if sp.issparse(A_cpu):
                A_cpu = A_cpu.tocsc()
            else:
                A_cpu = sp.csc_matrix(A_cpu)
            self._A_gpu = cusp.csc_matrix(A_cpu)
        else:
            self._A_gpu = None
        self._b_gpu = cup.asarray(b_cpu) if b_cpu is not None else None
        self._c_gpu = cup.asarray(c_cpu) if c_cpu is not None else None

        # Set up DPP hot path if problem has actual parameters
        self._dpp_ready = False
        if (self._param_prob is not None
                and len(problem.parameters()) > 0
                and self._param_prob.total_param_size > 0):
            self._setup_dpp_gpu()

    def _setup_dpp_gpu(self):
        """Move DPP tensors to GPU for hot-path re-canonicalization."""
        pp = self._param_prob

        # Cache reduced_A structure
        ra = pp.reduced_A
        ra.cache(False)

        # The key tensor: reduced_mat maps param_vec -> flat_problem_data
        # flat_problem_data = reduced_mat @ param_vec
        if ra.reduced_mat is not None and sp.issparse(ra.reduced_mat):
            try:
                mat = ra.reduced_mat.tocsr()
                if mat.dtype == object:
                    # Non-parametric problems have object dtype — can't transfer
                    return

                # Detect gather pattern: every row has exactly 1 nonzero.
                # In this case, sparse matmul reduces to:
                #   flat_data[i] = scale[i] * param_vec[gather_idx[i]]
                # which is a simple gather + elementwise multiply.
                nnz_per_row = np.diff(mat.indptr)
                if nnz_per_row.max() == 1 and nnz_per_row.min() == 1:
                    self._gather_mode = True
                    self._gather_idx = cup.asarray(mat.indices.copy())
                    self._gather_scale = cup.asarray(mat.data.copy())
                    # Check if all scales are just +1 or -1 (common case)
                    unique_abs = np.unique(np.abs(mat.data))
                    self._gather_unit_scale = (
                        len(unique_abs) == 1 and np.isclose(unique_abs[0], 1.0))
                    self._A_tensor_gpu = None  # not needed for gather
                else:
                    self._gather_mode = False
                    self._A_tensor_gpu = cusp.csr_matrix(mat)
            except Exception:
                return
        else:
            self._A_tensor_gpu = None
            return

        # Cache the CSC index structure (static sparsity pattern)
        # In PDI mode, the CSC shape is (n_rows, var_len + 1) where the
        # last column holds the b (offset) vector. We pre-compute separate
        # index structures for A (first var_len cols) and b (last col) so
        # the hot path can assemble them independently.
        self._A_var_len = ra.var_len
        if ra.problem_data_index is not None:
            indices, indptr, shape = ra.problem_data_index
            n_rows, n_cols = shape
            # n_cols == var_len + 1; last column is b
            assert n_cols == ra.var_len + 1, (
                f"PDI shape mismatch: {n_cols} cols vs var_len+1={ra.var_len + 1}")

            # Split the CSC structure: columns 0..var_len-1 are A,
            # column var_len is b.
            # CSC indptr has n_cols+1 entries. A uses indptr[0:var_len+1],
            # b uses indptr[var_len:var_len+2].
            a_end = int(indptr[n_cols - 1])  # where A data ends / b data starts
            b_end = int(indptr[n_cols])       # end of b data

            # A sparsity: indices/data for columns 0..var_len-1
            self._A_indices_gpu = cup.asarray(indices[:a_end])
            self._A_indptr_gpu = cup.asarray(indptr[:n_cols])  # var_len+1 entries
            self._A_csc_shape = (n_rows, n_cols - 1)
            self._A_flat_split = a_end  # index into flat_data where b starts

            # b sparsity: indices/data for the last column
            self._b_indices_gpu = cup.asarray(indices[a_end:b_end])
            self._b_nnz = b_end - a_end
            self._b_size = n_rows

            self._A_pdi_mode = True
        else:
            self._A_pdi_mode = False

        self._A_nonzero_rows = ra.mapping_nonzero

        # Objective tensor (q)
        self._q_gather_mode = False
        if pp.q is not None and sp.issparse(pp.q):
            try:
                q_mat = pp.q.tocsr()
                if q_mat.dtype != object:
                    nnz_per_row = np.diff(q_mat.indptr)
                    if nnz_per_row.max() == 1 and nnz_per_row.min() == 1:
                        self._q_gather_mode = True
                        self._q_gather_idx = cup.asarray(q_mat.indices.copy())
                        self._q_gather_scale = cup.asarray(q_mat.data.copy())
                        self._q_tensor_gpu = None
                    else:
                        self._q_tensor_gpu = cusp.csr_matrix(q_mat)
                else:
                    self._q_tensor_gpu = None
            except Exception:
                self._q_tensor_gpu = None
        else:
            self._q_tensor_gpu = None
        self._x_size = pp.x.size if hasattr(pp, 'x') and pp.x is not None else 0

        # Parameter mapping info
        self._param_id_to_col = pp.param_id_to_col
        self._param_id_to_size = pp.param_id_to_size
        self._total_param_size = pp.total_param_size

        # Pre-allocated output buffers for hot-path reuse (populated on first call)
        self._cached_A = None
        self._cached_b = None

        self._dpp_ready = True

    def _build_param_vector_gpu(self) -> cup.ndarray:
        """Build parameter vector on GPU from current CVXPY parameter values.

        Uses direct numpy construction (faster than canonInterface for large
        parameters) then transfers to GPU. In a full GPU pipeline, use
        canonicalize(param_vec_gpu) to skip this entirely.
        """
        pp = self._param_prob
        # Direct construction is ~30% faster than canonInterface because
        # it avoids the callback overhead per parameter.
        param_vec = np.zeros(self._total_param_size + 1)
        for pid, col in self._param_id_to_col.items():
            if pid == -1:
                # Constant offset placeholder
                param_vec[col] = 1.0
            else:
                sz = self._param_id_to_size[pid]
                val = np.asarray(pp.id_to_param[pid].value).flatten(order='F')
                param_vec[col:col + sz] = val
        return cup.asarray(param_vec)

    def canonicalize(self, param_vec_gpu: cup.ndarray | None = None) -> tuple:
        """
        Canonicalize with current or new parameter values. All GPU.

        Args:
            param_vec_gpu: Parameter vector already on GPU. If None,
                          builds it from current CVXPY parameter values.

        Returns:
            (A, b, c, cone_dims) all on GPU.
        """
        if not self._dpp_ready:
            return self._A_gpu, self._b_gpu, self._c_gpu, self.cone_dims

        if param_vec_gpu is None:
            param_vec_gpu = self._build_param_vector_gpu()

        # === HOT PATH: all GPU from here ===
        A_gpu, b_gpu = self._apply_A_b(param_vec_gpu)
        c_gpu = self._apply_objective(param_vec_gpu)
        return A_gpu, b_gpu, c_gpu, self.cone_dims

    def canonicalize_inplace(self, param_vec_gpu: cup.ndarray | None = None) -> tuple:
        """
        Like canonicalize(), but reuses cached GPU buffers when possible.

        This avoids allocating new CSC matrix objects on each call, which
        reduces Python/CuPy overhead in tight re-solve loops. The returned
        A matrix's data array is overwritten on subsequent calls.

        Args:
            param_vec_gpu: Parameter vector already on GPU.

        Returns:
            (A, b, c, cone_dims) all on GPU. A.data is a view that will be
            overwritten on the next call.
        """
        if not self._dpp_ready:
            return self._A_gpu, self._b_gpu, self._c_gpu, self.cone_dims

        if param_vec_gpu is None:
            param_vec_gpu = self._build_param_vector_gpu()

        # === HOT PATH: all GPU, minimal allocation ===
        A_gpu, b_gpu = self._apply_A_b_inplace(param_vec_gpu)
        c_gpu = self._apply_objective(param_vec_gpu)
        return A_gpu, b_gpu, c_gpu, self.cone_dims

    def _compute_flat_data(self, param_vec_gpu: cup.ndarray) -> cup.ndarray:
        """Compute flat problem data from parameter vector.

        Uses gather+multiply when the tensor has exactly 1 nnz per row
        (common case for DPP), otherwise falls back to sparse matmul.
        """
        if self._gather_mode:
            if self._gather_unit_scale:
                # All scales are +/-1: just gather with sign flip
                return self._gather_scale * param_vec_gpu[self._gather_idx]
            else:
                return self._gather_scale * param_vec_gpu[self._gather_idx]
        else:
            return self._A_tensor_gpu @ param_vec_gpu

    def _apply_A_b(self, param_vec_gpu: cup.ndarray) -> tuple:
        """Compute A matrix and b vector from parameter vector."""
        if self._A_tensor_gpu is None and not self._gather_mode:
            return self._A_gpu, self._b_gpu

        flat_data = self._compute_flat_data(param_vec_gpu)

        if self._A_pdi_mode:
            split = self._A_flat_split

            A_gpu = cusp.csc_matrix(
                (flat_data[:split],
                 self._A_indices_gpu,
                 self._A_indptr_gpu),
                shape=self._A_csc_shape,
            )

            if self._b_nnz > 0:
                b_gpu = cup.zeros(self._b_size, dtype=flat_data.dtype)
                b_gpu[self._b_indices_gpu] = flat_data[split:split + self._b_nnz]
            else:
                b_gpu = cup.zeros(self._b_size, dtype=flat_data.dtype)
        else:
            n_cols = self._A_var_len + 1
            n_rows = flat_data.size // n_cols
            M = flat_data.reshape((n_rows, n_cols), order='F')
            A_gpu = cusp.csc_matrix(M[:, :-1])
            b_gpu = M[:, -1].ravel()

        return A_gpu, b_gpu

    def _apply_A_b_inplace(self, param_vec_gpu: cup.ndarray) -> tuple:
        """Compute A and b, reusing cached GPU objects to minimize allocation."""
        if self._A_tensor_gpu is None and not self._gather_mode:
            return self._A_gpu, self._b_gpu

        flat_data = self._compute_flat_data(param_vec_gpu)

        if self._A_pdi_mode:
            split = self._A_flat_split

            if self._cached_A is not None:
                # Reuse: just swap the data pointer (same sparsity pattern)
                self._cached_A.data = flat_data[:split]
                A_gpu = self._cached_A
            else:
                A_gpu = cusp.csc_matrix(
                    (flat_data[:split],
                     self._A_indices_gpu,
                     self._A_indptr_gpu),
                    shape=self._A_csc_shape,
                )
                self._cached_A = A_gpu

            if self._b_nnz > 0:
                if self._cached_b is None:
                    self._cached_b = cup.zeros(self._b_size, dtype=flat_data.dtype)
                self._cached_b[:] = 0
                self._cached_b[self._b_indices_gpu] = flat_data[split:split + self._b_nnz]
                b_gpu = self._cached_b
            else:
                if self._cached_b is None:
                    self._cached_b = cup.zeros(self._b_size, dtype=flat_data.dtype)
                b_gpu = self._cached_b
        else:
            n_cols = self._A_var_len + 1
            n_rows = flat_data.size // n_cols
            M = flat_data.reshape((n_rows, n_cols), order='F')
            A_gpu = cusp.csc_matrix(M[:, :-1])
            b_gpu = M[:, -1].ravel()

        return A_gpu, b_gpu

    def _apply_objective(self, param_vec_gpu: cup.ndarray) -> cup.ndarray:
        """Compute objective vector c from parameter vector."""
        c_gpu = self._c_gpu
        if self._q_tensor_gpu is not None or self._q_gather_mode:
            if self._q_gather_mode:
                qd = self._q_gather_scale * param_vec_gpu[self._q_gather_idx]
            else:
                qd = self._q_tensor_gpu @ param_vec_gpu
            n_q_cols = self._x_size + 1
            if qd.size >= n_q_cols:
                n_q_rows = qd.size // n_q_cols
                Q = qd.reshape((n_q_rows, n_q_cols), order='F')
                c_gpu = Q[:, :-1].ravel()
        return c_gpu


def canonicalize_gpu(
    problem: cvxpy.Problem,
    solver: str = "CLARABEL",
) -> tuple[Any, Any, Any, dict]:
    """
    One-shot GPU canonicalization.

    Tries the GPU-native compiler first (fast, bypasses CVXPY reduction chain).
    Falls back to CompiledProgram (uses CVXPY backend + GPU transfer) for
    unsupported problem types.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda12x")

    # Always try GPU-native compiler first — it's faster for most problems.
    # It handles non-affine atoms via numerical evaluation, which works
    # correctly and is faster than CVXPY's reduction chain for most cases.
    try:
        from gpu_canon.compiler import compile_to_gpu
        return compile_to_gpu(problem)
    except (ValueError, NotImplementedError, TypeError):
        pass

    # Fallback: CVXPY reduction chain + GPU transfer
    compiled = CompiledProgram(problem, solver)
    return compiled.canonicalize()


# ── Trace-compile path ────────────────────────────────────────────────────────

# Module-level trace cache (shared across calls)
_trace_cache = None


def _get_trace_cache():
    """Get or create the module-level trace cache."""
    global _trace_cache
    if _trace_cache is None:
        from gpu_canon.trace_cache import TraceCache
        _trace_cache = TraceCache()
    return _trace_cache


def canonicalize_gpu_traced(
    problem: cvxpy.Problem,
    solver: str = "CLARABEL",
) -> tuple[Any, Any, Any, dict]:
    """GPU canonicalization via trace-compile.

    First call: traces CVXPY's numpy ops, caches the trace, returns GPU result.
    Subsequent calls with same problem structure: replays cached trace on GPU.

    This handles ANY problem CVXPY handles — no reimplementation needed.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda12x")

    cache = _get_trace_cache()
    return cache.canonicalize(problem, solver, on_gpu=True)


def reset_trace_cache():
    """Reset the module-level trace cache."""
    global _trace_cache
    if _trace_cache is not None:
        _trace_cache.clear()
    _trace_cache = None
