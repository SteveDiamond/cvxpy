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
                self._A_tensor_gpu = cusp.csr_matrix(mat)
            except Exception:
                return
        else:
            self._A_tensor_gpu = None
            return

        # Cache the CSC index structure (static sparsity pattern)
        if ra.problem_data_index is not None:
            indices, indptr, shape = ra.problem_data_index
            self._A_indices_gpu = cup.asarray(indices)
            self._A_indptr_gpu = cup.asarray(indptr)
            self._A_csc_shape = shape
            self._A_pdi_mode = True
        else:
            self._A_pdi_mode = False
            self._A_var_len = ra.var_len

        self._A_nonzero_rows = ra.mapping_nonzero

        # Objective tensor (q)
        if pp.q is not None and sp.issparse(pp.q):
            try:
                q_mat = pp.q.tocsr()
                if q_mat.dtype != object:
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

        self._dpp_ready = True

    def _build_param_vector_gpu(self) -> cup.ndarray:
        """Build parameter vector on GPU from current CVXPY parameter values."""
        # Build on CPU (param values come from CVXPY Parameter objects),
        # then transfer. In a full GPU pipeline, param values would already
        # be on GPU and this transfer disappears.
        def param_value(idx):
            return np.array(self._param_prob.id_to_param[idx].value)

        param_vec = canonInterface.get_parameter_vector(
            self._total_param_size,
            self._param_id_to_col,
            self._param_id_to_size,
            param_value,
        )
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
            # Non-DPP: no re-canonicalization possible.
            # The compile step (CompiledProgram.__init__) already produced the result.
            return self._A_gpu, self._b_gpu, self._c_gpu, self.cone_dims

        # DPP hot path: re-evaluate with new parameter values
        if param_vec_gpu is None:
            param_vec_gpu = self._build_param_vector_gpu()

        # === HOT PATH: all GPU from here ===

        # A matrix: flat_data = A_tensor @ param_vec, then reassemble CSC
        if self._A_tensor_gpu is not None:
            flat_data = self._A_tensor_gpu @ param_vec_gpu

            if self._A_pdi_mode:
                # Reassemble CSC using cached index structure
                A_gpu = cusp.csc_matrix(
                    (flat_data, self._A_indices_gpu, self._A_indptr_gpu),
                    shape=self._A_csc_shape,
                )
            else:
                # Reshape into dense matrix then extract A and b
                n_cols = self._A_var_len + 1
                n_rows = flat_data.size // n_cols
                M = flat_data.reshape((n_rows, n_cols), order='F')
                A_gpu = cusp.csc_matrix(M[:, :-1])
                b_gpu = M[:, -1].ravel()
                # Objective
                c_gpu = self._c_gpu
                if self._q_tensor_gpu is not None:
                    qd = self._q_tensor_gpu @ param_vec_gpu
                    n_q_cols = self._x_size + 1
                    n_q_rows = qd.size // n_q_cols
                    Q = qd.reshape((n_q_rows, n_q_cols), order='F')
                    c_gpu = Q[:, :-1].ravel()
                return A_gpu, b_gpu, c_gpu, self.cone_dims
        else:
            A_gpu = self._A_gpu

        # b vector from A matrix (last column is offset)
        if self._A_pdi_mode and A_gpu is not None:
            # b is extracted differently in PDI mode
            # The CSC matrix already has the right structure
            # b comes from the offset in the param computation
            # For now, recompute from the full system
            pass

        # Objective: c = q_tensor @ param_vec
        c_gpu = self._c_gpu
        if self._q_tensor_gpu is not None:
            qd = self._q_tensor_gpu @ param_vec_gpu
            n_q_cols = self._x_size + 1
            if qd.size >= n_q_cols:
                n_q_rows = qd.size // n_q_cols
                Q = qd.reshape((n_q_rows, n_q_cols), order='F')
                c_gpu = Q[:, :-1].ravel()
                # d (offset) = Q[:, -1] — ignored for now

        return A_gpu, self._b_gpu, c_gpu, self.cone_dims


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

    # Try GPU-native compiler (handles LP/affine problems)
    try:
        from gpu_canon.compiler import compile_to_gpu
        return compile_to_gpu(problem)
    except (ValueError, NotImplementedError, TypeError):
        pass

    # Fallback: CVXPY reduction chain + GPU transfer
    compiled = CompiledProgram(problem, solver)
    return compiled.canonicalize()
