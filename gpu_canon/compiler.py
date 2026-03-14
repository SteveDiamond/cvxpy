"""
GPU-native expression tree compiler.

Walks a CVXPY Problem's expression tree and extracts affine coefficients
directly, bypassing CVXPY's reduction chain entirely. Builds the
(A, b, c) matrices on GPU using COO triplet accumulation.
"""

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cup
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import cvxpy
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.promote import Promote
from cvxpy.constraints.nonpos import NonNeg, NonPos, Inequality
from cvxpy.constraints.zero import Zero, Equality
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable


class COOBuilder:
    """Accumulates COO triplets for sparse matrix construction."""

    def __init__(self):
        self.rows = []
        self.cols = []
        self.data = []

    def add_entry(self, row: int, col: int, val: float):
        if val != 0.0:
            self.rows.append(row)
            self.cols.append(col)
            self.data.append(val)

    def add_block(self, row_offset: int, col_offset: int,
                  matrix, scale: float = 1.0):
        """Add a dense or sparse matrix block at the given offset."""
        if sp.issparse(matrix):
            coo = matrix.tocoo()
            for i, j, v in zip(coo.row, coo.col, coo.data):
                sv = scale * v
                if sv != 0.0:
                    self.rows.append(row_offset + i)
                    self.cols.append(col_offset + j)
                    self.data.append(sv)
        else:
            mat = np.asarray(matrix)
            if mat.ndim == 1:
                for j, v in enumerate(mat):
                    sv = scale * v
                    if sv != 0.0:
                        self.rows.append(row_offset)
                        self.cols.append(col_offset + j)
                        self.data.append(sv)
            else:
                nz = np.nonzero(mat)
                for i, j in zip(nz[0], nz[1]):
                    self.rows.append(row_offset + int(i))
                    self.cols.append(col_offset + int(j))
                    self.data.append(scale * float(mat[i, j]))

    def to_scipy_csc(self, shape: tuple):
        """Build scipy CSC on CPU from accumulated triplets."""
        if not self.rows:
            return sp.csc_matrix(shape, dtype=np.float64)
        rows = np.array(self.rows, dtype=np.int32)
        cols = np.array(self.cols, dtype=np.int32)
        data = np.array(self.data, dtype=np.float64)
        return sp.coo_matrix((data, (rows, cols)), shape=shape).tocsc()

    def to_gpu_csc(self, shape: tuple):
        """Build CuPy CSC on GPU. Single bulk transfer of triplet arrays."""
        if not self.rows:
            return cusp.csc_matrix(shape, dtype=np.float64)
        # Build CSC on CPU first (scipy is fast at COO→CSC),
        # then transfer the 3 CSC arrays to GPU in one shot.
        csc_cpu = self.to_scipy_csc(shape)
        return cusp.csc_matrix(
            (cup.asarray(csc_cpu.data),
             cup.asarray(csc_cpu.indices),
             cup.asarray(csc_cpu.indptr)),
            shape=shape,
        )


def _to_dense_val(val):
    """Convert a CVXPY value (possibly sparse) to dense numpy array."""
    if val is None:
        return None
    if sp.issparse(val):
        return val.toarray()
    return np.asarray(val, dtype=np.float64)


def _extract_affine_coeffs(expr, var_id_to_col: dict, n_vars: int,
                            coo: COOBuilder, b_vec: np.ndarray,
                            row_offset: int, scale: float = 1.0):
    """Extract affine coefficients from expr into COO builder and b vector.

    For expr = A @ x + b, adds A entries to coo and b entries to b_vec.
    """
    if isinstance(expr, Variable):
        col = var_id_to_col[expr.id]
        for i in range(expr.size):
            coo.add_entry(row_offset + i, col + i, scale)
        return

    if isinstance(expr, (Constant, Parameter)):
        val = _to_dense_val(expr.value)
        if val is not None:
            flat = val.ravel()
            b_vec[row_offset:row_offset + len(flat)] += scale * flat
        return

    if isinstance(expr, NegExpression):
        _extract_affine_coeffs(expr.args[0], var_id_to_col, n_vars,
                                coo, b_vec, row_offset, -scale)
        return

    if isinstance(expr, Promote):
        _extract_affine_coeffs(expr.args[0], var_id_to_col, n_vars,
                                coo, b_vec, row_offset, scale)
        return

    if isinstance(expr, AddExpression):
        for arg in expr.args:
            _extract_affine_coeffs(arg, var_id_to_col, n_vars,
                                    coo, b_vec, row_offset, scale)
        return

    if isinstance(expr, MulExpression):
        lhs, rhs = expr.args

        # Constant @ Variable — the fast path
        if isinstance(lhs, Constant) and isinstance(rhs, Variable):
            mat = _to_dense_val(lhs.value)
            col = var_id_to_col[rhs.id]
            if mat.ndim == 1:
                # Vector: treat as 1×n matrix
                coo.add_block(row_offset, col, mat.reshape(1, -1), scale)
            else:
                coo.add_block(row_offset, col, mat, scale)
            return

        # Scalar constant * sub-expression
        if isinstance(lhs, Constant) and lhs.is_scalar():
            s = float(lhs.value)
            _extract_affine_coeffs(rhs, var_id_to_col, n_vars,
                                    coo, b_vec, row_offset, scale * s)
            return

        # Variable @ Constant (e.g. c @ x)
        if isinstance(rhs, Constant) and isinstance(lhs, Variable):
            mat = _to_dense_val(rhs.value)
            col = var_id_to_col[lhs.id]
            flat = mat.ravel()
            for i in range(len(flat)):
                coo.add_entry(row_offset, col + i, scale * flat[i])
            return

        # Constant matrix @ sub-expression
        if isinstance(lhs, Constant):
            mat = _to_dense_val(lhs.value)
            if mat.ndim < 2:
                s = float(mat.ravel()[0])
                _extract_affine_coeffs(rhs, var_id_to_col, n_vars,
                                        coo, b_vec, row_offset, scale * s)
                return
            # General matrix multiply: extract sub-expression to temp COO,
            # then multiply. This is the slow path.
            sub_coo = COOBuilder()
            sub_b = np.zeros(rhs.size, dtype=np.float64)
            _extract_affine_coeffs(rhs, var_id_to_col, n_vars,
                                    sub_coo, sub_b, 0, 1.0)
            # Build sub_A as scipy sparse
            if sub_coo.rows:
                sub_A = sp.coo_matrix(
                    (sub_coo.data, (sub_coo.rows, sub_coo.cols)),
                    shape=(rhs.size, n_vars),
                ).tocsc()
            else:
                sub_A = sp.csc_matrix((rhs.size, n_vars))
            result_A = mat @ sub_A  # dense @ sparse → dense
            result_b = mat @ sub_b
            coo.add_block(row_offset, 0, result_A, scale)
            b_vec[row_offset:row_offset + len(result_b)] += scale * result_b
            return

    if isinstance(expr, Sum):
        # sum(sub_expr) over all elements
        sub_coo = COOBuilder()
        sub_b = np.zeros(expr.args[0].size, dtype=np.float64)
        _extract_affine_coeffs(expr.args[0], var_id_to_col, n_vars,
                                sub_coo, sub_b, 0, 1.0)
        # Sum all rows into one row
        for r, c, d in zip(sub_coo.rows, sub_coo.cols, sub_coo.data):
            coo.add_entry(row_offset, c, scale * d)
        b_vec[row_offset] += scale * sub_b.sum()
        return

    # Fallback: numerical evaluation
    _extract_numerical(expr, var_id_to_col, n_vars, coo, b_vec, row_offset, scale)


def _extract_numerical(expr, var_id_to_col, n_vars, coo, b_vec, row_offset, scale):
    """Fallback: extract coefficients by numerical evaluation (O(n_vars))."""
    variables = expr.variables()
    saved = {v.id: v.value for v in variables}

    for v in variables:
        v.value = np.zeros(v.shape)
    offset = np.asarray(expr.value, dtype=np.float64).ravel()
    n_rows = len(offset)
    b_vec[row_offset:row_offset + n_rows] += scale * offset

    for v in variables:
        col = var_id_to_col[v.id]
        for j in range(v.size):
            for vv in variables:
                vv.value = np.zeros(vv.shape)
            e_j = np.zeros(v.shape)
            e_j.flat[j] = 1.0
            v.value = e_j
            col_vals = np.asarray(expr.value, dtype=np.float64).ravel() - offset
            for i in range(n_rows):
                coo.add_entry(row_offset + i, col + j, scale * col_vals[i])

    for v in variables:
        v.value = saved[v.id]


def compile_to_gpu(problem: cvxpy.Problem) -> tuple:
    """
    Compile a CVXPY Problem directly to (A, b, c, cone_dims) on GPU.

    Bypasses CVXPY's reduction chain entirely. Walks the expression tree,
    extracts affine coefficients as COO triplets, and builds sparse matrices
    on GPU.

    Returns:
        (A_gpu, b_gpu, c_gpu, cone_dims) — Clarabel format
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")

    # Step 1: Collect variables, assign column indices
    variables = problem.variables()
    var_id_to_col = {}
    col = 0
    for v in variables:
        var_id_to_col[v.id] = col
        col += v.size
    n_vars = col

    # Step 2: Extract objective (c vector)
    is_minimize = isinstance(problem.objective, cvxpy.Minimize)
    obj_coo = COOBuilder()
    obj_b = np.zeros(1, dtype=np.float64)
    _extract_affine_coeffs(problem.objective.expr, var_id_to_col, n_vars,
                           obj_coo, obj_b, 0, 1.0)
    c_vec = np.zeros(n_vars, dtype=np.float64)
    for _, c_col, val in zip(obj_coo.rows, obj_coo.cols, obj_coo.data):
        c_vec[c_col] += val
    if not is_minimize:
        c_vec = -c_vec

    # Step 3: Process constraints — accumulate COO triplets
    A_coo = COOBuilder()
    n_eq = 0
    n_ineq = 0
    current_row = 0

    # First pass: equalities (zero cone comes first in Clarabel)
    for constr in problem.constraints:
        if isinstance(constr, (Zero, Equality)):
            n_rows = constr.size
            b_eq = np.zeros(n_rows, dtype=np.float64)
            _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                   A_coo, b_eq, current_row, 1.0)
            # Store b for later (eq: expr == 0, so b = -offset)
            if not hasattr(compile_to_gpu, '_b_parts'):
                compile_to_gpu._b_parts = []
            current_row += n_rows
            n_eq += n_rows

    # Collect b values: need to redo this properly
    # Reset and do it right with a single b vector
    A_coo = COOBuilder()
    current_row = 0
    total_rows = sum(c.size for c in problem.constraints)
    b_full = np.zeros(total_rows, dtype=np.float64)
    n_eq = 0
    n_ineq = 0

    # Equalities first
    for constr in problem.constraints:
        if isinstance(constr, (Zero, Equality)):
            n_rows = constr.size
            _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                   A_coo, b_full, current_row, 1.0)
            # eq: expr == 0 → A@x = -b_offset
            b_full[current_row:current_row + n_rows] *= -1
            current_row += n_rows
            n_eq += n_rows

    # Then inequalities
    for constr in problem.constraints:
        if isinstance(constr, (NonNeg, NonPos, Inequality)):
            n_rows = constr.size
            if isinstance(constr, NonNeg):
                # expr >= 0: need -A@x <= b
                _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                       A_coo, b_full, current_row, -1.0)
                # b already has the right sign from -1 scale on offset
            elif isinstance(constr, (NonPos, Inequality)):
                # expr <= 0: A@x <= -b
                _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                       A_coo, b_full, current_row, 1.0)
                b_full[current_row:current_row + n_rows] *= -1
            current_row += n_rows
            n_ineq += n_rows
        elif not isinstance(constr, (Zero, Equality)):
            raise ValueError(f"Unsupported constraint type: {type(constr).__name__}")

    actual_rows = current_row

    # Step 4: Build on GPU
    A_gpu = A_coo.to_gpu_csc((actual_rows, n_vars))
    b_gpu = cup.asarray(b_full[:actual_rows])
    c_gpu = cup.asarray(c_vec)

    cone_dims = {'zero': n_eq, 'nonneg': n_ineq, 'soc': [], 'psd': []}

    return A_gpu, b_gpu, c_gpu, cone_dims
