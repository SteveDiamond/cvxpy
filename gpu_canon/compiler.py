"""
GPU-native expression tree compiler.

Walks a CVXPY Problem's expression tree and extracts affine coefficients
directly, bypassing CVXPY's reduction chain entirely. Builds the
(A, b, c) matrices on GPU using COO triplet accumulation.

Fast paths for common patterns (no recursive tree walk needed):
- Constant @ Variable <= Constant  (LP constraint)
- Variable >= 0  (nonnegativity)
- sum(Variable)  (linear objective)
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


def _to_dense(val):
    """Convert sparse/scalar to dense numpy array."""
    if sp.issparse(val):
        return np.asarray(val.toarray(), dtype=np.float64)
    return np.asarray(val, dtype=np.float64)


# ── Fast path: pattern match common constraint forms ─────────────────────

def _fast_extract_constraint(constr, var_id_to_col: dict, n_vars: int):
    """Try to extract (A_row, b_val, cone_type) without recursive tree walk.

    Returns (A_row_data, A_row_cols, b_val, n_rows) or None if pattern not matched.
    A_row_data and A_row_cols are arrays for COO construction.
    """
    expr = constr.expr

    if isinstance(constr, (NonNeg, NonPos, Inequality)):
        result = _fast_extract_affine(expr, var_id_to_col, n_vars)
        if result is not None:
            A_data, A_cols, b_offset, n_rows = result
            if isinstance(constr, NonNeg):
                # expr >= 0: -A@x <= b
                return -A_data, A_cols, b_offset, n_rows, "ineq"
            else:
                # expr <= 0: A@x <= -b
                return A_data, A_cols, -b_offset, n_rows, "ineq"

    elif isinstance(constr, (Zero, Equality)):
        result = _fast_extract_affine(expr, var_id_to_col, n_vars)
        if result is not None:
            A_data, A_cols, b_offset, n_rows = result
            return A_data, A_cols, -b_offset, n_rows, "eq"

    return None


def _fast_extract_affine(expr, var_id_to_col, n_vars):
    """Fast-path extraction for common affine expression patterns.

    Returns (A_data, A_cols, b_offset, n_rows) or None.
    A_data/A_cols are flat arrays: A[row_indices[i], A_cols[i]] = A_data[i].
    """

    # Pattern: AddExpression(MulExpression(Constant, Variable), NegExpression(Constant))
    # This is: A @ x - b, from constraint A @ x <= b
    if isinstance(expr, AddExpression) and len(expr.args) == 2:
        arg0, arg1 = expr.args

        # Check for Mul(Const, Var) + Neg(Const)
        if (isinstance(arg0, MulExpression) and isinstance(arg1, NegExpression)
                and len(arg0.args) == 2 and len(arg1.args) == 1):
            lhs, rhs = arg0.args
            neg_arg = arg1.args[0]

            if isinstance(lhs, (Constant, Parameter)) and isinstance(rhs, Variable) and isinstance(neg_arg, Constant):
                A_mat = _to_dense(lhs.value)
                b_vec = _to_dense(neg_arg.value).ravel()
                col_start = var_id_to_col[rhs.id]

                if A_mat.ndim == 1:
                    # a @ x (scalar result): 1 row
                    n_rows = 1
                    nz = np.nonzero(A_mat)[0]
                    A_data = A_mat[nz]
                    A_cols = nz + col_start
                    row_indices = np.zeros(len(nz), dtype=np.int32)
                    return A_data, A_cols, b_vec, n_rows
                else:
                    # A @ x (matrix result): m rows
                    n_rows = A_mat.shape[0]
                    nz_rows, nz_cols = np.nonzero(A_mat)
                    A_data = A_mat[nz_rows, nz_cols]
                    A_cols = nz_cols + col_start
                    return A_data, A_cols, b_vec, n_rows

        # Check for Promote(Const) + Neg(Var) — from x >= 0 rewritten
        if (isinstance(arg0, Promote) and isinstance(arg1, NegExpression)):
            if (isinstance(arg0.args[0], Constant) and isinstance(arg1.args[0], Variable)):
                const_val = float(_to_dense(arg0.args[0].value).ravel()[0])
                var = arg1.args[0]
                col_start = var_id_to_col[var.id]
                n_rows = var.size
                # expr = const - x, so A_data = -I, b = -const
                A_data = -np.ones(n_rows, dtype=np.float64)
                A_cols = np.arange(col_start, col_start + n_rows, dtype=np.int32)
                b_vec = np.full(n_rows, const_val, dtype=np.float64)
                return A_data, A_cols, b_vec, n_rows

    # Pattern: NegExpression(Variable) — from x >= 0 becoming -x <= 0
    if isinstance(expr, NegExpression) and isinstance(expr.args[0], Variable):
        var = expr.args[0]
        col_start = var_id_to_col[var.id]
        n_rows = var.size
        A_data = -np.ones(n_rows, dtype=np.float64)
        A_cols = np.arange(col_start, col_start + n_rows, dtype=np.int32)
        b_vec = np.zeros(n_rows, dtype=np.float64)
        return A_data, A_cols, b_vec, n_rows

    # Pattern: Variable (from x >= 0)
    if isinstance(expr, Variable):
        col_start = var_id_to_col[expr.id]
        n_rows = expr.size
        A_data = np.ones(n_rows, dtype=np.float64)
        A_cols = np.arange(col_start, col_start + n_rows, dtype=np.int32)
        b_vec = np.zeros(n_rows, dtype=np.float64)
        return A_data, A_cols, b_vec, n_rows

    return None


# ── Slow path: recursive affine extraction ───────────────────────────────

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
            nz = np.nonzero(mat)
            if mat.ndim == 1:
                for j in nz[0]:
                    self.rows.append(row_offset)
                    self.cols.append(col_offset + int(j))
                    self.data.append(scale * float(mat[j]))
            else:
                for i, j in zip(nz[0], nz[1]):
                    self.rows.append(row_offset + int(i))
                    self.cols.append(col_offset + int(j))
                    self.data.append(scale * float(mat[i, j]))

    def add_arrays(self, row_indices, col_indices, data_vals, row_offset=0):
        """Batch-add from numpy arrays (fast path)."""
        for r, c, d in zip(row_indices, col_indices, data_vals):
            self.rows.append(row_offset + int(r))
            self.cols.append(int(c))
            self.data.append(float(d))

    def to_scipy_csc(self, shape: tuple):
        if not self.rows:
            return sp.csc_matrix(shape, dtype=np.float64)
        rows = np.array(self.rows, dtype=np.int32)
        cols = np.array(self.cols, dtype=np.int32)
        data = np.array(self.data, dtype=np.float64)
        return sp.coo_matrix((data, (rows, cols)), shape=shape).tocsc()

    def to_gpu_csc(self, shape: tuple):
        if not self.rows:
            return cusp.csc_matrix(shape, dtype=np.float64)
        csc_cpu = self.to_scipy_csc(shape)
        return cusp.csc_matrix(
            (cup.asarray(csc_cpu.data),
             cup.asarray(csc_cpu.indices),
             cup.asarray(csc_cpu.indptr)),
            shape=shape,
        )


def _extract_affine_coeffs(expr, var_id_to_col, n_vars, coo, b_vec, row_offset, scale=1.0):
    """Recursively extract coefficients into COO builder and b vector."""

    if isinstance(expr, Variable):
        col = var_id_to_col[expr.id]
        for i in range(expr.size):
            coo.add_entry(row_offset + i, col + i, scale)
        return

    if isinstance(expr, (Constant, Parameter)):
        val = _to_dense(expr.value).ravel()
        b_vec[row_offset:row_offset + len(val)] += scale * val
        return

    if isinstance(expr, NegExpression):
        _extract_affine_coeffs(expr.args[0], var_id_to_col, n_vars, coo, b_vec, row_offset, -scale)
        return

    if isinstance(expr, Promote):
        _extract_affine_coeffs(expr.args[0], var_id_to_col, n_vars, coo, b_vec, row_offset, scale)
        return

    if isinstance(expr, AddExpression):
        for arg in expr.args:
            _extract_affine_coeffs(arg, var_id_to_col, n_vars, coo, b_vec, row_offset, scale)
        return

    if isinstance(expr, MulExpression):
        lhs, rhs = expr.args
        if isinstance(lhs, (Constant, Parameter)) and isinstance(rhs, Variable):
            val = lhs.value
            col = var_id_to_col[rhs.id]
            if sp.issparse(val):
                # Keep sparse — add_block handles it natively
                coo.add_block(row_offset, col, val, scale)
            else:
                mat = np.asarray(val, dtype=np.float64)
                if mat.ndim == 1:
                    mat = mat.reshape(1, -1)
                coo.add_block(row_offset, col, mat, scale)
            return
        if isinstance(lhs, (Constant, Parameter)) and lhs.is_scalar():
            _extract_affine_coeffs(rhs, var_id_to_col, n_vars, coo, b_vec, row_offset, scale * float(lhs.value))
            return
        if isinstance(rhs, (Constant, Parameter)) and isinstance(lhs, Variable):
            mat = _to_dense(rhs.value).ravel()
            col = var_id_to_col[lhs.id]
            for i in range(len(mat)):
                coo.add_entry(row_offset, col + i, scale * mat[i])
            return
        if isinstance(lhs, (Constant, Parameter)):
            mat = _to_dense(lhs.value)
            if mat.ndim < 2:
                _extract_affine_coeffs(rhs, var_id_to_col, n_vars, coo, b_vec, row_offset, scale * float(mat.ravel()[0]))
                return
            sub_coo = COOBuilder()
            sub_b = np.zeros(rhs.size, dtype=np.float64)
            _extract_affine_coeffs(rhs, var_id_to_col, n_vars, sub_coo, sub_b, 0, 1.0)
            if sub_coo.rows:
                sub_A = sp.coo_matrix((sub_coo.data, (sub_coo.rows, sub_coo.cols)), shape=(rhs.size, n_vars)).tocsc()
            else:
                sub_A = sp.csc_matrix((rhs.size, n_vars))
            result_A = mat @ sub_A
            result_b = mat @ sub_b
            coo.add_block(row_offset, 0, result_A, scale)
            b_vec[row_offset:row_offset + len(result_b)] += scale * result_b
            return

    if isinstance(expr, Sum):
        sub_coo = COOBuilder()
        sub_b = np.zeros(expr.args[0].size, dtype=np.float64)
        _extract_affine_coeffs(expr.args[0], var_id_to_col, n_vars, sub_coo, sub_b, 0, 1.0)
        for _, c, d in zip(sub_coo.rows, sub_coo.cols, sub_coo.data):
            coo.add_entry(row_offset, c, scale * d)
        b_vec[row_offset] += scale * sub_b.sum()
        return

    # Fallback: numerical evaluation
    _extract_numerical(expr, var_id_to_col, n_vars, coo, b_vec, row_offset, scale)


def _extract_numerical(expr, var_id_to_col, n_vars, coo, b_vec, row_offset, scale):
    """Fallback: extract by numerical evaluation (slow, O(n_vars))."""
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


# ── Main compiler ────────────────────────────────────────────────────────

def compile_to_gpu(problem: cvxpy.Problem) -> tuple:
    """
    Compile a CVXPY Problem to (A, b, c, cone_dims) on GPU.

    Uses fast-path pattern matching for common constraint forms,
    falls back to recursive extraction for complex expressions.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")

    # Step 1: Variable index mapping
    variables = problem.variables()
    var_id_to_col = {}
    col = 0
    for v in variables:
        var_id_to_col[v.id] = col
        col += v.size
    n_vars = col

    # Step 2: Objective
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

    # Step 3: Constraints — fast path first, slow path fallback
    # Separate into eq and ineq, equalities first (Clarabel convention)
    eq_constraints = []
    ineq_constraints = []
    for constr in problem.constraints:
        if isinstance(constr, (Zero, Equality)):
            eq_constraints.append(constr)
        else:
            ineq_constraints.append(constr)

    # Batch extraction using numpy arrays for fast-path constraints
    all_rows = []
    all_cols = []
    all_data = []
    all_b = []
    current_row = 0
    n_eq = 0
    n_ineq = 0

    # Use slow-path COO builder for constraints that don't match fast path
    slow_coo = COOBuilder()

    for constr_list, cone_type in [(eq_constraints, "eq"), (ineq_constraints, "ineq")]:
        for constr in constr_list:
            n_rows = constr.size

            # Try fast path
            fast = _fast_extract_constraint(constr, var_id_to_col, n_vars)
            if fast is not None:
                A_data, A_cols, b_vals, n_r, _ = fast
                # Build row indices
                if len(A_data) > 0:
                    if A_data.ndim == 0:
                        A_data = np.array([float(A_data)])
                    # For matrix patterns, we need proper row indices
                    if hasattr(A_cols, '__len__') and len(A_cols) == len(A_data):
                        # Compute row indices from the original matrix nonzeros
                        # For 1-row patterns, all rows are 0
                        if n_r == 1:
                            row_idx = np.zeros(len(A_data), dtype=np.int32) + current_row
                        else:
                            # Need to reconstruct row indices from the original extraction
                            # The fast path returns flat arrays from np.nonzero
                            # For A_mat case: nz_rows were captured
                            # We need to re-extract with row info
                            # Fall back to slow path for multi-row matrix patterns
                            # (fast path only reliable for single-row and diagonal patterns)
                            expr = constr.expr
                            b_slow = np.zeros(n_rows, dtype=np.float64)
                            if cone_type == "eq":
                                _extract_affine_coeffs(expr, var_id_to_col, n_vars,
                                                       slow_coo, b_slow, current_row, 1.0)
                                all_b.append(-b_slow)
                            elif isinstance(constr, NonNeg):
                                _extract_affine_coeffs(expr, var_id_to_col, n_vars,
                                                       slow_coo, b_slow, current_row, -1.0)
                                all_b.append(b_slow)
                            else:
                                _extract_affine_coeffs(expr, var_id_to_col, n_vars,
                                                       slow_coo, b_slow, current_row, 1.0)
                                all_b.append(-b_slow)

                            current_row += n_rows
                            if cone_type == "eq":
                                n_eq += n_rows
                            else:
                                n_ineq += n_rows
                            continue

                        all_rows.append(row_idx)
                        all_cols.append(np.asarray(A_cols, dtype=np.int32))
                        all_data.append(np.asarray(A_data, dtype=np.float64))

                all_b.append(np.asarray(b_vals, dtype=np.float64).ravel())
                current_row += n_rows
                if cone_type == "eq":
                    n_eq += n_rows
                else:
                    n_ineq += n_rows
                continue

            # Slow path
            b_slow = np.zeros(n_rows, dtype=np.float64)
            if cone_type == "eq":
                _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                       slow_coo, b_slow, current_row, 1.0)
                all_b.append(-b_slow)
            elif isinstance(constr, NonNeg):
                _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                       slow_coo, b_slow, current_row, -1.0)
                all_b.append(b_slow)
            else:
                _extract_affine_coeffs(constr.expr, var_id_to_col, n_vars,
                                       slow_coo, b_slow, current_row, 1.0)
                all_b.append(-b_slow)

            current_row += n_rows
            if cone_type == "eq":
                n_eq += n_rows
            else:
                n_ineq += n_rows

    # Merge fast-path arrays with slow-path COO
    if all_rows:
        merged_rows = np.concatenate(all_rows).tolist()
        merged_cols = np.concatenate(all_cols).tolist()
        merged_data = np.concatenate(all_data).tolist()
        slow_coo.rows.extend(merged_rows)
        slow_coo.cols.extend(merged_cols)
        slow_coo.data.extend(merged_data)

    actual_rows = current_row

    # Step 4: Build on GPU
    A_gpu = slow_coo.to_gpu_csc((actual_rows, n_vars))
    b_full = np.concatenate(all_b) if all_b else np.zeros(0, dtype=np.float64)
    b_gpu = cup.asarray(b_full[:actual_rows])
    c_gpu = cup.asarray(c_vec)

    cone_dims = {'zero': n_eq, 'nonneg': n_ineq, 'soc': [], 'psd': []}
    return A_gpu, b_gpu, c_gpu, cone_dims
