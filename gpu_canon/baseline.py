"""
CPU baseline canonicalization using CVXPY's own reduction chain.

Used for:
1. Correctness verification (GPU output should match this)
2. Speed comparison (GPU should beat this)
"""

from typing import Any

import numpy as np
import scipy.sparse as sp

import cvxpy


def canonicalize_cpu(
    problem: cvxpy.Problem,
    solver: str = "CLARABEL",
    backend: str = "SCIPY",
) -> tuple[Any, Any, Any, dict]:
    """
    Canonicalize a CVXPY Problem using CVXPY's CPU reduction chain.

    Args:
        problem: A cvxpy.Problem instance
        solver: Target solver
        backend: CVXPY canon backend ("CPP", "SCIPY", "COO")

    Returns:
        (A, b, c, cone_dims) as scipy/numpy arrays.
    """
    solver_cls = getattr(cvxpy, solver, None)
    if solver_cls is None:
        solver_cls = cvxpy.CLARABEL

    data, chain, inverse_data = problem.get_problem_data(
        solver_cls, canon_backend=backend
    )

    A_cpu = data.get("A")
    b_cpu = data.get("b")
    c_cpu = data.get("c")
    cone_dims = {}

    for key in ("dims", "cone_dims", "dims_dict"):
        if key in data:
            cone_dims = data[key]
            break

    if A_cpu is not None and not sp.issparse(A_cpu):
        A_cpu = sp.csc_matrix(A_cpu)

    return A_cpu, b_cpu, c_cpu, cone_dims
