"""
GPU canonicalization backend.

Takes a CVXPY Problem and produces (A, b, c, cone_dims) on GPU using CuPy.

This is the starter scaffold — intentionally simple. The autoresearch loop
iterates on this to make it fast.

Current approach (v0 — naive):
1. Walk the CVXPY expression tree to build IR (cpu_canon.tree)
2. Use CVXPY's own canonicalization to get the CPU (A, b, c) as a reference
3. Transfer to GPU via CuPy
4. Return GPU-resident arrays

This v0 is just the transfer baseline — it tells us the floor (how fast we
can go if we just copy the CPU result to GPU). The research loop replaces
steps 2-3 with actual GPU-native construction.
"""

from typing import Any

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import cvxpy

from gpu_canon.tree import walk, IRProblem


def canonicalize_gpu(
    problem: cvxpy.Problem,
    solver: str = "CLARABEL",
) -> tuple[Any, Any, Any, dict]:
    """
    Canonicalize a CVXPY Problem to (A, b, c, cone_dims) on GPU.

    Args:
        problem: A cvxpy.Problem instance
        solver: Target solver format (for cone ordering)

    Returns:
        (A, b, c, cone_dims) where A is a CuPy sparse matrix,
        b and c are CuPy dense arrays, and cone_dims is a dict.

    This v0 implementation walks the tree (building our IR) then falls back
    to CVXPY's CPU canonicalization and transfers to GPU. The research loop
    replaces this with GPU-native construction.
    """
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy not available. Install with: pip install cupy-cuda12x"
        )

    # Step 1: Walk the CVXPY expression tree to build our IR
    # (This is cheap and stays on CPU — it's just tree traversal)
    ir = walk(problem)

    # Step 2: For v0, use CVXPY's CPU canonicalization as the reference
    # The research loop replaces this with GPU-native matrix construction
    solver_cls = getattr(cvxpy, solver, None)
    if solver_cls is None:
        solver_cls = cvxpy.CLARABEL

    data, chain, inverse_data = problem.get_problem_data(
        solver_cls, canon_backend="SCIPY"
    )

    # Extract standard form matrices from solver data
    A_cpu = data.get("A")  # scipy sparse
    b_cpu = data.get("b")  # numpy array
    c_cpu = data.get("c")  # numpy array
    cone_dims = {}

    # Extract cone dimensions (solver-specific keys)
    for key in ("dims", "cone_dims", "dims_dict"):
        if key in data:
            cone_dims = data[key]
            break

    # Step 3: Transfer to GPU
    if A_cpu is not None:
        import scipy.sparse as sp
        if not sp.issparse(A_cpu):
            A_cpu = sp.csc_matrix(A_cpu)
        A_gpu = cusp.csc_matrix(A_cpu)
    else:
        A_gpu = None

    b_gpu = cp.asarray(b_cpu) if b_cpu is not None else None
    c_gpu = cp.asarray(c_cpu) if c_cpu is not None else None

    return A_gpu, b_gpu, c_gpu, cone_dims


def canonicalize_gpu_native(
    problem: cvxpy.Problem,
) -> tuple[Any, Any, Any, dict]:
    """
    GPU-native canonicalization — build (A, b, c) directly on GPU.

    THIS IS THE FUNCTION THE RESEARCH LOOP FILLS IN.

    The idea: walk the CVXPY expression tree, and instead of building
    scipy sparse matrices on CPU, build CuPy sparse matrices directly
    on GPU. This avoids the CPU→GPU transfer and lets us exploit GPU
    parallelism for matrix construction.

    Placeholder — raises NotImplementedError until the research loop
    implements it.
    """
    ir = walk(problem)
    raise NotImplementedError(
        "GPU-native canonicalization not yet implemented. "
        "The autoresearch loop builds this iteratively. "
        f"IR has {ir.total_vars} variables, {len(ir.constraints)} constraints."
    )
