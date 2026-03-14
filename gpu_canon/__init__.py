"""
GPU-native canonicalization for CVXPY problems.

Takes a cvxpy.Problem and compiles it to conic standard form (A, b, c)
entirely on the GPU, bypassing CVXPY's internal reduction chain.

Usage:
    import cvxpy as cp
    from gpu_canon import canonicalize_gpu, canonicalize_cpu

    prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b])

    # GPU path (our compiler)
    A_gpu, b_gpu, c_gpu, cone_dims = canonicalize_gpu(prob)

    # CPU baseline (CVXPY's own canonicalization, for comparison)
    A_cpu, b_cpu, c_cpu, cone_dims_cpu = canonicalize_cpu(prob)
"""

from gpu_canon.backend import canonicalize_gpu, CompiledProgram
from gpu_canon.baseline import canonicalize_cpu
from gpu_canon.direct import from_data

__all__ = ["canonicalize_gpu", "canonicalize_cpu", "CompiledProgram", "from_data"]
