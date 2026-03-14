from cvxpy.settings import (
    COO_CANON_BACKEND,
    CPP_CANON_BACKEND,
    SCIPY_CANON_BACKEND,
)
from cvxpy.utilities.warn import warn


def _auto_select_backend(problem) -> str:
    """Auto-select the best backend based on problem structure.

    - DPP problems (has parameters): COO is 30-50x faster than CPP
    - Non-DPP problems: CPP is 1.1-3.8x faster than COO
    """
    if problem.parameters():
        return COO_CANON_BACKEND
    return CPP_CANON_BACKEND


def get_canon_backend(problem, canon_backend: str) -> str:
    """
    Select the canonicalization backend.

    When canon_backend is None, auto-selects based on problem structure:
    COO for DPP (parametrized) problems, CPP for non-DPP.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    canon_backend : str
        'CPP' | 'SCIPY' | 'COO' | None (auto-select)
        Specifies which backend to use for canonicalization, which can affect
        compilation time. Defaults to None, i.e., auto-selecting the best
        backend.
    Returns
    -------
    canon_backend : str
        The canonicalization backend to use.
    """

    if not problem._supports_cpp():
        if canon_backend is None:
            return _auto_select_backend(problem) if problem.parameters() else SCIPY_CANON_BACKEND
        if canon_backend == CPP_CANON_BACKEND:
            raise ValueError(f"The {CPP_CANON_BACKEND} backend cannot be used with problems "
                             f"that have expressions which do not support it.")
        return canon_backend  # Use the specified backend (e.g., COO_CANON_BACKEND)

    if problem._max_ndim() > 2:
        if canon_backend is None:
            warn(
                f"The problem has an expression with dimension greater than 2. "
                f"Defaulting to the {SCIPY_CANON_BACKEND} backend for canonicalization.")
            return SCIPY_CANON_BACKEND
        if canon_backend == CPP_CANON_BACKEND:
            raise ValueError(
                f"Only the {COO_CANON_BACKEND} and {SCIPY_CANON_BACKEND} "
                f"backends are supported for problems "
                f"with expressions of dimension greater than 2."
            )

    if canon_backend is None:
        return _auto_select_backend(problem)
    return canon_backend
