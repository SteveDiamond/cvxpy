"""
CVXPY expression tree traversal and intermediate representation.

This module walks a cvxpy.Problem's expression tree (objective, constraints,
atoms, leaves) and produces a lightweight IR that the GPU backend can compile
into (A, b, c) matrices.

The IR is deliberately simple — the autoresearch loop will evolve it.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import cvxpy
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable


# ── IR node types ────────────────────────────────────────────────────────────

@dataclass
class IRVar:
    """A decision variable."""
    id: int
    shape: tuple
    size: int
    name: str = ""


@dataclass
class IRParam:
    """A parameter (value known at solve time, not at compile time)."""
    id: int
    shape: tuple
    size: int
    value: Any = None  # numpy array when evaluated


@dataclass
class IRConst:
    """A constant (value known at compile time)."""
    value: Any  # numpy array
    shape: tuple


@dataclass
class IRAffineOp:
    """An affine operation: result = coeff @ input + offset.

    This is the fundamental building block. The GPU backend assembles
    the full A matrix from these.
    """
    op_type: str  # "mul", "add", "neg", "reshape", "index", "sum", etc.
    args: list = field(default_factory=list)  # child IR nodes
    data: dict = field(default_factory=dict)  # op-specific data (coefficients, indices, etc.)
    shape: tuple = ()


@dataclass
class IRConstraint:
    """A conic constraint: expr in cone."""
    cone_type: str  # "zero", "nonneg", "soc", "psd"
    expr: Any  # IR node for the expression
    args: list = field(default_factory=list)  # for SOC: [t, x]
    size: int = 0


@dataclass
class IRProblem:
    """Complete IR for a conic problem."""
    objective: Any  # IR node for the objective expression
    constraints: list  # list of IRConstraint
    variables: dict  # var_id -> IRVar
    parameters: dict  # param_id -> IRParam
    minimize: bool = True
    total_vars: int = 0  # total number of scalar decision variables


# ── Tree walker ──────────────────────────────────────────────────────────────

class TreeWalker:
    """Walk a CVXPY Problem and produce an IRProblem.

    This is the starting scaffold. It handles:
    - Variables, Parameters, Constants as leaves
    - Affine atoms (multiply, add, neg, reshape, etc.)
    - Basic conic constraints (Zero, NonNeg/NonPos, SOC)

    The autoresearch loop extends this to handle more atoms and produce
    better IR for GPU execution.
    """

    def __init__(self):
        self.variables: dict[int, IRVar] = {}
        self.parameters: dict[int, IRParam] = {}

    def walk_problem(self, problem: cvxpy.Problem) -> IRProblem:
        """Convert a CVXPY Problem to IR."""
        # Collect all variables
        for var in problem.variables():
            self.variables[var.id] = IRVar(
                id=var.id,
                shape=var.shape,
                size=var.size,
                name=var.name(),
            )

        # Collect all parameters
        for param in problem.parameters():
            self.parameters[param.id] = IRParam(
                id=param.id,
                shape=param.shape,
                size=param.size,
                value=param.value,
            )

        # Walk objective
        is_minimize = isinstance(problem.objective, cvxpy.Minimize)
        obj_expr = problem.objective.expr
        obj_ir = self.walk_expr(obj_expr)

        # Walk constraints
        constraint_irs = []
        for constr in problem.constraints:
            constraint_irs.append(self.walk_constraint(constr))

        total_vars = sum(v.size for v in self.variables.values())

        return IRProblem(
            objective=obj_ir,
            constraints=constraint_irs,
            variables=self.variables,
            parameters=self.parameters,
            minimize=is_minimize,
            total_vars=total_vars,
        )

    def walk_expr(self, expr) -> Any:
        """Recursively walk a CVXPY expression and produce IR."""
        # Leaf nodes
        if isinstance(expr, Variable):
            return self.variables[expr.id]

        if isinstance(expr, Parameter):
            # Update value in case it changed
            self.parameters[expr.id].value = expr.value
            return self.parameters[expr.id]

        if isinstance(expr, Constant):
            return IRConst(
                value=np.array(expr.value),
                shape=expr.shape,
            )

        # Atom nodes — walk children first, then wrap in IRAffineOp
        if isinstance(expr, Atom):
            child_irs = [self.walk_expr(arg) for arg in expr.args]

            # Extract atom type name
            op_type = type(expr).__name__.lower()

            # Collect any constant data the atom carries
            data = {}
            # Common: check for .axis, .keepdims, .key (for indexing), etc.
            if hasattr(expr, "axis") and expr.axis is not None:
                data["axis"] = expr.axis
            if hasattr(expr, "keepdims"):
                data["keepdims"] = expr.keepdims

            return IRAffineOp(
                op_type=op_type,
                args=child_irs,
                data=data,
                shape=expr.shape,
            )

        # Fallback: try to evaluate numerically
        try:
            val = np.array(expr.value)
            return IRConst(value=val, shape=val.shape)
        except Exception:
            raise ValueError(f"Cannot walk expression of type {type(expr)}: {expr}")

    def walk_constraint(self, constr) -> IRConstraint:
        """Convert a CVXPY constraint to IR."""
        if isinstance(constr, Zero):
            expr_ir = self.walk_expr(constr.expr)
            return IRConstraint(
                cone_type="zero",
                expr=expr_ir,
                size=constr.size,
            )

        if isinstance(constr, (NonNeg, NonPos)):
            expr_ir = self.walk_expr(constr.expr)
            cone = "nonneg" if isinstance(constr, NonNeg) else "nonpos"
            return IRConstraint(
                cone_type=cone,
                expr=expr_ir,
                size=constr.size,
            )

        if isinstance(constr, SOC):
            arg_irs = [self.walk_expr(arg) for arg in constr.args]
            return IRConstraint(
                cone_type="soc",
                expr=arg_irs[0],
                args=arg_irs,
                size=constr.size,
            )

        # Fallback: treat as generic constraint, walk the expression
        expr_ir = self.walk_expr(constr.expr)
        return IRConstraint(
            cone_type="unknown",
            expr=expr_ir,
            size=constr.size,
        )


def walk(problem: cvxpy.Problem) -> IRProblem:
    """Convenience function: walk a CVXPY Problem and produce IR."""
    walker = TreeWalker()
    return walker.walk_problem(problem)
