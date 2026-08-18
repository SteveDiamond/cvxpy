"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT


class GUROBI(ConicSolver):
    """
    An interface for the Gurobi solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  4: s.INFEASIBLE_OR_UNBOUNDED,  # Triggers reoptimize.
                  5: s.UNBOUNDED,
                  6: s.SOLVER_ERROR,
                  7: s.USER_LIMIT, # ITERATION_LIMIT
                  8: s.USER_LIMIT, # NODE_LIMIT
                  9: s.USER_LIMIT,  # TIME_LIMIT
                  10: s.USER_LIMIT, # SOLUTION_LIMIT
                  11: s.USER_LIMIT, # INTERRUPTED
                  12: s.SOLVER_ERROR, # NUMERIC
                  13: s.USER_LIMIT, # SUBOPTIMAL
                  14: s.USER_LIMIT, # INPROGRESS
                  15: s.USER_LIMIT, # USER_OBJ_LIMIT
                  16: s.USER_LIMIT, # WORK_LIMIT
                  17: s.USER_LIMIT} # MEM_LIMIT

    def name(self):
        """The name of the solver.
        """
        return s.GUROBI

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import gurobipy  # noqa F401

    def accepts(self, problem) -> bool:
        """Can Gurobi solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        import gurobipy as grb
        data, inv_data = super(GUROBI, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        # Add initial guess.
        data['init_value'] = utilities.stack_vals(problem.variables, grb.GRB.UNDEFINED)

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        bar_iter_count = getattr(solution["model"], "BarIterCount", 0)

        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME],
                s.NUM_ITERS: bar_iter_count}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[GUROBI.VAR_ID]: solution['primal']}
            if "eq_dual" in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[GUROBI.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[GUROBI.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import gurobipy

        c = data[s.C]
        b = data[s.B]
        A = sp.csr_array(data[s.A])
        dims = dims_to_solver_dict(data[s.DIMS])
        lb = data[s.LOWER_BOUNDS]
        ub = data[s.UPPER_BOUNDS]

        n = c.shape[0]
        if lb is None:
            lb = np.full(n, -gurobipy.GRB.INFINITY)
        if ub is None:
            ub = np.full(n, gurobipy.GRB.INFINITY)

        # Create a new model
        if 'env' in solver_opts:
            # Specifies environment to create Gurobi model for control over licensing and parameters
            # https://www.gurobi.com/documentation/9.1/refman/environments.html
            default_env = solver_opts['env']
            del solver_opts['env']
            model = gurobipy.Model(env=default_env)
        else:
            # Create Gurobi model using default (unspecified) environment
            model = gurobipy.Model()

        # Pass through verbosity
        model.setParam("OutputFlag", verbose)

        # Add variables in bulk. Checking membership in BOOL_IDX and INT_IDX for
        # every variable makes model construction quadratic for mixed-integer
        # problems, while repeated addVar calls add substantial Python overhead.
        vtypes = [gurobipy.GRB.CONTINUOUS] * n
        for idx in data[s.BOOL_IDX]:
            vtypes[idx] = gurobipy.GRB.BINARY
        for idx in data[s.INT_IDX]:
            vtypes[idx] = gurobipy.GRB.INTEGER
        variable_names = [f"x_{i}" for i in range(n)]
        variables = list(model.addVars(
            n,
            obj=c,
            name=variable_names,
            vtype=vtypes,
            lb=lb,
            ub=ub,
        ).values())
        model.update()

        # Set the start value of Gurobi vars to user provided values.
        if warm_start and solver_cache is not None \
                and self.name() in solver_cache:
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status,
                                             s.SOLVER_ERROR)
            if (old_status in s.SOLUTION_PRESENT) or (old_model.solCount > 0):
                old_x = old_model.getVars()[:n]
                start = old_model.getAttr("X", old_x)
                model.setAttr("Start", variables, start)
        elif warm_start:
            model.setAttr("Start", variables, data['init_value'])

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        if hasattr(model, 'addMConstr'):
            # Code path for Gurobi v10.0-
            eq_constrs = []
            if leq_start > 0:
                eq_constrs = model.addMConstr(
                    A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start]
                ).tolist()
            ineq_constrs = []
            if leq_end > leq_start:
                ineq_constrs = model.addMConstr(
                    A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL,
                    b[leq_start:leq_end]).tolist()
        elif hasattr(model, 'addMConstrs'):
            # Code path for Gurobi v9.0-v9.5
            eq_constrs = []
            if leq_start > 0:
                eq_constrs = model.addMConstrs(
                    A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start])
            ineq_constrs = []
            if leq_end > leq_start:
                ineq_constrs = model.addMConstrs(
                    A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL,
                    b[leq_start:leq_end])
        else:
            eq_constrs = self.add_model_lin_constr(model, variables,
                                                   range(dims[s.EQ_DIM]),
                                                   gurobipy.GRB.EQUAL,
                                                   A, b)
            ineq_constrs = self.add_model_lin_constr(model, variables,
                                                     range(leq_start, leq_end),
                                                     gurobipy.GRB.LESS_EQUAL,
                                                     A, b)

        soc_constrs, new_leq_constrs, soc_variables = self.add_model_soc_constrs(
            model, variables, dims[s.SOC_DIM], leq_end, A, b
        )
        variables += soc_variables

        # Save file (*.mst, *.sol, ect.)
        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])

        # Set parameters
        # TODO user option to not compute duals.
        model.setParam("QCPDual", True)
        for key, value in solver_opts.items():
            # Ignore arguments unique to the CVXPY interface.
            if key not in self.INTERFACE_ARGS:
                model.setParam(key, value)

        solution = {}
        try:
            model.optimize()
            if model.Status == 4 and solver_opts.get('reoptimize', False):
                # INF_OR_UNBD. Solve again to get a definitive answer.
                model.setParam("DualReductions", 0)
                model.optimize()
            solution["value"] = model.ObjVal
            solution["primal"] = np.array(model.getAttr("X", variables))

            # Only add duals if not a MIP.
            # Not sure why we need to negate the following,
            # but need to in order to be consistent with other solvers.
            vals = []
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                lin_constrs = eq_constrs + ineq_constrs + new_leq_constrs
                vals += model.getAttr('Pi', lin_constrs)
                vals += model.getAttr('QCPi', soc_constrs)
                solution["y"] = -np.array(vals)
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]
        except Exception:
            pass
        solution[s.SOLVE_TIME] = model.Runtime
        solution["status"] = self.STATUS_MAP.get(model.Status,
                                                 s.SOLVER_ERROR)
        if solution["status"] == s.SOLVER_ERROR and model.SolCount:
            solution["status"] = s.OPTIMAL_INACCURATE
        if solution["status"] == s.USER_LIMIT and not model.SolCount:
            solution["status"] = s.INFEASIBLE_INACCURATE
        solution["model"] = model

        # Save model for warm start.
        if solver_cache is not None:
            solver_cache[self.name()] = model

        return solution

    def add_model_lin_constr(self, model, variables,
                             rows, ctype,
                             mat, vec):
        """Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : GUROBI model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        ctype : GUROBI constraint type
            The type of constraint.
        mat : SciPy CSR matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        list
            A list of constraints.
        """
        import gurobipy as gp

        constr = []
        for i in rows:
            start = mat.indptr[i]
            end = mat.indptr[i + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = mat.data[start:end]
            expr = gp.LinExpr(coeff, x)
            constr.append(model.addLConstr(expr, ctype, vec[i]))
        return constr

    def add_model_soc_constrs(self, model, variables, cone_dims,
                              row_start, mat, vec):
        """Adds all SOC constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : GUROBI model
            The problem model.
        variables : list
            The problem variables.
        cone_dims : list
            The dimensions of the second-order cones.
        row_start : int
            The first row of ``mat`` belonging to a second-order cone.
        mat : SciPy CSR matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        tuple
            A tuple of (list of QConstr, list of Constr, and list of variables).
        """
        import gurobipy as gp

        num_soc_rows = sum(cone_dims)
        if num_soc_rows == 0:
            return [], [], []

        # Introduce one auxiliary variable for each affine SOC row. The first
        # variable in every cone is nonnegative; the remaining variables are free.
        soc_lb = np.full(num_soc_rows, -gp.GRB.INFINITY)
        soc_names = []
        cone_start = 0
        for cone_dim in cone_dims:
            soc_lb[cone_start] = 0
            matrix_row = row_start + cone_start
            soc_names.append(f"soc_t_{matrix_row}")
            soc_names.extend(
                f"soc_x_{row}" for row in range(matrix_row + 1, matrix_row + cone_dim)
            )
            cone_start += cone_dim

        soc_vars = list(model.addVars(
            num_soc_rows,
            obj=0,
            name=soc_names,
            vtype=gp.GRB.CONTINUOUS,
            lb=soc_lb,
            ub=gp.GRB.INFINITY,
        ).values())
        model.update()

        # If y denotes the auxiliary variables, add A_soc @ x + y == b_soc
        # in one matrix operation instead of constructing one LinExpr per row.
        row_end = row_start + num_soc_rows
        soc_matrix = sp.hstack(
            (mat[row_start:row_end, :], sp.eye(num_soc_rows, format="csr")),
            format="csr",
        )
        if hasattr(model, "addMConstr"):
            new_lin_constrs = model.addMConstr(
                soc_matrix, None, gp.GRB.EQUAL, vec[row_start:row_end]
            ).tolist()
        elif hasattr(model, "addMConstrs"):
            new_lin_constrs = model.addMConstrs(
                soc_matrix, None, gp.GRB.EQUAL, vec[row_start:row_end]
            )
        else:
            all_variables = variables + soc_vars
            new_lin_constrs = self.add_model_lin_constr(
                model,
                all_variables,
                range(num_soc_rows),
                gp.GRB.EQUAL,
                soc_matrix,
                vec[row_start:row_end],
            )

        # Gurobi has no bulk API for independent quadratic constraints.
        soc_constrs = []
        cone_start = 0
        quadratic_coeffs = {}
        for cone_dim in cone_dims:
            cone_end = cone_start + cone_dim
            cone_vars = soc_vars[cone_start:cone_end]
            coeffs = quadratic_coeffs.get(cone_dim)
            if coeffs is None:
                coeffs = np.concatenate(([-1.0], np.ones(cone_dim - 1)))
                quadratic_coeffs[cone_dim] = coeffs
            expr = gp.QuadExpr()
            expr.addTerms(coeffs, cone_vars, cone_vars)
            soc_constrs.append(model.addQConstr(expr <= 0))
            cone_start = cone_end

        return soc_constrs, new_lin_constrs, soc_vars

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["GUROBI"]
