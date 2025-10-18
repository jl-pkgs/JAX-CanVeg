# 逆向速度慢2倍
import jax
from jax import grad, vjp, jacrev, jvp, jacfwd
import jax.numpy as jnp

import equinox as eqx
from typing import List
import lineax as lx
from .fixed_point import fixed_point


def fixed_point_reverse(
    states_guess,
    para,
    args,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter: int,
):
    """
    Reverse-mode implicit differentiation for fixed-point solver.

    Only static/constant values (functions, niter) are captured in closure.
    All traced values (states_guess, para, args) are passed as vjp arguments.
    """
    @eqx.filter_custom_vjp
    def _core_vjp(vjp_arg):
        """Core VJP function with only differentiable args."""
        states_guess, para, args_in = vjp_arg
        states_solution = fixed_point(iter_func, states_guess, para, niter, *args_in)
        substates_solution = get_substates_func(states_solution)
        return substates_solution

    @_core_vjp.def_fwd
    def _fwd(perturbed, vjp_arg):
        """Forward pass: compute output and save residuals."""
        states_guess, para, args_in = vjp_arg
        states_final = fixed_point(iter_func, states_guess, para, niter, *args_in)
        substates_final = get_substates_func(states_final)
        # Don't save states_final to avoid tracer leaks; save only non-traced values
        residuals = (para, states_guess, args_in)
        return substates_final, residuals

    @_core_vjp.def_bwd
    def _bwd(residuals, grad_out, perturbed, vjp_arg):
        """Backward pass: compute gradients via implicit differentiation."""
        states_guess, para, args_in = vjp_arg
        states_guess_perturbed, para_perturbed, args_perturbed = perturbed

        # Unpack residuals and recompute states_final to avoid tracer leaks
        para_saved, states_guess_saved, args_saved = residuals
        states_final = fixed_point(
            iter_func, states_guess_saved, para_saved, niter, *args_saved
        )
        substates_final = get_substates_func(states_final)
        v_bar = grad_out

        def each_iteration_state(substates):
            """F(substates) = g(f(update(states_final, substates)))"""
            states1 = update_substates_func(states_final, substates)
            states2 = iter_func(states1, para_saved, *args_saved)
            substates2 = get_substates_func(states2)
            return substates2

        # Solve (I - J^T) @ u_bar = v_bar using VJP-based matvec
        def J_T_matvec(v_in):
            _, vjp_func = jax.vjp(
                lambda substates: each_iteration_state(substates), substates_final
            )
            (J_T_v,) = vjp_func(v_in)
            return J_T_v

        J_T_op = lx.FunctionLinearOperator(J_T_matvec, jax.eval_shape(lambda: v_bar))
        I = lx.IdentityLinearOperator(J_T_op.in_structure())
        A_T = I - J_T_op
        u_bar = lx.linear_solve(
            A_T, v_bar, solver=lx.AutoLinearSolver(well_posed=False)
        ).value

        # Compute gradient wrt para
        para_bar = None
        if para_perturbed:

            def each_iteration_para(para_):
                """F(para) = g(f(states_final, para))"""
                states2 = iter_func(states_final, para_, *args_saved)
                substates2 = get_substates_func(states2)
                return substates2

            _, vjp_para = vjp(each_iteration_para, para_saved)
            (para_bar,) = vjp_para(u_bar)

        # Gradient wrt states_guess: create zero-structure matching states_guess
        # (converged fixed point is independent of initial guess)
        if states_guess_perturbed:
            states_guess_bar = jax.tree_util.tree_map(lambda _: None, states_guess)
        else:
            states_guess_bar = None

        # Gradient wrt args: always None (non-differentiable static parameters)
        # Must return structure matching vjp_arg
        if args_perturbed:
            args_bar = jax.tree_util.tree_map(lambda _: None, args_in)
        else:
            args_bar = None

        return (states_guess_bar, para_bar, args_bar)

    return _core_vjp((states_guess, para, args))
