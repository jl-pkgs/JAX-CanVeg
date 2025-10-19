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

        def each_iteration_para(para_):
            """F(para) = g(f(states_final, para))"""
            states2 = iter_func(states_final, para_, *args_saved)
            substates2 = get_substates_func(states2)
            return substates2

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

        # Compute gradient wrt para (only if para is being differentiated)
        if para_perturbed:
            _, vjp_para = vjp(each_iteration_para, para_saved)
            (para_bar,) = vjp_para(u_bar)
        else:
            para_bar = zeros_like_pytree(para)

        # state和 args 不是影响loss的主要原因
        states_guess_bar = zeros_like_pytree(states_guess)
        args_bar = zeros_like_pytree(args_in)

        return (states_guess_bar, para_bar, args_bar)

    return _core_vjp((states_guess, para, args))



def zeros_like_pytree(pytree):
    """Create a pytree of zeros with the same structure and shapes as the input pytree."""
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else None, pytree
    )
