import jax
from jax import grad, jvp, jacfwd

import equinox as eqx
from typing import List
import lineax as lx
from case import Para, fixed_point


@eqx.filter_custom_jvp
def implicit_func_fixed_point(
    states_guess: List,
    para: Para,
    args: List,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter: int,
):
    states_solution = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_solution = get_substates_func(states_solution)
    return substates_solution


@implicit_func_fixed_point.defjvp
def implicit_func_fixed_point_jvp(
    primals,
    tangents,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter,
):
    states_guess, para, args = primals[0], primals[1], primals[2]
    v = tangents[1]

    states_final = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_final = get_substates_func(states_final)

    def each_iteration_para(para):
        states2 = iter_func(states_final, para, *args)
        substates2 = get_substates_func(states2)
        return substates2

    def each_iteration_state(substates):
        states1 = update_substates_func(states_final, substates)
        states2 = iter_func(states1, para, *args)
        substates2 = get_substates_func(states2)
        return substates2

    
    _, u = jvp(each_iteration_para, (para,), (v,))
    _J = jacfwd(each_iteration_state, argnums=0)(substates_final)
    J = lx.PyTreeLinearOperator(_J, jax.eval_shape(lambda: u))
    I = lx.IdentityLinearOperator(J.in_structure())
    
    A = I - J
    tangent_out = lx.linear_solve(A, u, solver=lx.AutoLinearSolver(well_posed=False)).value
    return substates_final, tangent_out
