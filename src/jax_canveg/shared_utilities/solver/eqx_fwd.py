import jax
import jax.numpy as jnp
from jax import jvp

import equinox as eqx
import lineax as lx

from ...subjects import Para
from .fixed_point import fixed_point
from typing import Callable, List


@eqx.filter_custom_jvp
def fixed_point_forward(
    states_guess: List,
    para: Para,
    args: List,
    *,
    iter_func: Callable,
    update_substates_func: Callable,
    get_substates_func: Callable,
    niter: int,
):
    # jax.debug.print("Iterations: {i}", i=niter)
    states_solution = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_solution = get_substates_func(states_solution)
    return substates_solution


# @fixed_point_forward.def_jvp
@fixed_point_forward.def_jvp
def fixed_point_forward_jvp(
    # iter_func, update_substates_func, get_substate_func, niter, primals, tangents
    primals,
    tangents,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter,
):
    # states_guess, para, niter, args = primals[0], primals[1], primals[2], primals[3:]
    # states_guess, para, args = primals[0], primals[1], primals[2:]
    states_guess, para, args = primals[0], primals[1], primals[2]

    states_final = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_final = get_substates_func(states_final)

    v = recover_tangent(para, tangents[1])  # 这个变量为何理解, tan_para
    # jax.debug.print("debug para: {x}", x=para)
    # jax.debug.print("debug tan_para: {x}", x=tan_para)

    def each_iteration_para(para):
        states2 = iter_func(states_final, para, *args)
        substates2 = get_substates_func(states2)
        return substates2

    def each_iteration_state(substates):
        states1 = update_substates_func(states_final, substates)
        states2 = iter_func(states1, para, *args)
        substates2 = get_substates_func(states2)
        return substates2

    # Compute the Jacobian and the vectors
    # a -> w -> para, x -> state
    _, b = jvp(each_iteration_para, (para,), (v,))  # pdv(f, a) v

    def matvec(v_in):
        # 计算 J @ v_in，使用JVP而不是显式Jacobian
        _, Jv = jvp(each_iteration_state, (substates_final,), (v_in,))
        return Jv

    # 这里提前定义了一个矩阵乘法运算，about 2times faster
    J_op = lx.FunctionLinearOperator(matvec, jax.eval_shape(lambda: substates_final))

    I = lx.IdentityLinearOperator(J_op.in_structure())
    A = I - J_op  # (I - J)·v = v - J·v

    # _J = jacfwd(each_iteration_state, argnums=0)(substates_final)  # pdv(f, x)
    # J = lx.PyTreeLinearOperator(_J, jax.eval_shape(lambda: b)) # why ?
    # I = lx.IdentityLinearOperator(J.in_structure())  #
    # A = I - J

    # [ I - pdv(f, x) ] [pdv(x, w) v] = pdv(f, a) v,   (I - J) [pdv(x, w) v] = b
    tangent_out = lx.linear_solve(
        A, b, solver=lx.AutoLinearSolver(well_posed=False)
    ).value
    # tangent_out = lx.linear_solve(A, u, solver=lx.SVD()).value
    return substates_final, tangent_out


def recover_tangent(para, t_para):
    # 将 tangent 中的 None 转换为零数组, v 和 para 的树结构就匹配了
    return jax.tree.map(
        lambda p, t: jnp.zeros_like(p) if t is None else t,
        para,
        t_para,
        is_leaf=lambda x: x is None,
    )


# # @partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13))
# @eqx.filter_custom_jvp
# def fixed_point_forward_canveg_main(
#     iter_func: Callable,
#     update_substates_func: Callable,
#     get_substate_func: Callable,
#     # niter: int,
#     states_guess: List,
#     para: Para,
#     niter: int,
#     dij: Float_2D,
#     leaf_ang: LeafAng,
#     quantum: ParNir,
#     nir: ParNir,
#     lai: Lai,
#     n_can_layers: int,
#     stomata: int,
#     soil_mtime: int,
#     # *args,
# ):
#     """This is implicit function theorem implementation to canveg main iteration
#        to account for the variables/inputs needed to run the iteration.

#     Args:
#         iter_func (Callable): _description_
#         update_substates_func (Callable): _description_
#         get_substate_func (Callable): _description_
#         states_guess (List): _description_
#         para (Para): _description_
#         niter (int): _description_
#         n_can_layers (int): _description_
#         stomata (int): _description_
#         soil_mtime (int): _description_
#         dij (Float_2D): _description_
#         leaf_ang (LeafAng): _description_
#         quantum (ParNir): _description_
#         nir (ParNir): _description_
#         lai (Lai): _description_

#     Returns:
#         _type_: _description_
#     """
#     # jax.debug.print("Iterations: {i}", i=niter)
#     args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
#     states_solution = fixed_point(iter_func, states_guess, para, niter, *args)
#     substates_solution = get_substate_func(states_solution)
#     return substates_solution


# @fixed_point_forward_canveg_main.def_jvp
# def fixed_point_forward_canveg_main_jvp(
#     # iter_func, update_substates_func, get_substate_func, primals, tangents
#     iter_func,
#     update_substates_func,
#     get_substate_func,
#     niter,
#     dij,
#     leaf_ang,
#     quantum,
#     nir,
#     lai,
#     n_can_layers,
#     stomata,
#     soil_mtime,
#     primals,
#     tangents,
# ):
#     states_guess, para = primals[0], primals[1]
#     tan_para = tangents[1]

#     args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]

#     states_final = fixed_point(iter_func, states_guess, para, niter, *args)
#     substates_final = get_substate_func(states_final)

#     def each_iteration_para(para):
#         states2 = iter_func(states_final, para, *args)
#         substates2 = get_substate_func(states2)
#         return substates2

#     def each_iteration_state(substates):
#         states1 = update_substates_func(states_final, substates)
#         states2 = iter_func(states1, para, *args)
#         substates2 = get_substate_func(states2)
#         return substates2

#     # Compute the Jacobian and the vectors
#     _, u = jvp(each_iteration_para, (para,), (tan_para,), has_aux=False)
#     Jacobian_JAX = jacfwd(each_iteration_state, argnums=0, has_aux=False)
#     J = Jacobian_JAX(substates_final)
#     J = lx.PyTreeLinearOperator(J, jax.eval_shape(lambda: u))
#     I = lx.IdentityLinearOperator(J.in_structure())  # noqa: E741
#     A = I - J
#     tangent_out = lx.linear_solve(A, u).value
#     return substates_final, tangent_out
