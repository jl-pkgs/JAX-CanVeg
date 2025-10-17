import jax
from jax import grad, vjp, jacrev, jvp, jacfwd
import jax.numpy as jnp

import equinox as eqx
from typing import List
import lineax as lx
from case import Para, fixed_point, simple_iter_func, get_substates_func, update_substates_func, simple_iter_func



@eqx.filter_custom_jvp
def implicit_func_jvp(
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


@implicit_func_jvp.def_jvp
def implicit_func_jvp_rule(
    primals,
    tangents,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter,
):
    states_guess, para, args = primals[0], primals[1], primals[2]
    t_states_guess, t_para, t_args = tangents[0], tangents[1], tangents[2]

    # 前向计算
    states_final = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_final = get_substates_func(states_final)

    # 如果 para 没有被微分，直接返回零切向量
    if t_para is None:
        tangent_out = jnp.zeros_like(substates_final)
        return substates_final, tangent_out

    # 定义关于 para 的迭代函数
    def each_iteration_para(para_):
        states2 = iter_func(states_final, para_, *args)
        substates2 = get_substates_func(states2)
        return substates2

    # 定义关于 substates 的迭代函数
    def each_iteration_state(substates):
        states1 = update_substates_func(states_final, substates)
        states2 = iter_func(states1, para, *args)
        substates2 = get_substates_func(states2)
        return substates2

    # 计算 u = ∂F/∂para · t_para
    _, u = jvp(each_iteration_para, (para,), (t_para,))
    
    # 计算 Jacobian J = ∂F/∂substates
    _J = jacfwd(each_iteration_state, argnums=0)(substates_final)
    J = lx.PyTreeLinearOperator(_J, jax.eval_shape(lambda: u))
    I = lx.IdentityLinearOperator(J.in_structure())
    
    # 求解 (I - J) @ tangent_out = u
    A = I - J
    tangent_out = lx.linear_solve(
        A, u, solver=lx.AutoLinearSolver(well_posed=False)
    ).value
    
    return substates_final, tangent_out



@eqx.filter_custom_vjp
def implicit_func_fixed_point(
    vjp_arg,  # 第一个参数：需要求梯度的对象
    args,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter: int,
):
    """
    vjp_arg: (states_guess, para) - 需要求梯度的参数
    args: 其他参数（不可微）
    """
    states_guess, para = vjp_arg
    
    states_solution = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_solution = get_substates_func(states_solution)
    return substates_solution


@implicit_func_fixed_point.def_fwd
def implicit_func_fixed_point_fwd(
    perturbed,
    vjp_arg,
    args,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter,
):
    """前向传播：计算输出并保存 residuals"""
    states_guess, para = vjp_arg
    
    # 计算前向传播结果
    states_final = fixed_point(iter_func, states_guess, para, niter, *args)
    substates_final = get_substates_func(states_final)
    
    # 保存反向传播需要的中间值
    residuals = (states_final, para, args)
    
    return substates_final, residuals


@implicit_func_fixed_point.def_bwd
def implicit_func_fixed_point_bwd(
    residuals,
    grad_out,  # substates 的梯度（cotangent）
    perturbed,
    vjp_arg,
    args,
    *,
    iter_func,
    update_substates_func,
    get_substates_func,
    niter,
):
    """反向传播：计算梯度"""
    states_guess, para = vjp_arg
    states_guess_perturbed, para_perturbed = perturbed
    
    states_final, para_saved, args_saved = residuals
    substates_final = get_substates_func(states_final)
    
    # v_bar 是 substates 的 cotangent
    v_bar = grad_out
    
    # ============================================
    # 步骤 1: 求解隐式方程得到 u_bar (substates 的伴随变量)
    # 满足: u_bar = J^T @ u_bar + v_bar
    # 即: (I - J^T) @ u_bar = v_bar
    # ============================================
    
    def each_iteration_state(substates):
        """计算 F(substates) = g(f(update(states_final, substates)))"""
        states1 = update_substates_func(states_final, substates)
        states2 = iter_func(states1, para_saved, *args_saved)
        substates2 = get_substates_func(states2)
        return substates2
    
    # 计算 Jacobian J = ∂F/∂substates (使用反向模式更高效)
    _J = jacrev(each_iteration_state, argnums=0)(substates_final)
    J_T = lx.PyTreeLinearOperator(_J, jax.eval_shape(lambda: v_bar))
    I = lx.IdentityLinearOperator(J_T.in_structure())
    
    # 求解 (I - J^T) @ u_bar = v_bar
    A_T = I - J_T
    u_bar = lx.linear_solve(
        A_T, v_bar, solver=lx.AutoLinearSolver(well_posed=False)
    ).value
    
    # ============================================
    # 步骤 2: 计算对 para 的梯度
    # grad_para = ∂F/∂para^T @ u_bar
    # ============================================
    para_bar = None
    if para_perturbed:
        def each_iteration_para(para_):
            """计算 F(para) = g(f(states_final, para))"""
            states2 = iter_func(states_final, para_, *args_saved)
            substates2 = get_substates_func(states2)
            return substates2
        
        # 使用 vjp 计算 ∂F/∂para^T @ u_bar
        _, vjp_para = vjp(each_iteration_para, para_saved)
        para_bar, = vjp_para(u_bar)
    
    # ============================================
    # 步骤 3: 计算对 states_guess 的梯度
    # 通常 states_guess 只是初始猜测，不影响最终梯度
    # ============================================
    states_guess_bar = None
    if states_guess_perturbed:
        # 如果需要计算 states_guess 的梯度，通常设为零
        # 因为不动点求解的结果不依赖于初始猜测（收敛后）
        states_guess_bar = jax.tree_map(jnp.zeros_like, states_guess)
    
    return (states_guess_bar, para_bar)


# ============================================
# 使用示例
# ============================================
def example_usage():
    """使用示例"""
    # 假设你有这些函数定义
    # iter_func, update_substates_func, get_substates_func
    # 创建测试数据
    para = Para(
        weight=jnp.array(0.5),
        bias=jnp.array(3.0)
    )
    states_guess = [jnp.array(0.0)]
    args = []
    niter = 50

    # ========================================
    # 测试 1: JVP 版本
    # ========================================
    print("\n" + "-" * 70)
    print("测试 1: JVP 版本")
    print("-" * 70)
    
    result_jvp = implicit_func_jvp(
        states_guess, para, args,
        iter_func=simple_iter_func,
        update_substates_func=update_substates_func,
        get_substates_func=get_substates_func,
        niter=niter,
    )
    print(f"JVP 计算结果: {result_jvp}")

    # ========================================
    # 测试 2: VJP 版本
    # ========================================
    print("\n" + "-" * 70)
    print("测试 2: VJP 版本")
    print("-" * 70)
    
    result_vjp = implicit_func_fixed_point(
        (states_guess, para), args,
        iter_func=simple_iter_func,
        update_substates_func=update_substates_func,
        get_substates_func=get_substates_func,
        niter=niter,
    )
    print(f"VJP 计算结果: {result_vjp}")


example_usage()
