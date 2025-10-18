from functools import partial
import jax
import jax.numpy as jnp
from jax import grad, jvp, jacfwd, custom_jvp
from jax.lax import while_loop
from jax.flatten_util import ravel_pytree
import lineax as lx


@partial(custom_jvp, nondiff_argnums=(0,))  # 将第一个参数 f 设为静态（不参与微分）
def fixed_point_forward(f, a, x_guess):
    # max_iters, tol
    tol = 1e-6
    """前向模式的固定点求解器"""
    def cond_fun(carry):
        i, x_prev, x = carry
        return jnp.linalg.norm(x_prev - x) > tol
    
    def body_fun(carry):
        i, x_prev, x = carry
        return i + 1, x, f(a, x)
    
    _, _, x_star = while_loop(cond_fun, body_fun, (0, x_guess, f(a, x_guess)))
    return x_star


@fixed_point_forward.defjvp
def fixed_point_forward_jvp(f, primals, tangents):
    """前向模式的JVP规则"""
    a, x_guess = primals
    a_dot, x_guess_dot = tangents
    
    # 1. 计算固定点
    x_star = fixed_point_forward(f, a, x_guess)
    
    # 2. 在固定点处构造“扁平化”的函数，确保雅可比是( N, N )二维矩阵
    #    同时将右端项 ∂F/∂a · a_dot 扁平化到向量形状
    x_flat, unravel = ravel_pytree(x_star)

    def F_flat_from_x(x_flat_vec):
        x_tree = unravel(x_flat_vec)
        y_tree = f(a, x_tree)
        y_flat, _ = ravel_pytree(y_tree)
        return y_flat

    def F_flat_from_a(a_param):
        y_tree = f(a_param, x_star)
        y_flat, _ = ravel_pytree(y_tree)
        return y_flat

    # 3. 计算雅可比矩阵 J_x (N,N) 与右端项 F_a_dot_flat (N,)
    J_x = jacfwd(F_flat_from_x)(x_flat)                   # pdv(f, x), [N, N]
    _, F_a_dot_flat = jvp(F_flat_from_a, (a,), (a_dot,))  # b = pdv(F, a) v, (N,)

    # 4. 构造线性系统: (I - J_x) x_dot_flat = F_a_dot_flat
    N = x_flat.shape[0]
    I = jnp.eye(N, dtype=J_x.dtype)
    A_mat = I - J_x

    # 5. 使用 lineax 求解（允许病态情况），得到 x_dot_flat 并还原树形
    x_dot_flat = lx.linear_solve(
        lx.MatrixLinearOperator(A_mat),
        F_a_dot_flat,
        solver=lx.AutoLinearSolver(well_posed=False),
    ).value # pdv(x, w) v
    x_star_dot = unravel(x_dot_flat)
    return x_star, x_star_dot

# 示例1: 牛顿法求平方根
def newton_sqrt_forward(a):
  update = lambda a, x: 0.5 * (x + a / x)
  return fixed_point_forward(update, a, x_guess = a)

# %timeit
print(grad(newton_sqrt_forward)(2.))
