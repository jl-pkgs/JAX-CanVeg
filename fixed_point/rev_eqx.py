from functools import partial
from jax import grad, jvp, vjp
from jax.lax import while_loop
import jax.numpy as jnp
import equinox as eqx


## ==============================================================
@eqx.filter_custom_vjp
def fixed_point_reverse(vjp_arg, f):
  """
  vjp_arg: (a, x_guess) 元组，包含所有需要求梯度的参数
  f: 迭代函数 (非微分参数)
  """
  a, x_guess = vjp_arg
  
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.abs(x_prev - x) > 1e-6

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
  return x_star


@fixed_point_reverse.def_fwd
def fixed_point_fwd(perturbed, vjp_arg, f):
  a, x_guess = vjp_arg
  x_star = fixed_point_reverse(vjp_arg, f)
  
  # residuals: 反向传播需要的中间值
  residuals = (a, x_star, f)
  return x_star, residuals


@fixed_point_reverse.def_bwd
def fixed_point_rev(residuals, grad_out, perturbed, vjp_arg, f):
  a, x_guess = vjp_arg
  a_perturbed, x_guess_perturbed = perturbed
  
  a_saved, x_star, f_saved = residuals
  x_star_bar = grad_out
  
  # 求解 u_star
  def rev_iter_fn(packed, u):
    a_val, x_star_val = packed
    _, vjp_x = vjp(lambda x: f_saved(a_val, x), x_star_val)
    return u + vjp_x(u)[0]
  
  def solve_u_star(packed, u_init):
    def cond_fun(carry):
      u_prev, u = carry
      max_diff = jnp.max(jnp.abs(u_prev - u))
      return max_diff > 1e-6
    
    def body_fun(carry):
      _, u = carry
      return u, rev_iter_fn(packed, u)
    
    _, u_star = while_loop(cond_fun, body_fun, (u_init, rev_iter_fn(packed, u_init)))
    return u_star
  
  packed = (a_saved, x_star)
  u_star = solve_u_star(packed, x_star_bar)
  
  # 计算对 a 的梯度（只在 a 被微分时计算）
  if a_perturbed:
    _, vjp_a = vjp(lambda a_val: f_saved(a_val, x_star), a_saved)
    a_bar, = vjp_a(u_star)
  else:
    a_bar = None
  
  # x_guess 的梯度为零（因为它只是初始猜测）
  x_guess_bar = None if not x_guess_perturbed else jnp.zeros_like(x_guess)
  
  return (a_bar, x_guess_bar)


def newton_sqrt_reverse(a):
  update = lambda a, x: 0.5 * (x + a / x)
  return fixed_point_reverse((a, a), update)  # 将 a 和 x_guess 打包成元组


# 测试
print(grad(newton_sqrt_reverse)(2.))
