from functools import partial
from jax import grad, jvp, vjp, custom_vjp
from jax.lax import while_loop
import jax.numpy as jnp


## ==============================================================
@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point_reverse(f, a, x_guess):
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.abs(x_prev - x) > 1e-6

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
  return x_star


def fixed_point_fwd(f, a, x_init):
  x_star = fixed_point_reverse(f, a, x_init)
  return x_star, (a, x_star)


def rev_iter(f, packed, u):
  a, x_star, x_star_bar = packed
  _, vjp_x = vjp(lambda x: f(a, x), x_star) # pdv(f_i, x)
  return x_star_bar + vjp_x(u)[0] # u_star

def fixed_point_rev(f, res, x_star_bar):
  a, x_star = res # 已知最终的结果x_star
  _, vjp_a = vjp(lambda a: f(a, x_star), a) # u_star * pdv(f, a)
  
  packed = (a, x_star, x_star_bar)
  cal_ustar = lambda packed, u: rev_iter(f, packed, u)
  # cal_ustar = partial(rev_iter, f)
  a_bar, = vjp_a(fixed_point_reverse(cal_ustar, packed, x_star_bar))
  return a_bar, jnp.zeros_like(x_star)

fixed_point_reverse.defvjp(fixed_point_fwd, fixed_point_rev)



def newton_sqrt_reverse(a):
  update = lambda a, x: 0.5 * (x + a / x)
  return fixed_point_reverse(update, a, x_guess = a)


# %%
# %timeit 
print(grad(newton_sqrt_reverse)(2.))
