import jax
import equinox as eqx
import jax.numpy as jnp
from typing import List


class Para(eqx.Module):
    """使用 Equinox Module，自动支持 PyTree 和 hashable"""
    weight: jnp.ndarray
    bias: jnp.ndarray


@eqx.filter_jit
def fixed_point(
    func,
    states_initial: List,
    para: Para,
    niter: int,
    *args,
):  
    def iteration(c, i):
        cnew = func(c, para, *args)
        return cnew, None

    states_final, _ = jax.lax.scan(iteration, states_initial, xs=None, length=niter)
    return states_final


# ============================================================================
# 2. 定义简单的固定点迭代函数
# ============================================================================
def simple_iter_func(states, para: Para):
    """
    简单的固定点迭代：x_new = 0.5 * x + para.bias
    固定点解析解：x* = 2 * para.bias
    """
    x = states[0]
    x_new = 0.5 * x + para.bias
    return [x_new]


# ============================================================================
# 3. 定义辅助函数
# ============================================================================
def get_substates_func(states: List) -> jnp.ndarray:
    """提取我们关心的子状态"""
    return states[0]


def update_substates_func(states: List, substates: jnp.ndarray) -> List:
    """用新的子状态更新完整状态"""
    return [substates]

