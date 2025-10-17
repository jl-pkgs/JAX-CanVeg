import jax
import jax.numpy as jnp
from jax import grad, jvp
from typing import List
from forward_eqx import Para, fixed_point, fixed_point_forward
from case import *


# ============================================================================
# 5. 测试函数
# ============================================================================
def test_forward():
    """测试前向传播"""
    print("=" * 80)
    print("测试 1: 前向传播")
    print("=" * 80)
    
    batch_size = 4
    dim = 3
    niter = 50
    
    key = jax.random.PRNGKey(42)
    bias = jax.random.normal(key, (batch_size, dim))
    
    # 使用 eqx.Module 创建参数
    para = Para(
        weight=jnp.ones((batch_size, dim)),
        bias=bias
    )
    
    states_guess = [jnp.zeros((batch_size, dim))]
    
    result = fixed_point_forward(
        states_guess,
        para,
        [],
        iter_func=simple_iter_func,
        update_substates_func=update_substates_func,
        get_substates_func=get_substates_func,
        niter=niter,
    )
    
    expected = 2.0 * bias
    
    print(f"数值解:\n{result}")
    print(f"\n解析解:\n{expected}")
    print(f"\n最大误差: {jnp.max(jnp.abs(result - expected)):.6e}")
    
    assert jnp.allclose(result, expected, atol=1e-5), "前向传播测试失败！"
    print("\n✅ 前向传播测试通过！\n")


def test_jvp():
    """测试 JVP"""
    print("=" * 80)
    print("测试 2: JVP (前向模式自动微分)")
    print("=" * 80)
    
    batch_size = 2
    dim = 2
    niter = 50
    
    key = jax.random.PRNGKey(123)
    bias = jax.random.normal(key, (batch_size, dim))
    para = Para(
        weight=jnp.ones((batch_size, dim)),
        bias=bias
    )
    
    states_guess = [jnp.zeros((batch_size, dim))]
    
    def f(bias_param):
        para_temp = Para(weight=para.weight, bias=bias_param)
        return fixed_point_forward(
            states_guess,
            para_temp,
            [],
            iter_func=simple_iter_func,
            update_substates_func=update_substates_func,
            get_substates_func=get_substates_func,
            niter=niter,
        )
    
    v = jnp.ones_like(bias)
    result, tangent = jvp(f, (bias,), (v,))
    expected_tangent = 2.0 * v
    
    print(f"数值 JVP:\n{tangent}")
    print(f"\n解析 JVP:\n{expected_tangent}")
    print(f"\n最大误差: {jnp.max(jnp.abs(tangent - expected_tangent)):.6e}")
    
    assert jnp.allclose(tangent, expected_tangent, atol=1e-4), "JVP 测试失败！"
    print("\n✅ JVP 测试通过！\n")


def test_grad():
    """测试梯度计算"""
    print("=" * 80)
    print("测试 3: 梯度计算 (反向模式自动微分)")
    print("=" * 80)
    
    batch_size = 2
    dim = 2
    niter = 50
    
    key = jax.random.PRNGKey(456)
    bias = jax.random.normal(key, (batch_size, dim))
    para = Para(
        weight=jnp.ones((batch_size, dim)),
        bias=bias
    )
    
    states_guess = [jnp.zeros((batch_size, dim))]
    
    def loss_fn(bias_param):
        para_temp = Para(weight=para.weight, bias=bias_param)
        result = fixed_point_forward(
            states_guess,
            para_temp,
            [],
            iter_func=simple_iter_func,
            update_substates_func=update_substates_func,
            get_substates_func=get_substates_func,
            niter=niter,
        )
        return jnp.sum(result ** 2)
    
    grad_fn = grad(loss_fn)
    numerical_grad = grad_fn(bias)
    analytical_grad = 8.0 * bias
    
    print(f"数值梯度:\n{numerical_grad}")
    print(f"\n解析梯度:\n{analytical_grad}")
    print(f"\n最大误差: {jnp.max(jnp.abs(numerical_grad - analytical_grad)):.6e}")
    
    assert jnp.allclose(numerical_grad, analytical_grad, atol=1e-3), "梯度测试失败！"
    print("\n✅ 梯度测试通过！\n")


# ============================================================================
# 6. 运行所有测试
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🔥" * 40)
    print("开始测试 implicit_func_fixed_point (修复版)")
    print("🔥" * 40 + "\n")
    
    try:
        test_forward()
        test_jvp()
        test_grad()
        
        print("=" * 80)
        print("🎉 所有测试通过！")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
