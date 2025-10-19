import jax
import equinox as eqx
from typing import Callable, List
from ...subjects import Para

# import jax.numpy as jnp
# from ...shared_utilities.types import Float_2D
# from functools import partial
# from ...subjects import Para, Lai, ParNir, LeafAng
# from ...subjects import Met, Prof, SunAng, SunShadedCan
# from ...subjects import Veg, Soil, Qin, Ir, Can


# states_initial: list[
#     Met,
#     Prof,
#     Ir,
#     Qin,
#     SunAng,
#     SunShadedCan,
#     SunShadedCan,
#     Soil,
#     Veg,
#     Can,
# ],
@eqx.filter_jit
def fixed_point(
    func: Callable,
    states_initial: List,
    para: Para,
    niter: int,
    *args,
):
    def iteration(c, i):
        cnew = func(c, para, *args)
        return cnew, None

    # jax.debug.print("Iterations: {i}", i=niter)
    states_final, _ = jax.lax.scan(iteration, states_initial, xs=None, length=niter)
    return states_final
