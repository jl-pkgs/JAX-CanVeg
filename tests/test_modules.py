# %%
from jax_canveg import load_forcing

import jax
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug
from jax._src.tree_util import tree_flatten, tree_unflatten
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax_canveg.shared_utilities.optim.optim import loss_func_optim


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_chunk(xx, i=0):
    xs_flat, xs_tree = tree_flatten(xx)
    xs_slice = [x[i] for x in xs_flat]
    return tree_unflatten(xs_tree, xs_slice)


from jax_canveg.models.canveg_eqx import (
    canveg_initialize_states,
    canveg_each_iteration,
    implicit_func_fixed_point,
)


# %%
def test_modules():
    input = load_forcing()
    model = input["model"]
    output_funcs = input["output_funcs"]
    batched_met_train, batched_y_train = input["forcing"]["train"]

    ## 1. 单个batch
    met = get_chunk(batched_met_train)
    y = get_chunk(batched_y_train)
    print(met, y)
    print("Running one batch ...")

    update_substates_func = output_funcs[0]
    get_substates_func = output_funcs[1]

    self = model
    para, dij = self.para, self.dij
    niter = self.niter

    # Location parameters
    leafrh_func = self.leafrh_func
    soilresp_func = self.soilresp_func
    # Number of time steps from met

    kwargs = {
        "lat_deg": self.lat_deg,
        "long_deg": self.long_deg,
        "time_zone": self.time_zone,
        "leafangle": self.leafangle,
        "n_can_layers": self.n_can_layers,
        "n_total_layers": self.n_total_layers,
        "n_soil_layers": self.n_soil_layers,
        "time_batch_size": met.zL.size,  ## also named as ntime
        "dt_soil": self.dt_soil,
        "soil_mtime": self.soil_mtime,
    }

    # Initialization
    rnet, lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
        para, met, **kwargs
    )
    states_guess = initials

    # Forward runs
    args = [
        dij,
        sun_ang,
        leaf_ang,
        lai,
        self.n_can_layers,
        self.stomata,
        self.soil_mtime,
        leafrh_func,
        soilresp_func,
    ]
    states_final = implicit_func_fixed_point(
        states_guess,
        para,
        args,
        iter_func=canveg_each_iteration,
        update_substates_func=update_substates_func,
        get_substates_func=get_substates_func,
        niter=niter,
    )

    print(states_final)
    states_final, [rnet, sun_ang, leaf_ang, lai]


if __name__ == "__main__":
    test_modules()
