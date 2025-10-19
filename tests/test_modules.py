# %%
# from jax_canveg.models import CanvegIFT
from tests.forcing import load_forcing

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

(
    model,
    filter_model_spec,
    batched_met_train,
    batched_y_train,
    batched_met_test,
    batched_y_test,
    hyperparams,
    para_min,
    para_max,
    output_funcs,
    loss_func,
    optim,
    nsteps,
    configs,
) = load_forcing()

from jax_canveg.models.canveg_eqx import (
    canveg_initialize_states,
    canveg_each_iteration,
    implicit_func_fixed_point,
)

# %%
if __name__ == "__main__":

    # 加载数据
    _model = model.get_fixed_point_states
    _filter_model_spec = filter_model_spec.get_fixed_point_states
    batched_y = batched_y_train
    batched_met = batched_met_train

    diff_model, static_model = eqx.partition(_model, _filter_model_spec)

    ## 1. 单个batch
    met = get_chunk(batched_met_train)
    y = get_chunk(batched_y_train)
    print(met, y)
    print("Running one batch ...")

    model_args = output_funcs
    update_substates_func = output_funcs[0]
    get_substates_func = output_funcs[1]
    # model2 = eqx.combine(diff_model, static_model)

    # pred_y = _model(met, *model_args)
    # print(pred_y)

    self = model
    para, dij = self.para, self.dij
    # Location parameters
    lat_deg = self.lat_deg
    long_deg = self.long_deg
    time_zone = self.time_zone
    # Static parameters
    leafangle = self.leafangle
    stomata = self.stomata
    n_can_layers = self.n_can_layers
    n_total_layers = self.n_total_layers
    n_soil_layers = self.n_soil_layers
    dt_soil = self.dt_soil
    soil_mtime = self.soil_mtime
    niter = self.niter
    # Functions
    leafrh_func = self.leafrh_func
    soilresp_func = self.soilresp_func
    # Number of time steps from met
    ntime = met.zL.size

    # Initialization
    rnet, lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
        para,
        met,
        lat_deg,
        long_deg,
        time_zone,
        leafangle,
        n_can_layers,
        n_total_layers,
        n_soil_layers,
        ntime,
        dt_soil,
        soil_mtime,
    )
    states_guess = initials

    # Forward runs
    args = [
        dij,
        sun_ang,
        leaf_ang,
        lai,
        n_can_layers,
        stomata,
        soil_mtime,
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
    # _loss = loss_func(y, pred_y)
    # jax.debug.print("[one batch]: loss_value (direct compute): {x}", x=_loss)

    # # 都是同样的过程，为何调用 loss_func_optim 会失败？
    # run_batch(diff_model, static_model, y, met)

    ## 2. 数据batch
    # print("Running all batches ...")
    # run_batches()
    # print("Model updated.")
