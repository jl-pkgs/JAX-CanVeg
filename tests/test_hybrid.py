# %%
# from jax_canveg.models import CanvegIFT
from os import path
from jax_canveg import load_forcing

import jax
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug
from jax._src.tree_util import tree_flatten, tree_unflatten
import equinox as eqx


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_chunk(xx, i=0):
    xs_flat, xs_tree = tree_flatten(xx)
    xs_slice = [x[i] for x in xs_flat]
    return tree_unflatten(xs_tree, xs_slice)


def test_model_hybird():
    f_config = path.abspath("tests/Hybrid-1L-0.0/configs.json")
    input = load_forcing(f_config)

    model = input["model"]
    filter_model_spec = input["filter_model_spec"]
    output_funcs = input["output_funcs"]
    loss_func = input["loss_func"]
    batched_met_train, batched_y_train = input["forcing"]["train"]

    # 加载数据
    _model = model.get_fixed_point_states
    _filter_model_spec = filter_model_spec.get_fixed_point_states
    diff_model, static_model = eqx.partition(_model, _filter_model_spec)

    ## 1. 单个batch
    met = get_chunk(batched_met_train)
    y = get_chunk(batched_y_train)
    print(met, y)
    print("Running one batch ...")

    model_args = output_funcs
    model2 = eqx.combine(diff_model, static_model)

    pred_y = model2(met, *model_args)
    _loss = loss_func(y, pred_y) 
    jax.debug.print("[one batch]: loss_value (direct compute): {x}", x=_loss)

# %%
if __name__ == "__main__":
    test_model_hybird()
