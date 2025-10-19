# %%
import os
from os import path

import jax
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug
from jax_canveg.shared_utilities.optim import perform_optimization_batch

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

configs = path.abspath("tests/PB-1L-0.0/configs.json")
cwd = path.abspath(path.curdir)
cwd

indir = os.path.dirname(configs)
f_configs = os.path.basename(configs)
indir
os.chdir(indir)
path.abspath(path.curdir)


# %%
from jax_canveg.train_model import parse_config
(
    model,
    filter_model_spec,
    batched_met,
    batched_y,
    hyperparams,
    para_min,
    para_max,
    output_funcs,
    loss_func,
    optim,
    nsteps,
    configs,
) = parse_config(f_configs)
output_funcs

# %%
batched_y_train, batched_y_test = batched_y[0], batched_y[1]
batched_met_train, batched_met_test = batched_met[0], batched_met[1]

# %%
from jax._src.tree_util import tree_flatten, tree_unflatten

def get_chunk(xx, i=0):
    xs_flat, xs_tree = tree_flatten(xx)
    xs_slice = [x[i] for x in xs_flat]
    return tree_unflatten(xs_tree, xs_slice)

# xs = [batched_met[0], batched_y[0]]
x = get_chunk(batched_met_train)
y = get_chunk(batched_y_train)
print(x, y)

# %%
states_final, [rnet, sun_ang, leaf_ang, lai] = model(x)
# states_final: Met, Prof, ParNir, ParNir, Ir, Qin, SunShadedCan, SunShadedCan
#               Soil, Veg, Can
(
    model_new,
    loss_train_set,
    loss_test_set,
) = perform_optimization_batch(  # pyright: ignore
    model.get_fixed_point_states,  # pyright: ignore
    filter_model_spec.get_fixed_point_states,
    optim,
    nsteps,
    loss_func,
    batched_y_train,
    batched_met_train,
    batched_y_test,
    batched_met_test,
    para_min,
    para_max,
    *output_funcs,
)

# %%
pred_y = model(x)
loss_func(y, pred_y)
