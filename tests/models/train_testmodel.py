"""Train the models."""

import os
from os import path
from pathlib import Path
import logging

# Force JAX to use CPU before importing jax
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
from jax_canveg import train_model
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

################################################################
# Run the model
################################################################

def getwd(): return Path(path.abspath(path.dirname(__file__)))


f_Hybrid = getwd() / "Hybrid-1L-0.0/configs.json" # US-Whs
f_PB = getwd() / "PB-1L-0.0/configs.json" # US-Whs

# train_model(f_Hybrid, save_log_local=True)
# logging.basicConfig(level=logging.INFO)
train_model(f_PB, save_log_local=True)
