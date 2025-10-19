import os
from os import path
from .train_model import parse_config


default_config = path.abspath("tests/PB-1L-0.0/configs.json")


def load_forcing(f_config=default_config):
    cwd = path.abspath(path.curdir)  # 当前工作路径
    indir = os.path.dirname(f_config)
    os.chdir(indir)

    file = os.path.basename(f_config)

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
    ) = parse_config(file)
    os.chdir(cwd)

    batched_y_train, batched_y_test = batched_y[0], batched_y[1]
    batched_met_train, batched_met_test = batched_met[0], batched_met[1]

    return {
        "model": model,
        "filter_model_spec": filter_model_spec,
        "forcing": {
            "train": (batched_met_train, batched_y_train),
            "test": (batched_met_test, batched_y_test),
        },
        "hyperparams": hyperparams,
        "para_min": para_min,
        "para_max": para_max,
        "output_funcs": output_funcs,
        "loss_func": loss_func,
        "optim": optim,
        "nsteps": nsteps,
        "configs": configs,
    }
