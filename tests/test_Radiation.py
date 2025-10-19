# %%
import equinox as eqx
import jax
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax_canveg import load_forcing
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug
from jax_canveg.subjects.initialization_update import (
    initialize_model_states,
    initialize_profile,
)
from jax_canveg.shared_utilities.utils import dot
from jax_canveg.subjects import Met

from jax_canveg.physics.energy_fluxes import rad_tran_canopy, sky_ir
from jax_canveg.physics.energy_fluxes import diffuse_direct_radiation
from jax_canveg.physics.carbon_fluxes import angle, leaf_angle

# from jax_canveg.physics.energy_fluxes import compute_qin, ir_rad_tran_canopy
# from jax_canveg.physics.energy_fluxes import uz, soil_energy_balance

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
    canveg_initialize_states
)


# %%
def test_modules():
    f = "tests/PB-ML-0.0/configs.json"
    input = load_forcing(f)
    model = input["model"]
    batched_met_train, batched_y_train = input["forcing"]["train"]

    ## 1. 单个batch
    met = get_chunk(batched_met_train)
    y = get_chunk(batched_y_train)
    print(met, y)
    print("Running ...")

    self = model
    para = self.para
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

    ## =========================================================================
    jtot = self.n_can_layers
    ntime = met.zL.size

    prof = initialize_profile(met, para)

    lat_deg = self.lat_deg
    long_deg = self.long_deg
    time_zone = self.time_zone

    dt_soil = self.dt_soil
    soil_mtime = self.soil_mtime
    n_soil_layers = self.n_soil_layers
    leafangle = self.leafangle

    (soil, par, nir, ir, qin, rnet, sun, shade, veg, lai, can) = (
        initialize_model_states(
            met, para, ntime, jtot, dt_soil, soil_mtime, n_soil_layers
        )
    )

    sun_ang = angle(lat_deg, long_deg, time_zone, met.day, met.hhour) # passed
    leaf_ang = leaf_angle(sun_ang, para, leafangle, lai)  # passed

    #                     Compute direct and diffuse radiations                    #
    ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = diffuse_direct_radiation(
        sun_ang.sin_beta, met.rglobal, met.parin, met.P_kPa
    )

    par = eqx.tree_at(lambda t: (t.inbeam, t.indiffuse), par, (par_beam, par_diffuse))
    nir = eqx.tree_at(lambda t: (t.inbeam, t.indiffuse), nir, (nir_beam, nir_diffuse))

    # ---------------------------------------------------------------------------- #
    #                     Initialize IR fluxes with air temperature                #
    ir_in = sky_ir(met.T_air_K, ratrad, para.sigma)
    # ir_in = sky_ir_v2(met, ratrad, para.sigma)
    ir_dn = dot(ir_in, ir.ir_dn)
    ir_up = dot(ir_in, ir.ir_up)
    ir = eqx.tree_at(lambda t: (t.ir_in, t.ir_dn, t.ir_up), ir, (ir_in, ir_dn, ir_up))

    # PAR
    par = rad_tran_canopy(
        sun_ang, leaf_ang, par, para, lai,
        para.par_reflect,
        para.par_trans,
        para.par_soil_refl,
        niter=niter
    )
    # NIR
    nir = rad_tran_canopy(
        sun_ang, leaf_ang, nir, para, lai,
        para.nir_reflect,
        para.nir_trans,
        para.nir_soil_refl,
        niter=niter
    )

    states_initial = [met, prof, par, nir, ir, qin, sun, shade, soil, veg, can]


if __name__ == "__main__":
    test_modules()
