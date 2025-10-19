"""
改进的 Para 类 - 使用类属性定义默认值
优势：
1. 所有可选参数的类型和默认值一目了然
2. 参数按功能分组，易于查找和维护
3. 减少 __init__ 中的重复代码
"""

from typing import Optional
import jax.numpy as jnp
import jax
import equinox as eqx

# 假设这些类型已定义
Float_0D = float
Float_1D = list


class VarStats:
    """变量统计信息"""

    pass


class MLP(eqx.Module):
    """多层感知机"""

    pass


class MLP2(eqx.Module):
    """多层感知机2"""

    pass


class MLP3(eqx.Module):
    """多层感知机3"""

    pass


class Para(eqx.Module):
    """
    生态系统模型参数类

    必需参数（5个）：
        - zht1, zht2, delz1, delz2: 垂直剖面
        - soil_depth: 土壤深度
        - leaf_clumping_factor: 叶片聚集因子

    可选参数（60+）：按功能分组定义
    """

    # ========================================================================
    # 必需参数（需要在初始化时显式指定，不在这里定义默认值）
    # ========================================================================
    # zht1, zht2, delz1, delz2, soil_depth, leaf_clumping_factor

    # ========================================================================
    # 可选参数：辐射相关 (PAR and NIR)
    # ========================================================================
    par_reflect: Float_0D = 0.05
    par_trans: Float_0D = 0.05
    par_soil_refl: Float_0D = 0.05
    nir_reflect: Float_0D = 0.60
    nir_trans: Float_0D = 0.20
    nir_soil_refl: Float_0D = 0.10

    # ========================================================================
    # 可选参数：光合作用模型 (Farquhar et al.)
    # ========================================================================
    vcopt: Float_0D = 171.0  # carboxylation rate at 25°C, µmol m⁻² s⁻¹
    jmopt: Float_0D = 259.0  # electron transport rate at 25°C, µmol m⁻² s⁻¹
    rd25: Float_0D = 2.68  # dark respiration at 25°C
    hkin: Float_0D = 200000.0  # enthalpy term, J mol⁻¹
    skin: Float_0D = 710.0  # entropy term, J K⁻¹ mol⁻¹
    ejm: Float_0D = 55000.0  # activation energy for electron transport
    evc: Float_0D = 55000.0  # activation energy for carboxylation

    # ========================================================================
    # 可选参数：酶常数 (Enzyme constants)
    # ========================================================================
    kc25: Float_0D = 274.6  # kinetic coef for CO2 at 25°C, microbars
    ko25: Float_0D = 419.8  # kinetic coef for O2 at 25°C, millibars
    o2: Float_0D = 210.0  # oxygen concentration, mmol mol⁻¹
    ekc: Float_0D = 80500.0  # activation energy for K of CO2
    eko: Float_0D = 14500.0  # activation energy for K of O2
    erd: Float_0D = 38000.0  # activation energy for dark respiration
    ektau: Float_0D = -29000.0  # J mol⁻¹

    # ========================================================================
    # 可选参数：温度优化参数
    # ========================================================================
    toptvc: Float_0D = 303.0  # optimum temp for max carboxylation, K
    toptjm: Float_0D = 303.0  # optimum temp for max electron transport, K

    # ========================================================================
    # 可选参数：气孔导度 (Stomatal conductance)
    # ========================================================================
    kball: Float_0D = 8.17  # Ball-Berry stomatal coefficient
    bprime: Float_0D = 0.05  # Ball-Berry intercept, mol m⁻² s⁻¹ H2O
    rsm: Float_0D = 145.0  # minimum stomatal resistance, s m⁻¹
    brs: Float_0D = 60.0  # curvature coefficient for light response
    qalpha: Float_0D = 0.22  # leaf quantum yield, electrons
    lleaf: Float_0D = 0.04  # leaf length, m

    # ========================================================================
    # 可选参数：土壤水分阈值
    # ========================================================================
    theta_min: Float_0D = 0.03  # wilting point
    theta_max: Float_0D = 0.2  # field capacity

    # ========================================================================
    # 可选参数：Q10 参数（土壤呼吸）
    # ========================================================================
    q10a: Float_0D = 5.0
    q10b: Float_0D = 1.7
    q10c: Float_0D = 0.8

    # ========================================================================
    # 可选参数：物理常数（通常不需要修改）
    # ========================================================================
    sigma: Float_0D = 5.670367e-08  # Stefan-Boltzmann constant, W m⁻² K⁻⁴
    ep: Float_0D = 0.98  # emissivity of leaves
    epsoil: Float_0D = 0.98  # emissivity of soil
    rugc: Float_0D = 8.314  # universal gas constant, J mol⁻¹ K⁻¹
    rgc1000: Float_0D = 8314.0  # gas constant × 1000
    Cp: Float_0D = 1005.0  # specific heat of air, J kg⁻¹ K⁻¹
    tk_25: Float_0D = 298.16  # absolute temp at 25°C, K

    # ========================================================================
    # 可选参数：扩散系数 (Diffusivity at STP, 273K and 1013mb)
    # ========================================================================
    nuvisc: Float_0D = 13.27  # molecular viscosity, mm² s⁻¹
    dc: Float_0D = 13.81  # diffusivity of CO2, mm² s⁻¹
    dh: Float_0D = 18.69  # diffusivity of heat, mm² s⁻¹
    dv: Float_0D = 21.78  # diffusivity of water vapor, mm² s⁻¹
    do3: Float_0D = 14.44  # diffusivity of ozone, mm² s⁻¹
    betfac: Float_0D = 1.5  # boundary layer sheltering factor

    # ========================================================================
    # 可选参数：其他常数
    # ========================================================================
    Mair: Float_0D = 28.97  # molecular weight of air
    dLdT: Float_0D = -2370.0  # temperature derivative
    extinct: Float_0D = 2.0  # extinction coefficient for wind in canopy

    # ========================================================================
    # 可选参数：气象统计数据
    # ========================================================================
    var_mean: Optional[VarStats] = None
    var_std: Optional[VarStats] = None
    var_max: Optional[VarStats] = None
    var_min: Optional[VarStats] = None

    # ========================================================================
    # 可选参数：深度学习模型
    # ========================================================================
    RsoilDL: Optional[eqx.Module] = None  # soil respiration model
    LeafRHDL: Optional[eqx.Module] = None  # leaf relative humidity model
    bprimeDL: Optional[eqx.Module] = None  # bprime model
    gscoefDL: Optional[eqx.Module] = None  # stomatal conductance coef model

    def __init__(
        self,
        # ============ 必需参数（共6个）============
        zht1: Float_1D,
        zht2: Float_1D,
        delz1: Float_1D,
        delz2: Float_1D,
        soil_depth: Float_0D,
        leaf_clumping_factor: Float_0D,
        # ============ 可选参数通过 **kwargs 传入 ============
        **kwargs,
    ) -> None:
        """
        初始化 Para 参数对象

        Args:
            zht1: 垂直高度剖面1
            zht2: 垂直高度剖面2
            delz1: 层厚度1
            delz2: 层厚度2
            soil_depth: 土壤深度
            leaf_clumping_factor: 叶片聚集因子
            **kwargs: 其他可选参数，会覆盖类属性的默认值

        Example:
            >>> para = Para(
            ...     zht1=jnp.array([0.1, 0.5, 1.0]),
            ...     zht2=jnp.array([0.3, 0.7, 1.5]),
            ...     delz1=jnp.array([0.1, 0.4, 0.5]),
            ...     delz2=jnp.array([0.1, 0.4, 0.5]),
            ...     soil_depth=2.0,
            ...     leaf_clumping_factor=0.85,
            ...     vcopt=180.0,  # 覆盖默认值
            ...     learning_rate=0.01
            ... )
        """
        # 1. 设置必需参数
        self.zht1 = zht1
        self.zht2 = zht2
        self.delz1 = delz1
        self.delz2 = delz2
        self.soil_depth = jnp.array(soil_depth)
        self.leaf_clumping_factor = jnp.array(leaf_clumping_factor)

        # 2. 从类属性继承所有可选参数的默认值
        self._initialize_defaults()

        # 3. 用 kwargs 覆盖默认值
        self._apply_kwargs(kwargs)

        # 4. 初始化深度学习模型（如果未在 kwargs 中指定）
        self._initialize_dl_models()

    def _initialize_defaults(self):
        """从类属性继承所有可选参数的默认值"""
        for key in dir(self.__class__):
            if key.startswith("_"):
                continue

            # 跳过必需参数
            if key in [
                "zht1",
                "zht2",
                "delz1",
                "delz2",
                "soil_depth",
                "leaf_clumping_factor",
            ]:
                continue

            class_attr = getattr(self.__class__, key)

            # 只复制数据属性，不复制方法
            if not callable(class_attr):
                # 对于需要转换为 jax array 的参数
                if class_attr is not None and not isinstance(class_attr, type):
                    if isinstance(class_attr, (int, float)):
                        setattr(self, key, jnp.array(class_attr))
                    else:
                        setattr(self, key, class_attr)
                else:
                    setattr(self, key, class_attr)

    def _apply_kwargs(self, kwargs: dict):
        """应用 kwargs 中的参数，覆盖默认值"""
        # 获取所有有效的可选参数名
        valid_optional_params = set()
        for key in dir(self.__class__):
            if not key.startswith("_"):
                class_attr = getattr(self.__class__, key)
                if not callable(class_attr) and key not in [
                    "zht1",
                    "zht2",
                    "delz1",
                    "delz2",
                    "soil_depth",
                    "leaf_clumping_factor",
                ]:
                    valid_optional_params.add(key)

        # 应用 kwargs
        for key, value in kwargs.items():
            if key in valid_optional_params:
                # 对数值参数自动转换为 jax array
                if isinstance(value, (int, float)):
                    setattr(self, key, jnp.array(value))
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(
                    f"Unknown parameter: '{key}'. "
                    f"Valid optional parameters: {sorted(valid_optional_params)}"
                )

    def _initialize_dl_models(self):
        """初始化深度学习模型（如果未被 kwargs 指定）"""
        key = jax.random.PRNGKey(0)
        if self.RsoilDL is None:
            self.RsoilDL = MLP(in_size=2, out_size=1, width_size=6, depth=2, key=key)

        if self.LeafRHDL is None:
            self.LeafRHDL = MLP2(in_size=2, out_size=1, width_size=6, depth=2, key=key)

        if self.bprimeDL is None:
            self.bprimeDL = MLP2(in_size=4, out_size=1, width_size=6, depth=2, key=key)

        if self.gscoefDL is None:
            self.gscoefDL = MLP3(in_size=3, out_size=2, width_size=6, depth=2, key=key)

    def get_params(self) -> dict:
        """返回所有参数的字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def update(self, **kwargs):
        """
        更新参数值

        Example:
            >>> para.update(vcopt=180.0, jmopt=270.0)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, (int, float)):
                    setattr(self, key, jnp.array(value))
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: '{key}'")
        return self


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("改进后的 Para 类使用示例")
    print("=" * 70)

    # 示例1: 只使用必需参数，其他全部使用默认值
    para1 = Para(
        zht1=jnp.array([0.1, 0.5, 1.0]),
        zht2=jnp.array([0.3, 0.7, 1.5]),
        delz1=jnp.array([0.1, 0.4, 0.5]),
        delz2=jnp.array([0.1, 0.4, 0.5]),
        soil_depth=2.0,
        leaf_clumping_factor=0.85,
    )
    print(f"\n示例1 - 全部使用默认值:")
    print(f"vcopt: {para1.vcopt}")
    print(f"par_reflect: {para1.par_reflect}")

    # 示例2: 覆盖部分可选参数
    para2 = Para(
        zht1=jnp.array([0.1, 0.5, 1.0]),
        zht2=jnp.array([0.3, 0.7, 1.5]),
        delz1=jnp.array([0.1, 0.4, 0.5]),
        delz2=jnp.array([0.1, 0.4, 0.5]),
        soil_depth=2.0,
        leaf_clumping_factor=0.85,
        vcopt=180.0,  # 覆盖默认值
        jmopt=270.0,
        par_reflect=0.08,
        theta_min=0.05,
    )
    print(f"\n示例2 - 覆盖部分参数:")
    print(f"vcopt: {para2.vcopt} (覆盖)")
    print(f"jmopt: {para2.jmopt} (覆盖)")
    print(f"par_reflect: {para2.par_reflect} (覆盖)")
    print(f"par_trans: {para2.par_trans} (默认)")
    print(f"sigma: {para2.sigma} (物理常数，默认)")
    print(f"Cp: {para2.Cp} (物理常数，默认)")

    # 示例3: 动态更新参数
    print(f"\n示例3 - 动态更新:")
    print(f"更新前 vcopt: {para2.vcopt}")
    para2.update(vcopt=190.0, kball=9.0)
    print(f"更新后 vcopt: {para2.vcopt}")
    print(f"更新后 kball: {para2.kball}")

    # 示例4: 甚至可以覆盖物理常数（虽然很少这样做）
    para3 = Para(
        zht1=jnp.array([0.1, 0.5, 1.0]),
        zht2=jnp.array([0.3, 0.7, 1.5]),
        delz1=jnp.array([0.1, 0.4, 0.5]),
        delz2=jnp.array([0.1, 0.4, 0.5]),
        soil_depth=2.0,
        leaf_clumping_factor=0.85,
        sigma=5.67e-08,  # 可以覆盖物理常数（如果需要的话）
    )
    print(f"\n示例4 - 覆盖物理常数:")
    print(f"sigma: {para3.sigma} (自定义)")

    print(f"\n✅ 所有参数数量: {len(para2.get_params())}")
