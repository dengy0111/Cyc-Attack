"""CPU-only checks of 10_16 sign-PGD and shared spatial selection."""
import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stages/06_attack_core.py"


def funcs(names, namespace):
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def setup(selectors=("wind",)):
    ns = {"np": np, "jnp": jnp, "PGD_STEP_SIZE": 0.01,
          "GC_MEAN": {}, "GRADIENT_DILATION_VARS": selectors,
          "broadcast_weight": lambda w, d: jnp.broadcast_to(w, d.shape),
          "gradient_with_optional_dilation": lambda k, g, m: g * m,
          "_masked_mean_abs": lambda g, m: jnp.sum(jnp.abs(g * m)) / jnp.maximum(m.sum(), 1.)}
    return funcs({"input_gradient_dilation_mask", "variable_distance_weight", "input_reg_masked", "update_delta_pgd"}, ns)


def test_default_wind_selected_for_both_gradient_distance_and_reg_distance():
    ns = setup()
    wind = "10m_u_component_of_wind"
    temp = "temperature"
    delta = {wind: jnp.ones((2, 2)), temp: jnp.ones((2, 2))}
    grads = {k: jnp.ones_like(v) for k, v in delta.items()}
    masks = {k: jnp.ones_like(v) for k, v in delta.items()}
    spatial_grad = jnp.array([[0., 0.5], [1., 1.]])
    spatial_reg = jnp.array([[0., 0.], [0., 0.]])
    result, lr, grads_abs = ns["update_delta_pgd"](delta, grads, spatial_grad, masks)
    np.testing.assert_allclose(np.asarray(result[wind]), 1 - .01 * np.asarray(spatial_grad))
    np.testing.assert_allclose(np.asarray(result[temp]), np.full((2, 2), .99))
    assert lr == .01 and grads_abs[wind] == 1.
    # Wind gets w_reg=0, other variables get plain unit-weight L2.
    reg = ns["input_reg_masked"](delta, spatial_reg, masks)
    assert np.isclose(float(reg), 0.5)


def test_pressure_level_selector_controls_both_spatial_weights():
    ns = setup(("temperature@300",))
    ns["GC_MEAN"] = {"temperature": xr.DataArray(np.ones(3), dims=("level",),
                                                coords={"level": [300, 500, 700]})}
    tensor = jnp.ones((2, 3, 2, 2))
    chosen = ns["variable_distance_weight"]("temperature", tensor, jnp.zeros((2, 2)))
    np.testing.assert_array_equal(np.asarray(chosen[:, 0]), 0)
    np.testing.assert_array_equal(np.asarray(chosen[:, 1:]), 1)
    delta = {"temperature": tensor}
    mask = {"temperature": jnp.ones_like(tensor)}
    reg = ns["input_reg_masked"](delta, jnp.zeros((2, 2)), mask)
    assert np.isclose(float(reg), 2 / 3)
    updated, _, _ = ns["update_delta_pgd"](delta, delta, jnp.zeros((2, 2)), mask)
    np.testing.assert_array_equal(np.asarray(updated["temperature"][:, 0]), 1.)
    np.testing.assert_allclose(np.asarray(updated["temperature"][:, 1:]), .99, atol=1e-6)


def test_pgd_sign_only_without_momentum_or_normalization():
    ns = setup(("none",))
    delta = {"temperature": jnp.zeros((2, 2))}
    grads = {"temperature": jnp.array([[1e-9, -1000.], [0., 2.]])}
    mask = {"temperature": jnp.ones((2, 2))}
    updated, lr, _ = ns["update_delta_pgd"](delta, grads, jnp.zeros((2, 2)), mask)
    np.testing.assert_allclose(np.asarray(updated["temperature"]), [[-.01, .01], [0, -.01]], atol=1e-6)
    assert lr == .01
    source = SOURCE.read_text()
    assert "MOMENTUM *" not in source
    assert "safe_rms" not in source


def test_cli_and_default_config():
    from sys import path
    path.insert(0, str(ROOT))
    from config import ProjectConfig
    cfg = ProjectConfig.from_json(ROOT / "config.json")
    cfg.validate()
    assert cfg.optimizer == "pgd"
    assert cfg.pgd_step_size == .01
    assert cfg.gradient_dilation_vars == ["wind"]
