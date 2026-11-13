"""CPU-only tests of equal-weight TC guidance, Adam spatial weighting, and std exports."""
import ast
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import jax
import jax.numpy as jnp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path(__file__).resolve().parents[1] / "stages" / "06_attack_core.py"


def extract(names, namespace):
    source = ast.parse(SOURCE.read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace


def test_equal_temporal_weights_and_guide_equal_components():
    namespace = {"jnp": jnp, "REG_LAMBDA": 0.001}
    extract({"time_average", "combine_loss"}, namespace)
    tgt = jnp.array([2., 100., 6., 200.])
    orig = jnp.array([100., 4., 200., 8.])
    geo = {"has_tgt": jnp.array([1., 0., 1., 0.]), "has_orig": jnp.array([0., 1., 0., 1.])}
    alpha = jnp.ones((4,))
    total, terms = namespace["combine_loss"]({"tgt": tgt, "orig": orig}, 0., geo, alpha, alpha, 1., 1.)
    assert np.isclose(float(terms["tgt"]), 4.)
    assert np.isclose(float(terms["orig"]), 6.)
    assert np.isclose(float(total), 5.)
    geo["has_orig"] = jnp.zeros(4)
    total, _ = namespace["combine_loss"]({"tgt": tgt, "orig": orig}, 0., geo, alpha, alpha, 1., 1.)
    assert np.isclose(float(total), 4.)
    assert "np.ones(FORECAST_STEPS" in SOURCE.read_text()


def test_projected_adam_spatial_mask_and_moments():
    namespace = {"jnp": jnp, "np": np, "ADAM_BETA1": .9, "ADAM_BETA2": .999,
                 "ADAM_EPS": 1e-8, "TI_SIGMA_CELLS": 1.,
                 "cosine_lr": lambda step: .01,
                 "ti_smooth": lambda grad, sigma: grad,
                 "gradient_with_optional_dilation": lambda var, grad, mask: grad * mask,
                 "_masked_mean_abs": lambda x, mask: jnp.sum(jnp.abs(x * mask)) / jnp.maximum(jnp.sum(mask), 1),
                 "broadcast_weight": lambda w, x: jnp.broadcast_to(w, x.shape),
                 "variable_distance_weight": lambda var, x, w: jnp.broadcast_to(w, x.shape)}
    extract({"adam_state_zero", "update_delta_adam"}, namespace)
    delta = {"var": jnp.zeros((2, 2), dtype=jnp.float32)}
    grad = {"var": jnp.ones((2, 2), dtype=jnp.float32)}
    mask = {"var": jnp.array([[1., 1.], [1., 0.]])}
    spatial = jnp.array([[1., .2], [0., 1.]])
    state = namespace["adam_state_zero"](delta)
    result, state, lr, gmean = namespace["update_delta_adam"](delta, grad, state, 0, spatial, mask)
    np.testing.assert_allclose(np.array(result["var"]), [[-.01, -.002], [0, 0]], atol=1e-6)
    assert np.array(state["m"]["var"])[1, 1] == 0
    assert np.array(state["v"]["var"])[1, 1] == 0
    assert lr == .01 and gmean["var"] > 0


def test_standardized_downstream_differences_are_scaled():
    ns = {"np": np, "DOWNSTREAM_STD": torch.tensor([2., 4., 10., 1.]),
          "standardized_to_raw": None}
    extract({"_downstream_comparison_maps"}, ns)
    ctx = {"clean_s_np": np.zeros((2,4,2,2), dtype=np.float32),
           "guide_np": np.ones((2,4,2,2), dtype=np.float32)}
    adv = np.full((2,4,2,2), 3., dtype=np.float32)
    standard = ns["_downstream_comparison_maps"](ctx,adv,standardized=True)
    raw = ns["_downstream_comparison_maps"](ctx,adv,standardized=False)
    np.testing.assert_allclose(standard[1][1][1], np.full((2,2),3.))
    np.testing.assert_allclose(raw[1][1][1], np.full((2,2),12.))
    np.testing.assert_allclose(standard[2][1][2], np.full((2,2),2.))
    np.testing.assert_allclose(raw[2][1][2], np.full((2,2),20.))
