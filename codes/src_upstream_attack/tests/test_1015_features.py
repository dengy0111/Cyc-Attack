"""CPU-only regression tests for the 10_15 changes (no GraphCast/TE needed)."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import jax.numpy as jnp
import xarray as xr
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ProjectConfig


def load_function(file, name, namespace):
    tree = ast.parse((ROOT / file).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), "exec"), namespace)
    return namespace[name]


def test_config_and_loss_weight_flags_are_validated():
    cfg = ProjectConfig.from_json(ROOT / 'config.json')
    cfg.validate()
    assert cfg.gradient_dilation_vars == ['wind']
    assert cfg.original_weight == 1.0
    cfg.gradient_dilation_vars = ['temperature@300', '10m_u_component_of_wind']
    cfg.guide_loss_weight, cfg.trajectory_loss_weight, cfg.reg_lambda = 0.5, 2.0, 0.01
    cfg.validate()
    assert cfg.gradient_dilation_vars == ['temperature@300', '10m_u_component_of_wind']
    cfg.gradient_dilation_vars = ['wind', 'none']
    try:
        cfg.validate()
        assert False, 'Expected validation error for contradictory selectors'
    except ValueError:
        pass


def test_gradient_dilation_default_wind_not_pressure_or_temperature():
    levels = [300., 500., 700., 850.]
    ns = {'jnp': jnp, 'np': np, 'GRADIENT_DILATION_VARS': ['wind'],
          'GC_MEAN': {'temperature': xr.DataArray(np.ones(4), dims=('level',), coords={'level': levels})},
          'TI_SIGMA_CELLS': 1., 'ti_smooth': lambda g, sigma: g * 2}
    load_function('stages/06_attack_core.py', 'input_gradient_dilation_mask', ns)
    grad_func = load_function('stages/06_attack_core.py', 'gradient_with_optional_dilation', ns)
    g = jnp.ones((2, 4, 3, 4))
    mask = jnp.ones_like(g)
    np.testing.assert_array_equal(grad_func('10m_u_component_of_wind', g, mask), g * 2)
    np.testing.assert_array_equal(grad_func('temperature', g, mask), g)
    ns['GRADIENT_DILATION_VARS'] = ['temperature@300']
    mixed = np.asarray(grad_func('temperature', g, mask))
    np.testing.assert_array_equal(mixed[:, 0], 2 * np.ones((2, 3, 4)))
    np.testing.assert_array_equal(mixed[:, 1:], np.ones((2, 3, 3, 4)))
    ns['GRADIENT_DILATION_VARS'] = ['none']
    np.testing.assert_array_equal(grad_func('10m_u_component_of_wind', g, mask), g)


def test_cumulative_summary_at_each_epsilon(tmp_path):
    stages = [3., 6., 8., 10.]
    ns = {'pd': pd, 'WARMUP_SUMMARY_DIR': tmp_path / 'summaries',
          'ATTACK_STAGE_SCHEDULE': [{'input_epsilon': x, 'guide_epsilon': 10.} for x in stages],
          'number_tag': lambda n: str(int(n)), 'END2END_EPSILON': 10.}
    ns['WARMUP_SUMMARY_DIR'].mkdir()
    ns['end2end_sample_dir'] = lambda case, tag, guide, eps: tmp_path / f'e{int(eps)}'
    for eps in stages:
        d = tmp_path / f'e{int(eps)}'
        (d / 'objective_best').mkdir(parents=True)
        for selection, filename in [('last', d / 'last_checkpoint_losses.csv'),
                                    ('objective_best', d / 'objective_best' / 'selected_checkpoint_losses.csv')]:
            pd.DataFrame([{'sample':'sample', 'target_tag':'distance_1of3', 'input_epsilon':eps,
                           'selection':selection, 'total_loss':eps + (0 if selection=='last' else -1),
                           'guide_tgt':1., 'guide_orig':2., 'traj_core':3., 'reg':4.}]).to_csv(filename,index=False)
    write = load_function('stages/08_run_attack.py', 'write_cumulative_loss_summaries', ns)
    for index, eps in enumerate(stages,1):
        ns['END2END_EPSILON'] = eps
        frame = write('sample', 'distance_1of3', tmp_path / f'e{int(eps)}', index)
        assert len(frame) == index * 2
        assert list(frame['input_epsilon'].unique()) == stages[:index][::-1]
        assert set(frame['selection']) == {'objective_best','last'}
        assert (tmp_path / f'e{int(eps)}' / 'loss_summary_cumulative.csv').exists()
    assert (ns['WARMUP_SUMMARY_DIR'] / 'loss_summary_through_epsilon_10__sample__distance_1of3.csv').exists()


def test_10_15_warmup_uses_objective_not_te():
    src = (ROOT / 'stages/06_attack_core.py').read_text()
    assert '/ "objective_best" / "best_delta_std.npz"' in src
    assert '"selection": "objective_best"' in src
    assert '"warm_start_selection": "objective_best"' in src
    assert '"root_output_selection": "te_best"' in src
    agg = (ROOT / 'stages/10_aggregate.py').read_text()
    assert 'summary.get("warm_start_selection") == "objective_best"' in agg


def test_guide_original_weight_is_effective():
    ns = {'jnp':jnp, 'REG_LAMBDA':0.001}
    load_function('stages/06_attack_core.py','time_average', ns)
    combine=load_function('stages/06_attack_core.py','combine_loss', ns)
    geo={'has_tgt':jnp.array([1.,0.]), 'has_orig':jnp.array([0.,1.])}
    steps={'tgt':jnp.array([2.,100.]), 'orig':jnp.array([100.,8.])}
    alpha=jnp.ones(2)
    total,_=combine(steps, 0.,geo,alpha,alpha,1.,1.)
    assert np.isclose(float(total),5.)
    weighted,_=combine(steps,0.,geo,alpha,alpha,1.,3.)
    assert np.isclose(float(weighted),6.5)
