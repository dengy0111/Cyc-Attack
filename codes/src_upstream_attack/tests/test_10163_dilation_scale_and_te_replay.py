"""CPU-safe regression tests for family multipliers and final TE replay."""
import ast
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
ATTACK_SOURCE = ROOT / 'stages/06_attack_core.py'


def extract_functions(names, ns):
    tree = ast.parse(ATTACK_SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ATTACK_SOURCE), 'exec'), ns)
    return ns


def test_config_accepts_and_normalizes_family_scales():
    from config import ProjectConfig
    cfg = ProjectConfig.from_json(ROOT / 'config.json')
    cfg.gradient_dilation_vars = ['all']
    cfg.dilation_scale = ['wind:1', 'others:10']
    cfg.validate()
    assert cfg.dilation_scale == ['wind:1', 'others:10']
    assert cfg.optimizer == 'pgd'


@pytest.mark.parametrize('bad', [
    ['wind:0'], ['others:-1'], ['wind:nan'], ['others:inf'],
    ['wind:1', 'wind:2'], ['pressure_wind:2'], ['wrong'],
])
def test_config_rejects_invalid_scales(bad):
    from config import ProjectConfig
    cfg = ProjectConfig.from_json(ROOT / 'config.json')
    cfg.dilation_scale = bad
    with pytest.raises(ValueError):
        cfg.validate()


def test_family_specific_ti_sigma_and_selected_mask():
    ns = {'np':np, 'jnp':jnp, 'TI_SIGMA_CELLS':1.0,
          'DILATION_SCALE_MAP':{'wind':1.,'others':10.},
          'GRADIENT_DILATION_VARS':('all',), 'GC_MEAN':{},
          'ti_smooth':lambda g,sigma: g * sigma}
    extract_functions({'input_gradient_dilation_mask', 'gradient_with_optional_dilation'}, ns)
    g = jnp.ones((3,3))
    mask = jnp.ones_like(g)
    np.testing.assert_allclose(np.asarray(ns['gradient_with_optional_dilation']('10m_u_component_of_wind',g,mask)), 1.)
    np.testing.assert_allclose(np.asarray(ns['gradient_with_optional_dilation']('temperature',g,mask)), 10.)
    ns['GRADIENT_DILATION_VARS']=('wind',)
    np.testing.assert_allclose(np.asarray(ns['gradient_with_optional_dilation']('temperature',g,mask)), 1.)


def test_gradient_and_regularization_share_family_specific_weights():
    ns={'np':np,'jnp':jnp,'PGD_STEP_SIZE':.01,
        'GRADIENT_DILATION_VARS':('all',), 'GC_MEAN':{},
        'broadcast_weight':lambda w,x:jnp.broadcast_to(w,x.shape),
        'gradient_with_optional_dilation':lambda var,g,mask:g*mask,
        '_masked_mean_abs':lambda g,mask:jnp.sum(jnp.abs(g*mask))/jnp.maximum(mask.sum(),1.)}
    extract_functions({'input_gradient_dilation_mask','variable_distance_weight',
                       'input_reg_masked','update_delta_pgd'}, ns)
    key_w='10m_u_component_of_wind'; key_o='temperature'
    vals={key_w:jnp.ones((2,2)),key_o:jnp.ones((2,2))}
    masks={k:jnp.ones_like(v) for k,v in vals.items()}
    gw={'wind':jnp.full((2,2),.2),'others':jnp.full((2,2),.8)}
    rw={'wind':jnp.full((2,2),.9),'others':jnp.full((2,2),.3)}
    out,_,_=ns['update_delta_pgd'](vals,vals,gw,masks)
    np.testing.assert_allclose(np.asarray(out[key_w]),.998,atol=1e-6)
    np.testing.assert_allclose(np.asarray(out[key_o]),.992,atol=1e-6)
    reg=float(ns['input_reg_masked'](vals,rw,masks))
    assert np.isclose(reg,(.9**2+.3**2)/2)
    ns['GRADIENT_DILATION_VARS']=('wind',)
    out,_,_=ns['update_delta_pgd'](vals,vals,gw,masks)
    np.testing.assert_allclose(np.asarray(out[key_o]),.99,atol=1e-6)
    reg=float(ns['input_reg_masked'](vals,rw,masks))
    assert np.isclose(reg,(.9**2+1.)/2)


def test_both_final_selected_bundles_replay_te_and_plot():
    source=ATTACK_SOURCE.read_text()
    # Verify final replay is inside save_solution_bundle (called for both names).
    begin=source.index('    def save_solution_bundle(')
    end=source.index('    objective_bundle = save_solution_bundle(',begin)
    body=source[begin:end]
    for expected in ('_te_downstream_to_nc(selected_tensor, sample, replay_nc)',
                     '_te_run(replay_nc, replay_nodes, replay_tracks)',
                     'bundle_dir / "te_track_map.png"',
                     'bundle_dir / "te_tracks.txt"',
                     'bundle_dir / "te_nodes.txt"',
                     'bundle_dir / "te_metrics.json"',
                     'selected_tracks = read_track_file(replay_tracks)'):
        assert expected in body
    assert '    te_bundle = save_solution_bundle(' in source
    assert '    objective_bundle = save_solution_bundle(' in source


def test_separable_ti_gaussian_matches_previous_reference():
    import jax
    import math
    ns = {'np': np, 'jnp': jnp, 'jax': jax, 'math': math}
    extract_functions({'_gauss_kernel_1d', 'ti_smooth'}, ns)
    sample = np.random.default_rng(19).normal(size=(2, 3, 19, 70)).astype('float32')
    for sigma in (1., 10.):
        radius = int(np.ceil(3 * sigma))
        grid = np.arange(-radius, radius+1)
        kernel = np.exp(-grid ** 2/(2*sigma**2))
        kernel /= kernel.sum()
        expanded = np.concatenate([sample[..., -radius:], sample, sample[..., :radius]], axis=-1)
        horizontal = sum(kernel[i] * expanded[..., i:i+70] for i in range(2*radius+1))
        expanded = np.pad(horizontal, ((0,0),(0,0),(radius,radius),(0,0)), mode='edge')
        reference = sum(kernel[i] * expanded[..., i:i+19,:] for i in range(2*radius+1))
        result = np.asarray(ns['ti_smooth'](jnp.asarray(sample), sigma))
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=1e-6)
