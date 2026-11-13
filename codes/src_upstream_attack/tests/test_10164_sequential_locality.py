"""CPU-only 10_16_4 regression tests; no GraphCast/TE installations required."""
import ast
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "stages/06_attack_core.py"
RUN = ROOT / "stages/08_run_attack.py"


def funcs(names, ns):
    tree = ast.parse(CORE.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CORE), "exec"), ns)
    return ns


def test_split_uses_disjoint_primary_objectives_with_shared_collateral_penalty():
    ns = dict(jnp=jnp, np=np, jax=jax, PREFIX_WEIGHTS=np.ones(2), GUIDE_CHANNELS=[0,1,2],
              GUIDE_LOSS_WEIGHT=2.0, TRAJECTORY_LOSS_WEIGHT=5.0, ORIGINAL_WEIGHT=1.0,
              REG_LAMBDA=0.3, UNRELATED_LOSS_WEIGHT=0.1,
              ACTIVE_OBJECTIVE_PHASE="guidance",
              build_adv_ds=lambda ctx, d:d,
              region_step_losses=lambda s,clean,guide,geo:{"tgt":jnp.ones((2,))*3,"orig":jnp.ones((2,))*5},
              time_average=lambda values,a,has:(values * has).sum()/jnp.maximum(has.sum(),1),
              surrogate_bridge=lambda s:jnp.zeros((2,2,2,2)),
              trajectory_focal_loss=lambda *args:jnp.asarray(7.),
              input_reg_masked=lambda *args:jnp.asarray(11.))
    funcs({"combined_objective", "active_loss_weights"},ns)
    ctx=dict(pipeline=lambda ds:(jnp.ones((2,3,2,2)), {}, None), clean_s=jnp.zeros((2,3,2,2)),
             guide=jnp.zeros((2,3,2,2)),geo=dict(has_tgt=jnp.ones((2,)),has_orig=jnp.ones((2,)),
                  outside_mask=jnp.ones((2,2,2)),w_reg_family_hw={}),
             target_soft=None,trajectory_calibration=None,perturb_masks={})
    expected_unrelated = 1.0
    for phase,primary in (("guidance",2*4),("trajectory",5*7),("joint",2*4+5*7)):
        ns["ACTIVE_OBJECTIVE_PHASE"]=phase
        loss,(parts,*_) = ns["combined_objective"](ctx,{})
        assert np.isclose(float(loss), primary+0.3*11+0.1*expected_unrelated)
        wg,wt=ns["active_loss_weights"]()
        assert wg==(2.0 if phase!='trajectory' else 0.0)
        assert wt==(5.0 if phase!='guidance' else 0.0)
        assert np.isclose(float(parts['unrelated']),expected_unrelated)


def test_projection_strictly_zeros_input_outside_spatial_mask():
    ns=dict(jnp=jnp, END2END_EPSILON=3.,NONNEGATIVE_INPUT_VARS=set())
    funcs({"project_delta"},ns)
    mask=jnp.asarray([[[[1,0],[0,1]]]],dtype=jnp.float32)
    ctx=dict(perturb_masks={'temperature':mask})
    arr=jnp.full((1,1,2,2),10.0)
    out=ns['project_delta'](ctx,{'temperature':arr})
    np.testing.assert_allclose(np.asarray(out['temperature']), [[[[3,0],[0,3]]]])


def test_unrelated_metrics_exact_separate_selected_checkpoint(tmp_path):
    ns=dict(np=np,pd=pd, INPUT_MASK_RADIUS_DEG=20.,UNRELATED_EXCLUSION_RADIUS_DEG=10.,
            GUIDE_CHANNELS=[0,1,2])
    funcs({'save_unrelated_diagnostics'},ns)
    orig=np.zeros((2,3,2,2),dtype=np.float32)
    adv=orig.copy();adv[:,:,0,0]=4.;adv[:,:,1,1]=2.
    outside=np.zeros((2,2,2),dtype=np.float32);outside[:,1,1]=1
    mask=np.ones((1,1,2,2),dtype=np.float32);mask[...,0,1]=0
    ctx=dict(geo={'outside_mask':outside,'w_tgt':np.zeros_like(outside),
                  'w_orig':np.zeros_like(outside)},clean_s_np=orig,
             perturb_masks={'temperature':mask})
    output=ns['save_unrelated_diagnostics'](ctx,adv,{'temperature':np.zeros_like(mask)},tmp_path)
    assert np.isclose(output['unrelated_rmse'],2)
    assert output['input_mask_violation_max_abs']==0
    assert len(pd.read_csv(tmp_path/'unrelated_region_by_step.csv'))==2
    assert (tmp_path/'unrelated_region_metrics.csv').exists()


def test_stage_runner_two_phases_and_handoff_are_explicit():
    s=RUN.read_text()
    assert 'objective_phase="guidance", output_dir=guide_dir' in s
    assert 'objective_phase="trajectory", warm_start_checkpoint=guide_checkpoint' in s
    assert '"next_epsilon_warmup_checkpoint": str(out_dir / "objective_best" / "best_delta_std.npz")' in s
    assert 'guidance_phase' in s and 'warmup_transition_manifest' in s


def test_config_validates_new_spatial_controls_and_phase_steps():
    import importlib.util
    spec=importlib.util.spec_from_file_location("project_config_10164",ROOT/"config.py")
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    ProjectConfig=mod.ProjectConfig
    cfg=ProjectConfig.from_json(ROOT/'config.json')
    cfg.validate()
    cfg.guidance_steps=100
    cfg.trajectory_steps=200
    cfg.validate()
    for attr,value in [('guidance_steps',0),('trajectory_steps',0),('unrelated_loss_weight',-1),
                       ('unrelated_exclusion_radius_deg',2.),('input_mask_radius_deg',-1)]:
        clone=ProjectConfig.from_json(ROOT/'config.json')
        setattr(clone,attr,value)
        with pytest.raises(ValueError):clone.validate()
