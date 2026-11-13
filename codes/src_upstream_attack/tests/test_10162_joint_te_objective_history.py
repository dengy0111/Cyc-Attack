"""10_16_2 regression tests; no HPC datasets, GraphCast or TempestExtremes needed."""
import ast
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import jax

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stages/06_attack_core.py"


def load_names(namespace, *names):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def tracks():
    now = datetime(2020, 10, 26, 6)
    times = [now + timedelta(hours=6*i) for i in range(4)]
    def track(lon, count=4):
        return {"nodes": [{"datetime": t, "lon": lon, "lat": 15.0} for t in times[:count]]}
    orig, target = track(10), track(25)
    reference = {"target_tracks": [target], "original_tracks": [orig],
                 "source_indices": [0], "changed_indices": [0],
                 "removed_indices": [0], "target_mode": "distance_1of3"}
    return orig, target, reference


def score_ns(weight=0.5):
    import math
    ns = {"np":np, "math":math, "TE_GENERATION_WEIGHT":weight, "TE_TOL_DEG":1.0}
    return load_names(ns, "_haversine_deg", "_direct_track_overlap",
                      "_direct_best_track_match", "score_tracks_direct", "te_result_better")


def test_joint_score_rewards_removal_even_with_unchanged_generation():
    ns = score_ns()
    orig, target, ref = tracks()
    baseline = ns['score_tracks_direct']([orig], ref)
    removed = ns['score_tracks_direct']([], ref)
    assert baseline['mean_target_overlap'] == removed['mean_target_overlap'] == 0
    assert baseline['mean_removed_original_overlap'] == 1
    assert removed['mean_removed_original_overlap'] == 0
    assert baseline['te_joint_score'] == 0
    assert removed['te_joint_score'] == .5
    assert ns['te_result_better'](removed, baseline)


def test_joint_score_rewards_generation_even_with_unchanged_removal():
    ns = score_ns()
    orig, target, ref = tracks()
    before = ns['score_tracks_direct']([orig], ref)
    after = ns['score_tracks_direct']([orig, target], ref)
    assert before['mean_removed_original_overlap'] == after['mean_removed_original_overlap'] == 1
    assert after['mean_target_overlap'] == 1
    assert after['te_joint_score'] == .5
    assert ns['te_result_better'](after, before)


def test_both_better_than_either_one_and_threshold_metric_retained():
    ns = score_ns()
    orig, target, ref = tracks()
    best = ns['score_tracks_direct']([target], ref)
    assert best['te_generation_score'] == 1
    assert best['te_removal_score'] == 1
    assert best['te_joint_score'] == 1
    assert best['DR_changed'] == 1
    assert best['remove_success'] == 1
    assert best['te_joint_success'] == 1
    same = ns['score_tracks_direct']([target], ref)
    assert not ns['te_result_better'](same, best)
    half_gen = score_ns(.75)['score_tracks_direct']([target], ref)
    assert half_gen['te_joint_score'] == 1


def test_remove_only_keeps_original_mode_policy():
    ns = score_ns()
    orig, target, ref = tracks()
    ref['target_mode'] = 'remove_changed_original'
    exists = ns['score_tracks_direct']([orig], ref)
    missing = ns['score_tracks_direct']([], ref)
    assert exists['primary_best_metric'] == 0
    assert missing['primary_best_metric'] == 1
    assert ns['te_result_better'](missing, exists)


def test_config_defaults_and_bounds():
    from config import ProjectConfig
    config = ProjectConfig.from_json(ROOT / 'config.json')
    config.validate()
    assert config.optimizer == 'pgd'
    assert config.objective_best_te_eval is True
    assert config.te_generation_weight == .5
    config.te_generation_weight = 1.1
    with pytest.raises(ValueError):
        config.validate()


def test_te_check_also_archives_non_te_best_objective_epoch(tmp_path):
    ns = score_ns()
    orig, target, ref = tracks()
    ns.update({'Path':Path,'np':np,'json':json,'jax':jax,'shutil':shutil,
               'datetime':datetime,'timedelta':timedelta,
               'TE_GENERATION_WEIGHT':.5, 'DOWNSTREAM_EPSILON':10.,
               'END2END_EPSILON':3.0})
    saved = []
    def fake_artifacts(*args, **kwargs):
        saved.append(('artifacts', kwargs['artifact_name']))
        p = Path(kwargs['artifact_dir'])
        p.mkdir(exist_ok=True, parents=True)
        for filename in ('objective_best.png', 'objective_best_tracks.txt', 'objective_best_nodes.txt',
                         'objective_best_adversarial_x_minus_original_x.png',
                         'objective_best_adversarial_x_minus_original_x_std.png',
                         'objective_best_adversarial_downstream_vs_guidance.png',
                         'objective_best_adversarial_downstream_vs_guidance_std.png'):
            (p / filename).write_text('test output')
    def fake_prob(*args, **kwargs):
        saved.append(('prob',kwargs.get('full_array')))
        p=Path(args[2]); np.save(p/'surrogate_probability.npy',np.ones((4,2,2)))
    ns.update({
        '_te_downstream_to_nc': lambda raw, case, p: Path(p).write_text('fake'),
        '_te_run': lambda netcdf, nodes, path: (Path(nodes).write_text('fake'),Path(path).write_text('fake')),
        'read_track_file': lambda path: [orig],
        '_read_detected_nodes': lambda nodes, start: [],
        '_print_end2end_track_times': lambda *args: None,
        '_te_save_check_artifacts': fake_artifacts,
        '_save_best_probability': fake_prob,
        '_unique_best_dir': lambda out, typ, step: Path(out)/'history'/typ/f'step{step}',
        '_append_best_manifest': lambda *args: None,
    })
    # In this synthetic case TE is NOT better than previous best, yet the
    # Objective checkpoint must receive its complete epoch-specific TE artifacts.
    load_names(ns,'te_check')
    baseline=ns['score_tracks_direct']([target],ref)
    work_dir=tmp_path/'te_checks'/'step_0002'
    objective_dir=tmp_path/'best_history'/'objective'/'epoch_0003'
    objective_dir.mkdir(parents=True)
    result=ns['te_check'](np.zeros((4,4,2,2)), '2020-10-26T0000', ref, work_dir,
       ctx={'te_reference':ref}, delta={'wind':jax.numpy.zeros((1,))}, s_adv_std=np.zeros((4,4,2,2)),
       gc_output='placeholder', best_result=baseline, step=2, logits='placeholder',
       objective_history_dir=objective_dir)
    assert result['is_new_te_best'] is False
    assert ('artifacts','objective_best') in saved
    assert ('prob',True) in saved
    for filename in ('te_track_map.png','te_tracks.txt','adversarial_x_minus_original_x_std.png',
                     'adversarial_downstream_vs_guidance_std.png', 'surrogate_probability.npy',
                     'objective_te_metrics.json'):
        assert (objective_dir/filename).exists(), filename
    assert json.loads((objective_dir/'objective_te_metrics.json').read_text())['te_evaluated_at_exact_objective_epoch']


def test_loop_wires_objective_checkpoint_to_real_te_evaluation():
    source = SOURCE.read_text()
    assert 'or (objective_improved and OBJECTIVE_BEST_TE_EVAL)' in source
    assert 'objective_history_dir=objective_history_dir if objective_improved else None' in source
    assert 'objective_history_dir=event_dir if final_loss_f < best_objective_loss_before_final else None' in source
    assert 'for best_type in ("te", "objective"):' in source
