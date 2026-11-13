"""CPU-only smoke tests for standalone reporting helpers, without HPCC datasets/GraphCast."""
import ast
import json
from datetime import datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "stages" / "06_attack_core.py"
NAMES = {
    "_unique_best_dir", "_append_best_manifest", "_probability_from_logits",
    "plot_surrogate_probability", "_tevis_rel_lon", "_tevis_center_lon", "_tevis_track_lonlat",
    "_tevis_extent", "_tevis_focus_geometry", "_tevis_crop_field",
    "ensure_two_class_logits", "standardized_to_raw", "_save_best_probability",
    "_downstream_comparison_maps", "_downstream_color_scales", "plot_downstream_guidance_difference",
    "save_loss_components", "_raw_downstream_to_standardized", "_render_stage_colorbars",
    "_load_saved_tensor_cpu",
}


def helpers():
    module = ast.parse(SOURCE.read_text())
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    namespace = {
        "np": np, "jax": jax, "jnp": jnp, "torch": torch, "plt": plt, "pd": pd,
        "json": json, "Path": Path, "datetime": datetime, "timedelta": timedelta,
        "GRID_H": 181, "GRID_W": 360, "LAT_DEG": np.linspace(-90, 90, 181),
        "LON_DEG": np.arange(360), "CH_WIND": 0, "CH_THICK": 1, "CH_MSLP": 2,
        "TE_VIS_DPI": 60, "DOWNSTREAM_STD": torch.ones(4),
        "DOWNSTREAM_MEAN": torch.zeros(4), "BEST_SAVE_PROBABILITY": True,
        "DOWNSTREAM_STAGE_COLORBAR": True, "TE_SAVE_ATTACK_DIFF_VIS": True,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace


def reference():
    time = datetime(2020, 10, 26, 6)
    original = {"nodes": [{"lon": 270, "lat": 20, "datetime": time}, {"lon": 272, "lat": 22, "datetime": time + timedelta(hours=6)}]}
    target = {"nodes": [{"lon": 270, "lat": 20, "datetime": time}, {"lon": 265, "lat": 23, "datetime": time + timedelta(hours=6)}]}
    return {"original_tracks": [original], "changed_target_pairs": [{"source_index": 0, "target_track": target}],
            "removed_original_tracks": []}


def context():
    return {"clean_s_np": np.zeros((2, 4, 181, 360), dtype=np.float32),
            "guide_np": np.full((2, 4, 181, 360), 0.5, dtype=np.float32)}


def test_unique_epoch_and_manifest(tmp_path):
    ns = helpers()
    save_dir = ns["_unique_best_dir"](tmp_path, "te", 9)
    next_dir = ns["_unique_best_dir"](tmp_path, "te", 9)
    assert save_dir != next_dir and "epoch_0010" in str(save_dir)
    row = {"stage_index": 2, "stage_kind": "warmup", "guide_delta": 10, "input_epsilon": 6,
           "type": "te", "epoch": 10, "step": 9, "directory": str(save_dir)}
    ns["_append_best_manifest"](tmp_path, row)
    ns["_append_best_manifest"](tmp_path, {**row, "directory": str(next_dir)})
    assert len(pd.read_csv(tmp_path / "best_manifest.csv")) == 2
    assert save_dir.exists()


def test_probability_is_softmax_and_image(tmp_path):
    ns = helpers()
    logits = jnp.stack([jnp.zeros((2, 181, 360)), jnp.ones((2, 181, 360))], axis=1)
    prob = ns["_probability_from_logits"](logits)
    assert prob.shape == (2, 181, 360)
    assert np.allclose(prob, 1 / (1 + np.exp(-1)), atol=1e-6)
    ns["_save_best_probability"](logits, reference(), tmp_path, "Test probability", full_array=True)
    assert (tmp_path / "surrogate_probability.npy").exists()
    assert (tmp_path / "surrogate_probability_heatmap.png").stat().st_size > 10_000


def test_loss_exports_have_separate_csv_and_plot(tmp_path):
    ns = helpers()
    rows = []
    for step in range(4):
        rows.append({"step": step, "epoch": step + 1, "total": 5 - step,
                     "guide_tgt": 1.0, "guide_orig": 2.0, "guide_core": 3.0,
                     "traj_core": 1.5, "reg": .5, "weighted_guide": 3.0 - step,
                     "weighted_traj": 1.5, "weighted_reg": .5,
                     "weighted_guide_tgt": 1.0, "weighted_guide_orig": 2.0})
    ns["save_loss_components"](rows, tmp_path, 1, 2)
    d = tmp_path / "loss_components"
    assert (d / "loss_reg.csv").exists() and (d / "loss_reg.png").stat().st_size > 500
    assert len(pd.read_csv(d / "loss_traj_core.csv")) == 4
    assert json.loads((d / "loss_identity_check.json").read_text())["max_absolute_difference"] < 1e-9


def test_shared_colorbars_across_rows_and_best_epochs(tmp_path):
    ns = helpers()
    ctx, ref = context(), reference()
    a = np.zeros((2, 4, 181, 360), dtype=np.float32)
    a[:, 0] = 3.0
    a[:, 1] = 10.0
    a[:, 2] = 20.0
    b = a * 2
    scale_a = ns["_downstream_color_scales"](ctx, a, ref)
    scale_b = ns["_downstream_color_scales"](ctx, b, ref)
    assert scale_a[0] < scale_b[0]
    folder = tmp_path / "best_history" / "te" / "epoch_0010_step_0009"
    folder.mkdir(parents=True)
    torch.save(torch.from_numpy(a), folder / "adversarial_downstream_input.pt")
    obj_dir, te_dir = tmp_path / "objective_best", tmp_path / "te_best"
    for directory, arr in [(obj_dir, b), (te_dir, a)]:
        directory.mkdir()
        torch.save(torch.from_numpy(arr), directory / "final_downstream_input.pt")
    scales = ns["_render_stage_colorbars"](tmp_path, ctx, ref, obj_dir, te_dir)
    assert scales[0] == scale_b[0]
    assert (folder / "te_best_adversarial_downstream_vs_guidance.png").exists()
    assert (obj_dir / "adversarial_downstream_vs_guidance.png").exists()
    assert json.loads((tmp_path / "downstream_colorbar_scale.json").read_text())["wind_ms"] == scale_b[0]


def test_te_best_persists_multiple_versions_and_updates_legacy_pointer(tmp_path):
    import os
    import shutil
    ns = helpers()
    module = ast.parse(SOURCE.read_text())
    selected = {"te_result_better", "te_check", "_update_legacy_te_artifacts"}
    additional = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    exec(compile(ast.fix_missing_locations(ast.Module(body=additional, type_ignores=[])), str(SOURCE), "exec"), ns)
    ns.update({"shutil": shutil, "os": os, "DOWNSTREAM_EPSILON": 10.0, "END2END_EPSILON": 3.0})
    call_count = [0]

    def fake_te_in(raw, sample, path):
        path.write_text("mock TE input")

    def fake_run(te_nc, nodes, tracks):
        nodes.write_text("")
        tracks.write_text("")

    def fake_score(pred, ref, candidates):
        call_count[0] += 1
        return {"primary_best_metric": 0.1 * call_count[0],
                "secondary_best_metric": 0.0, "DR_changed": 0.1 * call_count[0]}

    def fake_save(*args, **kwargs):
        folder = kwargs["artifact_dir"]
        (folder / "te_best.png").write_bytes(b"fake-map")
        torch.save(torch.zeros(2, 4, 181, 360), folder / "adversarial_downstream_input.pt")

    ns.update({"_te_downstream_to_nc": fake_te_in, "_te_run": fake_run,
               "read_track_file": lambda path: [], "_read_detected_nodes": lambda *args: [],
               "score_tracks_direct": fake_score, "_print_end2end_track_times": lambda *args: None,
               "_te_save_check_artifacts": fake_save, "_save_best_probability": lambda *args, **kwargs: None,
               "surrogate_bridge": lambda _: jnp.zeros((2, 2, 181, 360))})
    base = tmp_path / "te_checks"
    kwargs = {"downstream_raw": torch.zeros(2, 4, 181, 360), "sample": "2020-10-26T0000",
              "te_reference": reference(), "ctx": context(), "delta": {"one": jnp.zeros((2,))},
              "s_adv_std": np.zeros((2, 4, 181, 360)), "gc_output": object(),
              "stage_meta": {"stage_index": 1, "stage_kind": "warmup", "input_epsilon": 3, "guide_delta": 10}}
    result1 = ns["te_check"](work_dir=base / "stage_initial", step=-1, best_result=None, **kwargs)
    result2 = ns["te_check"](work_dir=base / "step_0009", step=9, best_result=result1, **kwargs)
    assert result1["is_new_te_best"] and result2["is_new_te_best"]
    assert Path(result1["history_dir"]).exists() and Path(result2["history_dir"]).exists()
    assert Path(result1["history_dir"]) != Path(result2["history_dir"])
    assert (tmp_path / "te_saved_states" / "te_best").resolve() == Path(result2["history_dir"])
    assert len(pd.read_csv(tmp_path / "best_manifest.csv")) == 2
    assert (tmp_path / "te_vis" / "te_best.png").read_bytes() == b"fake-map"
