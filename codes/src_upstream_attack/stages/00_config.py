# Stage 00: environment, paths, target-focused method configuration, resource profiling
PROJECT_CONFIG = globals().get("PROJECT_CONFIG", {})

SAVED_DOWNSTREAM_DELTAS = [float(x) for x in PROJECT_CONFIG.get("saved_downstream_deltas", [3.0, 6.0, 8.0, 10.0])]
FINAL_DELTAS = [float(x) for x in PROJECT_CONFIG.get("selected_deltas", [10.0])]
WARMUP_DELTAS = [float(x) for x in PROJECT_CONFIG.get("warmup_deltas", [])]
WARMUP_STEPS = int(PROJECT_CONFIG.get("warmup_steps", 300))
GUIDE_DELTA_CONFIG = PROJECT_CONFIG.get("guide_delta")
GUIDE_DELTA_CONFIG = None if GUIDE_DELTA_CONFIG is None else float(GUIDE_DELTA_CONFIG)

if WARMUP_DELTAS and len(FINAL_DELTAS) != 1:
    raise ValueError("Explicit warmup requires exactly one final selected delta.")
if WARMUP_DELTAS:
    _final_eps = float(FINAL_DELTAS[0])
    FIXED_GUIDE_EPSILON = float(GUIDE_DELTA_CONFIG if GUIDE_DELTA_CONFIG is not None else _final_eps)
    ATTACK_STAGE_SCHEDULE = [
        {"kind": "warmup", "input_epsilon": float(eps), "guide_epsilon": FIXED_GUIDE_EPSILON, "steps": WARMUP_STEPS}
        for eps in WARMUP_DELTAS
    ] + [
        {"kind": "final", "input_epsilon": _final_eps, "guide_epsilon": FIXED_GUIDE_EPSILON,
         "steps": int(PROJECT_CONFIG.get("end2end_steps", 1000))}
    ]
else:
    ATTACK_STAGE_SCHEDULE = []
    for eps in FINAL_DELTAS:
        guide_eps = float(GUIDE_DELTA_CONFIG if GUIDE_DELTA_CONFIG is not None else eps)
        ATTACK_STAGE_SCHEDULE.append({
            "kind": "final", "input_epsilon": float(eps), "guide_epsilon": guide_eps,
            "steps": int(PROJECT_CONFIG.get("end2end_steps", 1000)),
        })
    FIXED_GUIDE_EPSILON = None if GUIDE_DELTA_CONFIG is None else float(GUIDE_DELTA_CONFIG)

DELTA_SCHEDULE = [float(x["input_epsilon"]) for x in ATTACK_STAGE_SCHEDULE]
if not ATTACK_STAGE_SCHEDULE:
    raise ValueError("Attack stage schedule is empty.")

# Default globals point to the final stage so --only evaluate/aggregate resolves the final run.
_FINAL_STAGE = ATTACK_STAGE_SCHEDULE[-1]
DOWNSTREAM_EPSILON = float(_FINAL_STAGE["guide_epsilon"])
END2END_EPSILON = float(_FINAL_STAGE["input_epsilon"])
END2END_STEPS = int(_FINAL_STAGE["steps"])
END2END_VARIANT = "target_focused_surrogate"

METHOD_KEYS = [
    "guide_source", "guide_background_sigma_deg", "vortex_radius_deg",
    "core_loss", "region_radius_deg", "core_radius_deg", "moved_node_keep_deg",
    "center_weight_sigma_deg", "center_weight_floor",
    "channel_weight_wind", "channel_weight_thickness", "channel_weight_mslp",
    "prefix_tau_steps", "prefix_floor", "guide_loss_weight", "trajectory_loss_weight",
    "trajectory_target_radius", "trajectory_target_sigma", "trajectory_focal_gamma",
    "trajectory_zero_original_only_after_dilation", "reg_lambda", "reg_sigma_rad", "grad_sigma_rad",
    "input_core_radius_deg", "original_weight", "input_mask_radius_deg", "unrelated_exclusion_radius_deg", "unrelated_loss_weight",
    "surrogate_model_name", "surrogate_backbone_name", "surrogate_output_stride",
    "optimizer", "adam_beta1", "adam_beta2", "adam_eps", "adaptive_step_rejection",
    "pgd_step_size", "ti_sigma_cells", "gradient_dilation_vars", "dilation_scale", "lr_start", "lr_end", "grad_floor", "fp32_graphcast",
    "te_generation_weight", "objective_best_te_eval", "te_every", "te_tol_deg", "te_save_vis", "te_keep_text", "te_vis_dpi",
    "te_save_attack_diff_vis", "te_save_state_artifacts",
    "best_history_save_probability", "best_history_objective_figures",
    "best_history_te_state", "downstream_colorbar_stage_consistent",
]
_missing_method_keys = [k for k in METHOD_KEYS if k not in PROJECT_CONFIG]
if _missing_method_keys:
    raise KeyError(f"Missing method settings in PROJECT_CONFIG: {_missing_method_keys}")
METHOD_HYPERPARAMETERS = {k: PROJECT_CONFIG[k] for k in METHOD_KEYS}

GUIDE_SOURCE = str(PROJECT_CONFIG["guide_source"])
GUIDE_BACKGROUND_SIGMA_DEG = float(PROJECT_CONFIG["guide_background_sigma_deg"])
VORTEX_RADIUS_DEG = float(PROJECT_CONFIG["vortex_radius_deg"])
CORE_LOSS = str(PROJECT_CONFIG["core_loss"])
REGION_RADIUS_DEG = float(PROJECT_CONFIG["region_radius_deg"])
CORE_RADIUS_DEG = float(PROJECT_CONFIG["core_radius_deg"])
MOVED_NODE_KEEP_DEG = float(PROJECT_CONFIG["moved_node_keep_deg"])
CENTER_WEIGHT_SIGMA_DEG = float(PROJECT_CONFIG["center_weight_sigma_deg"])
CENTER_WEIGHT_FLOOR = float(PROJECT_CONFIG["center_weight_floor"])
CHANNEL_WEIGHTS = {
    "wind": float(PROJECT_CONFIG["channel_weight_wind"]),
    "thickness": float(PROJECT_CONFIG["channel_weight_thickness"]),
    "mslp": float(PROJECT_CONFIG["channel_weight_mslp"]),
}
PREFIX_TAU_STEPS = float(PROJECT_CONFIG["prefix_tau_steps"])
PREFIX_FLOOR = float(PROJECT_CONFIG["prefix_floor"])
GUIDE_LOSS_WEIGHT = float(PROJECT_CONFIG["guide_loss_weight"])
TRAJECTORY_LOSS_WEIGHT = float(PROJECT_CONFIG["trajectory_loss_weight"])
TRAJECTORY_TARGET_RADIUS = int(PROJECT_CONFIG["trajectory_target_radius"])
TRAJECTORY_TARGET_SIGMA = float(PROJECT_CONFIG["trajectory_target_sigma"])
TRAJECTORY_FOCAL_GAMMA = float(PROJECT_CONFIG["trajectory_focal_gamma"])
TRAJECTORY_ZERO_ORIGINAL_ONLY_AFTER_DILATION = bool(PROJECT_CONFIG["trajectory_zero_original_only_after_dilation"])
REG_LAMBDA = float(PROJECT_CONFIG["reg_lambda"])
REG_SIGMA_RAD = float(PROJECT_CONFIG["reg_sigma_rad"])
GRAD_SIGMA_RAD = float(PROJECT_CONFIG["grad_sigma_rad"])
INPUT_CORE_RADIUS_DEG = float(PROJECT_CONFIG["input_core_radius_deg"])
INPUT_MASK_RADIUS_DEG = float(PROJECT_CONFIG["input_mask_radius_deg"])
UNRELATED_EXCLUSION_RADIUS_DEG = float(PROJECT_CONFIG["unrelated_exclusion_radius_deg"])
UNRELATED_LOSS_WEIGHT = float(PROJECT_CONFIG["unrelated_loss_weight"])
GUIDANCE_PHASE_STEPS = PROJECT_CONFIG.get("guidance_steps")
TRAJECTORY_PHASE_STEPS = PROJECT_CONFIG.get("trajectory_steps")
ACTIVE_OBJECTIVE_PHASE = "joint"
ORIGINAL_WEIGHT = float(PROJECT_CONFIG["original_weight"])

SURROGATE_MODEL_NAME = str(PROJECT_CONFIG["surrogate_model_name"])
SURROGATE_BACKBONE_NAME = str(PROJECT_CONFIG["surrogate_backbone_name"])
SURROGATE_OUTPUT_STRIDE = int(PROJECT_CONFIG["surrogate_output_stride"])

OPTIMIZER = str(PROJECT_CONFIG["optimizer"])
ADAM_BETA1 = float(PROJECT_CONFIG["adam_beta1"])
ADAM_BETA2 = float(PROJECT_CONFIG["adam_beta2"])
ADAM_EPS = float(PROJECT_CONFIG["adam_eps"])
ADAPTIVE_STEP_REJECTION = bool(PROJECT_CONFIG["adaptive_step_rejection"])
PGD_STEP_SIZE = float(PROJECT_CONFIG["pgd_step_size"])
TI_SIGMA_CELLS = float(PROJECT_CONFIG["ti_sigma_cells"])
GRADIENT_DILATION_VARS = tuple(PROJECT_CONFIG["gradient_dilation_vars"])
DILATION_SCALE_MAP = {"wind": 1.0, "others": 1.0}
for _item in PROJECT_CONFIG["dilation_scale"]:
    _family, _, _value = _item.partition(":")
    DILATION_SCALE_MAP[_family] = float(_value)
LR_START = float(PROJECT_CONFIG["lr_start"])
LR_END = float(PROJECT_CONFIG["lr_end"])
GRAD_FLOOR = float(PROJECT_CONFIG["grad_floor"])
FP32_GRAPHCAST = bool(PROJECT_CONFIG["fp32_graphcast"])

TE_GENERATION_WEIGHT = float(PROJECT_CONFIG["te_generation_weight"])
OBJECTIVE_BEST_TE_EVAL = bool(PROJECT_CONFIG["objective_best_te_eval"])
TE_EVERY = int(PROJECT_CONFIG["te_every"])
TE_TOL_DEG = float(PROJECT_CONFIG["te_tol_deg"])
TE_SAVE_VIS = bool(PROJECT_CONFIG["te_save_vis"])
TE_KEEP_TEXT = bool(PROJECT_CONFIG["te_keep_text"])
TE_VIS_DPI = int(PROJECT_CONFIG["te_vis_dpi"])
TE_SAVE_ATTACK_DIFF_VIS = bool(PROJECT_CONFIG["te_save_attack_diff_vis"])
TE_SAVE_STATE_ARTIFACTS = bool(PROJECT_CONFIG["te_save_state_artifacts"])
BEST_SAVE_PROBABILITY = bool(PROJECT_CONFIG["best_history_save_probability"])
BEST_OBJECTIVE_FIGURES = bool(PROJECT_CONFIG["best_history_objective_figures"])
BEST_TE_STATE = bool(PROJECT_CONFIG["best_history_te_state"])
DOWNSTREAM_STAGE_COLORBAR = bool(PROJECT_CONFIG["downstream_colorbar_stage_consistent"])

TRACK_MATCH_TOLERANCE_DEG = TE_TOL_DEG
TRACK_MATCH_TIME_TOLERANCE_HOURS = 0.0

import contextlib
import copy
import dataclasses
import functools
import gc
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

RUN_TAG = str(
    PROJECT_CONFIG.get("run_tag")
    or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
).strip()
if not RUN_TAG:
    raise ValueError("run_tag cannot be empty")
if not all(c.isalnum() or c in "._-" for c in RUN_TAG):
    raise ValueError(
        "run_tag may contain only letters, numbers, '.', '_' and '-'"
    )
RUN_SUFFIX = f"__run_{RUN_TAG}"
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import psutil
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from IPython.display import display

BUILD_TARGETS = bool(PROJECT_CONFIG.get("build_targets", False))
OVERWRITE = bool(PROJECT_CONFIG.get("overwrite", False))
SEED = int(PROJECT_CONFIG.get("seed", 42))

HPCC_SCRATCH_BASE = Path("/mnt/scratch/dengyue1")
HPCC_SCRATCH_ROOT = Path("/mnt/scratch/dengyue1/My_Drive")
HPCC_HOME_ROOT = Path("/mnt/ffs24/home/dengyue1")

SCRATCH_BASE = Path(PROJECT_CONFIG.get("scratch_base") or HPCC_SCRATCH_BASE).expanduser()
SCRATCH_ROOT = Path(PROJECT_CONFIG.get("scratch_root") or HPCC_SCRATCH_ROOT).expanduser()
SOFTWARE_ROOT = Path(PROJECT_CONFIG.get("software_root") or (HPCC_HOME_ROOT / "software")).expanduser()

CLIMATE_ROOT = SCRATCH_ROOT / "Climate-data"
DATA_DIR = CLIMATE_ROOT / "Graphcast_1deg"
EXP_ROOT = DATA_DIR / "case_study_end2end_all"

GRAPHCAST_PACKAGE_ROOT = SOFTWARE_ROOT / "graphcast-paper-used"
TEMPEST_ROOT = SOFTWARE_ROOT / "tempestextremes-master"
DETECT_BIN = TEMPEST_ROOT / "build" / "src" / "nodes" / "DetectNodes"
STITCH_BIN = TEMPEST_ROOT / "build" / "src" / "nodes" / "StitchNodes"
GRAPHCAST_ASSET_ROOT = SOFTWARE_ROOT / "graphcast_assets"

_default_surrogate_candidates = [
    SOFTWARE_ROOT / "TempestExtremes_surrogate" / "DeepLabV3Plus-Pytorch-master",
    SOFTWARE_ROOT / "DeepLabV3Plus-Pytorch-master",
    SOFTWARE_ROOT.parent / "TempestExtremes_surrogate" / "DeepLabV3Plus-Pytorch-master",
    SCRATCH_ROOT / "TempestExtremes_surrogate" / "DeepLabV3Plus-Pytorch-master",
    SCRATCH_ROOT / "DeepLabV3Plus-Pytorch-master",
]
if PROJECT_CONFIG.get("surrogate_repo"):
    SURROGATE_REPO = Path(PROJECT_CONFIG["surrogate_repo"]).expanduser()
else:
    SURROGATE_REPO = next((p for p in _default_surrogate_candidates if p.exists()), _default_surrogate_candidates[0])
SURROGATE_CKPT = Path(
    PROJECT_CONFIG.get("surrogate_checkpoint")
    or (DATA_DIR / f"best_{SURROGATE_MODEL_NAME}_{SURROGATE_BACKBONE_NAME}_focal_trajectory_with_heatmap2.pt")
).expanduser()

ELEVATION_NC = DATA_DIR / "elevation" / "graphcast_surface_elevation.nc"

for p in [EXP_ROOT, EXP_ROOT / "summaries", EXP_ROOT / "figures", EXP_ROOT / "configs", GRAPHCAST_ASSET_ROOT]:
    p.mkdir(parents=True, exist_ok=True)
for p in [GRAPHCAST_PACKAGE_ROOT, SURROGATE_REPO]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TC2_CASES = [
    "2008-04-27T0000", "2010-10-12T0000", "2013-11-03T0000",
    "2018-09-20T0000", "2019-08-02T0000", "2020-10-26T0000",
    "2022-09-09T0000", "2017-08-17T0000", "2017-09-16T0000",
    "2011-08-21T0000",
]
RUN_CASES = list(PROJECT_CONFIG.get("run_cases") or TC2_CASES)
unknown_cases = [x for x in RUN_CASES if x not in TC2_CASES]
if unknown_cases:
    raise ValueError(f"Unknown run_cases: {unknown_cases}")
FORECAST_STEPS = 38
GRID_H, GRID_W = 181, 360
PREFIX = "graphcast"
PREDICTION_GOAL = "trajectory"

ALL_TARGET_SCENARIOS = {
    "distance_1of3": {"mode": "euclidean", "ratio": 1 / 3},
    "distance_1of2": {"mode": "euclidean", "ratio": 1 / 2},
    "distance_2of3": {"mode": "euclidean", "ratio": 2 / 3},
    "remove_changed_original": {"mode": "remove_changed_original", "ratio": None},
    "distance_1of12": {"mode": "euclidean", "ratio": 1 / 12},
    "distance_1of6": {"mode": "euclidean", "ratio": 1 / 6},
    "distance_1of4": {"mode": "euclidean", "ratio": 1 / 4},
    "submission_full": {"mode": "submission_full", "ratio": None},
    "distance_manual_extra": {"mode": "distance", "ratio": None},
}

TARGET_IDS = {
    "distance_1of3": "target_01",
    "distance_1of2": "target_02",
    "distance_2of3": "target_03",
    "remove_changed_original": "target_04",
    "distance_1of12": "target_05",
    "distance_1of6": "target_06",
    "distance_1of4": "target_07",
    "submission_full": "target_08",
    "distance_manual_extra": "target_09",
}

TARGET_TAGS_ALL = [
    "distance_1of3",
    "distance_1of2",
    "distance_2of3",
    "remove_changed_original",
    "distance_1of12",
    "distance_1of6",
    "distance_1of4",
    "submission_full",
    "distance_manual_extra",
]

DIFFICULTY_TO_TARGET_TAG = {
    "easy": "distance_1of3",
    "medium": "distance_1of2",
    "hard": "distance_2of3",
    "remove_only": "remove_changed_original",
}

_configured_target_tags = PROJECT_CONFIG.get("selected_targets")
_configured_difficulties = PROJECT_CONFIG.get("selected_difficulties")

if _configured_target_tags:
    _selected_target_tags = list(dict.fromkeys(str(x) for x in _configured_target_tags))
    unknown_targets = [x for x in _selected_target_tags if x not in TARGET_TAGS_ALL]
    if unknown_targets:
        raise ValueError(
            f"Unknown selected_targets: {unknown_targets}. "
            f"Choose from {TARGET_TAGS_ALL}."
        )
    _selected_difficulties = []
elif _configured_difficulties:
    _selected_difficulties = list(_configured_difficulties)
    unknown_difficulties = [x for x in _selected_difficulties if x not in DIFFICULTY_TO_TARGET_TAG]
    if unknown_difficulties:
        raise ValueError(
            f"Unknown difficulty labels {unknown_difficulties}. "
            f"Choose from {list(DIFFICULTY_TO_TARGET_TAG)}."
        )
    _selected_target_tags = [DIFFICULTY_TO_TARGET_TAG[x] for x in _selected_difficulties]
else:
    _selected_difficulties = []
    _selected_target_tags = [tag for tag in TARGET_TAGS_ALL if tag not in {"submission_full", "distance_manual_extra"}]

TARGET_SCENARIOS = {tag: ALL_TARGET_SCENARIOS[tag] for tag in _selected_target_tags}
SELECTED_DIFFICULTIES = list(_selected_difficulties)
SELECTED_TARGET_TAGS = list(_selected_target_tags)

print("Configured target tags:", TARGET_TAGS_ALL)
print("Selected target tags:", SELECTED_TARGET_TAGS)

# ---------------------------------------------------------------------------
# TempestExtremes input channels (order used everywhere in this project)
# ---------------------------------------------------------------------------
DOWNSTREAM_MEAN = torch.tensor([6.3266, 383.84, 35398.0, 100930.0], dtype=torch.float32)
DOWNSTREAM_STD = torch.tensor([3.7596, 856.8956, 1829.8486, 1372.3270], dtype=torch.float32)
DOWNSTREAM_VARS = ["wind_speed_10m", "elevation", "thickness", "mean_sea_level_pressure"]
CH_WIND, CH_ELEV, CH_THICK, CH_MSLP = 0, 1, 2, 3
GUIDE_CHANNELS = (CH_WIND, CH_THICK, CH_MSLP)   # elevation is static and never guided

END2END_STEPS = int(PROJECT_CONFIG.get("end2end_steps", 1000))
END2END_VERBOSE_EVERY = int(PROJECT_CONFIG.get("verbose_every", 10))

# ---------------------------------------------------------------------------
# Attackable GraphCast inputs: full field, with configured pressure levels.
# ---------------------------------------------------------------------------

#PERTURB_INPUT_VARS = [
#    "10m_u_component_of_wind",
#    "10m_v_component_of_wind",
#    "mean_sea_level_pressure",
#    "geopotential",
#    "u_component_of_wind",
#    "v_component_of_wind",
#    "temperature",
#    "specific_humidity",
#]
#
#PERTURB_LEVELS = {
#    "geopotential": (300, 500),
#    "u_component_of_wind": (200, 300, 500, 700, 850),
#    "v_component_of_wind": (200, 300, 500, 700, 850),
#    "temperature": (300, 400, 500),
#    "specific_humidity": (700, 850),
#}
#NONNEGATIVE_INPUT_VARS = {"specific_humidity"}

PERTURB_INPUT_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
]

PERTURB_LEVELS = {
    "geopotential": (300, 500),
    "u_component_of_wind": (300, 500, 700, 850),
    "v_component_of_wind": (300, 500, 700, 850),
    "temperature": (300, 500),
}

NONNEGATIVE_INPUT_VARS = set()

_ATTACK_CONFIG_PAYLOAD = {
    "variant": END2END_VARIANT,
    "method_hyperparameters": METHOD_HYPERPARAMETERS,
    "attack_stage_schedule": ATTACK_STAGE_SCHEDULE,
    "final_deltas": list(FINAL_DELTAS),
    "warmup_deltas": list(WARMUP_DELTAS),
    "warmup_steps": int(WARMUP_STEPS),
    "guide_delta_config": GUIDE_DELTA_CONFIG,
    "surrogate_model_name": SURROGATE_MODEL_NAME,
    "surrogate_backbone_name": SURROGATE_BACKBONE_NAME,
    "surrogate_checkpoint": str(SURROGATE_CKPT),
    "perturb_input_vars": PERTURB_INPUT_VARS,
    "perturb_levels": {k: list(v) for k, v in PERTURB_LEVELS.items()},
    "allowed_deltas": list(SAVED_DOWNSTREAM_DELTAS),
    "selected_difficulties": list(SELECTED_DIFFICULTIES),
    "selected_target_tags": list(SELECTED_TARGET_TAGS),
    "target_scenarios": sorted(TARGET_SCENARIOS),
    "run_cases": list(RUN_CASES),
    "final_end2end_steps": int(PROJECT_CONFIG.get("end2end_steps", 1000)),
}
ATTACK_CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(_ATTACK_CONFIG_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()[:10]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def _gpu_used_mb():
    """Approximate GPU memory owned by this Python PID, in MiB."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pid = os.getpid()
        total = 0.0
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) == pid:
                total += float(parts[1])
        return total
    except Exception:
        return 0.0


class ResourceProfiler:
    def __init__(self, stage, sample, target_tag, out_dir, interval=0.25):
        self.stage, self.sample, self.target_tag = stage, sample, target_tag
        self.out_dir, self.interval = Path(out_dir), interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.peak_rss = self.process.memory_info().rss
        self.peak_gpu = _gpu_used_mb()

    def _poll(self):
        while not self.stop_event.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.peak_gpu = max(self.peak_gpu, _gpu_used_mb())
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.start_rss = self.process.memory_info().rss
        self.start_gpu = _gpu_used_mb()
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        elapsed = time.perf_counter() - self.start
        end_rss = self.process.memory_info().rss
        end_gpu = _gpu_used_mb()
        peak_cuda = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        self.record = {
            "stage": self.stage,
            "sample": self.sample,
            "target_id": get_target_id(self.target_tag) if self.target_tag in TARGET_IDS else self.target_tag,
            "target_tag": self.target_tag,
            "elapsed_seconds": elapsed,
            "rss_start_mb": self.start_rss / 2**20,
            "rss_end_mb": end_rss / 2**20,
            "rss_peak_mb": self.peak_rss / 2**20,
            "rss_increment_peak_mb": max(0.0, (self.peak_rss - self.start_rss) / 2**20),
            "gpu_start_mb": self.start_gpu,
            "gpu_end_mb": end_gpu,
            "gpu_peak_poll_mb": self.peak_gpu,
            "torch_cuda_peak_allocated_mb": peak_cuda,
        }
        with open(self.out_dir / "runtime_memory.json", "w") as f:
            json.dump(self.record, f, indent=2)
        return False


def require_paths(*paths):
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def file_sha256(path, chunk_bytes=2**20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sample_target_dir(sample, target_tag):
    return EXP_ROOT / sample / target_tag


def number_tag(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def downstream_delta_tag(epsilon=None):
    epsilon = DOWNSTREAM_EPSILON if epsilon is None else epsilon
    return f"delta_{number_tag(epsilon)}"


def end2end_epsilon_tag(epsilon=None):
    epsilon = END2END_EPSILON if epsilon is None else epsilon
    return f"input_epsilon_{number_tag(epsilon)}"


def get_target_id(target_tag):
    if target_tag not in TARGET_IDS:
        raise KeyError(f"Unknown target tag: {target_tag}")
    return TARGET_IDS[target_tag]


def target_experiment_tag(target_tag):
    return f"{get_target_id(target_tag)}__{target_tag}"


def downstream_sample_dir(sample, target_tag, epsilon=None):
    return sample_target_dir(sample, target_tag) / "downstream" / downstream_delta_tag(epsilon)


def end2end_run_tag(downstream_epsilon=None, end2end_epsilon=None):
    base = f"{downstream_delta_tag(downstream_epsilon)}__{end2end_epsilon_tag(end2end_epsilon)}"
    return (
        f"{base}__{END2END_VARIANT}_{ATTACK_CONFIG_FINGERPRINT}"
        f"{RUN_SUFFIX}"
    )


def set_attack_stage(input_epsilon, guide_epsilon, steps):
    global DOWNSTREAM_EPSILON, END2END_EPSILON, END2END_STEPS
    END2END_EPSILON = float(input_epsilon)
    DOWNSTREAM_EPSILON = float(guide_epsilon)
    END2END_STEPS = int(steps)
    if END2END_EPSILON <= 0 or DOWNSTREAM_EPSILON <= 0 or END2END_STEPS <= 0:
        raise ValueError("Attack stage epsilon/guide epsilon/steps must be positive.")
    return END2END_EPSILON


def end2end_experiment_tag(target_tag, downstream_epsilon=None, end2end_epsilon=None):
    return f"{target_experiment_tag(target_tag)}__{end2end_run_tag(downstream_epsilon, end2end_epsilon)}"


def end2end_sample_dir(sample, target_tag, downstream_epsilon=None, end2end_epsilon=None):
    return sample_target_dir(sample, target_tag) / "end2end" / end2end_run_tag(downstream_epsilon, end2end_epsilon)


def end2end_summary_dir(downstream_epsilon=None, end2end_epsilon=None):
    return EXP_ROOT / "summaries" / "end2end" / end2end_run_tag(downstream_epsilon, end2end_epsilon)


def end2end_figure_dir(downstream_epsilon=None, end2end_epsilon=None):
    return EXP_ROOT / "figures" / "end2end" / end2end_run_tag(downstream_epsilon, end2end_epsilon)


def warmup_schedule_tag():
    parts = [f"x{number_tag(s['input_epsilon'])}s{int(s['steps'])}" for s in ATTACK_STAGE_SCHEDULE]
    guides = sorted(set(float(s["guide_epsilon"]) for s in ATTACK_STAGE_SCHEDULE))
    gtag = "guide_" + "_".join(number_tag(x) for x in guides)
    return "surrogate_warmup_" + "_".join(parts) + "__" + gtag


WARMUP_SUMMARY_DIR = EXP_ROOT / "summaries" / "warmup" / f"{warmup_schedule_tag()}__{END2END_VARIANT}_{ATTACK_CONFIG_FINGERPRINT}{RUN_SUFFIX}"
WARMUP_FIGURE_DIR = EXP_ROOT / "figures" / "warmup" / f"{warmup_schedule_tag()}__{END2END_VARIANT}_{ATTACK_CONFIG_FINGERPRINT}{RUN_SUFFIX}"
WARMUP_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
WARMUP_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

for p in [end2end_summary_dir(), end2end_figure_dir()]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = EXP_ROOT / "configs" / f"end2end_config_{warmup_schedule_tag()}__{END2END_VARIANT}_{ATTACK_CONFIG_FINGERPRINT}{RUN_SUFFIX}.json"
config = {
    "run_tag": RUN_TAG,
    "strategy": "target_focused_guide_plus_surrogate_trajectory",
    "tc2_cases": TC2_CASES,
    "run_cases": RUN_CASES,
    "target_scenarios": TARGET_SCENARIOS,
    "selected_difficulties": SELECTED_DIFFICULTIES,
    "selected_target_tags": SELECTED_TARGET_TAGS,
    "target_ids": TARGET_IDS,
    "attack_stage_schedule": ATTACK_STAGE_SCHEDULE,
    "final_deltas": list(FINAL_DELTAS),
    "warmup_deltas": list(WARMUP_DELTAS),
    "warmup_steps": int(WARMUP_STEPS),
    "guide_delta_config": GUIDE_DELTA_CONFIG,
    "surrogate_model_name": SURROGATE_MODEL_NAME,
    "surrogate_backbone_name": SURROGATE_BACKBONE_NAME,
    "surrogate_checkpoint": str(SURROGATE_CKPT),
    "allowed_deltas": list(SAVED_DOWNSTREAM_DELTAS),
    "upstream_delta_continuation": len(ATTACK_STAGE_SCHEDULE) > 1,
    "warm_start_selection": "objective_best",
    "fixed_saved_downstream_guidance_during_explicit_warmup": bool(WARMUP_DELTAS),
    "end2end_epsilon_initial": END2END_EPSILON,
    "end2end_steps": END2END_STEPS,
    "end2end_run_tag": end2end_run_tag(),
    "variant": END2END_VARIANT,
    "attack_config_fingerprint": ATTACK_CONFIG_FINGERPRINT,
    "method_hyperparameters": METHOD_HYPERPARAMETERS,
    "perturb_input_vars": PERTURB_INPUT_VARS,
    "perturb_levels": {k: list(v) for k, v in PERTURB_LEVELS.items()},
    "gradient_dilation_vars": list(GRADIENT_DILATION_VARS),
    "shared_distance_weighting_vars": list(GRADIENT_DILATION_VARS),
    "nonnegative_input_vars": sorted(NONNEGATIVE_INPUT_VARS),
    "TE_used_in_learning": False,
    "python_executable": sys.executable,
    "scratch_base": str(SCRATCH_BASE),
    "scratch_root": str(SCRATCH_ROOT),
    "software_root": str(SOFTWARE_ROOT),
    "data_dir": str(DATA_DIR),
    "experiment_root": str(EXP_ROOT),
    "graphcast_package_root": str(GRAPHCAST_PACKAGE_ROOT),
    "tempest_root": str(TEMPEST_ROOT),
    "graphcast_asset_root": str(GRAPHCAST_ASSET_ROOT),
    "surrogate_repo": str(SURROGATE_REPO),
    "surrogate_checkpoint": str(SURROGATE_CKPT),
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print("Python:", sys.executable)
print("DEVICE:", DEVICE)
print("Strategy: target-focused guide + downstream surrogate trajectory loss | guide source:", GUIDE_SOURCE, "| core loss:", CORE_LOSS, "| original weight:", ORIGINAL_WEIGHT)
print("Attack stage schedule:", ATTACK_STAGE_SCHEDULE)
print("JAX devices:", jax.devices())
print("SCRATCH root:", SCRATCH_ROOT)
print("Software root:", SOFTWARE_ROOT)
print("Data dir:", DATA_DIR)
print("Experiment root:", EXP_ROOT)
print("Final upstream input delta:", end2end_epsilon_tag())
print("Fixed/final saved-downstream guide delta:", downstream_delta_tag())
print("Surrogate repo:", SURROGATE_REPO)
print("Surrogate checkpoint:", SURROGATE_CKPT)
print("Attack variables (configured levels, full field):", PERTURB_INPUT_VARS)
print("Attack config fingerprint:", ATTACK_CONFIG_FINGERPRINT)
print("Run tag:", RUN_TAG)
print("End-to-end run:", end2end_run_tag())
print("Configuration saved to:", CONFIG_PATH)
