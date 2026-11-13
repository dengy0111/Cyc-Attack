from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DIFFICULTY_TO_TARGET_TAG = {
    "easy": "distance_1of3",
    "medium": "distance_1of2",
    "hard": "distance_2of3",
    "remove_only": "remove_changed_original",
}

ALL_TARGET_TAGS = [
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

DEFAULT_SAVED_DELTAS = [3.0, 6.0, 8.0, 10.0]
GUIDE_SOURCES = ("relocation", "saved_downstream")
CORE_LOSSES = ("hinge", "mse")


@dataclass
class ProjectConfig:
    selected_difficulties: list[str] | None = None
    selected_targets: list[str] | None = None
    selected_deltas: list[float] = field(default_factory=lambda: [10.0])
    saved_downstream_deltas: list[float] = field(default_factory=lambda: DEFAULT_SAVED_DELTAS.copy())
    guide_delta: float | None = None
    warmup_deltas: list[float] = field(default_factory=list)
    warmup_steps: int = 300
    guidance_steps: int | None = None
    trajectory_steps: int | None = None
    run_cases: list[str] | None = None
    build_targets: bool = False
    overwrite: bool = False
    seed: int = 42
    end2end_steps: int = 1000
    verbose_every: int = 10
    run_tag: str | None = None
    scratch_base: str | None = None
    scratch_root: str | None = None
    software_root: str | None = None
    surrogate_repo: str | None = None
    surrogate_checkpoint: str | None = None

    guide_source: str = "saved_downstream"
    guide_background_sigma_deg: float = 8.0
    vortex_radius_deg: float = 6.0

    core_loss: str = "mse"
    region_radius_deg: float = 6.0
    core_radius_deg: float = 2.0
    moved_node_keep_deg: float = 3.0
    center_weight_sigma_deg: float = 3.0
    center_weight_floor: float = 0.1
    channel_weight_wind: float = 1.0
    channel_weight_thickness: float = 1.0
    channel_weight_mslp: float = 1.0
    prefix_tau_steps: float = 8.0
    prefix_floor: float = 1.0
    guide_loss_weight: float = 1.0
    trajectory_loss_weight: float = 1.0
    trajectory_target_radius: int = 1
    trajectory_target_sigma: float = 1.0
    trajectory_focal_gamma: float = 2.0
    trajectory_zero_original_only_after_dilation: bool = True
    reg_lambda: float = 1e-3
    reg_sigma_rad: float = 0.03
    grad_sigma_rad: float = 0.03
    input_core_radius_deg: float = 1.0
    input_mask_radius_deg: float = 20.0  # 0 disables strict input spatial mask
    unrelated_exclusion_radius_deg: float = 10.0
    unrelated_loss_weight: float = 0.05
    original_weight: float = 1.0

    surrogate_model_name: str = "deeplabv3plus"
    surrogate_backbone_name: str = "xception"
    surrogate_output_stride: int = 2

    optimizer: str = "pgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adaptive_step_rejection: bool = False
    pgd_step_size: float = 0.01
    ti_sigma_cells: float = 1.0
    gradient_dilation_vars: list[str] = field(default_factory=lambda: ["wind"])
    dilation_scale: list[str] = field(default_factory=lambda: ["wind:1", "others:1"])
    lr_start: float = 0.001
    lr_end: float = 0.0001
    grad_floor: float = 1e-8
    fp32_graphcast: bool = False

    te_generation_weight: float = 0.5
    objective_best_te_eval: bool = True
    te_every: int = 10
    te_tol_deg: float = 1.0
    te_save_vis: bool = True
    te_keep_text: bool = True
    te_vis_dpi: int = 130
    te_save_attack_diff_vis: bool = True
    te_save_state_artifacts: bool = True
    best_history_save_probability: bool = True
    best_history_objective_figures: bool = True
    best_history_te_state: bool = True
    downstream_colorbar_stage_consistent: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        data = json.loads(path.read_text())
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {unknown}")
        return cls(**data)

    def validate(self) -> None:
        if self.selected_difficulties:
            unknown = [d for d in self.selected_difficulties if d not in DIFFICULTY_TO_TARGET_TAG]
            if unknown:
                raise ValueError(f"Unknown difficulties {unknown}; choose from {list(DIFFICULTY_TO_TARGET_TAG)}")
        if self.selected_targets is not None:
            if not self.selected_targets:
                raise ValueError("selected_targets cannot be empty; use null to run all configured targets")
            unknown_targets = [t for t in self.selected_targets if t not in ALL_TARGET_TAGS]
            if unknown_targets:
                raise ValueError(f"Unknown targets {unknown_targets}; choose from {ALL_TARGET_TAGS}")

        self.saved_downstream_deltas = [float(x) for x in self.saved_downstream_deltas]
        self.selected_deltas = [float(x) for x in self.selected_deltas]
        self.warmup_deltas = [float(x) for x in self.warmup_deltas]
        self.guide_delta = None if self.guide_delta is None else float(self.guide_delta)

        if not self.selected_deltas:
            raise ValueError("selected_deltas cannot be empty")
        if any(x <= 0 for x in self.selected_deltas):
            raise ValueError("selected_deltas must be positive")
        if any(x <= 0 for x in self.warmup_deltas):
            raise ValueError("warmup_deltas must be positive")
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        for phase_steps in ("guidance_steps", "trajectory_steps"):
            value = getattr(self, phase_steps)
            if value is not None and value <= 0:
                raise ValueError(f"{phase_steps} must be positive")

        if self.warmup_deltas:
            if len(self.selected_deltas) != 1:
                raise ValueError("Explicit warmup requires exactly one final --delta value")
            final_eps = float(self.selected_deltas[0])
            if sorted(self.warmup_deltas) != self.warmup_deltas or len(set(self.warmup_deltas)) != len(self.warmup_deltas):
                raise ValueError("warmup_deltas must be strictly increasing")
            if any(x >= final_eps for x in self.warmup_deltas):
                raise ValueError("Every warmup delta must be smaller than the final delta")

        if self.guide_source == "saved_downstream":
            if self.guide_delta is not None:
                if self.guide_delta not in self.saved_downstream_deltas:
                    raise ValueError(
                        f"guide_delta={self.guide_delta:g} is unavailable; saved downstream deltas: {self.saved_downstream_deltas}"
                    )
            elif self.warmup_deltas:
                final_eps = float(self.selected_deltas[0])
                if final_eps not in self.saved_downstream_deltas:
                    raise ValueError(
                        "With explicit warmup and guide_delta=null, the final delta is used as the fixed guide delta; "
                        f"{final_eps:g} is not in saved_downstream_deltas={self.saved_downstream_deltas}."
                    )
            else:
                missing = [x for x in self.selected_deltas if x not in self.saved_downstream_deltas]
                if missing:
                    raise ValueError(
                        f"Selected deltas {missing} do not have matching saved downstream guidance. "
                        "Set guide_delta explicitly to use a fixed saved guide."
                    )

        if self.end2end_steps <= 0:
            raise ValueError("end2end_steps must be positive")
        if self.verbose_every <= 0:
            raise ValueError("verbose_every must be positive")
        if self.run_tag is not None:
            self.run_tag = str(self.run_tag).strip()
            if not self.run_tag:
                self.run_tag = None
            elif not all(c.isalnum() or c in "._-" for c in self.run_tag):
                raise ValueError("run_tag may contain only letters, numbers, '.', '_' and '-'")

        if self.guide_source not in GUIDE_SOURCES:
            raise ValueError(f"guide_source must be one of {GUIDE_SOURCES}")
        if self.core_loss not in CORE_LOSSES:
            raise ValueError(f"core_loss must be one of {CORE_LOSSES}")
        positive = [
            "guide_background_sigma_deg", "vortex_radius_deg", "region_radius_deg",
            "core_radius_deg", "center_weight_sigma_deg", "prefix_tau_steps",
            "trajectory_target_sigma",
            "trajectory_focal_gamma", "reg_sigma_rad", "grad_sigma_rad",
            "input_core_radius_deg", "lr_start", "lr_end", "te_tol_deg",
        ]
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for weight in ("guide_loss_weight", "trajectory_loss_weight"):
            if float(getattr(self, weight)) < 0:
                raise ValueError(f"{weight} must be non-negative")
        if self.guide_loss_weight == self.trajectory_loss_weight == self.reg_lambda == 0:
            raise ValueError("At least one loss weight must be positive")
        for channel_weight in ("channel_weight_wind", "channel_weight_thickness", "channel_weight_mslp"):
            if float(getattr(self, channel_weight)) < 0:
                raise ValueError(f"{channel_weight} must be non-negative")
        if self.trajectory_target_radius < 0:
            raise ValueError("trajectory_target_radius must be >= 0")
        if self.core_radius_deg > self.region_radius_deg:
            raise ValueError("core_radius_deg must not exceed region_radius_deg")
        for name in ["center_weight_floor", "prefix_floor"]:
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.input_mask_radius_deg < 0 or self.input_mask_radius_deg > 180:
            raise ValueError("input_mask_radius_deg must lie in [0, 180]; 0 disables the hard mask")
        if self.unrelated_exclusion_radius_deg <= 0 or self.unrelated_exclusion_radius_deg > 180:
            raise ValueError("unrelated_exclusion_radius_deg must lie in (0, 180]")
        if self.unrelated_exclusion_radius_deg < self.region_radius_deg:
            raise ValueError("unrelated_exclusion_radius_deg must be >= region_radius_deg")
        if self.unrelated_loss_weight < 0:
            raise ValueError("unrelated_loss_weight must be non-negative")
        if self.guide_loss_weight <= 0 or self.trajectory_loss_weight <= 0:
            raise ValueError("Split optimization requires strictly positive guidance and trajectory coefficients")
        if self.original_weight < 0:
            raise ValueError("original_weight must be non-negative")
        if self.optimizer not in {"pgd", "adam"}:
            raise ValueError("optimizer must be pgd or adam")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta parameters must be in [0,1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.pgd_step_size <= 0:
            raise ValueError("pgd_step_size must be positive")
        if self.surrogate_output_stride <= 0:
            raise ValueError("surrogate_output_stride must be positive")
        if not isinstance(self.gradient_dilation_vars, list) or not self.gradient_dilation_vars:
            raise ValueError("gradient_dilation_vars must be a nonempty list, e.g. ['wind'] or ['none']")
        allowed = {"wind", "wind_speed", "all", "none", "10m_u_component_of_wind", "10m_v_component_of_wind",
                   "u_component_of_wind", "v_component_of_wind", "mean_sea_level_pressure",
                   "geopotential", "temperature", "specific_humidity", "total_precipitation_6hr"}
        normalized = []
        for token in self.gradient_dilation_vars:
            if not isinstance(token, str):
                raise ValueError("Each gradient_dilation_vars entry must be a string")
            name, separator, level = token.strip().partition("@")
            if name not in allowed or (separator and (name in {"wind", "wind_speed", "all", "none"} or not level)):
                raise ValueError(f"Invalid gradient dilation selector {token!r}; use 'wind', 'none', 'all', a variable name, or variable@level")
            if separator:
                try:
                    if not float(level).is_integer():
                        raise ValueError
                except ValueError as exc:
                    raise ValueError(f"Invalid pressure level in gradient dilation selector {token!r}") from exc
                token = f"{name}@{int(float(level))}"
            else:
                token = name
            normalized.append(token)
        if ("all" in normalized or "none" in normalized) and len(normalized) != 1:
            raise ValueError("'all' and 'none' cannot be combined with other gradient dilation selectors")
        # Independent multipliers of the *three existing Gaussian length scales*
        # (TI, gradient distance, regularization distance). One shared variable
        # selector still controls which channels receive these mechanisms.
        if not isinstance(self.dilation_scale, list) or not self.dilation_scale:
            raise ValueError("dilation_scale must contain 'wind:S' and/or 'others:S'")
        scale_by_family = {"wind": 1.0, "others": 1.0}
        seen_families = set()
        for entry in self.dilation_scale:
            if not isinstance(entry, str) or ":" not in entry:
                raise ValueError(f"Invalid dilation scale {entry!r}; use wind:1 others:10")
            family, sep, value = entry.partition(":")
            family = family.strip().lower()
            if family not in scale_by_family or family in seen_families:
                raise ValueError(f"Unknown or repeated dilation family in {entry!r}")
            try:
                import math
                scale = float(value.strip())
            except ValueError as exc:
                raise ValueError(f"Non-numeric dilation multiplier: {entry!r}") from exc
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Dilation multiplier must be positive and finite: {entry!r}")
            scale_by_family[family] = scale
            seen_families.add(family)
        self.dilation_scale = [f"{family}:{scale_by_family[family]:g}" for family in ("wind", "others")]

        self.gradient_dilation_vars = list(dict.fromkeys(normalized))
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
        if not 0 <= self.te_generation_weight <= 1:
            raise ValueError("te_generation_weight must be between 0 and 1")
        if self.te_every <= 0:
            raise ValueError("te_every must be positive")

    def to_runtime_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
