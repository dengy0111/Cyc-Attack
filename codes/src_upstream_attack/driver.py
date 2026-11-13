#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig
from runner import ExperimentRunner


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run the target-focused GraphCast upstream attack with fixed saved-downstream guidance, "
            "a differentiable TC-trajectory surrogate loss, optional epsilon continuation, and "
            "TempestExtremes used only for evaluation/current-best saving."
        )
    )
    p.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    p.add_argument("--difficulty", nargs="+", choices=["easy", "medium", "hard", "remove_only"])
    p.add_argument("--target", nargs="+")
    p.add_argument("--all-targets", action="store_true")
    p.add_argument("--delta", nargs="+", type=float,
                   help="Final upstream epsilon(s). With --warmup-delta, exactly one final epsilon is required.")
    p.add_argument("--guide-delta", type=float,
                   help="Saved downstream delta used as the fixed guidance target. With warmup, defaults to final --delta.")
    p.add_argument("--warmup-delta", nargs="+", type=float,
                   help="Strictly increasing upstream epsilon warmup stages before the final --delta.")
    p.add_argument("--warmup-steps", type=int,
                   help="Legacy per-phase step count for warmup epsilons, unless phase-specific steps override.")
    p.add_argument("--guidance-steps", type=int, help="PGD steps in Guidance-only phase at every epsilon; default uses warmup-steps/steps")
    p.add_argument("--trajectory-steps", type=int, help="PGD steps in Trajectory-only phase at every epsilon; default uses warmup-steps/steps")
    p.add_argument("--input-mask-radius-deg", type=float, help="Hard input perturbation radius from original TC center; 0 disables")
    p.add_argument("--unrelated-exclusion-radius-deg", type=float, help="Forecast TC corridor radius excluded from unrelated-region penalty")
    p.add_argument("--unrelated-loss-weight", type=float, help="Unrelated forecast RMSE^2 penalty coefficient in both phases")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--optimizer", choices=["pgd", "adam"], help="Sign-PGD (default) or optional Projected Adam")
    p.add_argument("--pgd-step-size", type=float, help="Fixed per-step Sign-PGD step size in standardized input units (default: 0.01)")
    p.add_argument("--adaptive-step-rejection", action="store_true", help="Optional loss-aware backtracking; requires extra GraphCast forward evaluations")
    p.add_argument("--dilation-vars", "--weighted-vars", "--gradient-dilation-vars", nargs="+", dest="gradient_dilation_vars",
                   metavar="VAR", help="One shared selector for TI smoothing + distance gradient weighting + distance regularization: wind (default), none, all, variable or variable@level")
    p.add_argument("--dilation-scale", nargs="+", metavar="FAMILY:MULTIPLIER",
                   help="TI/gradient-distance/reg-distance length-scale multiplier by family; e.g. wind:1 others:10 (use --dilation-vars all)")
    p.add_argument("--guide-loss-weight", type=float)
    p.add_argument("--trajectory-loss-weight", type=float)
    p.add_argument("--reg-lambda", type=float)
    p.add_argument("--original-weight", type=float, help="Original-region guidance relative weight (1.0 = target/original 1:1)")
    p.add_argument("--channel-weight-wind", type=float)
    p.add_argument("--channel-weight-thickness", type=float)
    p.add_argument("--channel-weight-mslp", type=float)
    p.add_argument("--te-generation-weight", type=float, help="TE relocation best score: weight of target generation; original removal gets 1-weight (default: 0.5)")
    p.add_argument("--no-objective-best-te-eval", action="store_true", help="Skip EXTRA TE checks on Objective Best-only epochs; full Objective TE track visualizations will then be unavailable on those epochs")
    p.add_argument("--steps", type=int,
                   help="Optimization steps for the final epsilon stage (or per stage when no explicit warmup is used).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--run-tag")
    p.add_argument("--build-targets", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--through", choices=["prepare", "attack", "evaluate", "aggregate"], default="aggregate")
    p.add_argument("--only", choices=["prepare", "diagnostic", "attack", "evaluate", "aggregate"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ProjectConfig.from_json(args.config)
    if args.difficulty is not None:
        cfg.selected_difficulties = list(args.difficulty)
        cfg.selected_targets = None
    if args.target is not None:
        cfg.selected_targets = list(args.target)
        cfg.selected_difficulties = None
    if args.all_targets:
        cfg.selected_targets = None
        cfg.selected_difficulties = None
    if args.delta is not None:
        cfg.selected_deltas = [float(x) for x in args.delta]
    if args.guide_delta is not None:
        cfg.guide_delta = float(args.guide_delta)
    if args.warmup_delta is not None:
        cfg.warmup_deltas = [float(x) for x in args.warmup_delta]
    if args.warmup_steps is not None:
        cfg.warmup_steps = int(args.warmup_steps)
    for setting in ("guidance_steps", "trajectory_steps", "input_mask_radius_deg", "unrelated_exclusion_radius_deg", "unrelated_loss_weight"):
        value = getattr(args, setting)
        if value is not None:
            setattr(cfg, setting, value)
    if args.cases is not None:
        cfg.run_cases = list(args.cases)
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.pgd_step_size is not None:
        cfg.pgd_step_size = args.pgd_step_size
    if args.adaptive_step_rejection:
        cfg.adaptive_step_rejection = True
    if args.dilation_scale is not None:
        cfg.dilation_scale = list(args.dilation_scale)
    for setting in ("gradient_dilation_vars", "guide_loss_weight", "trajectory_loss_weight", "reg_lambda",
                    "original_weight", "channel_weight_wind", "channel_weight_thickness", "channel_weight_mslp"):
        value = getattr(args, setting)
        if value is not None:
            setattr(cfg, setting, value)
    if args.te_generation_weight is not None:
        cfg.te_generation_weight = args.te_generation_weight
    if args.no_objective_best_te_eval:
        cfg.objective_best_te_eval = False
    if args.steps is not None:
        cfg.end2end_steps = int(args.steps)
    if args.overwrite:
        cfg.overwrite = True
    if args.build_targets:
        cfg.build_targets = True
    if args.run_tag is not None:
        cfg.run_tag = str(args.run_tag)
    elif not cfg.run_tag:
        if args.only in {"evaluate", "aggregate"}:
            raise SystemExit(
                "--only evaluate/aggregate requires --run-tag for an existing run. "
                "Use the Run tag printed when that attack was created."
            )
        cfg.run_tag = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")

    cfg.validate()
    fixed_guide = cfg.guide_delta
    if cfg.warmup_deltas and fixed_guide is None:
        fixed_guide = float(cfg.selected_deltas[0])

    print("Run tag:", cfg.run_tag)
    print("Selected target tags:", cfg.selected_targets or "all configured targets")
    print("Final upstream delta(s):", cfg.selected_deltas)
    print("Warmup upstream deltas:", cfg.warmup_deltas or "none")
    print("Warmup steps/epsilon:", cfg.warmup_steps)
    print("Final steps/phase (if not overridden):", cfg.end2end_steps)
    print("Separate steps: guidance=", cfg.guidance_steps, "trajectory=", cfg.trajectory_steps)
    print("Sequential epsilon phases: guidance-only -> trajectory-only -> next epsilon")
    print("Hard input mask radius(deg):", cfg.input_mask_radius_deg, "| forecast unrelated corridor(deg):", cfg.unrelated_exclusion_radius_deg, "| penalty:", cfg.unrelated_loss_weight)
    print("Optimizer:", cfg.optimizer)
    print("Warm-start checkpoint: objective_best (surrogate loss minimum)")
    print("Shared TI/gradient-distance/reg-distance variable selectors:", cfg.gradient_dilation_vars)
    print("Shared spatial length-scale multipliers:", cfg.dilation_scale)
    if cfg.optimizer == "pgd":
        print("PGD: fixed sign step size:", cfg.pgd_step_size, "| momentum and gradient normalization OFF")
    print("Phase-specific loss weights (never combined): guide=", cfg.guide_loss_weight, "trajectory=", cfg.trajectory_loss_weight,
          "regularization=", cfg.reg_lambda)
    print("Guidance: valid TC steps only, equal temporal weights, equal target/original contributions")
    print("Fixed saved-downstream guide delta:", fixed_guide if fixed_guide is not None else "stage-matched")
    print("Run cases:", cfg.run_cases or "all TC2 cases")
    print("TE Best: generation weight=", cfg.te_generation_weight,
          "removal weight=", 1.0 - cfg.te_generation_weight,
          "| evaluate every Objective Best:", cfg.objective_best_te_eval)
    print("TE role: evaluation/current-best saving only; no TE feedback enters the loss or horizon.")

    runner = ExperimentRunner(cfg)
    if args.only is None:
        runner.run(through=args.through, run_diagnostic=args.diagnostic)
        return
    if args.only == "prepare":
        runner.prepare()
    elif args.only == "diagnostic":
        runner.diagnostic()
    elif args.only == "attack":
        runner.attack()
    elif args.only == "evaluate":
        runner.evaluate()
    elif args.only == "aggregate":
        runner.aggregate()


if __name__ == "__main__":
    main()
