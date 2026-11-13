# Stage 10: Aggregate target-focused guide+surrogate results across explicit attack stages.

require_paths(WARMUP_SUMMARY_DIR / "end2end_metrics_all_deltas.csv")
metrics = pd.read_csv(WARMUP_SUMMARY_DIR / "end2end_metrics_all_deltas.csv")
resources_path = WARMUP_SUMMARY_DIR / "runtime_memory_all_stages.csv"
resources = pd.read_csv(resources_path) if resources_path.exists() else pd.DataFrame()

metric_candidates = [
    "target_success", "DR_all", "DR_changed", "remove_success", "mean_target_overlap",
    "mean_removed_original_overlap", "TPR", "FNR", "FPR", "FAR", "FAR_global",
    "input_perturbation_magnitude_normalized_mean",
]
metric_cols = [c for c in metric_candidates if c in metrics.columns]
group_cols = [
    c for c in ["stage_index", "stage_kind", "input_epsilon", "guide_delta", "selection_type", "target_id", "target_tag"]
    if c in metrics.columns
]
if group_cols and metric_cols:
    aggregate = metrics.groupby(group_cols, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
else:
    aggregate = metrics.copy()
aggregate.to_csv(WARMUP_SUMMARY_DIR / "metrics_by_stage_target.csv", index=False)

if not resources.empty:
    resource_group_cols = [c for c in ["component", "stage_kind", "input_epsilon", "guide_delta"] if c in resources.columns]
    resource_metrics = [
        c for c in [
            "elapsed_seconds", "rss_peak_mb", "rss_increment_peak_mb",
            "gpu_peak_poll_mb", "torch_cuda_peak_allocated_mb",
        ] if c in resources.columns
    ]
    if resource_group_cols and resource_metrics:
        resource_summary = resources.groupby(resource_group_cols, dropna=False)[resource_metrics].agg(["mean", "max"]).reset_index()
        resource_summary.columns = [
            "__".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
            for col in resource_summary.columns
        ]
        resource_summary.to_csv(WARMUP_SUMMARY_DIR / "runtime_memory_by_stage.csv", index=False)
        display(resource_summary)

# Compact stage-wise plots. Plotting remains aggregate only; no per-forecast-step figures.
for metric in ["target_success", "DR_changed", "mean_target_overlap", "remove_success", "FAR", "TPR", "input_perturbation_magnitude_normalized_mean"]:
    if metric not in metrics.columns or "input_epsilon" not in metrics.columns:
        continue
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_groups = ["input_epsilon"]
    if "target_tag" in metrics.columns:
        plot_groups.append("target_tag")
    if "selection_type" in metrics.columns:
        plot_groups.append("selection_type")
    grouped = metrics.groupby(plot_groups, dropna=False)[metric].mean().reset_index()
    label_cols = [c for c in ["target_tag", "selection_type"] if c in grouped.columns]
    if label_cols:
        for key, g in grouped.groupby(label_cols, sort=False, dropna=False):
            g = g.sort_values("input_epsilon")
            label = " / ".join(map(str, key if isinstance(key, tuple) else (key,)))
            ax.plot(g["input_epsilon"], g[metric], marker="o", label=label)
        ax.legend(fontsize=7)
    else:
        grouped = grouped.sort_values("input_epsilon")
        ax.plot(grouped["input_epsilon"], grouped[metric], marker="o")
    ax.set_xlabel("Upstream input epsilon")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs upstream input epsilon")
    fig.tight_layout()
    fig.savefig(WARMUP_FIGURE_DIR / f"{metric}_vs_input_epsilon.png", bbox_inches="tight")
    plt.close(fig)

# Hard audit: TWO checkpoint transitions per epsilon and exact phase-specific steps.
audit_rows = []
for i, spec in enumerate(ATTACK_STAGE_SCHEDULE):
    input_epsilon = float(spec["input_epsilon"])
    guide_epsilon = float(spec["guide_epsilon"])
    gsteps = int(GUIDANCE_PHASE_STEPS if GUIDANCE_PHASE_STEPS is not None else spec["steps"])
    tsteps = int(TRAJECTORY_PHASE_STEPS if TRAJECTORY_PHASE_STEPS is not None else spec["steps"])
    prev = ATTACK_STAGE_SCHEDULE[i - 1] if i > 0 else None
    expected_prior = None if prev is None else {
        "guide_delta": float(prev["guide_epsilon"]),
        "input_epsilon": float(prev["input_epsilon"]),
    }
    for target_tag in TARGET_SCENARIOS:
        for sample in RUN_CASES:
            out_dir = end2end_sample_dir(sample, target_tag, guide_epsilon, input_epsilon)
            summary_path = out_dir / "attack_summary.json"
            guide_dir = out_dir / "guidance_phase"
            guide_summary_path = guide_dir / "attack_summary.json"
            phase_manifest_path = out_dir / "sequential_phase_manifest.json"
            require_paths(summary_path, guide_summary_path, phase_manifest_path)
            summary = json.loads(summary_path.read_text())
            guide_summary = json.loads(guide_summary_path.read_text())
            manifest = json.loads(phase_manifest_path.read_text())
            guidance_best = guide_dir / "objective_best" / "best_delta_std.npz"
            expected_main_warm = {
                "guide_delta": float(prev["guide_epsilon"]) if prev is not None else guide_epsilon,
                "input_epsilon": float(prev["input_epsilon"]) if prev is not None else input_epsilon,
                "intra_epsilon_checkpoint": str(guidance_best),
            }
            prior_checkpoint = (None if prev is None else
                end2end_sample_dir(sample, target_tag, float(prev["guide_epsilon"]), float(prev["input_epsilon"]))
                / "objective_best" / "best_delta_std.npz")
            main_audit = out_dir / "warmup_source.json"
            guidance_audit = guide_dir / "warmup_source.json"
            source_ok = False
            if main_audit.exists():
                recorded = json.loads(main_audit.read_text())
                source_ok = (recorded.get("selection") == "objective_best"
                             and recorded.get("unchanged_after_projection") is True
                             and Path(recorded.get("checkpoint_path", "")).resolve() == guidance_best.resolve())
            prior_ok = (not guidance_audit.exists()) if prev is None else False
            if prev is not None and guidance_audit.exists():
                record = json.loads(guidance_audit.read_text())
                prior_ok = (record.get("unchanged_after_projection") is True
                            and record.get("selection") == "objective_best"
                            and Path(record.get("checkpoint_path", "")).resolve() == prior_checkpoint.resolve())
            passed = bool(
                abs(float(summary["guide_delta"]) - guide_epsilon) <= 1e-12
                and abs(float(summary["end2end_epsilon"]) - input_epsilon) <= 1e-12
                and int(summary["steps_this_stage"]) == tsteps
                and int(guide_summary["steps_this_stage"]) == gsteps
                and summary.get("objective_phase") == "trajectory"
                and guide_summary.get("objective_phase") == "guidance"
                and summary.get("strategy") == "sequential_guidance_then_trajectory"
                and summary.get("warm_start_from") == expected_main_warm
                and guide_summary.get("warm_start_from") == expected_prior
                and summary.get("warm_start_selection") == "objective_best"
                and summary.get("root_output_selection") == "te_best"
                and summary.get("TE_used_in_learning") is False
                and manifest.get("objective_switch") == "guidance_only -> trajectory_only"
                and Path(manifest.get("next_epsilon_warmup_checkpoint", "")).exists()
                and source_ok and prior_ok
            )
            audit_rows.append({
                "sample": sample, "target_tag": target_tag,
                "stage_index": i, "stage_kind": spec["kind"],
                "input_epsilon": input_epsilon, "guide_delta": guide_epsilon,
                "guidance_steps": gsteps, "trajectory_steps": tsteps,
                "prior_epsilon_warm_start": expected_prior,
                "intra_epsilon_checkpoint": str(guidance_best),
                "source_provenance_ok": source_ok,
                "previous_stage_handoff_ok": prior_ok,
                "passed": passed,
            })

audit = pd.DataFrame(audit_rows)
audit.to_csv(WARMUP_SUMMARY_DIR / "continuation_consistency_audit.csv", index=False)
if not audit.empty and not audit["passed"].all():
    raise AssertionError(
        "Continuation audit failed:\n" + audit.loc[~audit["passed"]].to_string(index=False)
    )

te_loss_file = WARMUP_SUMMARY_DIR / "te_best_loss_summary.csv"
if te_loss_file.exists():
    te_loss_summary = pd.read_csv(te_loss_file)
    if not te_loss_summary.empty:
        if te_loss_summary["loss_identity_error"].abs().max() > 1e-3:
            raise AssertionError("TE Best component sum inconsistent with Total Loss")
        plot_terms = [x for x in ("total_loss", "weighted_guide", "weighted_traj", "weighted_reg", "weighted_unrelated") if x in te_loss_summary]
        grouped = te_loss_summary.groupby(["input_epsilon", "target_tag"])[plot_terms].mean().reset_index()
        for target_tag, group in grouped.groupby("target_tag"):
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
            for term in plot_terms:
                ax.plot(group["input_epsilon"], group[term], marker="o", label=term)
            ax.set_xlabel("Input epsilon")
            ax.set_ylabel("Loss at selected TE Best")
            ax.set_title(f"Selected TE Best loss components: {target_tag}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(WARMUP_FIGURE_DIR / f"te_best_loss_vs_input_epsilon_{target_tag}.png", bbox_inches="tight")
            plt.close(fig)

selection_rows = []
for spec in ATTACK_STAGE_SCHEDULE:
    input_epsilon = float(spec["input_epsilon"])
    guide_epsilon = float(spec["guide_epsilon"])
    for target_tag in TARGET_SCENARIOS:
        for sample in RUN_CASES:
            out_dir = end2end_sample_dir(sample, target_tag, guide_epsilon, input_epsilon)
            summary_path = out_dir / "attack_summary.json"
            te_path = out_dir / "te_check_log.csv"
            if not summary_path.exists():
                continue
            summary = json.load(open(summary_path))
            row = {
                "sample": sample,
                "target_tag": target_tag,
                "stage_kind": spec["kind"],
                "input_epsilon": input_epsilon,
                "guide_delta": guide_epsilon,
                "TE_used_in_learning": summary.get("TE_used_in_learning"),
                "objective_best_step": summary.get("objective_best", {}).get("selected_step"),
                "objective_best_loss": summary.get("objective_best", {}).get("loss"),
                "te_best_step": summary.get("te_best", {}).get("selected_step"),
                "te_best_DR_changed": summary.get("te_best", {}).get("DR_changed"),
                "te_best_mean_target_overlap": summary.get("te_best", {}).get("mean_target_overlap"),
                "te_best_remove_success": summary.get("te_best", {}).get("remove_success"),
                "te_best_mean_removed_original_overlap": summary.get("te_best", {}).get("mean_removed_original_overlap"),
                "te_best_joint_score": summary.get("te_best", {}).get("te_joint_score"),
                "te_best_generation_score": summary.get("te_best", {}).get("te_generation_score"),
                "te_best_removal_score": summary.get("te_best", {}).get("te_removal_score"),
                "te_best_joint_success": summary.get("te_best", {}).get("te_joint_success"),
            }
            if te_path.exists():
                te_df = pd.read_csv(te_path)
                if not te_df.empty:
                    last = te_df.iloc[-1]
                    for col in ["DR_changed", "mean_target_overlap", "remove_success", "mean_removed_original_overlap", "te_joint_score", "te_generation_score", "te_removal_score", "te_joint_success", "n_pred_tracks"]:
                        if col in te_df.columns:
                            row[f"last_te_{col}"] = last[col]
            selection_rows.append(row)

if selection_rows:
    pd.DataFrame(selection_rows).to_csv(
        WARMUP_SUMMARY_DIR / "dual_checkpoint_summary.csv", index=False
    )

manifest_rows = []
for p in EXP_ROOT.rglob("*"):
    if p.is_file():
        manifest_rows.append({
            "relative_path": str(p.relative_to(EXP_ROOT)),
            "size_bytes": p.stat().st_size,
        })
pd.DataFrame(manifest_rows).to_csv(WARMUP_SUMMARY_DIR / "manifest.csv", index=False)

print("Final target-focused summaries:", WARMUP_SUMMARY_DIR)
print("Final target-focused figures:", WARMUP_FIGURE_DIR)
display(aggregate)
