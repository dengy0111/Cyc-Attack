# Stage 08: Run target-focused upstream attack with explicit epsilon continuation.
#
# When --warmup-delta is supplied, only the upstream feasible-set epsilon changes
# across warmup stages; the saved downstream guidance delta remains fixed.

warmup_stage_rows = []
all_resource_rows = []
all_best_rows = []
selected_best_rows = []
te_best_loss_rows = []
objective_and_last_rows = []
transition_rows = []


def write_cumulative_loss_summaries(sample, target_tag, out_dir, stage_index):
    """One row per (epsilon, last/objective_best) for every completed epsilon up to this stage."""
    current = []
    for spec in ATTACK_STAGE_SCHEDULE[:stage_index]:
        ep = float(spec["input_epsilon"])
        directory = end2end_sample_dir(sample, target_tag, float(spec["guide_epsilon"]), ep)
        for selection, relative in (("last", "last_checkpoint_losses.csv"),
                                    ("objective_best", "objective_best/selected_checkpoint_losses.csv")):
            source = directory / relative
            if not source.exists():
                raise FileNotFoundError(f"Missing {selection} losses for ε={ep:g}: {source}")
            frame = pd.read_csv(source)
            if len(frame) != 1:
                raise ValueError(f"Expected exactly one row in {source}, got {len(frame)}")
            row = frame.iloc[0].to_dict()
            row.update({"selection": selection, "input_epsilon": ep,
                        "summary_through_epsilon": float(END2END_EPSILON),
                        "loss_source_file": str(source)})
            current.append(row)
    frame = pd.DataFrame(current).sort_values(["input_epsilon", "selection"], ascending=[False, True])
    if frame.duplicated(["input_epsilon", "selection"]).any():
        raise AssertionError("Cumulative summary contains duplicate epsilon/selection entries")
    frame.to_csv(out_dir / "loss_summary_cumulative.csv", index=False)
    frame.to_csv(WARMUP_SUMMARY_DIR / f"loss_summary_through_epsilon_{number_tag(END2END_EPSILON)}__{sample}__{target_tag}.csv", index=False)
    return frame



def collect_te_best_stage_outputs(sample, target_tag, out_dir, previous_pair, stage_index):
    """Record exact selected TE loss + auditable warm-start source for every case and stage."""
    loss_file = out_dir / "te_best_loss_summary.csv"
    if not loss_file.exists():
        raise FileNotFoundError(f"Missing selected TE best loss components: {loss_file}")
    row = pd.read_csv(loss_file).iloc[0].to_dict()
    te_best_loss_rows.append(row)
    # Two audited transitions exist: prior epsilon trajectory -> current guidance,
    # then same epsilon guidance -> trajectory. Main warmup_source refers to latter.
    source_file = out_dir / "guidance_phase" / "warmup_source.json"
    intra_file = out_dir / "warmup_source.json"
    if not intra_file.exists():
        raise FileNotFoundError(f"Missing phase handoff checkpoint audit: {intra_file}")
    intra = json.loads(intra_file.read_text())
    expected_handoff = out_dir / "guidance_phase" / "objective_best" / "best_delta_std.npz"
    if (not intra.get("unchanged_after_projection") or
            Path(intra.get("checkpoint_path", "")).resolve() != expected_handoff.resolve()):
        raise AssertionError(f"Invalid intra-epsilon Guidance->Trajectory handoff: {intra_file}")
    if previous_pair is not None:
        if not source_file.exists():
            raise FileNotFoundError(f"Missing prior-epsilon warmup checkpoint audit: {source_file}")
        source = json.loads(source_file.read_text())
        if source.get("selection") != "objective_best" or not source.get("unchanged_after_projection"):
            raise AssertionError(f"Invalid previous-epsilon Objective best warm-start: {source_file}")
    else:
        source = {"selection": "original_input", "checkpoint_path": None, "unchanged_after_projection": True}
    cumulative = write_cumulative_loss_summaries(sample, target_tag, out_dir, stage_index)
    objective_and_last_rows.extend(cumulative.loc[cumulative["input_epsilon"] == float(row["input_epsilon"])].to_dict("records"))
    transition_rows.append({"sample": sample, "target_tag": target_tag,
                            "stage_index": int(stage_index), "input_epsilon": float(row["input_epsilon"]),
                            "warmup_checkpoint": source.get("checkpoint_path"),
                            "warmup_selection": source.get("selection"),
                            "guidance_to_trajectory_checkpoint": intra.get("checkpoint_path"),
                            "unchanged_after_projection": source.get("unchanged_after_projection")})

print("Attack stage schedule:")
for i, spec in enumerate(ATTACK_STAGE_SCHEDULE):
    print(
        f"  {i + 1}: kind={spec['kind']} input_epsilon={spec['input_epsilon']:g} "
        f"guide_delta={spec['guide_epsilon']:g} steps={int(spec['steps'])}"
    )
print("Attack variables (configured levels, full field):", PERTURB_INPUT_VARS, PERTURB_LEVELS)
print("Guide source:", GUIDE_SOURCE)
print("TE role: evaluation/current-best saving only; TE never changes the differentiable objective.")

previous_stage = None

for stage_index, spec in enumerate(ATTACK_STAGE_SCHEDULE):
    input_epsilon = float(spec["input_epsilon"])
    guide_epsilon = float(spec["guide_epsilon"])
    stage_steps = int(spec["steps"])
    previous_pair = None
    if previous_stage is not None:
        previous_pair = (
            float(previous_stage["guide_epsilon"]),
            float(previous_stage["input_epsilon"]),
        )

    # Every epsilon now runs TWO separate optimization objectives.
    guide_steps = int(GUIDANCE_PHASE_STEPS if GUIDANCE_PHASE_STEPS is not None else stage_steps)
    traj_steps = int(TRAJECTORY_PHASE_STEPS if TRAJECTORY_PHASE_STEPS is not None else stage_steps)
    set_attack_stage(input_epsilon, guide_epsilon, traj_steps)
    stage_wall_start = time.perf_counter()

    print("\n" + "=" * 100)
    print(
        f"ATTACK STAGE {stage_index + 1}/{len(ATTACK_STAGE_SCHEDULE)}: "
        f"kind={spec['kind']}; upstream epsilon={input_epsilon:g}; "
        f"fixed guidance delta={guide_epsilon:g}; guide_steps={guide_steps}; trajectory_steps={traj_steps}; "
        f"warm-start={previous_pair}"
    )
    print("=" * 100)

    if GUIDE_SOURCE == "saved_downstream":
        validate_saved_downstream_stage(guide_epsilon)

    upstream_records = []

    for target_tag in TARGET_SCENARIOS:
        for sample in RUN_CASES:
            out_dir = end2end_sample_dir(sample, target_tag, guide_epsilon, input_epsilon)
            final_path = out_dir / "final_downstream_input.pt"
            summary_path = out_dir / "attack_summary.json"

            if final_path.exists() and summary_path.exists() and not OVERWRITE:
                saved = json.load(open(summary_path))
                expected = {
                    "run_tag": RUN_TAG,
                    "target_id": get_target_id(target_tag),
                    "target_tag": target_tag,
                    "guide_delta": float(guide_epsilon),
                    "end2end_epsilon": float(input_epsilon),
                    "steps_this_stage": int(traj_steps),
                    "objective_phase": "trajectory",
                    "variant": END2END_VARIANT,
                    "attack_config_fingerprint": ATTACK_CONFIG_FINGERPRINT,
                    "warm_start_from": {
                        "guide_delta": float(previous_pair[0]) if previous_pair is not None else float(guide_epsilon),
                        "input_epsilon": float(previous_pair[1]) if previous_pair is not None else float(input_epsilon),
                        "intra_epsilon_checkpoint": str(out_dir / "guidance_phase" / "objective_best" / "best_delta_std.npz"),
                    },
                }
                mismatch = {
                    k: (saved.get(k), v)
                    for k, v in expected.items()
                    if saved.get(k) != v
                }
                if mismatch:
                    raise RuntimeError(
                        f"Existing upstream stage mismatch at {out_dir}: {mismatch}"
                    )

                if not (out_dir / "guidance_phase" / "objective_best" / "best_delta_std.npz").exists():
                    raise FileNotFoundError("Trajectory completed without corresponding Guidance phase")
                print("Reusing completed two-phase upstream stage:", out_dir)
                runtime_path = out_dir / "runtime_memory.json"
                if runtime_path.exists():
                    rec = json.load(open(runtime_path))
                    upstream_records.append(rec)
                    all_resource_rows.append({
                        **rec,
                        "input_epsilon": float(input_epsilon),
                        "guide_delta": float(guide_epsilon),
                        "stage_kind": str(spec["kind"]),
                        "component": "upstream",
                    })
                manifest_path = out_dir / "best_manifest.csv"
                if manifest_path.exists():
                    for item in pd.read_csv(manifest_path).to_dict("records"):
                        all_best_rows.append({"sample": sample, "target_tag": target_tag, **item})
                for kind in ("te_best", "objective_best"):
                    if kind in saved:
                        selected_best_rows.append({"sample": sample, "target_tag": target_tag, "stage_index": stage_index + 1,
                                                   "stage_kind": spec["kind"], "input_epsilon": input_epsilon,
                                                   "guide_delta": guide_epsilon, "best_type": kind, **saved[kind]})
                collect_te_best_stage_outputs(sample, target_tag, out_dir, previous_pair, stage_index + 1)
                continue

            # Phase A: minimize ONLY local guidance (+ optional protection penalties).
            # Its Objective Best, not TE Best or last iterate, seeds Phase B.
            guide_dir = out_dir / "guidance_phase"
            guide_summary = guide_dir / "attack_summary.json"
            guide_checkpoint = guide_dir / "objective_best" / "best_delta_std.npz"
            if guide_summary.exists() and guide_checkpoint.exists() and not OVERWRITE:
                meta = json.loads(guide_summary.read_text())
                if (meta.get("run_tag") != RUN_TAG
                        or meta.get("attack_config_fingerprint") != ATTACK_CONFIG_FINGERPRINT
                        or meta.get("objective_phase") != "guidance"
                        or meta.get("steps_this_stage") != guide_steps):
                    raise RuntimeError(f"Incompatible existing Guidance phase at {guide_dir}")
                print("Reusing completed Guidance-only phase:", guide_dir)
            else:
                print(f"[SEQUENTIAL] ε={input_epsilon:g} PHASE A: GUIDANCE ONLY, steps={guide_steps}")
                set_attack_stage(input_epsilon, guide_epsilon, guide_steps)
                run_target_focused_case(
                    sample, target_tag, warm_start_from=previous_pair,
                    stage_index=stage_index + 1, stage_kind=spec["kind"],
                    objective_phase="guidance", output_dir=guide_dir,
                )
            if not guide_checkpoint.exists():
                raise FileNotFoundError(f"Guidance Objective Best checkpoint absent: {guide_checkpoint}")
            print(f"[SEQUENTIAL] ε={input_epsilon:g} PHASE B: TRAJECTORY ONLY, "
                  f"steps={traj_steps}, seed={guide_checkpoint}")
            set_attack_stage(input_epsilon, guide_epsilon, traj_steps)
            with ResourceProfiler(
                "sequential_trajectory_only",
                sample,
                target_tag,
                out_dir,
                interval=1.0,
            ) as profiler:
                run_target_focused_case(
                    sample, target_tag, warm_start_from=previous_pair,
                    stage_index=stage_index + 1, stage_kind=spec["kind"],
                    objective_phase="trajectory", warm_start_checkpoint=guide_checkpoint,
                )
            # A completed epsilon always passes its TRAJECTORY Objective Best to the next epsilon.
            (out_dir / "sequential_phase_manifest.json").write_text(json.dumps({
                "input_epsilon": input_epsilon,
                "guidance_steps": guide_steps,
                "trajectory_steps": traj_steps,
                "guidance_objective_best_checkpoint": str(guide_checkpoint),
                "trajectory_start_checkpoint": str(guide_checkpoint),
                "next_epsilon_warmup_checkpoint": str(out_dir / "objective_best" / "best_delta_std.npz"),
                "objective_switch": "guidance_only -> trajectory_only",
                "te_role": "evaluation_only",
            }, indent=2))

            profiler.record.update({
                "run_tag": RUN_TAG,
                "target_id": get_target_id(target_tag),
                "target_tag": target_tag,
                "stage_kind": str(spec["kind"]),
                "input_epsilon": float(input_epsilon),
                "guide_delta": float(guide_epsilon),
                "steps_this_stage": int(traj_steps),
                "warm_start_from": (
                    None if previous_pair is None else {
                        "guide_delta": float(previous_pair[0]),
                        "input_epsilon": float(previous_pair[1]),
                    }
                ),
                "guide_source": GUIDE_SOURCE,
                "variant": END2END_VARIANT,
                "attack_config_fingerprint": ATTACK_CONFIG_FINGERPRINT,
            })

            with open(out_dir / "runtime_memory.json", "w") as f:
                json.dump(profiler.record, f, indent=2)

            manifest_path = out_dir / "best_manifest.csv"
            if manifest_path.exists():
                for item in pd.read_csv(manifest_path).to_dict("records"):
                    all_best_rows.append({"sample": sample, "target_tag": target_tag, **item})
            stage_summary = json.loads((out_dir / "attack_summary.json").read_text())
            for kind in ("te_best", "objective_best"):
                selected_best_rows.append({"sample": sample, "target_tag": target_tag, "stage_index": stage_index + 1,
                                           "stage_kind": spec["kind"], "input_epsilon": input_epsilon,
                                           "guide_delta": guide_epsilon, "best_type": kind, **stage_summary[kind]})
            collect_te_best_stage_outputs(sample, target_tag, out_dir, previous_pair, stage_index + 1)
            upstream_records.append(profiler.record)
            all_resource_rows.append({
                **profiler.record,
                "input_epsilon": float(input_epsilon),
                "guide_delta": float(guide_epsilon),
                "stage_kind": str(spec["kind"]),
                "component": "upstream",
            })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    stage_elapsed = time.perf_counter() - stage_wall_start
    stage_summary_dir = end2end_summary_dir(guide_epsilon, input_epsilon)
    stage_summary_dir.mkdir(parents=True, exist_ok=True)

    if upstream_records:
        pd.DataFrame(upstream_records).to_csv(
            stage_summary_dir / "end2end_runtime_memory.csv",
            index=False,
        )

    stage_row = {
        "run_tag": RUN_TAG,
        "stage_index": int(stage_index),
        "stage_kind": str(spec["kind"]),
        "input_epsilon": float(input_epsilon),
        "guide_delta": float(guide_epsilon),
        "warm_start_from_guide_delta": None if previous_pair is None else float(previous_pair[0]),
        "warm_start_from_input_epsilon": None if previous_pair is None else float(previous_pair[1]),
        "steps": int(guide_steps + traj_steps),
        "guidance_steps": guide_steps,
        "trajectory_steps": traj_steps,
        "stage_wall_seconds": float(stage_elapsed),
        "guide_source": GUIDE_SOURCE,
        "completed": True,
    }
    warmup_stage_rows.append(stage_row)

    pd.DataFrame(warmup_stage_rows).to_csv(
        WARMUP_SUMMARY_DIR / "warmup_stage_manifest.csv",
        index=False,
    )
    if all_best_rows:
        pd.DataFrame(all_best_rows).to_csv(WARMUP_SUMMARY_DIR / "best_history_all_stages.csv", index=False)
    if selected_best_rows:
        pd.DataFrame(selected_best_rows).to_csv(WARMUP_SUMMARY_DIR / "best_selected_all_stages.csv", index=False)
    if objective_and_last_rows:
        pd.DataFrame(objective_and_last_rows).to_csv(WARMUP_SUMMARY_DIR / "loss_summary_all_completed_stages.csv", index=False)
    if te_best_loss_rows:
        pd.DataFrame(te_best_loss_rows).to_csv(WARMUP_SUMMARY_DIR / "te_best_loss_summary.csv", index=False)
    if transition_rows:
        pd.DataFrame(transition_rows).to_csv(WARMUP_SUMMARY_DIR / "warmup_transition_manifest.csv", index=False)
    if all_resource_rows:
        pd.DataFrame(all_resource_rows).to_csv(
            WARMUP_SUMMARY_DIR / "runtime_memory_all_stages.csv",
            index=False,
        )

    print(
        f"Completed stage {stage_index + 1}: upstream epsilon={input_epsilon:g}, "
        f"guide delta={guide_epsilon:g}, steps={stage_steps}, "
        f"elapsed={stage_elapsed / 60:.2f} min"
    )
    previous_stage = dict(spec)

# Restore final-stage globals for evaluate/aggregate stages in the same driver run.
_final = ATTACK_STAGE_SCHEDULE[-1]
set_attack_stage(_final["input_epsilon"], _final["guide_epsilon"], _final["steps"])

print("\nAll upstream continuation stages finished.")
display(pd.DataFrame(warmup_stage_rows))
