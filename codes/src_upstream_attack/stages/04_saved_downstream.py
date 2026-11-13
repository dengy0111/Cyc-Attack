# Stage 04: Saved downstream adversarial forecasts used as a fixed guide during upstream continuation.

def saved_downstream_artifact(sample, target_tag, epsilon):
    return downstream_sample_dir(sample, target_tag, epsilon) / "adv_hat_Z_raw.pt"


def validate_saved_downstream_stage(epsilon):
    epsilon = float(epsilon)
    rows = []
    for target_tag in TARGET_SCENARIOS:
        for sample in RUN_CASES:
            adv_path = saved_downstream_artifact(sample, target_tag, epsilon)
            if not adv_path.exists():
                raise FileNotFoundError(
                    "guide_source=saved_downstream requires the saved downstream forecast "
                    f"for guide delta={epsilon:g}:\n{adv_path}"
                )
            rows.append({
                "sample": sample,
                "target_id": get_target_id(target_tag),
                "target_tag": target_tag,
                "guide_delta": epsilon,
                "adv_hat_Z_raw": str(adv_path),
            })
    return rows


if GUIDE_SOURCE == "saved_downstream":
    saved_downstream_rows = []
    guide_epsilons = sorted(set(float(stage["guide_epsilon"]) for stage in ATTACK_STAGE_SCHEDULE))
    for epsilon in guide_epsilons:
        saved_downstream_rows.extend(validate_saved_downstream_stage(epsilon))
    pd.DataFrame(saved_downstream_rows).to_csv(
        WARMUP_SUMMARY_DIR / "saved_downstream_manifest.csv", index=False
    )
    print("Validated fixed saved-downstream guide(s):", guide_epsilons)
else:
    print("Guide source is 'relocation': saved downstream forecasts are not used.")
