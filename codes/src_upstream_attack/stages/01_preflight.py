# Stage 01: HPCC preflight: HOME software + SCRATCH data
print("Python executable:", sys.executable)
print("Scratch base:", SCRATCH_BASE)
print("SCRATCH data root:", SCRATCH_ROOT)
print("HOME software root:", SOFTWARE_ROOT)
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch GPU:", torch.cuda.get_device_name(0))
print("JAX devices:", jax.devices())

required_software_paths = [
    GRAPHCAST_PACKAGE_ROOT,
    TEMPEST_ROOT,
    DETECT_BIN,
    STITCH_BIN,
    SURROGATE_REPO,
]

required_scratch_paths = [
    DATA_DIR,
    ELEVATION_NC,
    DATA_DIR / f"{PREFIX}_hat_Z_case_study.pt",
    SURROGATE_CKPT,
]
for sample in RUN_CASES:
    sample_dir = DATA_DIR / sample
    required_scratch_paths.extend([
        sample_dir / f"{PREFIX}_input_range_{sample}.nc",
        sample_dir / f"{PREFIX}_tempestextremes_{sample}.nc",
        sample_dir / f"{PREFIX}_cyclone_tracks_{sample}.txt",
    ])
    if not BUILD_TARGETS:
        for target_tag in TARGET_SCENARIOS:
            target_dir = sample_target_dir(sample, target_tag) / "target"
            required_scratch_paths.extend([
                target_dir / f"target_mask_{target_tag}.pt",
                target_dir / f"target_tracks_{target_tag}.txt",
                target_dir / "target_metadata.json",
            ])
            if GUIDE_SOURCE == "saved_downstream":
                for guide_eps in sorted(set(float(stage["guide_epsilon"]) for stage in ATTACK_STAGE_SCHEDULE)):
                    required_scratch_paths.append(
                        downstream_sample_dir(sample, target_tag, guide_eps) / "adv_hat_Z_raw.pt"
                    )

missing_software = [Path(p) for p in required_software_paths if not Path(p).exists()]
missing_scratch = [Path(p) for p in required_scratch_paths if not Path(p).exists()]

print(f"Required HOME software paths checked: {len(required_software_paths)}")
print(f"Missing HOME software paths: {len(missing_software)}")
for p in missing_software:
    print("  MISSING SOFTWARE:", p)

print(f"Required SCRATCH/data/model paths checked: {len(required_scratch_paths)}")
print(f"Missing SCRATCH/data/model paths: {len(missing_scratch)}")
for p in missing_scratch[:100]:
    print("  MISSING DATA/MODEL:", p)
if len(missing_scratch) > 100:
    print(f"  ... and {len(missing_scratch) - 100} more")

if missing_software or missing_scratch:
    raise FileNotFoundError(
        "HPCC preflight failed. Check HOME software paths and SCRATCH data/model paths above."
    )

print("TempestExtremes binaries (evaluation/current-best selection only):", DETECT_BIN, STITCH_BIN, sep="\n  ")
print("Differentiable downstream trajectory surrogate:", SURROGATE_CKPT)
print("HPCC HOME-software / SCRATCH-data preflight passed.")
