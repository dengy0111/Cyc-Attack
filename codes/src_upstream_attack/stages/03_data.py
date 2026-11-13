# Stage 03: Load the clean TC2 downstream tensors and the per-case frame layout.
# The clean tensor is used for figures and, in the saved_downstream ablation,
# as the reference of the saved guide. No detector model is loaded.

CLEAN_DOWNSTREAM_PATH = DATA_DIR / f"{PREFIX}_hat_Z_case_study.pt"
DATES_PATH = DATA_DIR / f"{PREFIX}_adv_dates_case_study_{PREDICTION_GOAL}.csv"
require_paths(CLEAN_DOWNSTREAM_PATH)

clean_hat_Z = torch.load(CLEAN_DOWNSTREAM_PATH, map_location="cpu").float()
if clean_hat_Z.ndim != 4 or clean_hat_Z.shape[1:] != (4, GRID_H, GRID_W):
    raise ValueError(f"Unexpected clean downstream shape: {clean_hat_Z.shape}")

# The supplied tensor contains 10 × 38 = 380 frames, while an older dates CSV may
# contain only 8 × 38 = 304 rows. When the tensor has the complete TC2 packed
# layout, its authoritative order is TC2_CASES and the stale CSV is not used for
# slicing. This avoids dropping the last two cases or shifting case boundaries.
expected_packed_frames = len(TC2_CASES) * FORECAST_STEPS
csv_dates = []
if DATES_PATH.exists():
    dates_df = pd.read_csv(DATES_PATH)
    if "dates" in dates_df.columns:
        csv_dates = dates_df["dates"].astype(str).tolist()

CASE_SLICES = {}
CASE_FRAME_DATES = {}
DATA_LAYOUT_MODE = None

if clean_hat_Z.shape[0] == expected_packed_frames:
    DATA_LAYOUT_MODE = "packed_tc2_case_order"
    for case_index, sample in enumerate(TC2_CASES):
        start = case_index * FORECAST_STEPS
        CASE_SLICES[sample] = slice(start, start + FORECAST_STEPS)
        sample_dt = datetime.strptime(sample, "%Y-%m-%dT%H%M")
        CASE_FRAME_DATES[sample] = [
            (sample_dt + timedelta(hours=6 * (lead + 1))).strftime("%Y-%m-%dT%H%M")
            for lead in range(FORECAST_STEPS)
        ]
    if csv_dates and len(csv_dates) != clean_hat_Z.shape[0]:
        print(
            f"Warning: ignoring stale {DATES_PATH.name}: {len(csv_dates)} rows, "
            f"but clean_hat_Z has {clean_hat_Z.shape[0]} frames. "
            "Using contiguous 38-frame blocks in TC2_CASES order."
        )
elif csv_dates and len(csv_dates) == clean_hat_Z.shape[0]:
    DATA_LAYOUT_MODE = "date_csv"
    for sample in TC2_CASES:
        sample_dt = datetime.strptime(sample, "%Y-%m-%dT%H%M")
        candidates = [
            sample_dt.strftime("%Y-%m-%dT%H%M"),
            (sample_dt + timedelta(hours=6)).strftime("%Y-%m-%dT%H%M"),
        ]
        starts = [csv_dates.index(x) for x in candidates if x in csv_dates]
        if not starts:
            raise ValueError(f"Neither {candidates[0]} nor {candidates[1]} was found in {DATES_PATH}")
        start = starts[0]
        stop = start + FORECAST_STEPS
        if stop > clean_hat_Z.shape[0]:
            raise IndexError(f"Insufficient frames for {sample}: slice({start}, {stop})")
        CASE_SLICES[sample] = slice(start, stop)
        CASE_FRAME_DATES[sample] = csv_dates[start:stop]
else:
    raise ValueError(
        f"Cannot infer TC2 layout: clean_hat_Z has {clean_hat_Z.shape[0]} frames; "
        f"expected packed layout {expected_packed_frames}, while dates CSV has {len(csv_dates)} rows."
    )



def case_slice(sample):
    if sample not in CASE_SLICES:
        raise KeyError(f"Unknown TC2 case: {sample}")
    return CASE_SLICES[sample]


for sample in RUN_CASES:
    sl = case_slice(sample)
    labels = CASE_FRAME_DATES[sample]
    print(sample, sl, labels[0], labels[-1])

print("DATA_LAYOUT_MODE:", DATA_LAYOUT_MODE)
print("clean_hat_Z:", tuple(clean_hat_Z.shape))
