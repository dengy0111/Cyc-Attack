# Stage 09: TempestExtremes evaluation of target-focused guide+surrogate attack, direct downstream trajectory metrics, and input closeness statistics

require_paths(DETECT_BIN, STITCH_BIN)
os.chmod(DETECT_BIN, 0o755)
os.chmod(STITCH_BIN, 0o755)


def downstream_tensor_to_netcdf(raw_tensor, sample, out_path):
    source = DATA_DIR / sample / f"{PREFIX}_tempestextremes_{sample}.nc"
    require_paths(source)
    source_dataset = xr.open_dataset(source, decode_times=False)
    time_steps = min(source_dataset.sizes["time"], raw_tensor.shape[0])
    raw = raw_tensor[:time_steps].cpu().numpy().astype(np.float32)
    raw[:, 0] = np.maximum(raw[:, 0], 0.0)
    dataset = xr.Dataset(coords={
        "time": source_dataset["time"],
        "lat": source_dataset["lat"],
        "lon": source_dataset["lon"],
    })
    for channel, variable_name in enumerate(DOWNSTREAM_VARS):
        dataset[variable_name] = xr.DataArray(
            raw[:, channel], dims=("time", "lat", "lon"), coords=dataset.coords
        )
        if variable_name in source_dataset and "units" in source_dataset[variable_name].attrs:
            dataset[variable_name].attrs["units"] = source_dataset[variable_name].attrs["units"]
    dataset["time"].encoding.update(source_dataset["time"].encoding)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(out_path)
    dataset.close()
    source_dataset.close()


def run_tempestextremes(te_nc, nodes_path, tracks_path):
    subprocess.run([
        str(DETECT_BIN), "--in_data", str(te_nc), "--out", str(nodes_path),
        "--searchbymin", "mean_sea_level_pressure",
        "--closedcontourcmd", "mean_sea_level_pressure,200.0,5.5,0;thickness,-58.8,6.5,1.0",
        "--mergedist", "6.0",
        "--outputcmd", "mean_sea_level_pressure,min,0;wind_speed_10m,max,2.00000001;elevation,min,0",
    ], check=True)
    subprocess.run([
        str(STITCH_BIN), "--in", str(nodes_path), "--out", str(tracks_path),
        "--in_fmt", "lon,lat,slp,wind,zs", "--range", "8.0", "--mintime", "54h",
        "--maxgap", "24h",
        "--threshold", "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<=,150.0,10",
    ], check=True)


def haversine_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    value = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.rad2deg(
        2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(1e-15, 1 - value)))
    )


def track_overlap(target_track, pred_track, tolerance_deg=1.0, time_tolerance_hours=0.0):
    pred_nodes = pred_track["nodes"]
    hits = 0
    hit_times = []
    for target_idx, target_node in enumerate(target_track["nodes"]):
        candidates = []
        for pred_node in pred_nodes:
            time_difference = abs(
                (pred_node["datetime"] - target_node["datetime"]).total_seconds()
            ) / 3600.0
            if time_difference <= float(time_tolerance_hours) + 1e-9:
                candidates.append((pred_node["lon"], pred_node["lat"]))
        if candidates:
            distance = min(
                haversine_deg(target_node["lon"], target_node["lat"], lon, lat)
                for lon, lat in candidates
            )
            if distance < tolerance_deg:
                hits += 1
                hit_times.append(target_idx)
    total = max(1, len(target_track["nodes"]))
    return hits / total, hits, hit_times


def best_track_match(reference_track, pred_tracks, tolerance_deg, time_tolerance_hours):
    matches = [
        track_overlap(reference_track, pred_track, tolerance_deg, time_tolerance_hours)
        for pred_track in pred_tracks
    ]
    if not matches:
        return 0.0, 0, []
    return max(matches, key=lambda item: (item[1], item[0]))


def trajectory_metrics_target_aware(
    target_tracks,
    original_tracks,
    pred_tracks,
    metadata,
    tolerance_deg=1.0,
    time_tolerance_hours=0.0,
):
    target_mode = metadata["target_mode"]
    changed_indices = [int(idx) for idx in metadata["changed_track_indices"]]
    removed_indices = [int(idx) for idx in metadata.get("removed_original_track_indices", [])] or changed_indices
    source_indices = [int(idx) for idx in metadata.get("target_track_source_indices", range(len(target_tracks)))]

    target_scores = []
    for target_track in target_tracks:
        score, _, _ = best_track_match(
            target_track, pred_tracks, tolerance_deg, time_tolerance_hours
        )
        target_scores.append(score)

    dr_all = (
        float(np.mean([score >= 0.5 for score in target_scores]))
        if target_scores else np.nan
    )
    changed_target_scores = [
        target_scores[target_idx]
        for target_idx, source_idx in enumerate(source_indices)
        if source_idx in changed_indices
    ]
    dr_changed = (
        float(np.mean([score >= 0.5 for score in changed_target_scores]))
        if changed_target_scores else np.nan
    )

    removed_original_scores = []
    for original_idx in removed_indices:
        score, _, _ = best_track_match(
            original_tracks[original_idx], pred_tracks, tolerance_deg, time_tolerance_hours
        )
        removed_original_scores.append(score)
    remove_success = (
        float(np.mean([score < 0.5 for score in removed_original_scores]))
        if removed_original_scores else np.nan
    )
    target_success = remove_success if target_mode == "remove_changed_original" else dr_changed

    classifications = []
    false_alarm_count = 0
    residual_removed_original_count = 0
    true_unrelated_false_alarm_count = 0

    for pred_idx, pred_track in enumerate(pred_tracks):
        desired_matches = [
            track_overlap(target_track, pred_track, tolerance_deg, time_tolerance_hours)
            for target_track in target_tracks
        ]
        if desired_matches:
            best_target_idx = max(
                range(len(desired_matches)),
                key=lambda idx: (desired_matches[idx][1], desired_matches[idx][0]),
            )
            desired_score, desired_hits, desired_hit_times = desired_matches[best_target_idx]
            best_target_source_idx = source_indices[best_target_idx]
        else:
            best_target_idx = None
            best_target_source_idx = None
            desired_score, desired_hits, desired_hit_times = 0.0, 0, []

        is_false_alarm = (
            desired_hits == 0
            or (
                desired_hits == 1
                and desired_hit_times
                and desired_hit_times[0] == 0
            )
        )

        removed_matches = []
        for original_idx in removed_indices:
            score, hits, hit_times = track_overlap(
                original_tracks[original_idx],
                pred_track,
                tolerance_deg,
                time_tolerance_hours,
            )
            removed_matches.append((score, hits, hit_times, original_idx))
        best_removed = max(removed_matches, default=(0.0, 0, [], None), key=lambda item: (item[1], item[0]))

        if not is_false_alarm:
            category = "desired_target_match"
        elif best_removed[0] >= 0.5:
            category = "residual_removed_original"
            false_alarm_count += 1
            residual_removed_original_count += 1
        else:
            category = "true_unrelated_false_alarm"
            false_alarm_count += 1
            true_unrelated_false_alarm_count += 1

        classifications.append({
            "pred_track_index": pred_idx,
            "classification": category,
            "best_desired_target_index": best_target_idx,
            "best_desired_target_source_original_index": best_target_source_idx,
            "best_desired_overlap": desired_score,
            "best_desired_hits": desired_hits,
            "best_removed_original_index": best_removed[3],
            "best_removed_original_overlap": best_removed[0],
            "best_removed_original_hits": best_removed[1],
            "time_tolerance_hours": time_tolerance_hours,
        })

    num_pred_tracks = len(pred_tracks)
    far = false_alarm_count / num_pred_tracks if num_pred_tracks else 0.0
    return {
        "num_target_tracks": len(target_tracks),
        "num_pred_tracks": num_pred_tracks,
        "DR_all": dr_all,
        "DR_changed": dr_changed,
        "remove_success": remove_success,
        "target_success": target_success,
        "FAR": far,
        "false_alarm_tracks": false_alarm_count,
        "residual_removed_original_tracks": residual_removed_original_count,
        "true_unrelated_false_alarm_tracks": true_unrelated_false_alarm_count,
        "FAR_residual_removed_original": (
            residual_removed_original_count / num_pred_tracks if num_pred_tracks else 0.0
        ),
        "FAR_true_unrelated": (
            true_unrelated_false_alarm_count / num_pred_tracks if num_pred_tracks else 0.0
        ),
        "mean_target_overlap": float(np.mean(target_scores)) if target_scores else np.nan,
        "mean_removed_original_overlap": (
            float(np.mean(removed_original_scores)) if removed_original_scores else np.nan
        ),
    }, classifications


def mask_rates(target_mask, pred_mask):
    target = target_mask.bool()
    prediction = pred_mask.bool()
    true_positive = int((target & prediction).sum())
    true_negative = int((~target & ~prediction).sum())
    false_positive = int((~target & prediction).sum())
    false_negative = int((target & ~prediction).sum())
    return {
        "TPR": true_positive / max(1, true_positive + false_negative),
        "FNR": false_negative / max(1, true_positive + false_negative),
        "FPR": false_positive / max(1, false_positive + true_negative),
        "TNR": true_negative / max(1, false_positive + true_negative),
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
    }


def _track_lon_lat(track):
    lon = _plot_lon([node["lon"] for node in track["nodes"]])
    lat = np.asarray([node["lat"] for node in track["nodes"]], dtype=float)
    return lon, lat


def _changed_focus_tracks(original_tracks, target_tracks, metadata, removed_indices):
    changed_indices = [int(idx) for idx in metadata.get("changed_track_indices", [])]
    focus_indices = changed_indices or [int(idx) for idx in removed_indices]
    focus_original = [
        original_tracks[idx]
        for idx in focus_indices
        if 0 <= idx < len(original_tracks)
    ]

    source_indices = metadata.get("target_track_source_indices")
    if source_indices is not None and len(source_indices) == len(target_tracks):
        focus_target = [
            track
            for track, source_idx in zip(target_tracks, source_indices)
            if int(source_idx) in focus_indices
        ]
    elif metadata.get("target_mode") == "remove_changed_original":
        focus_target = []
    else:
        focus_target = list(target_tracks)

    if not focus_original and original_tracks:
        focus_original = [original_tracks[0]]
    if not focus_target and metadata.get("target_mode") != "remove_changed_original":
        focus_target = list(target_tracks)
    return focus_original, focus_target, focus_indices


def _zoom_extent_from_tracks(focus_tracks, lon_pad=4.0, lat_pad=4.0, min_lon_span=20.0, min_lat_span=14.0):
    lons, lats = [], []
    for track in focus_tracks:
        lon, lat = _track_lon_lat(track)
        if len(lon):
            lons.extend(lon.tolist())
            lats.extend(lat.tolist())
    if not lons:
        return None

    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    if lon_max - lon_min > 120.0:
        return None

    lon_center = 0.5 * (lon_min + lon_max)
    lat_center = 0.5 * (lat_min + lat_max)
    lon_half = max(0.5 * (lon_max - lon_min) + lon_pad, 0.5 * min_lon_span)
    lat_half = max(0.5 * (lat_max - lat_min) + lat_pad, 0.5 * min_lat_span)

    west = max(-180.0, lon_center - lon_half)
    east = min(180.0, lon_center + lon_half)
    south = max(-80.0, lat_center - lat_half)
    north = min(80.0, lat_center + lat_half)

    if east - west < min_lon_span:
        missing = min_lon_span - (east - west)
        west = max(-180.0, west - missing / 2.0)
        east = min(180.0, east + missing / 2.0)
    if north - south < min_lat_span:
        missing = min_lat_span - (north - south)
        south = max(-80.0, south - missing / 2.0)
        north = min(80.0, north + missing / 2.0)
    return [west, east, south, north]


def _track_intersects_extent(track, extent, extra_pad=2.0):
    if extent is None:
        return True
    west, east, south, north = extent
    lon, lat = _track_lon_lat(track)
    if len(lon) == 0:
        return False
    inside = (
        (lon >= west - extra_pad)
        & (lon <= east + extra_pad)
        & (lat >= south - extra_pad)
        & (lat <= north + extra_pad)
    )
    return bool(np.any(inside))


def plot_track_comparison(
    sample,
    original_tracks,
    target_tracks,
    pred_tracks,
    removed_indices,
    metadata,
    out_path,
    title,
):
    focus_original, focus_target, focus_indices = _changed_focus_tracks(
        original_tracks, target_tracks, metadata, removed_indices
    )
    focus_for_extent = list(focus_original) + list(focus_target)
    extent = _zoom_extent_from_tracks(focus_for_extent)
    local_pred_tracks = [
        track for track in pred_tracks if _track_intersects_extent(track, extent)
    ]

    fig = plt.figure(figsize=(10.5, 6.6), dpi=170)
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="#fffdf0")
    axis.add_feature(cfeature.OCEAN, facecolor="#f4fbff")
    axis.coastlines(linewidth=0.65)
    axis.add_feature(cfeature.BORDERS, linewidth=0.35)

    groups = [
        (focus_original, "Original", "#1f4e99", "-", "o", 1.5, 3.2),
        (focus_target, "Desired target", "#b2182b", "--", "^", 1.7, 3.5),
        (local_pred_tracks, "Adversarial prediction", "#222222", "-", "s", 1.5, 3.0),
    ]
    for tracks, label, color, linestyle, marker, linewidth, markersize in groups:
        for track_idx, track in enumerate(tracks):
            lon, lat = _track_lon_lat(track)
            axis.plot(
                lon,
                lat,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                transform=ccrs.Geodetic(),
                label=label if track_idx == 0 else None,
                zorder=4,
            )

    for plot_idx, original_idx in enumerate(focus_indices):
        if not (0 <= original_idx < len(original_tracks)):
            continue
        track = original_tracks[original_idx]
        lon, lat = _track_lon_lat(track)
        axis.plot(
            lon,
            lat,
            color="#000000",
            linestyle=":",
            marker="x",
            markersize=3.4,
            linewidth=1.35,
            transform=ccrs.Geodetic(),
            label="Original path to remove" if plot_idx == 0 else None,
            zorder=5,
        )

    if extent is not None:
        axis.set_extent(extent, ccrs.PlateCarree())
    else:
        axis.set_global()

    grid = axis.gridlines(draw_labels=True, linewidth=0.25, alpha=0.45, linestyle="--")
    grid.top_labels = False
    grid.right_labels = False
    grid.xlabel_style = {"size": 8}
    grid.ylabel_style = {"size": 8}

    axis.set_title(title, fontsize=10, pad=10)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    display(fig)
    plt.close(fig)

    print(
        f"Zoom plot: sample={sample} changed_indices={focus_indices} "
        f"local_pred_tracks={len(local_pred_tracks)}/{len(pred_tracks)} extent={extent}"
    )


def _np_stats(array):
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        return {"mean": np.nan, "max": np.nan, "min": np.nan, "range": np.nan}
    if not np.isfinite(array).all():
        raise ValueError("Input statistics contain NaN or infinity.")
    mn, mx = float(array.min()), float(array.max())
    return {"mean": float(array.mean()), "max": mx, "min": mn, "range": mx - mn}


def _flatten_stats_dict(stats):
    out = {}
    for prefix, values in stats.items():
        for name, value in values.items():
            out[f"input_{prefix}_{name}"] = value
    return out


def input_closeness_statistics(clean_inputs, adversarial_inputs):
    global_groups = {
        "perturbed_value_raw": [],
        "perturbed_value_normalized": [],
        "perturbation_magnitude_raw": [],
        "perturbation_magnitude_normalized": [],
    }
    per_variable_rows = []

    for var in PERTURB_INPUT_VARS:
        if var not in clean_inputs or var not in adversarial_inputs:
            continue

        clean_da = clean_inputs[var]
        adv_da = adversarial_inputs[var]
        clean_raw = np.asarray(clean_da.values, dtype=np.float32)
        adv_raw = np.asarray(adv_da.values, dtype=np.float32)
        if clean_raw.shape != adv_raw.shape:
            raise ValueError(f"Input variable shape mismatch for {var}: {clean_raw.shape} != {adv_raw.shape}")

        mean_v, std_v = stat_for_var(var, clean_da)
        mean_v = np.asarray(mean_v, dtype=np.float32)
        std_v = np.asarray(std_v, dtype=np.float32)
        clean_norm = (clean_raw - mean_v) / std_v
        adv_norm = (adv_raw - mean_v) / std_v

        attack_mask = np.asarray(build_perturb_mask(var, clean_da)) > 0.5
        if not attack_mask.any():
            continue

        groups = {
            "perturbed_value_raw": adv_raw[attack_mask],
            "perturbed_value_normalized": adv_norm[attack_mask],
            "perturbation_magnitude_raw": np.abs(adv_raw - clean_raw)[attack_mask],
            "perturbation_magnitude_normalized": np.abs(adv_norm - clean_norm)[attack_mask],
        }

        row = {
            "variable": var,
            "attackable_element_count": int(attack_mask.sum()),
            "attackable_fraction": float(attack_mask.mean()),
            "pressure_levels": ",".join(str(x) for x in PERTURB_LEVELS[var]) if var in PERTURB_LEVELS else "all",
        }
        for prefix, arr in groups.items():
            stats = _np_stats(arr)
            global_groups[prefix].append(np.asarray(arr).reshape(-1))
            for name, value in stats.items():
                row[f"input_{prefix}_{name}"] = value
        per_variable_rows.append(row)

    global_stats = {}
    for prefix, arrays in global_groups.items():
        if arrays:
            global_stats[prefix] = _np_stats(np.concatenate(arrays))
        else:
            global_stats[prefix] = {
                "mean": np.nan, "max": np.nan, "min": np.nan, "range": np.nan
            }
    return _flatten_stats_dict(global_stats), per_variable_rows

def evaluate_end2end_stage(
    sample,
    target_tag,
    stage_dir,
    target_mask,
    target_track_path,
    selection_type,
):
    stage_dir = Path(stage_dir)
    pred_track_path = stage_dir / "cyclone_tracks.txt"
    original_tracks = read_track_file(
        DATA_DIR
        / sample
        / f"{PREFIX}_cyclone_tracks_{sample}.txt"
    )
    target_tracks = read_track_file(target_track_path)
    pred_tracks = (
        read_track_file(pred_track_path)
        if pred_track_path.exists()
        and pred_track_path.stat().st_size
        else []
    )

    metadata_path = (
        sample_target_dir(sample, target_tag)
        / "target"
        / "target_metadata.json"
    )
    metadata = json.load(open(metadata_path))
    changed_indices = [
        int(idx)
        for idx in metadata["changed_track_indices"]
    ]
    removed_indices = [
        int(idx)
        for idx in metadata.get(
            "removed_original_track_indices",
            [],
        )
    ] or changed_indices

    pred_mask = tracks_to_mask_per_step(
        pred_tracks,
        sample,
    )
    rates = mask_rates(
        target_mask,
        pred_mask,
    )

    trajectory_exact, classifications = (
        trajectory_metrics_target_aware(
            target_tracks,
            original_tracks,
            pred_tracks,
            metadata,
            tolerance_deg=TRACK_MATCH_TOLERANCE_DEG,
            time_tolerance_hours=0.0,
        )
    )
    trajectory_tolerant, _ = (
        trajectory_metrics_target_aware(
            target_tracks,
            original_tracks,
            pred_tracks,
            metadata,
            tolerance_deg=TRACK_MATCH_TOLERANCE_DEG,
            time_tolerance_hours=(
                TRACK_MATCH_TIME_TOLERANCE_HOURS
            ),
        )
    )

    final_downstream = torch.load(
        stage_dir / "final_downstream_input.pt",
        map_location="cpu",
    ).float()
    _, clean_inputs, _, _ = prepare_graphcast_case(
        sample
    )
    adv_inputs = xr.open_dataset(
        stage_dir / "adv_inputs_xr_attacked.nc"
    )
    input_stats, input_variable_rows = (
        input_closeness_statistics(
            clean_inputs,
            adv_inputs,
        )
    )

    original_mask = tracks_to_mask_per_step(
        original_tracks,
        sample,
    ).bool()
    unchanged_positive_positions = (
        original_mask & target_mask.bool()
    )
    unchanged_counted_as_fp = int(
        (
            unchanged_positive_positions
            & (~target_mask.bool())
            & pred_mask.bool()
        ).sum()
    )
    if unchanged_counted_as_fp != 0:
        raise AssertionError(
            "Unchanged original-target positions "
            "were counted as false positives."
        )

    metrics = {
        "sample": sample,
        "selection_type": selection_type,
        "target_id": get_target_id(target_tag),
        "target_tag": target_tag,
        "ratio_tag": target_tag,
        "target_mode": metadata["target_mode"],
        "downstream_epsilon": float(
            DOWNSTREAM_EPSILON
        ),
        "downstream_delta_tag": downstream_delta_tag(),
        "end2end_epsilon": float(
            END2END_EPSILON
        ),
        "end2end_epsilon_tag": end2end_epsilon_tag(),
        **rates,
        **trajectory_exact,
        "paper_target_success": trajectory_exact[
            "target_success"
        ],
        "paper_DR_changed": trajectory_exact[
            "DR_changed"
        ],
        "paper_remove_success": trajectory_exact[
            "remove_success"
        ],
        "target_success_time_tolerant_6h": (
            trajectory_tolerant["target_success"]
        ),
        "FAR_time_tolerant_6h": (
            trajectory_tolerant["FAR"]
        ),
        "DR_changed_time_tolerant_6h": (
            trajectory_tolerant["DR_changed"]
        ),
        "remove_success_time_tolerant_6h": (
            trajectory_tolerant["remove_success"]
        ),
        "unchanged_positive_position_count": int(
            unchanged_positive_positions.sum()
        ),
        "unchanged_counted_as_fp": (
            unchanged_counted_as_fp
        ),
        "false_alarm_validation_passed": (
            unchanged_counted_as_fp == 0
        ),
        "wind_min_raw": float(
            final_downstream[:, 0].min()
        ),
        **input_stats,
    }

    pd.DataFrame([metrics]).to_csv(
        stage_dir / "metrics.csv",
        index=False,
    )
    pd.DataFrame(classifications).to_csv(
        stage_dir
        / "predicted_track_classification.csv",
        index=False,
    )

    for row in input_variable_rows:
        row.update({
            "sample": sample,
            "selection_type": selection_type,
            "target_id": get_target_id(
                target_tag
            ),
            "target_tag": target_tag,
            "target_mode": metadata[
                "target_mode"
            ],
            "downstream_epsilon": float(
                DOWNSTREAM_EPSILON
            ),
            "end2end_epsilon": float(
                END2END_EPSILON
            ),
        })

    pd.DataFrame(input_variable_rows).to_csv(
        stage_dir
        / "input_closeness_per_variable.csv",
        index=False,
    )

    plot_track_comparison(
        sample,
        original_tracks,
        target_tracks,
        pred_tracks,
        removed_indices,
        metadata,
        stage_dir / "trajectory_comparison.png",
        (
            f"End-to-end ({selection_type}): {sample} — "
            f"{get_target_id(target_tag)} / {target_tag} / "
            f"{end2end_run_tag()}"
        ),
    )
    adv_inputs.close()
    return metrics, input_variable_rows

def ensure_current_end2end_tempestextremes(stage_dir, raw_path, sample, target_tag):
    stage_dir = Path(stage_dir)
    te_nc = stage_dir / "adv_tempestextremes.nc"
    nodes = stage_dir / "detected_nodes.txt"
    tracks = stage_dir / "cyclone_tracks.txt"
    fingerprint_path = stage_dir / "tempestextremes_input_sha256.txt"
    current_fingerprint = file_sha256(raw_path)
    previous_fingerprint = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
    must_run = OVERWRITE or not tracks.exists() or previous_fingerprint != current_fingerprint
    profile_record = None
    if must_run:
        profile_dir = stage_dir / "te_profile"
        with ResourceProfiler("end2end_tempestextremes", sample, target_tag, profile_dir) as profiler:
            downstream_tensor_to_netcdf(torch.load(raw_path, map_location="cpu"), sample, te_nc)
            run_tempestextremes(te_nc, nodes, tracks)
        profile_record = profiler.record
        profile_record.update({
            "target_id": get_target_id(target_tag),
            "target_tag": target_tag,
            "downstream_epsilon": DOWNSTREAM_EPSILON,
            "downstream_delta_tag": downstream_delta_tag(),
            "end2end_epsilon": END2END_EPSILON,
            "end2end_epsilon_tag": end2end_epsilon_tag(),
            "adv_input_sha256": current_fingerprint,
        })
        with open(profile_dir / "runtime_memory.json", "w") as f:
            json.dump(profile_record, f, indent=2)
        fingerprint_path.write_text(current_fingerprint)
    else:
        print("Reusing TempestExtremes output with matching end-to-end downstream hash:", tracks)
    return profile_record

# Evaluate both saved checkpoint families for every explicit attack stage.
all_end2end_metrics = []
all_input_variable_rows = []
all_te_profile_rows = []

for stage_index, spec in enumerate(ATTACK_STAGE_SCHEDULE):
    input_epsilon = float(spec["input_epsilon"])
    guide_epsilon = float(spec["guide_epsilon"])
    stage_steps = int(spec["steps"])
    set_attack_stage(input_epsilon, guide_epsilon, stage_steps)
    stage_metrics = []
    stage_input_rows = []
    stage_te_profiles = []
    print(
        f"\nEvaluating stage {stage_index + 1}/{len(ATTACK_STAGE_SCHEDULE)}: "
        f"input_epsilon={input_epsilon:g}, guide_delta={guide_epsilon:g}, steps={stage_steps}"
    )

    for target_tag in TARGET_SCENARIOS:
        for sample in RUN_CASES:
            run_dir = end2end_sample_dir(sample, target_tag, guide_epsilon, input_epsilon)
            target_dir = sample_target_dir(sample, target_tag) / "target"
            target_mask = torch.load(
                target_dir / f"target_mask_{target_tag}.pt",
                map_location="cpu",
            ).float()
            target_track_path = target_dir / f"target_tracks_{target_tag}.txt"

            selection_dirs = [
                ("objective_best", run_dir / "objective_best"),
                ("te_best", run_dir / "te_best"),
            ]
            if not any(d.exists() for _, d in selection_dirs):
                selection_dirs = [("te_best", run_dir)]

            for selection_type, result_dir in selection_dirs:
                if not result_dir.exists():
                    continue
                raw_path = result_dir / "final_downstream_input.pt"
                adv_input_path = result_dir / "adv_inputs_xr_attacked.nc"
                require_paths(raw_path, adv_input_path)

                print(f"  {sample} {target_tag} [{selection_type}]")
                profile = ensure_current_end2end_tempestextremes(
                    result_dir, raw_path, sample, target_tag
                )
                if profile:
                    profile.update({
                        "stage_index": int(stage_index),
                        "stage_kind": str(spec["kind"]),
                        "input_epsilon": input_epsilon,
                        "guide_delta": guide_epsilon,
                        "selection_type": selection_type,
                    })
                    stage_te_profiles.append(profile)
                    all_te_profile_rows.append(profile)

                metrics, input_rows = evaluate_end2end_stage(
                    sample,
                    target_tag,
                    result_dir,
                    target_mask,
                    target_track_path,
                    selection_type,
                )
                metrics.update({
                    "stage_index": int(stage_index),
                    "stage_kind": str(spec["kind"]),
                    "input_epsilon": input_epsilon,
                    "guide_delta": guide_epsilon,
                    "steps_this_stage": int(TRAJECTORY_PHASE_STEPS if TRAJECTORY_PHASE_STEPS is not None else stage_steps),
                    "guidance_steps": int(GUIDANCE_PHASE_STEPS if GUIDANCE_PHASE_STEPS is not None else stage_steps),
                    "objective_phase": "trajectory",
                    # Backward-compatible alias: this now means upstream epsilon only.
                    "synchronized_delta": input_epsilon,
                })
                stage_metrics.append(metrics)
                all_end2end_metrics.append(metrics)

                for row in input_rows:
                    row.update({
                        "stage_index": int(stage_index),
                        "stage_kind": str(spec["kind"]),
                        "input_epsilon": input_epsilon,
                        "guide_delta": guide_epsilon,
                        "synchronized_delta": input_epsilon,
                    })
                    stage_input_rows.append(row)
                    all_input_variable_rows.append(row)

                # Root-level evaluation outputs correspond to the real-TE-selected result.
                if selection_type == "te_best" and result_dir != run_dir:
                    for name in [
                        "metrics.csv",
                        "predicted_track_classification.csv",
                        "input_closeness_per_variable.csv",
                        "trajectory_comparison.png",
                    ]:
                        src = result_dir / name
                        if src.exists():
                            shutil.copy2(src, run_dir / name)

    summary_dir = end2end_summary_dir(guide_epsilon, input_epsilon)
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stage_metrics).to_csv(
        summary_dir / "end2end_metrics_summary.csv", index=False
    )
    pd.DataFrame(stage_input_rows).to_csv(
        summary_dir / "input_closeness_per_variable_all_cases.csv", index=False
    )
    if stage_te_profiles:
        pd.DataFrame(stage_te_profiles).to_csv(
            summary_dir / "tempestextremes_runtime_memory.csv", index=False
        )

# Restore final-stage globals.
_final = ATTACK_STAGE_SCHEDULE[-1]
set_attack_stage(_final["input_epsilon"], _final["guide_epsilon"], _final["steps"])

all_metrics_df = pd.DataFrame(all_end2end_metrics)
all_metrics_df.to_csv(
    WARMUP_SUMMARY_DIR / "end2end_metrics_all_deltas.csv", index=False
)
pd.DataFrame(all_input_variable_rows).to_csv(
    WARMUP_SUMMARY_DIR / "input_closeness_all_deltas.csv", index=False
)
if all_te_profile_rows:
    pd.DataFrame(all_te_profile_rows).to_csv(
        WARMUP_SUMMARY_DIR / "tempestextremes_runtime_memory_all_deltas.csv", index=False
    )

display(all_metrics_df)
