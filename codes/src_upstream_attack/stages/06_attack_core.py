# Stage 06: Target-focused upstream attack core
#
# Differentiable objective = saved/relocated downstream guide matching + the same
# dilated focal trajectory-generation loss used by the downstream surrogate attack
# + input regularization. Real TempestExtremes is NEVER used in the gradient path;
# it runs periodically only to evaluate the actual stitched trajectory and overwrite
# the rolling current TE-best checkpoint/visualization when the downstream metric improves.

require_paths(DETECT_BIN, STITCH_BIN)
os.chmod(DETECT_BIN, 0o755)
os.chmod(STITCH_BIN, 0o755)


def _latlon_grids_jax(H, W):
    lat = jnp.asarray(np.deg2rad(np.linspace(-90.0, 90.0, H)), dtype=jnp.float32)
    lon = jnp.asarray(np.deg2rad(np.linspace(0.0, 360.0, W, endpoint=False)), dtype=jnp.float32)
    return jnp.meshgrid(lat, lon, indexing="ij")


def min_distance_jax(centers, max_points=256, far=10.0):
    centers = jnp.asarray(centers, dtype=jnp.float32)
    squeeze = centers.ndim == 2
    if squeeze:
        centers = centers[None]
    if centers.ndim == 4:
        centers = centers[:, 0]
    if centers.ndim != 3:
        raise ValueError(f"Expected [T,H,W] centers, got {centers.shape}")

    n_max = int(np.asarray((centers > 0.5).sum(axis=(1, 2))).max()) if centers.shape[0] else 0
    if n_max > max_points:
        raise ValueError(f"{n_max} nodes in one frame > max_points={max_points}")

    T, H, W = centers.shape
    lat_grid, lon_grid = _latlon_grids_jax(H, W)
    lat_f, lon_f = lat_grid.reshape(-1), lon_grid.reshape(-1)
    sin_lat, cos_lat = jnp.sin(lat_f), jnp.cos(lat_f)

    def one(mask):
        idx = jnp.nonzero(mask.reshape(-1) > 0.5, size=max_points, fill_value=-1)[0]
        valid = idx >= 0
        safe = jnp.where(valid, idx, 0)
        cos_d = (
            sin_lat[:, None] * sin_lat[safe][None]
            + cos_lat[:, None] * cos_lat[safe][None]
            * jnp.cos(lon_f[:, None] - lon_f[safe][None])
        )
        d = jnp.arccos(jnp.clip(cos_d, -1.0, 1.0))
        d = jnp.where(valid[None], d, far)
        return d.min(axis=1).reshape(H, W)

    out = jax.lax.map(one, centers)
    return out[0] if squeeze else out


def distance_weight_fast(centers, sigma):
    d = min_distance_jax(centers)
    return jnp.exp(-(d ** 2) / (2.0 * sigma ** 2))


def broadcast_weight(weight, tensor):
    if weight.shape == tensor.shape:
        return weight
    if weight.ndim == 2:
        w = weight
        while w.ndim < tensor.ndim:
            w = w[None]
        return jnp.broadcast_to(w, tensor.shape)
    if weight.ndim == 3 and tensor.ndim == 4:
        return jnp.broadcast_to(weight[:, None], tensor.shape)
    if weight.ndim == 3 and tensor.ndim == 5:
        return jnp.broadcast_to(weight[None, :, None], tensor.shape)
    raise ValueError(f"Cannot broadcast {weight.shape} to {tensor.shape}")


def stat_for_var(var_name, data_array):
    mean_da, std_da = GC_MEAN[var_name], GC_STD[var_name]
    if "level" in data_array.dims and "level" in mean_da.dims:
        levels = data_array.coords["level"]
        mean_da, std_da = mean_da.sel(level=levels), std_da.sel(level=levels)
    mean_np = np.asarray(mean_da.values, np.float32)
    std_np = np.asarray(std_da.values, np.float32)
    if "level" in data_array.dims and mean_np.ndim == 1:
        shape = [1] * data_array.ndim
        shape[data_array.dims.index("level")] = mean_np.shape[0]
        mean_np, std_np = mean_np.reshape(shape), std_np.reshape(shape)
    return jnp.asarray(mean_np), jnp.asarray(std_np)


def save_gc_output(gc_output, sample, out_path):
    out = jax.device_get(gc_output)
    out.to_netcdf(out_path)



def build_perturb_mask(var, data_array):
    shape = tuple(data_array.shape)
    if var not in PERTURB_LEVELS:
        return jnp.ones(shape, dtype=jnp.float32)
    if "level" not in data_array.dims:
        raise ValueError(f"{var} has restricted levels but no level dimension: {data_array.dims}")

    requested = tuple(float(x) for x in PERTURB_LEVELS[var])
    levels = np.asarray(data_array.coords["level"].values, dtype=float)
    missing = [x for x in requested if not np.any(np.isclose(levels, x))]
    if missing:
        raise ValueError(f"{var}: missing requested levels {missing}; available={levels.tolist()}")

    selected_1d = np.zeros_like(levels, dtype=np.float32)
    for requested_level in requested:
        selected_1d[np.isclose(levels, requested_level)] = 1.0

    level_axis = data_array.dims.index("level")
    reshape = [1] * data_array.ndim
    reshape[level_axis] = levels.size
    mask = jnp.asarray(selected_1d.reshape(reshape), dtype=jnp.float32)
    return jnp.broadcast_to(mask, shape)


def validate_perturb_masks(base_ds, perturb_masks):
    summary = {}
    for var, mask in perturb_masks.items():
        row = {
            "shape": list(mask.shape),
            "active_fraction": float(np.asarray(mask).mean()),
        }
        if var in PERTURB_LEVELS:
            level_axis = base_ds[var].dims.index("level")
            levels = np.asarray(base_ds[var]["level"].values)
            reduce_axes = tuple(i for i in range(mask.ndim) if i != level_axis)
            per_level_active = np.asarray(mask).max(axis=reduce_axes)
            active_levels = levels[per_level_active > 0.5].tolist()
            expected = sorted([float(x) for x in PERTURB_LEVELS[var]])
            observed = sorted([float(x) for x in active_levels])
            if observed != expected:
                raise AssertionError(f"{var}: attackable levels {observed} != expected {expected}")
            row["active_levels_hpa"] = observed
        summary[var] = row
    return summary



def _te_downstream_to_nc(raw_tensor, sample, out_path):
    source = DATA_DIR / sample / f"{PREFIX}_tempestextremes_{sample}.nc"
    require_paths(source)
    src = xr.open_dataset(source, decode_times=False)
    n_t = min(src.sizes["time"], raw_tensor.shape[0])
    raw = raw_tensor[:n_t].cpu().numpy().astype(np.float32)
    raw[:, 0] = np.maximum(raw[:, 0], 0.0)
    ds = xr.Dataset(coords={"time": src["time"], "lat": src["lat"], "lon": src["lon"]})
    for ch, name in enumerate(DOWNSTREAM_VARS):
        ds[name] = xr.DataArray(raw[:, ch], dims=("time", "lat", "lon"), coords=ds.coords)
        if name in src and "units" in src[name].attrs:
            ds[name].attrs["units"] = src[name].attrs["units"]
    ds["time"].encoding.update(src["time"].encoding)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    ds.close()
    src.close()


def _te_run(te_nc, nodes_path, tracks_path):
    subprocess.run([
        str(DETECT_BIN), "--in_data", str(te_nc), "--out", str(nodes_path),
        "--searchbymin", "mean_sea_level_pressure",
        "--closedcontourcmd", "mean_sea_level_pressure,200.0,5.5,0;thickness,-58.8,6.5,1.0",
        "--mergedist", "6.0",
        "--outputcmd", "mean_sea_level_pressure,min,0;wind_speed_10m,max,2.00000001;elevation,min,0",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([
        str(STITCH_BIN), "--in", str(nodes_path), "--out", str(tracks_path),
        "--in_fmt", "lon,lat,slp,wind,zs", "--range", "8.0", "--mintime", "54h",
        "--maxgap", "24h",
        "--threshold", "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<=,150.0,10",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _haversine_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    v = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.rad2deg(2 * np.arctan2(np.sqrt(v), np.sqrt(np.maximum(1e-15, 1 - v))))


def _track_nodes_by_time(track):
    by_time = {}
    for node in track.get("nodes", []):
        by_time.setdefault(node["datetime"], []).append(
            (float(node["lon"]) % 360.0, float(node["lat"]))
        )
    return by_time


def _end2end_track_time_lines(pred_tracks, forecast_start, check_name):
    lines = []
    lines.append(
        f"[te_track_times] {check_name}: "
        f"{len(pred_tracks)} stitched end2end adversarial downstream track(s)"
    )
    if not pred_tracks:
        lines.append(f"[te_track_times] {check_name}: no stitched trajectory")
        return lines

    for track_index, track in enumerate(pred_tracks, start=1):
        nodes = list(track.get("nodes", []))
        lines.append(
            f"[te_track_times] {check_name}: track={track_index} "
            f"points={len(nodes)}"
        )
        for point_index, node in enumerate(nodes, start=1):
            dt = node["datetime"]
            hours = (dt - forecast_start).total_seconds() / 3600.0
            step0 = int(round(hours / 6.0))
            step1 = step0 + 1
            lines.append(
                f"[te_track_times] {check_name}: "
                f"track={track_index} point={point_index} "
                f"forecast_step={step1} "
                f"time={dt.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"lon={float(node['lon']) % 360.0:.3f} "
                f"lat={float(node['lat']):.3f}"
            )
    return lines


def _print_end2end_track_times(pred_tracks, forecast_start, check_name):
    lines = _end2end_track_time_lines(pred_tracks, forecast_start, check_name)
    for line in lines:
        print(line)
    return lines


def _best_distance_to_points(lon, lat, pts):
    if not pts:
        return np.inf
    return min(_haversine_deg(lon, lat, a, b) for a, b in pts)



def build_te_reference(sample, target_track_path, metadata_path):
    """Build a changed-track-aware TE reference.

    Only target tracks whose source original index is listed in
    changed_track_indices contribute to target-hit scoring. Original-removal
    scoring uses the changed/removed source original tracks themselves rather
    than a raster-mask heuristic.
    """
    original_track_path = (
        DATA_DIR
        / sample
        / f"{PREFIX}_cyclone_tracks_{sample}.txt"
    )
    require_paths(original_track_path, target_track_path, metadata_path)

    original_tracks = read_track_file(original_track_path)
    target_tracks = read_track_file(target_track_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    changed_indices = [
        int(x)
        for x in metadata.get("changed_track_indices", [])
    ]
    source_indices = [
        int(x)
        for x in metadata.get(
            "target_track_source_indices",
            range(len(target_tracks)),
        )
    ]
    removed_indices = [
        int(x)
        for x in metadata.get(
            "removed_original_track_indices",
            [],
        )
    ] or list(changed_indices)

    if len(source_indices) != len(target_tracks):
        raise ValueError(
            f"target_track_source_indices length mismatch in {metadata_path}: "
            f"{len(source_indices)} != {len(target_tracks)}"
        )

    changed_target_pairs = []
    for target_index, (target_track, source_index) in enumerate(
        zip(target_tracks, source_indices)
    ):
        if source_index not in changed_indices:
            continue
        if not 0 <= source_index < len(original_tracks):
            raise IndexError(
                f"Invalid source original index {source_index} for "
                f"{target_track_path}"
            )
        changed_target_pairs.append({
            "target_index": int(target_index),
            "source_index": int(source_index),
            "target_track": target_track,
            "original_track": original_tracks[source_index],
        })

    removed_original_tracks = []
    for source_index in removed_indices:
        if not 0 <= source_index < len(original_tracks):
            raise IndexError(
                f"Invalid removed original index {source_index} for "
                f"{metadata_path}"
            )
        removed_original_tracks.append({
            "source_index": int(source_index),
            "original_track": original_tracks[source_index],
        })

    return {
        "sample": sample,
        "target_mode": metadata.get("target_mode"),
        "metadata": metadata,
        "original_tracks": original_tracks,
        "target_tracks": target_tracks,
        "changed_indices": changed_indices,
        "source_indices": source_indices,
        "removed_indices": removed_indices,
        "changed_target_pairs": changed_target_pairs,
        "removed_original_tracks": removed_original_tracks,
    }


def _read_detected_nodes(path, forecast_start):
    """Read DetectNodes candidates and attach their forecast-step datetimes."""
    path = Path(path)
    out = []
    if not path.exists() or path.stat().st_size == 0:
        return out
    lines = path.read_text().splitlines()
    i = 0
    step = 0
    while i < len(lines):
        header = lines[i].split()
        i += 1
        if len(header) < 4:
            step += 1
            continue
        try:
            n_nodes = int(header[3])
        except (ValueError, TypeError):
            step += 1
            continue
        dt = forecast_start + timedelta(hours=6 * step)
        for _ in range(n_nodes):
            if i >= len(lines):
                break
            cols = lines[i].split()
            i += 1
            if len(cols) < 4:
                continue
            try:
                lon = float(cols[2]) % 360.0
                lat = float(cols[3])
            except (ValueError, TypeError):
                continue
            out.append({
                "datetime": dt,
                "lon": lon,
                "lat": lat,
            })
        step += 1
    return out



# ---------------------------------------------------------------------------
# Per-TE-check visualization and attack-state diagnostics
# ---------------------------------------------------------------------------
def _tevis_rel_lon(lon, center):
    """Longitude relative to `center`, wrapped to [-180, 180)."""
    return ((np.asarray(lon, dtype=float) - center + 180.0) % 360.0) - 180.0


def _tevis_center_lon(lons):
    """Circular mean longitude in [0, 360); handles tracks crossing 0/180."""
    if len(lons) == 0:
        return 0.0
    rad = np.deg2rad(np.asarray(lons, dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def _tevis_track_lonlat(track):
    lon = np.asarray([node["lon"] for node in track.get("nodes", [])], dtype=float)
    lat = np.asarray([node["lat"] for node in track.get("nodes", [])], dtype=float)
    return lon, lat


def _tevis_extent(lons, lats, center, lon_pad=6.0, lat_pad=5.0,
                  min_lon_span=24.0, min_lat_span=16.0, max_lon_span=150.0):
    """Zoom extent in coordinates relative to `center`; None means global."""
    if len(lons) == 0:
        return None
    rel = _tevis_rel_lon(lons, center)
    lo, hi = float(rel.min()), float(rel.max())
    if hi - lo > max_lon_span:
        return None
    la0, la1 = float(np.min(lats)), float(np.max(lats))
    cx, cy = 0.5 * (lo + hi), 0.5 * (la0 + la1)
    hx = max(0.5 * (hi - lo) + lon_pad, 0.5 * min_lon_span)
    hy = max(0.5 * (la1 - la0) + lat_pad, 0.5 * min_lat_span)
    return [max(-180.0, cx - hx), min(180.0, cx + hx),
            max(-85.0, cy - hy), min(85.0, cy + hy)]


def _tevis_in_extent(lon, lat, center, extent, pad=2.0):
    if extent is None:
        return np.ones(len(lon), dtype=bool)
    rel = _tevis_rel_lon(lon, center)
    w, e, s, n = extent
    return (rel >= w - pad) & (rel <= e + pad) & (lat >= s - pad) & (lat <= n + pad)


def _tevis_prediction_time_footer(pred_tracks, forecast_start, max_chars=118):
    """Return wrapped footer lines listing the time of every black TE-track point."""
    if not pred_tracks:
        return ["Black TE prediction point times: no stitched trajectory"]

    lines = ["Black TE prediction point times"]
    for track_index, track in enumerate(pred_tracks, start=1):
        items = []
        for point_index, node in enumerate(track.get("nodes", []), start=1):
            dt = node["datetime"]
            hours = (dt - forecast_start).total_seconds() / 3600.0
            step1 = int(round(hours / 6.0)) + 1
            items.append(
                f"P{point_index:02d}/S{step1:02d}={dt.strftime('%Y-%m-%d %H:%M')}"
            )

        prefix = f"Track {track_index}: "
        if not items:
            lines.append(prefix + "no points")
            continue

        current = prefix
        continuation = " " * len(prefix)
        for item in items:
            token = item if current.endswith(prefix) else " | " + item
            if len(current) + len(token) <= max_chars:
                current += token
            else:
                lines.append(current)
                current = continuation + item
        lines.append(current)
    return lines


def plot_te_check(te_reference, pred_tracks, candidates, result,
                     forecast_start, out_path, title):
    """One map per TE check.

    blue solid  : changed original track(s) that should disappear
    red dashed  : desired target track(s)          (numbers = forecast step)
    black solid : tracks TempestExtremes actually stitched (numbers = step)
    grey dots   : DetectNodes candidates (before StitchNodes)
    light blue  : unchanged original tracks (context)
    """
    changed_sources = {int(r["source_index"]) for r in te_reference["removed_original_tracks"]}
    changed_sources |= {int(p["source_index"]) for p in te_reference["changed_target_pairs"]}
    original_tracks = te_reference["original_tracks"]
    changed_orig = [original_tracks[i] for i in sorted(changed_sources)
                    if 0 <= i < len(original_tracks)]
    other_orig = [t for i, t in enumerate(original_tracks) if i not in changed_sources]
    target_tracks = [p["target_track"] for p in te_reference["changed_target_pairs"]]

    focus = changed_orig + target_tracks or list(original_tracks)
    f_lon = np.concatenate([_tevis_track_lonlat(t)[0] for t in focus]) if focus else np.array([])
    f_lat = np.concatenate([_tevis_track_lonlat(t)[1] for t in focus]) if focus else np.array([])
    center = _tevis_center_lon(f_lon)
    extent = _tevis_extent(f_lon, f_lat, center)
    proj = ccrs.PlateCarree(central_longitude=center)

    def step_index(dt):
        return int(round((dt - forecast_start).total_seconds() / 21600.0))

    time_footer_lines = _tevis_prediction_time_footer(pred_tracks, forecast_start)
    footer_line_count = max(1, len(time_footer_lines))
    footer_height_in = 0.22 * footer_line_count + 0.18
    fig_height = 6.0 + footer_height_in
    footer_fraction = min(0.42, max(0.10, footer_height_in / fig_height))

    fig, ax = plt.subplots(
        figsize=(9.0, fig_height),
        dpi=TE_VIS_DPI,
        subplot_kw={"projection": proj},
    )
    ax.add_feature(cfeature.LAND, facecolor="#fffdf0")
    ax.add_feature(cfeature.OCEAN, facecolor="#f4fbff")
    ax.coastlines(linewidth=0.6)

    def draw(tracks, color, ls, marker, lw, ms, label, annotate=False, z=4,
             only_in_view=False):
        first = True
        for tr in tracks:
            lon, lat = _tevis_track_lonlat(tr)
            if len(lon) == 0:
                continue
            if only_in_view and not _tevis_in_extent(lon, lat, center, extent).any():
                continue
            ax.plot(lon, lat, color=color, linestyle=ls, marker=marker,
                    markersize=ms, linewidth=lw, transform=ccrs.Geodetic(),
                    label=label if first else None, zorder=z)
            first = False
            if annotate:
                for node in tr["nodes"][::2]:
                    ax.text(node["lon"], node["lat"], str(step_index(node["datetime"]) + 1),
                            fontsize=5.5, color=color, transform=ccrs.PlateCarree(),
                            ha="left", va="bottom", zorder=z + 1)

    draw(other_orig, "#9ecae1", "-", None, 0.9, 0, "Unchanged original", z=2,
         only_in_view=True)
    if candidates:
        c_lon = np.asarray([c["lon"] for c in candidates], dtype=float)
        c_lat = np.asarray([c["lat"] for c in candidates], dtype=float)
        keep = _tevis_in_extent(c_lon, c_lat, center, extent)
        if keep.any():
            ax.scatter(c_lon[keep], c_lat[keep], s=7, c="#7f7f7f", alpha=0.55,
                       transform=ccrs.PlateCarree(), label="DetectNodes candidates",
                       zorder=3)
    draw(changed_orig, "#1f4e99", "-", "o", 1.4, 3.0, "Original (to change)")
    draw(target_tracks, "#b2182b", "--", "^", 1.6, 3.4, "Target", annotate=True, z=5)
    draw(pred_tracks, "#111111", "-", "s", 1.4, 2.8, "TE prediction", annotate=True,
         z=6, only_in_view=True)

    if extent is not None:
        ax.set_extent(extent, crs=proj)
    else:
        ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.45, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    r = result
    info = (
        f"DR_changed {float(r.get('DR_changed', float('nan'))):.2f}, "
        f"mean target overlap {float(r.get('mean_target_overlap', float('nan'))):.2f}\n"
        f"remove success {float(r.get('remove_success', float('nan'))):.2f}, "
        f"mean original overlap {float(r.get('mean_removed_original_overlap', float('nan'))):.2f}\n"
        f"joint TE {float(r.get('te_joint_score', float('nan'))):.3f} "
        f"(gen {float(r.get('te_generation_score', float('nan'))):.3f}, "
        f"remove {float(r.get('te_removal_score', float('nan'))):.3f})\n"
        f"tracks {r.get('n_pred_tracks', 0)}, candidates {r.get('n_detected_candidates', 0)}"
    )
    ax.text(0.01, 0.01, info, transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), zorder=10)
    ax.set_title(title, fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout(rect=(0.0, footer_fraction, 1.0, 0.985))
    fig.text(
        0.015,
        0.015,
        "\n".join(time_footer_lines),
        ha="left",
        va="bottom",
        fontsize=6.0,
        family="monospace",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def _tevis_focus_geometry(te_reference):
    changed_sources = {int(r["source_index"]) for r in te_reference["removed_original_tracks"]}
    changed_sources |= {int(p["source_index"]) for p in te_reference["changed_target_pairs"]}
    original_tracks = te_reference["original_tracks"]
    changed_orig = [original_tracks[i] for i in sorted(changed_sources) if 0 <= i < len(original_tracks)]
    target_tracks = [p["target_track"] for p in te_reference["changed_target_pairs"]]
    focus = changed_orig + target_tracks or list(original_tracks)
    if not focus:
        return 0.0, None
    f_lon = np.concatenate([_tevis_track_lonlat(t)[0] for t in focus])
    f_lat = np.concatenate([_tevis_track_lonlat(t)[1] for t in focus])
    center = _tevis_center_lon(f_lon)
    return center, _tevis_extent(f_lon, f_lat, center)


def _tevis_crop_field(field_hw, center, extent):
    field_hw = np.asarray(field_hw, dtype=float)
    rel_lon = _tevis_rel_lon(LON_DEG, center)
    order = np.argsort(rel_lon)
    x = rel_lon[order]
    y = LAT_DEG
    z = field_hw[:, order]
    if extent is not None:
        w, e, s, n = extent
        keep_x = (x >= w) & (x <= e)
        keep_y = (y >= s) & (y <= n)
        if keep_x.any() and keep_y.any():
            x = x[keep_x]
            y = y[keep_y]
            z = z[np.ix_(keep_y, keep_x)]
    return z, x, y


def _unique_best_dir(out_dir, best_type, step):
    """Create a never-overwritten checkpoint directory, including after an interrupted rerun."""
    epoch = int(step) + 1
    name = "epoch_0000_initial" if step < 0 else f"epoch_{epoch:04d}_step_{step:04d}"
    base = Path(out_dir) / "best_history" / best_type
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / name
    suffix = 1
    while candidate.exists():
        candidate = base / f"{name}_repeat_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _append_best_manifest(out_dir, entry):
    path = Path(out_dir) / "best_manifest.csv"
    columns = ["stage_index", "stage_kind", "guide_delta", "input_epsilon", "type", "epoch", "step", "loss", "primary_best_metric", "secondary_best_metric", "directory"]
    row = {k: entry.get(k, np.nan) for k in columns}
    import csv
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _probability_from_logits(logits):
    x = np.asarray(jax.device_get(ensure_two_class_logits(logits)), dtype=np.float32).copy()
    x -= np.max(x, axis=1, keepdims=True)
    np.exp(x, out=x)
    return (x[:, 1] / np.maximum(x.sum(axis=1), 1e-20)).astype(np.float32)


def plot_surrogate_probability(probability, te_reference, out_path, title, pred_tracks=None):
    """Fixed [0,1] TC probability heatmap, annotated with target/original/TE tracks."""
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[1:] != (GRID_H, GRID_W):
        raise ValueError(f"Expected surrogate TC probabilities [T,{GRID_H},{GRID_W}], got {probability.shape}")
    center, extent = _tevis_focus_geometry(te_reference)
    prob_max = probability.max(axis=0)
    prob_mean = probability.mean(axis=0)
    changed = {int(p["source_index"]) for p in te_reference["changed_target_pairs"]}
    changed.update(int(p["source_index"]) for p in te_reference["removed_original_tracks"])
    orig = [te_reference["original_tracks"][i] for i in sorted(changed) if i < len(te_reference["original_tracks"])]
    if not orig:
        orig = te_reference["original_tracks"]
    targets = [x["target_track"] for x in te_reference["changed_target_pairs"]]
    figs, axes = plt.subplots(1, 2, figsize=(13, 4.7), dpi=TE_VIS_DPI, squeeze=False)
    for ax, data, caption in zip(axes[0], (prob_max, prob_mean), ("Max over forecast steps", "Mean over forecast steps")):
        z, lon, lat = _tevis_crop_field(data, center, extent)
        im = ax.imshow(z, origin="lower", aspect="auto", extent=[lon[0]-.5, lon[-1]+.5, lat[0]-.5, lat[-1]+.5], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        for tracks, color, line_style, label in [(orig, "#64c9ff", "-", "Changed original"), (targets, "#54ff8c", "--", "Target"), (pred_tracks or [], "white", "-", "TempestExtremes")]:
            first = True
            for tr in tracks:
                tlon, tlat = _tevis_track_lonlat(tr)
                if not tlon.size:
                    continue
                ax.plot(_tevis_rel_lon(tlon, center), tlat, color=color, linestyle=line_style, linewidth=1.3, marker="o", markersize=1.7, label=label if first else None)
                first = False
        ax.set_title(caption)
        ax.set_xlabel(f"Longitude relative to {center:.1f}°E")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.15)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.75)
        fig_cb = figs.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        fig_cb.set_label("Surrogate TC probability [0, 1]", fontsize=8)
    figs.suptitle(title, fontsize=10)
    figs.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figs.savefig(out_path, bbox_inches="tight")
    plt.close(figs)


def _save_best_probability(logits, te_reference, target_dir, title, pred_tracks=None, full_array=False):
    if not BEST_SAVE_PROBABILITY:
        return
    probability = _probability_from_logits(logits)
    if full_array:
        np.save(Path(target_dir) / "surrogate_probability.npy", probability)
    plot_surrogate_probability(probability, te_reference, Path(target_dir) / "surrogate_probability_heatmap.png", title, pred_tracks)


def _downstream_comparison_maps(ctx, s_adv_std, standardized=False):
    clean = np.asarray(ctx["clean_s_np"], dtype=np.float32)
    adv = np.asarray(s_adv_std, dtype=np.float32)
    guide = clean + np.asarray(ctx["guide_np"], dtype=np.float32)
    if not standardized:
        std = DOWNSTREAM_STD.numpy()[None, :, None, None]
        clean, adv, guide = clean * std, adv * std, guide * std
    return [
        ("Mean |Guidance − Clean|", np.abs(guide - clean).mean(axis=0)),
        ("Mean |Adversarial downstream − Clean|", np.abs(adv - clean).mean(axis=0)),
        ("Mean |Adversarial downstream − Guidance|", np.abs(adv - guide).mean(axis=0)),
    ]


def _downstream_color_scales(ctx, s_adv_std, te_reference, standardized=False):
    center, extent = _tevis_focus_geometry(te_reference)
    maps = _downstream_comparison_maps(ctx, s_adv_std, standardized=standardized)
    scales = {}
    for ch in (CH_WIND, CH_THICK, CH_MSLP):
        q = []
        for _, values in maps:
            crop, _, _ = _tevis_crop_field(values[ch], center, extent)
            finite = crop[np.isfinite(crop)]
            if finite.size:
                q.append(float(np.percentile(finite, 99.5)))
        scales[ch] = max(max(q, default=0.0), 1e-12)
    return scales


def _raw_downstream_to_standardized(raw):
    raw = np.asarray(raw, dtype=np.float32)
    return (raw - DOWNSTREAM_MEAN.numpy()[None, :, None, None]) / DOWNSTREAM_STD.numpy()[None, :, None, None]


def _load_saved_tensor_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _render_stage_colorbars(out_dir, ctx, te_reference, objective_dir, te_dir):
    """Use a shared raw and standardized scale per variable across all bests in a stage."""
    import shutil
    if not DOWNSTREAM_STAGE_COLORBAR or not TE_SAVE_ATTACK_DIFF_VIS:
        return {}
    sources = []
    for best_type in ("te", "objective"):
        for history_dir in sorted((Path(out_dir) / "best_history" / best_type).glob("epoch_*")):
            tensor = history_dir / "adversarial_downstream_input.pt"
            if tensor.exists():
                sources.append((history_dir, tensor,
                                f"{best_type}_best_adversarial_downstream_vs_guidance"))
    for bundle in (objective_dir, te_dir):
        tensor = Path(bundle) / "final_downstream_input.pt"
        if tensor.exists():
            sources.append((Path(bundle), tensor, "adversarial_downstream_vs_guidance"))
    scale_table = {}
    for standardized, key in ((False, "raw"), (True, "standardized")):
        scales = {ch: 1e-12 for ch in (CH_WIND, CH_THICK, CH_MSLP)}
        for _, tensor, _ in sources:
            raw = _load_saved_tensor_cpu(tensor).numpy()
            this = _downstream_color_scales(ctx, _raw_downstream_to_standardized(raw), te_reference,
                                            standardized=standardized)
            for ch, value in this.items():
                scales[ch] = max(scales[ch], value)
        for directory, tensor, stem in sources:
            raw = _load_saved_tensor_cpu(tensor).numpy()
            suffix = "_std.png" if standardized else ".png"
            updated_plot = directory / f"{stem}{suffix}"
            plot_downstream_guidance_difference(ctx, _raw_downstream_to_standardized(raw), te_reference,
                                                updated_plot,
                                                f"{directory.name} | stage-wide fixed scale", scales,
                                                standardized=standardized)
            if directory.parent.name in ("objective", "te"):
                shutil.copy2(updated_plot, directory / f"adversarial_downstream_vs_guidance{suffix}")
        scale_table[key] = {"wind": scales[CH_WIND], "thickness": scales[CH_THICK], "mslp": scales[CH_MSLP]}
    with open(Path(out_dir) / "downstream_colorbar_scale.json", "w") as f:
        json.dump({**scale_table, "wind_ms": scale_table["raw"]["wind"],
                   "thickness_m2s2": scale_table["raw"]["thickness"],
                   "mslp_pa": scale_table["raw"]["mslp"], "method": "stage-wide max of cropped 99.5th percentile; shared across 3 rows; vmin=0"}, f, indent=2)
    return {CH_WIND: scale_table["raw"]["wind"], CH_THICK: scale_table["raw"]["thickness"], CH_MSLP: scale_table["raw"]["mslp"]}


def _tevis_input_delta_rows(ctx, delta, standardized=False):
    rows = []
    for var in PERTURB_INPUT_VARS:
        da = ctx["base_ds"][var]
        raw_delta = np.asarray(delta[var] if standardized else delta[var] * ctx["std_arrays"][var], dtype=np.float32)
        coords = {d: da.coords[d] for d in da.dims if d in da.coords}
        arr = xr.DataArray(raw_delta, dims=da.dims, coords=coords)
        if "batch" in arr.dims:
            arr = arr.isel(batch=0)
        levels = [None]
        if "level" in arr.dims:
            levels = list(PERTURB_LEVELS.get(var, tuple(float(x) for x in arr["level"].values)))
        for level in levels:
            x = arr
            label = var
            if level is not None:
                x = x.sel(level=float(level), method="nearest")
                label = f"{var} @ {float(level):g} hPa"
            if "time" not in x.dims:
                x = x.expand_dims(time=[0])
            x = x.transpose("time", "lat", "lon")
            units = "Standardized units (σ)" if standardized else str(da.attrs.get("units", "raw units"))
            rows.append((label, units, np.asarray(x.values, dtype=np.float32)))
    return rows


def plot_input_delta_check(ctx, delta, te_reference, out_path, title, standardized=False):
    """Visualize X_adv - X_original in raw or GraphCast-standardized units."""
    rows = _tevis_input_delta_rows(ctx, delta, standardized=standardized)
    if not rows:
        return
    n_frames = max(x.shape[0] for _, _, x in rows)
    center, extent = _tevis_focus_geometry(te_reference)
    fig, axes = plt.subplots(
        len(rows), n_frames,
        figsize=(5.2 * n_frames, max(3.0, 2.45 * len(rows))),
        dpi=TE_VIS_DPI,
        squeeze=False,
    )
    for i, (label, units, frames) in enumerate(rows):
        cropped = []
        for t in range(frames.shape[0]):
            z, xlon, ylat = _tevis_crop_field(frames[t], center, extent)
            cropped.append((z, xlon, ylat))
        values = np.concatenate([np.abs(z).ravel() for z, _, _ in cropped]) if cropped else np.array([0.0])
        vmax = float(np.nanpercentile(values, 99.5)) if np.isfinite(values).any() else 0.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1e-12
        for t in range(n_frames):
            ax = axes[i, t]
            if t >= len(cropped):
                ax.axis("off")
                continue
            z, xlon, ylat = cropped[t]
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[float(xlon[0]), float(xlon[-1]), float(ylat[0]), float(ylat[-1])],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"{label} | input frame {t}", fontsize=7)
            ax.set_ylabel("Latitude", fontsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel(f"Longitude relative to {center:.1f}°E", fontsize=7)
            ax.tick_params(labelsize=6)
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=5)
            cb.set_label(units, fontsize=5)
    fig.suptitle(title + (" | Standardized ΔX" if standardized else " | Adversarial X − Original X"), fontsize=10, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_downstream_guidance_difference(ctx, s_adv_std, te_reference, out_path, title, fixed_scales=None, standardized=False):
    """Compare the current adversarial downstream state with the absolute guide target."""
    maps = _downstream_comparison_maps(ctx, s_adv_std, standardized=standardized)
    if fixed_scales is None:
        fixed_scales = _downstream_color_scales(ctx, s_adv_std, te_reference, standardized=standardized)
    channels = [
        (CH_WIND, "10 m wind speed"),
        (CH_THICK, "Thickness"),
        (CH_MSLP, "MSLP"),
    ]
    center, extent = _tevis_focus_geometry(te_reference)
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 10.5), dpi=TE_VIS_DPI, squeeze=False)
    for r, (row_name, values) in enumerate(maps):
        for c, (ch, ch_name) in enumerate(channels):
            z, xlon, ylat = _tevis_crop_field(values[ch], center, extent)
            vmax = float(fixed_scales[ch])
            ax = axes[r, c]
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[float(xlon[0]), float(xlon[-1]), float(ylat[0]), float(ylat[-1])],
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"{row_name}\n{ch_name}", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=7)
            if r == 2:
                ax.set_xlabel(f"Longitude relative to {center:.1f}°E", fontsize=7)
            ax.tick_params(labelsize=6)
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=5)
    fig.suptitle(title + (" | Standardized downstream differences (σ)" if standardized else " | Downstream guidance comparison"), fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _te_save_state_artifacts(
    work_dir,
    sample,
    ctx,
    delta,
    downstream_raw,
    gc_output,
    result,
    pred_tracks,
    forecast_start,
    artifact_name=None,
    artifact_dir=None,
):
    """Save the exact attack state associated with one retained TE check.

    Files are written under the immutable history event directory:
      <stage out_dir>/best_history/te/epoch_XXXX_step_YYYY/

    The bundle contains:
      adversarial_upstream_input.nc
          X_adv actually fed to GraphCast at this check.
      adversarial_upstream_delta_std.npz
          Standardized perturbation delta X for exact reconstruction/debugging.
      adversarial_upstream_output.nc
          Full GraphCast forecast f(X_adv).
      adversarial_downstream_input.pt
          Raw [T,4,H,W] TempestExtremes-channel tensor.
      adversarial_downstream_input_te.nc
          The same downstream tensor converted to the NetCDF format actually
          passed to TempestExtremes.
      te_metrics.json
          TE metrics and save-decision metadata.
      adversarial_downstream_trajectory_times.txt
          Human-readable time, forecast-step, longitude, and latitude for every
          point of every stitched end-to-end adversarial downstream trajectory.
    """
    name = str(artifact_name or Path(work_dir).name)
    state_dir = Path(artifact_dir) if artifact_dir is not None else Path(work_dir).parent.parent / "te_saved_states" / name
    state_dir.mkdir(parents=True, exist_ok=True)

    adv_ds = build_adv_ds(ctx, delta)
    adv_ds.to_netcdf(state_dir / "adversarial_upstream_input.nc")

    np.savez_compressed(
        state_dir / "adversarial_upstream_delta_std.npz",
        **{k: np.asarray(jax.device_get(v), dtype=np.float32) for k, v in delta.items()},
    )

    if gc_output is None:
        raise ValueError("gc_output is required when te_save_state_artifacts=true")
    save_gc_output(
        gc_output,
        sample,
        state_dir / "adversarial_upstream_output.nc",
    )

    if isinstance(downstream_raw, torch.Tensor):
        downstream_cpu = downstream_raw.detach().cpu().float()
    else:
        downstream_cpu = torch.from_numpy(
            np.asarray(downstream_raw, dtype=np.float32).copy()
        ).float()

    torch.save(
        downstream_cpu,
        state_dir / "adversarial_downstream_input.pt",
    )
    _te_downstream_to_nc(
        downstream_cpu,
        sample,
        state_dir / "adversarial_downstream_input_te.nc",
    )

    trajectory_time_lines = _end2end_track_time_lines(
        pred_tracks, forecast_start, name
    )
    (state_dir / "adversarial_downstream_trajectory_times.txt").write_text(
        "\n".join(trajectory_time_lines) + "\n"
    )

    metrics = {
        "check": name,
        "sample": sample,
        "target_mode": ctx["te_reference"].get("target_mode"),
        "guide_delta": float(DOWNSTREAM_EPSILON),
        "end2end_epsilon": float(END2END_EPSILON),
        "selected_step": int(result.get("step", -1)),
        "selected_epoch": int(result.get("step", -1)) + 1,
        "DR_changed": float(result.get("DR_changed", np.nan)),
        "mean_target_overlap": float(result.get("mean_target_overlap", np.nan)),
        "remove_success": float(result.get("remove_success", np.nan)),
        "mean_removed_original_overlap": float(result.get("mean_removed_original_overlap", np.nan)),
        "te_generation_score": float(result.get("te_generation_score", np.nan)),
        "te_removal_score": float(result.get("te_removal_score", np.nan)),
        "te_joint_score": float(result.get("te_joint_score", np.nan)),
        "te_joint_success": float(result.get("te_joint_success", np.nan)),
        "target_success": float(result.get("target_success", np.nan)),
        "num_pred_tracks": int(result.get("n_pred_tracks", 0)),
        "num_detected_candidates": int(result.get("n_detected_candidates", 0)),
        "is_new_te_best": bool(result.get("is_new_te_best", False)),
        "metric_definition": "direct downstream track overlap; exact forecast time; no changed-node disambiguation",
    }
    with open(state_dir / "te_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def _te_save_check_artifacts(
    work_dir,
    nodes,
    tracks,
    te_reference,
    pred,
    candidates,
    result,
    forecast_start,
    sample,
    ctx=None,
    delta=None,
    s_adv_std=None,
    downstream_raw=None,
    gc_output=None,
    artifact_name=None,
    artifact_dir=None,
    logits=None,
):
    """Persist artifacts for a TE check selected by the save policy.

    Full state and visuals are stored in the immutable history event directory.
    Legacy rolling te_vis and te_saved_states names are compatibility aliases.
    """
    save_vis = TE_SAVE_VIS
    keep_text = TE_KEEP_TEXT
    save_attack_diff = (
        TE_SAVE_ATTACK_DIFF_VIS
        and ctx is not None
        and delta is not None
        and s_adv_std is not None
    )
    save_state = (
        (TE_SAVE_STATE_ARTIFACTS or BEST_TE_STATE)
        and ctx is not None
        and delta is not None
        and downstream_raw is not None
        and gc_output is not None
    )
    if not (save_vis or keep_text or save_attack_diff or save_state):
        return

    name = str(artifact_name or Path(work_dir).name)
    vis_dir = Path(artifact_dir) if artifact_dir is not None else Path(work_dir).parent.parent / "te_vis"

    try:
        if save_vis or keep_text or save_attack_diff:
            vis_dir.mkdir(parents=True, exist_ok=True)
        if keep_text:
            if tracks.exists():
                shutil.copy2(tracks, vis_dir / f"{name}_tracks.txt")
            if nodes.exists():
                shutil.copy2(nodes, vis_dir / f"{name}_nodes.txt")
        if save_vis:
            title = (
                f"{sample} | {te_reference.get('target_mode')} | "
                f"eps={float(END2END_EPSILON):g} | {name}"
            )
            plot_te_check(
                te_reference,
                pred,
                candidates,
                result,
                forecast_start,
                vis_dir / f"{name}.png",
                title,
            )
        if save_attack_diff:
            title = (
                f"{sample} | {te_reference.get('target_mode')} | "
                f"eps={float(END2END_EPSILON):g} | {name}"
            )
            plot_input_delta_check(
                ctx,
                delta,
                te_reference,
                vis_dir / f"{name}_adversarial_x_minus_original_x.png",
                title,
            )
            plot_downstream_guidance_difference(
                ctx, s_adv_std, te_reference,
                vis_dir / f"{name}_adversarial_downstream_vs_guidance.png", title,
            )
            plot_input_delta_check(
                ctx, delta, te_reference,
                vis_dir / f"{name}_adversarial_x_minus_original_x_std.png", title, standardized=True,
            )
            plot_downstream_guidance_difference(
                ctx, s_adv_std, te_reference,
                vis_dir / f"{name}_adversarial_downstream_vs_guidance_std.png", title,
                standardized=True,
            )
    except Exception as exc:
        print(f"[te_vis] warning: could not save visualization/text artifacts for {name}: {exc!r}")
        plt.close("all")
        if name == "objective_best":
            # Objective Best history is required to be complete, not silently
            # downgraded to a perturbation-only checkpoint.
            raise RuntimeError("Objective Best history visualization failed") from exc

    if save_state:
        try:
            _te_save_state_artifacts(
                work_dir=work_dir,
                sample=sample,
                ctx=ctx,
                delta=delta,
                downstream_raw=downstream_raw,
                gc_output=gc_output,
                result=result,
                pred_tracks=pred,
                forecast_start=forecast_start,
                artifact_name=artifact_name,
                artifact_dir=artifact_dir,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to save mandatory TE-best state for {name}") from exc


def _update_legacy_te_artifacts(out_dir, event_dir):
    """Maintain old te_vis/te_best* names as small copies, and a full-state relative symlink."""
    out_dir, event_dir = Path(out_dir), Path(event_dir)
    vis_dir = out_dir / "te_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for suffix, name in [(".png", "te_best.png"), ("_tracks.txt", "te_best_tracks.txt"),
                         ("_nodes.txt", "te_best_nodes.txt"),
                         ("_adversarial_x_minus_original_x.png", "te_best_adversarial_x_minus_original_x.png"),
                         ("_adversarial_downstream_vs_guidance.png", "te_best_adversarial_downstream_vs_guidance.png"),
                         ("_adversarial_x_minus_original_x_std.png", "te_best_adversarial_x_minus_original_x_std.png"),
                         ("_adversarial_downstream_vs_guidance_std.png", "te_best_adversarial_downstream_vs_guidance_std.png")]:
        src = event_dir / f"te_best{suffix}"
        if src.exists():
            shutil.copy2(src, vis_dir / name)
    state_root = out_dir / "te_saved_states"
    state_root.mkdir(exist_ok=True)
    link = state_root / "te_best"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        archive = _unique_best_dir(out_dir, "te", -1)
        archive.rmdir()
        shutil.move(str(link), str(archive))
    link.symlink_to(os.path.relpath(event_dir, state_root), target_is_directory=True)


def _direct_track_overlap(target_track, pred_track, tolerance_deg=None):
    """Paper/downstream-style exact-time track overlap with no source-track disambiguation."""
    tolerance_deg = TE_TOL_DEG if tolerance_deg is None else float(tolerance_deg)
    pred_by_time = {}
    for node in pred_track.get("nodes", []):
        pred_by_time.setdefault(node["datetime"], []).append(
            (float(node["lon"]) % 360.0, float(node["lat"]))
        )
    hits = 0
    hit_indices = []
    for i, node in enumerate(target_track.get("nodes", [])):
        pts = pred_by_time.get(node["datetime"], [])
        if not pts:
            continue
        d = min(
            _haversine_deg(float(node["lon"]) % 360.0, float(node["lat"]), lo, la)
            for lo, la in pts
        )
        if d < tolerance_deg:
            hits += 1
            hit_indices.append(i)
    total = max(1, len(target_track.get("nodes", [])))
    return hits / total, hits, hit_indices


def _direct_best_track_match(reference_track, pred_tracks):
    matches = [_direct_track_overlap(reference_track, p) for p in pred_tracks]
    if not matches:
        return 0.0, 0, []
    return max(matches, key=lambda x: (x[1], x[0]))


def score_tracks_direct(pred_tracks, te_reference, candidates=None):
    """Evaluate exact-time downstream TE tracks with both continuous and threshold metrics.

    Relocation target Best uses TE_GENERATION_WEIGHT * target overlap plus
    (1 - TE_GENERATION_WEIGHT) * (1 - original overlap), tie-broken by
    the corresponding target/removal success fractions. All component metrics
    remain available independently for comparisons to previous experiments.

    Remove-only Best keeps removal success then negative original overlap.
    """
    target_tracks = list(te_reference.get("target_tracks", []))
    original_tracks = list(te_reference.get("original_tracks", []))
    changed_indices = [int(x) for x in te_reference.get("changed_indices", [])]
    source_indices = [int(x) for x in te_reference.get("source_indices", range(len(target_tracks)))]
    removed_indices = [int(x) for x in te_reference.get("removed_indices", [])] or changed_indices
    target_mode = te_reference.get("target_mode")

    target_scores = []
    target_hits = []
    for tr in target_tracks:
        score, hits, hit_idx = _direct_best_track_match(tr, pred_tracks)
        target_scores.append(score)
        target_hits.append((hits, hit_idx))

    changed_target_positions = [
        i for i, src in enumerate(source_indices)
        if src in changed_indices and 0 <= i < len(target_tracks)
    ]
    changed_scores = [target_scores[i] for i in changed_target_positions]
    dr_changed = (
        float(np.mean([x >= 0.5 for x in changed_scores]))
        if changed_scores else np.nan
    )
    mean_target_overlap = float(np.mean(changed_scores)) if changed_scores else np.nan
    target_hit_nodes = int(sum(target_hits[i][0] for i in changed_target_positions))
    changed_target_nodes = int(sum(len(target_tracks[i].get("nodes", [])) for i in changed_target_positions))

    removed_scores = []
    removed_hits = []
    for idx in removed_indices:
        if 0 <= idx < len(original_tracks):
            score, hits, hit_idx = _direct_best_track_match(original_tracks[idx], pred_tracks)
            removed_scores.append(score)
            removed_hits.append((hits, hit_idx, idx))
    mean_removed = float(np.mean(removed_scores)) if removed_scores else np.nan
    remove_success = (
        float(np.mean([x < 0.5 for x in removed_scores]))
        if removed_scores else np.nan
    )
    residual_original_nodes = int(sum(x[0] for x in removed_hits))
    original_reference_nodes = int(sum(len(original_tracks[idx].get("nodes", [])) for _, _, idx in removed_hits))

    false_alarm_tracks = 0
    for pred in pred_tracks:
        desired = [_direct_track_overlap(t, pred) for t in target_tracks]
        if not desired:
            false_alarm_tracks += 1
            continue
        best = max(desired, key=lambda x: (x[1], x[0]))
        if best[1] == 0 or (best[1] == 1 and best[2] and best[2][0] == 0):
            false_alarm_tracks += 1
    far = false_alarm_tracks / len(pred_tracks) if pred_tracks else 0.0

    # TE is an evaluation metric only: do not differentiate or optimize this score.
    # In relocation experiments, reward BOTH generating the changed target TC
    # and removing its original TC, using continuous overlap (rather than only
    # the 50%-threshold DR_changed indicator). The original metrics remain intact.
    generation_score = float(np.clip(mean_target_overlap, 0, 1)) if np.isfinite(mean_target_overlap) else 0.0
    removal_score = float(1.0 - np.clip(mean_removed, 0, 1)) if np.isfinite(mean_removed) else 0.0
    joint_score = (TE_GENERATION_WEIGHT * generation_score
                   + (1.0 - TE_GENERATION_WEIGHT) * removal_score)
    joint_success = (
        TE_GENERATION_WEIGHT * (float(dr_changed) if np.isfinite(dr_changed) else 0.0)
        + (1.0 - TE_GENERATION_WEIGHT) * (float(remove_success) if np.isfinite(remove_success) else 0.0)
    )
    if target_mode == "remove_changed_original":
        target_success = remove_success
        primary = remove_success
        secondary = -mean_removed if np.isfinite(mean_removed) else -np.inf
    else:
        target_success = dr_changed
        primary = joint_score
        secondary = joint_success

    return {
        "target_mode": target_mode,
        "DR_changed": float(dr_changed) if np.isfinite(dr_changed) else np.nan,
        "mean_target_overlap": float(mean_target_overlap) if np.isfinite(mean_target_overlap) else np.nan,
        "remove_success": float(remove_success) if np.isfinite(remove_success) else np.nan,
        "mean_removed_original_overlap": float(mean_removed) if np.isfinite(mean_removed) else np.nan,
        "te_generation_score": generation_score,
        "te_removal_score": removal_score,
        "te_joint_score": float(joint_score),
        "te_joint_success": float(joint_success),
        "te_generation_weight": float(TE_GENERATION_WEIGHT),
        "target_success": float(target_success) if np.isfinite(target_success) else np.nan,
        "primary_best_metric": float(primary) if np.isfinite(primary) else -np.inf,
        "secondary_best_metric": float(secondary) if np.isfinite(secondary) else -np.inf,
        "target_hit_nodes": int(target_hit_nodes),
        "changed_target_nodes": int(changed_target_nodes),
        "residual_original_nodes": int(residual_original_nodes),
        "original_reference_nodes": int(original_reference_nodes),
        "residual_original_ratio": float(mean_removed) if np.isfinite(mean_removed) else 0.0,
        "fa_nodes": int(false_alarm_tracks),
        "fa_ratio": float(far),
    }


def te_result_better(candidate, best, tol=1e-12):
    """Strictly improve the configured TE score (joint generation/removal for relocation)."""
    if best is None:
        return True
    cp = float(candidate.get("primary_best_metric", -np.inf))
    bp = float(best.get("primary_best_metric", -np.inf))
    if cp > bp + tol:
        return True
    if cp < bp - tol:
        return False
    cs = float(candidate.get("secondary_best_metric", -np.inf))
    bs = float(best.get("secondary_best_metric", -np.inf))
    return cs > bs + tol


def te_check(
    downstream_raw,
    sample,
    te_reference,
    work_dir,
    ctx=None,
    delta=None,
    s_adv_std=None,
    gc_output=None,
    best_result=None,
    step=-1,
    logits=None,
    stage_meta=None,
    objective_history_dir=None,
):
    """Run real TE once; archive each new TE best and any matching objective best.

    objective_history_dir is an independent, immutable Objective-Best history
    checkpoint. It is populated even if its TE score does not improve TE Best.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    te_nc = work_dir / "adv_tempestextremes.nc"
    nodes = work_dir / "detected_nodes.txt"
    tracks = work_dir / "cyclone_tracks.txt"

    _te_downstream_to_nc(downstream_raw, sample, te_nc)
    _te_run(te_nc, nodes, tracks)
    te_nc.unlink(missing_ok=True)

    pred = read_track_file(tracks) if tracks.exists() and tracks.stat().st_size else []
    start = datetime.strptime(sample, "%Y-%m-%dT%H%M") + timedelta(hours=6)
    candidates = _read_detected_nodes(nodes, start)
    result = score_tracks_direct(pred, te_reference, candidates=candidates)
    result["n_detected_candidates"] = int(len(candidates))
    result["n_pred_tracks"] = int(len(pred))
    result["step"] = int(step)

    _print_end2end_track_times(pred, start, work_dir.name)

    is_new_best = te_result_better(result, best_result)
    result["is_new_te_best"] = bool(is_new_best)
    result["artifacts_saved"] = bool(is_new_best)

    if is_new_best:
        out_dir = work_dir.parent.parent
        event_dir = _unique_best_dir(out_dir, "te", step)
        result["history_dir"] = str(event_dir)
        try:
            delta_host = {k: np.asarray(jax.device_get(v), dtype=np.float32) for k, v in delta.items()}
            np.savez_compressed(event_dir / "best_delta_std.npz", **delta_host)
            with open(event_dir / "best_event.json", "w") as f:
                json.dump({"type": "te", "epoch": int(step) + 1, "step": int(step),
                           "metric_primary": float(result["primary_best_metric"]),
                           "metric_secondary": float(result["secondary_best_metric"]),
                           "te_generation_score": float(result.get("te_generation_score", np.nan)),
                           "te_removal_score": float(result.get("te_removal_score", np.nan)),
                           "te_joint_score": float(result.get("te_joint_score", np.nan)),
                           "te_joint_success": float(result.get("te_joint_success", np.nan)),
                           "stage": stage_meta}, f, indent=2)
            _te_save_check_artifacts(work_dir, nodes, tracks, te_reference, pred, candidates,
                                     result, start, sample, ctx=ctx, delta=delta,
                                     s_adv_std=s_adv_std, downstream_raw=downstream_raw,
                                     gc_output=gc_output, artifact_name="te_best",
                                     artifact_dir=event_dir, logits=logits)
            if logits is None:
                logits = surrogate_bridge(s_adv_std)
            _save_best_probability(logits, te_reference, event_dir,
                                   f"TE best | epoch {step + 1} | {sample} | {work_dir.parent.parent.name}",
                                   pred_tracks=pred, full_array=True)
            _update_legacy_te_artifacts(out_dir, event_dir)
            _append_best_manifest(out_dir, {**(stage_meta or {}), "type": "te", "epoch": int(step) + 1,
                                            "step": int(step), "primary_best_metric": result["primary_best_metric"],
                                            "secondary_best_metric": result["secondary_best_metric"],
                                            "directory": str(event_dir)})
        except Exception:
            print(f"[te_best] Failed to persist new best at epoch {step + 1}: {event_dir}")
            raise
        print(f"[te_best] {work_dir.name}: saved epoch={step + 1}; "
              f"primary={result['primary_best_metric']:.3f}, secondary={result['secondary_best_metric']:.3f}")
    else:
        print(
            f"[te_best] {work_dir.name}: no improvement; "
            f"primary={result['primary_best_metric']:.3f}, secondary={result['secondary_best_metric']:.3f}"
        )

    if objective_history_dir is not None:
        # Unlike the old probability-only checkpoint, EVERY Objective Best gets
        # an exact-epoch real-TE track map, all 4 difference plots, full forecast,
        # downstream input, probability array and TE diagnostics. The separate
        # TE-best selection remains unaffected by the objective-loss decision.
        objective_history_dir = Path(objective_history_dir)
        if ctx is None or delta is None or s_adv_std is None or gc_output is None:
            raise ValueError("Objective Best full archive requires the exact-epoch forecast state")
        _te_save_check_artifacts(
            work_dir, nodes, tracks, te_reference, pred, candidates, result, start,
            sample, ctx=ctx, delta=delta, s_adv_std=s_adv_std,
            downstream_raw=downstream_raw, gc_output=gc_output,
            artifact_name="objective_best", artifact_dir=objective_history_dir,
            logits=logits,
        )
        if logits is None:
            logits = surrogate_bridge(s_adv_std)
        _save_best_probability(
            logits, te_reference, objective_history_dir,
            f"Objective best | epoch {step + 1} | {sample} | {te_reference.get('target_mode')}",
            pred_tracks=pred, full_array=True,
        )
        # Consistent convenient names, in addition to the historical TE-compatible
        # objective_best_* filenames saved by _te_save_check_artifacts.
        for source, alias in (
            ("objective_best.png", "te_track_map.png"),
            ("objective_best_tracks.txt", "te_tracks.txt"),
            ("objective_best_nodes.txt", "te_nodes.txt"),
            ("objective_best_adversarial_x_minus_original_x.png", "adversarial_x_minus_original_x.png"),
            ("objective_best_adversarial_x_minus_original_x_std.png", "adversarial_x_minus_original_x_std.png"),
            ("objective_best_adversarial_downstream_vs_guidance.png", "adversarial_downstream_vs_guidance.png"),
            ("objective_best_adversarial_downstream_vs_guidance_std.png", "adversarial_downstream_vs_guidance_std.png"),
        ):
            src = objective_history_dir / source
            if src.exists():
                shutil.copy2(src, objective_history_dir / alias)
        (objective_history_dir / "objective_te_metrics.json").write_text(json.dumps(
            {"objective_best_epoch": int(step) + 1,
             "te_evaluated_at_exact_objective_epoch": True,
             "te_is_new_best": bool(is_new_best),
             **{k: float(result[k]) for k in (
                 "DR_changed", "mean_target_overlap", "remove_success",
                 "mean_removed_original_overlap", "te_joint_score", "te_joint_success",
                 "primary_best_metric", "secondary_best_metric")}}, indent=2,
        ))
        print(f"[objective_best] full epoch={step + 1} bundle saved: {objective_history_dir}")

    shutil.rmtree(work_dir, ignore_errors=True)
    return result



def _gauss_kernel_1d(sigma):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return jnp.asarray(k / k.sum()), r


def ti_smooth(g, sigma_cells):
    """Separable Gaussian TI filter: periodic longitude, edge-repeated latitude.

    Use two XLA convolutions instead of unrolling ~122 Python/JAX additions
    at sigma=10, which would otherwise dramatically increase compilation time.
    There is no mixing across batch, time, variable, or pressure levels.
    """
    if sigma_cells <= 0:
        return g
    k, r = _gauss_kernel_1d(sigma_cells)
    H, W = g.shape[-2], g.shape[-1]
    if r >= W:
        raise ValueError("TI Gaussian radius must be smaller than longitude width")
    # All leading dimensions are treated as independent samples, never channels.
    x = jnp.reshape(g, (-1, 1, H, W))
    periodic_lon = jnp.concatenate([x[..., -r:], x, x[..., :r]], axis=-1)
    conv = jax.lax.conv_general_dilated
    dims = ("NCHW", "OIHW", "NCHW")
    lon = conv(periodic_lon, jnp.reshape(k, (1, 1, 1, 2 * r + 1)),
               window_strides=(1, 1), padding="VALID", dimension_numbers=dims)
    edge_lat = jnp.pad(lon, ((0, 0), (0, 0), (r, r), (0, 0)), mode="edge")
    result = conv(edge_lat, jnp.reshape(k, (1, 1, 2 * r + 1, 1)),
                  window_strides=(1, 1), padding="VALID", dimension_numbers=dims)
    return jnp.reshape(result, g.shape)



def input_gradient_dilation_mask(var, tensor, selected_levels=None):
    """Select Gaussian TI smoothing by GraphCast input variable/pressure level.

    A value of one selects the original TI smoothing; zero retains the raw gradient.
    This does NOT change trajectory-label kernel dilation or geodesic weighting.
    """
    rules = set(GRADIENT_DILATION_VARS)
    if "all" in rules:
        return jnp.ones_like(tensor)
    if "none" in rules:
        return jnp.zeros_like(tensor)
    is_wind = var in {"10m_u_component_of_wind", "10m_v_component_of_wind",
                      "u_component_of_wind", "v_component_of_wind"}
    if var in rules or (is_wind and ("wind" in rules or "wind_speed" in rules)):
        return jnp.ones_like(tensor)
    chosen = [int(x.partition("@")[2]) for x in rules if x.startswith(var + "@")]
    if not chosen:
        return jnp.zeros_like(tensor)
    if selected_levels is None:
        raise ValueError(f"Need pressure levels to interpret gradient dilation rule for {var}")
    levels = np.asarray(selected_levels, dtype=float)
    # GraphCast inputs have [batch,time,level,lat,lon] or [time,level,lat,lon].
    # The level dimension is immediately before the last two spatial dimensions.
    if tensor.ndim < 3 or tensor.shape[-3] != len(levels):
        raise ValueError(f"{var}: level axis does not match tensor shape {tensor.shape}")
    selection = np.isin(levels, chosen).astype(np.float32)
    if not selection.any():
        raise ValueError(f"{var}: configured dilation levels {chosen} not found in {list(levels)}")
    shape = (1,) * (tensor.ndim - 3) + (len(levels), 1, 1)
    return jnp.broadcast_to(jnp.asarray(selection.reshape(shape)), tensor.shape)


def variable_distance_weight(var, tensor, spatial_weight):
    """Distance kernel on selected variables/levels; unit weight on everything else.

    One GRADIENT_DILATION_VARS selection controls TI smoothing, the gradient
    distance weight, and the regularization distance weight. Non-selected
    variables still update and receive a normal, unweighted L2 penalty.
    """
    levels = (np.asarray(GC_MEAN[var].coords["level"].values)
              if var in GC_MEAN and "level" in GC_MEAN[var].dims else None)
    selected = input_gradient_dilation_mask(var, tensor, levels)
    if isinstance(spatial_weight, dict):
        wind_names = {"10m_u_component_of_wind", "10m_v_component_of_wind",
                      "u_component_of_wind", "v_component_of_wind"}
        family = "wind" if var in wind_names else "others"
        spatial_weight = spatial_weight[family]
    spatial = broadcast_weight(spatial_weight, tensor)
    return jnp.where(selected > 0, spatial, jnp.ones_like(tensor))


def gradient_with_optional_dilation(var, gradient, mask):
    """Gaussian TI smooth selected channels only; others retain unblurred gradient."""
    selected_levels = (np.asarray(GC_MEAN[var].coords["level"].values)
                       if var in GC_MEAN and "level" in GC_MEAN[var].dims else None)
    dilate_mask = input_gradient_dilation_mask(var, gradient, selected_levels)
    clean_gradient = gradient * mask
    if TI_SIGMA_CELLS <= 0:
        return clean_gradient
    wind_names = {"10m_u_component_of_wind", "10m_v_component_of_wind",
                  "u_component_of_wind", "v_component_of_wind"}
    family = "wind" if var in wind_names else "others"
    scale = globals().get("DILATION_SCALE_MAP", {}).get(family, 1.0)
    smoothed = ti_smooth(clean_gradient, TI_SIGMA_CELLS * scale)
    return (jnp.where(dilate_mask > 0, smoothed, clean_gradient) * mask)


def _masked_mean_abs(x, mask):
    return (jnp.abs(x) * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def _masked_rms(x, mask):
    return jnp.sqrt(((x ** 2) * mask).sum() / jnp.maximum(mask.sum(), 1.0))




# ===========================================================================
# Guide-only method: geometry on the 1-degree GraphCast grid
# ===========================================================================
LAT_DEG = np.linspace(-90.0, 90.0, GRID_H)
LON_DEG = np.arange(GRID_W, dtype=float) * (360.0 / GRID_W)
_LAT_RAD_2D = np.deg2rad(LAT_DEG)[:, None]
_LON_RAD_2D = np.deg2rad(LON_DEG)[None, :]


def forecast_start_of(sample):
    return datetime.strptime(sample, "%Y-%m-%dT%H%M") + timedelta(hours=6)


def step_of(dt, start):
    hours = (dt - start).total_seconds() / 3600.0
    k = int(round(hours / 6.0))
    if 0 <= k < FORECAST_STEPS and abs(hours - 6.0 * k) < 1e-4:
        return k
    return None


def nodes_by_step(track, start):
    """{forecast step k: (lon, lat)} for one track (first node per step)."""
    out = {}
    if track is None:
        return out
    for node in track.get("nodes", []):
        k = step_of(node["datetime"], start)
        if k is not None and k not in out:
            out[k] = (float(node["lon"]) % 360.0, float(node["lat"]))
    return out


def grid_index(lon, lat):
    iy = int(np.clip(int(round(lat + 90.0)), 0, GRID_H - 1))
    ix = int(round(lon % 360.0)) % GRID_W
    return iy, ix


def grid_distance_deg(lon, lat):
    """Great-circle distance (deg) from (lon, lat) to every grid point, [H, W]."""
    lat0, lon0 = np.deg2rad(lat), np.deg2rad(lon)
    c = (
        np.sin(lat0) * np.sin(_LAT_RAD_2D)
        + np.cos(lat0) * np.cos(_LAT_RAD_2D) * np.cos(_LON_RAD_2D - lon0)
    )
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def rasterize_tracks(tracks, start):
    """[T, H, W] uint8 node mask built from lon/lat (same grid convention as GraphCast)."""
    mask = np.zeros((FORECAST_STEPS, GRID_H, GRID_W), dtype=np.uint8)
    for track in tracks:
        for k, (lon, lat) in nodes_by_step(track, start).items():
            iy, ix = grid_index(lon, lat)
            mask[k, iy, ix] = 1
    return mask


def changed_track_items(te_reference):
    """One item per changed / removed source track: original track + its target (or None)."""
    target_by_source = {}
    for pair in te_reference["changed_target_pairs"]:
        target_by_source.setdefault(int(pair["source_index"]), pair["target_track"])
    sources = sorted(
        set(int(i) for i in te_reference["removed_indices"])
        | set(int(i) for i in te_reference["changed_indices"])
        | set(target_by_source)
    )
    originals = te_reference["original_tracks"]
    return [
        {
            "source_index": i,
            "original_track": originals[i],
            "target_track": target_by_source.get(i),
        }
        for i in sources
        if 0 <= i < len(originals)
    ]


# ===========================================================================
# Guide construction
# ===========================================================================
def _cosine_taper(dist_deg, radius_deg):
    return np.where(dist_deg < radius_deg, 0.5 * (1.0 + np.cos(np.pi * dist_deg / radius_deg)), 0.0)


def _shift_patch(field, dy, dx):
    """Shift [C, H, W] by dy rows (zero fill, no wrap) and dx columns (periodic in longitude)."""
    out = np.roll(field, int(dx), axis=-1)
    dy = int(dy)
    if dy == 0:
        return out
    shifted = np.zeros_like(out)
    if dy > 0:
        shifted[..., dy:, :] = out[..., :-dy, :]
    else:
        shifted[..., :dy, :] = out[..., -dy:, :]
    return shifted


def build_relocation_guide(clean_std, te_reference, sample):
    """Vortex-relocation guide in standardized TE-channel space, [T, 4, H, W].

    For every changed source track and forecast step k:
        A_k = (s_k - G_sigma * s_k) * taper(|x - p_k| / R_v)      (clean vortex anomaly)
        g_k = -A_k + shift_{p_k -> p*_k}(A_k)                      (move it to the target)
    Steps without a target node (remove-only targets) keep only -A_k.
    Elevation is never guided.
    """
    clean_std = np.asarray(clean_std, dtype=np.float32)
    guide = np.zeros_like(clean_std)
    ch = list(GUIDE_CHANNELS)
    background = np.empty_like(clean_std[:, ch])
    for j, c in enumerate(ch):
        background[:, j] = scipy.ndimage.gaussian_filter(
            clean_std[:, c],
            sigma=(0.0, GUIDE_BACKGROUND_SIGMA_DEG, GUIDE_BACKGROUND_SIGMA_DEG),
            mode=("nearest", "nearest", "wrap"),
        )
    anomaly = clean_std[:, ch] - background

    start = forecast_start_of(sample)
    info = {"relocated_steps": 0, "removed_only_steps": 0, "target_steps_without_original": 0,
            "changed_sources": []}
    for item in changed_track_items(te_reference):
        orig_steps = nodes_by_step(item["original_track"], start)
        tgt_steps = nodes_by_step(item["target_track"], start)
        info["changed_sources"].append(int(item["source_index"]))
        for k, (lon, lat) in orig_steps.items():
            patch = anomaly[k] * _cosine_taper(grid_distance_deg(lon, lat), VORTEX_RADIUS_DEG)[None]
            guide[k, ch] -= patch
            if k in tgt_steps:
                tlon, tlat = tgt_steps[k]
                iy0, ix0 = grid_index(lon, lat)
                iy1, ix1 = grid_index(tlon, tlat)
                guide[k, ch] += _shift_patch(patch, iy1 - iy0, ix1 - ix0)
                info["relocated_steps"] += 1
            else:
                info["removed_only_steps"] += 1
        info["target_steps_without_original"] += len(set(tgt_steps) - set(orig_steps))
    guide[:, CH_ELEV] = 0.0
    return guide, info


def build_saved_downstream_guide(sample, target_tag, clean_std):
    """Saved downstream guidance relative to the same fixed clean reference used by the attack."""
    path = saved_downstream_artifact(sample, target_tag, DOWNSTREAM_EPSILON)
    require_paths(path)
    adv = torch.load(path, map_location="cpu").float()
    mean = DOWNSTREAM_MEAN[None, :, None, None]
    std = DOWNSTREAM_STD[None, :, None, None]
    adv_std = ((adv - mean) / std).numpy().astype(np.float32)
    clean_std = np.asarray(clean_std, dtype=np.float32)
    if adv_std.shape != clean_std.shape:
        raise ValueError(
            f"Saved downstream guidance shape {adv_std.shape} != clean reference {clean_std.shape}"
        )
    guide = adv_std - clean_std
    guide[:, CH_ELEV] = 0.0
    return guide, {
        "source": str(path),
        "clean_reference": "same fixed GraphCast clean run used in the end-to-end objective",
    }


# ===========================================================================
# Regions, centre weights, core masks, input-space weights
# ===========================================================================
def input_locality_center(te_reference, sample):
    """Return the first valid node of the changed original TC track in the forecast window."""
    start = forecast_start_of(sample)
    items = changed_track_items(te_reference)
    preferred = [it for it in items if it.get("target_track") is not None] or items
    candidates = []
    for item in preferred:
        source_index = int(item["source_index"])
        for node in item["original_track"].get("nodes", []):
            k = step_of(node["datetime"], start)
            if k is None:
                continue
            candidates.append((
                int(k), node["datetime"], source_index,
                float(node["lon"]) % 360.0, float(node["lat"]),
            ))
    if not candidates:
        raise ValueError("No valid changed-original TC node exists inside the forecast window.")
    k, dt, source_index, lon, lat = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    return {
        "source_index": int(source_index),
        "forecast_step": int(k),
        "datetime": dt,
        "lon": float(lon),
        "lat": float(lat),
    }


def build_loss_geometry(te_reference, sample):
    start = forecast_start_of(sample)
    items = changed_track_items(te_reference)
    target_tracks = [it["target_track"] for it in items if it["target_track"] is not None]
    original_tracks = [it["original_track"] for it in items]

    tgt_nodes = rasterize_tracks(target_tracks, start)
    orig_nodes = rasterize_tracks(original_tracks, start)

    rad2deg = 180.0 / np.pi
    d_tgt = np.asarray(min_distance_jax(jnp.asarray(tgt_nodes, dtype=jnp.float32))) * rad2deg
    orig_seed = (orig_nodes > 0) & (d_tgt >= MOVED_NODE_KEEP_DEG)
    d_orig = np.asarray(min_distance_jax(jnp.asarray(orig_seed, dtype=jnp.float32))) * rad2deg

    target_region = d_tgt < REGION_RADIUS_DEG
    orig_region = (d_orig < REGION_RADIUS_DEG) & (~target_region)
    # ALL original/target TC nodes define the related forecast corridor,
    # including original nodes excluded from the *original guidance* term.
    d_orig_all = np.asarray(min_distance_jax(jnp.asarray(orig_nodes, dtype=jnp.float32))) * rad2deg
    outside_mask = ((d_tgt >= UNRELATED_EXCLUSION_RADIUS_DEG)
                    & (d_orig_all >= UNRELATED_EXCLUSION_RADIUS_DEG)).astype(np.float32)

    def centre_weight(d):
        return np.maximum(
            np.exp(-(d ** 2) / (2.0 * CENTER_WEIGHT_SIGMA_DEG ** 2)),
            CENTER_WEIGHT_FLOOR,
        )

    w_tgt = target_region * centre_weight(d_tgt)
    w_orig = orig_region * centre_weight(d_orig)

    input_center = input_locality_center(te_reference, sample)
    d_input_deg = grid_distance_deg(input_center["lon"], input_center["lat"])
    d_outside_core_deg = np.maximum(d_input_deg - INPUT_CORE_RADIUS_DEG, 0.0)
    d_outside_core_rad = np.deg2rad(d_outside_core_deg).astype(np.float32)
    w_grad_hw = np.exp(-(d_outside_core_rad ** 2) / (2.0 * GRAD_SIGMA_RAD ** 2)).astype(np.float32)
    w_reg_hw = (1.0 - np.exp(-(d_outside_core_rad ** 2) / (2.0 * REG_SIGMA_RAD ** 2))).astype(np.float32)
    # Recompute kernels from actual distances. Taking roots of the baseline
    # float32 kernel loses far-field support due to numerical underflow.
    grad_family_hw = {}
    reg_family_hw = {}
    for family in ("wind", "others"):
        scale = DILATION_SCALE_MAP[family]
        grad_family_hw[family] = np.exp(
            -(d_outside_core_rad ** 2) / (2.0 * (GRAD_SIGMA_RAD * scale) ** 2)
        ).astype(np.float32)
        reg_family_hw[family] = (1.0 - np.exp(
            -(d_outside_core_rad ** 2) / (2.0 * (REG_SIGMA_RAD * scale) ** 2)
        )).astype(np.float32)

    lam = np.asarray(
        [CHANNEL_WEIGHTS["wind"], CHANNEL_WEIGHTS["thickness"], CHANNEL_WEIGHTS["mslp"]],
        dtype=np.float32,
    )
    kappa_tgt = np.asarray([1.0, 1.0, -1.0], dtype=np.float32)

    f32 = lambda x: jnp.asarray(np.asarray(x, dtype=np.float32))
    geo = {
        "w_tgt": f32(w_tgt),
        "w_orig": f32(w_orig),
        "core_tgt": jnp.asarray(target_region & (d_tgt <= CORE_RADIUS_DEG)),
        "core_orig": jnp.asarray(orig_region & (d_orig <= CORE_RADIUS_DEG)),
        "has_tgt": f32(target_region.any(axis=(1, 2))),
        "has_orig": f32(orig_region.any(axis=(1, 2))),
        "lam": f32(lam),
        "kappa_tgt": f32(kappa_tgt),
        "w_grad_hw": f32(w_grad_hw),
        "w_reg_hw": f32(w_reg_hw),
        "w_grad_family_hw": {name: f32(arr) for name, arr in grad_family_hw.items()},
        "w_reg_family_hw": {name: f32(arr) for name, arr in reg_family_hw.items()},
        "has_targets": bool(target_region.any()),
        "outside_mask": f32(outside_mask),
        "input_center": input_center,
    }
    diag = {
        "changed_sources": [int(it["source_index"]) for it in items],
        "target_nodes": int(tgt_nodes.sum()),
        "original_nodes": int(orig_nodes.sum()),
        "removal_seed_nodes": int(orig_seed.sum()),
        "moved_original_nodes_not_removed": int(((orig_nodes > 0) & ~orig_seed).sum()),
        "unrelated_exclusion_radius_deg": float(UNRELATED_EXCLUSION_RADIUS_DEG),
        "input_mask_radius_deg": float(INPUT_MASK_RADIUS_DEG),
        "unrelated_pixels_by_step": outside_mask.sum(axis=(1, 2)).astype(int).tolist(),
        "steps_with_target_region": int(target_region.any(axis=(1, 2)).sum()),
        "steps_with_original_region": int(orig_region.any(axis=(1, 2)).sum()),
        "target_core_pixels": int((target_region & (d_tgt <= CORE_RADIUS_DEG)).sum()),
        "original_core_pixels": int((orig_region & (d_orig <= CORE_RADIUS_DEG)).sum()),
        "input_locality_center": {
            "source_index": int(input_center["source_index"]),
            "forecast_step_zero_based": int(input_center["forecast_step"]),
            "datetime": input_center["datetime"].isoformat(),
            "lon": float(input_center["lon"]),
            "lat": float(input_center["lat"]),
        },
        "input_core_radius_deg": float(INPUT_CORE_RADIUS_DEG),
        "grad_sigma_rad": float(GRAD_SIGMA_RAD),
        "reg_sigma_rad": float(REG_SIGMA_RAD),
        "dilation_scale": dict(DILATION_SCALE_MAP),
        "effective_spatial_scales": {
            family: {"ti_sigma_cells": TI_SIGMA_CELLS * scale,
                     "grad_sigma_rad": GRAD_SIGMA_RAD * scale,
                     "reg_sigma_rad": REG_SIGMA_RAD * scale}
            for family, scale in DILATION_SCALE_MAP.items()
        },
    }
    return geo, diag


# ===========================================================================
# Differentiable downstream trajectory surrogate loss (same logic as downstream)
# ===========================================================================
def build_gaussian_heatmap_torch(masks, radius, sigma=None, clamp=True, circular_support=False):
    if radius <= 0:
        return masks.float()
    if sigma is None:
        sigma = max(1.0, radius / 3.0)
    masks = masks.float()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"masks must be [T,H,W] or [H,W], got {tuple(masks.shape)}")
    T, H, W = masks.shape
    c = torch.arange(-radius, radius + 1, device=masks.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    d2 = xx.square() + yy.square()
    kernel = torch.exp(-d2 / (2.0 * float(sigma) ** 2))
    if circular_support:
        kernel = kernel * (d2 <= radius ** 2).float()
    kernel = kernel / kernel.max().clamp_min(1e-12)
    patches = F.unfold(masks[:, None], kernel_size=2 * radius + 1, padding=radius)
    heatmap = (patches * kernel.reshape(1, -1, 1)).amax(dim=1).reshape(T, H, W)
    return heatmap.clamp(0.0, 1.0) if clamp else heatmap


def ensure_two_class_logits(logits):
    logits = jnp.asarray(logits)
    if logits.ndim == 3:
        logits = logits[:, None]
    if logits.shape[1] == 2:
        return logits
    if logits.shape[1] == 1:
        return jnp.concatenate([jnp.zeros_like(logits), logits], axis=1)
    raise ValueError(f"Unexpected surrogate logits shape: {logits.shape}")


def logits_to_mask_jax(logits):
    probs = jax.nn.softmax(
        jnp.transpose(ensure_two_class_logits(logits), (0, 2, 3, 1)), axis=-1
    )[..., 1]
    return (probs >= 0.5).astype(jnp.float32)


def _masked_mean_jax(values, mask, eps=1e-12):
    m = jnp.asarray(mask, dtype=values.dtype)
    m = jnp.broadcast_to(m, values.shape)
    denom = m.sum()
    return jnp.where(denom > eps, (values * m).sum() / (denom + eps), 0.0)


def trajectory_focal_loss(logits, target_soft, calibration):
    """Exact downstream-style calibrated focal loss used for trajectory generation."""
    logits = ensure_two_class_logits(logits)
    y1 = jnp.asarray(target_soft, dtype=logits.dtype)
    tol = 1e-6
    mask_zero = (y1 <= tol).astype(logits.dtype)
    mask_one = (y1 >= 1.0 - tol).astype(logits.dtype)
    mask_soft = 1.0 - mask_zero - mask_one
    log_probs = jax.nn.log_softmax(jnp.transpose(logits, (0, 2, 3, 1)), axis=-1)
    log_p0 = log_probs[..., 0]
    log_p1 = log_probs[..., 1]
    cross_entropy = -(
        mask_zero * log_p0
        + mask_one * log_p1
        + mask_soft * y1 * log_p1
    )
    p_t = jnp.exp(-cross_entropy)
    gamma_map = jnp.where(
        jnp.asarray(calibration) > 0.5,
        0.0,
        float(TRAJECTORY_FOCAL_GAMMA),
    )
    focal = ((1.0 - p_t) ** gamma_map) * cross_entropy
    return focal.sum(axis=(1, 2)).mean()


# ===========================================================================
# Loss
# ===========================================================================
PREFIX_WEIGHTS = np.ones(FORECAST_STEPS, dtype=np.float32)  # Equal weighting on TC-valid steps only


def region_step_losses(s_adv, s_clean, guide, geo):
    """Per-step target/original guide-matching losses, each [T]."""
    ch = jnp.asarray(GUIDE_CHANNELS)
    e = (s_adv - s_clean)[:, ch] - guide[:, ch]
    lam = geo["lam"][None, :, None, None]
    sym = (lam * e ** 2).sum(axis=1)
    if CORE_LOSS == "hinge":
        kap = geo["kappa_tgt"][None, :, None, None]
        hinge_tgt = (lam * jax.nn.relu(-kap * e) ** 2).sum(axis=1)
        hinge_orig = (lam * jax.nn.relu(kap * e) ** 2).sum(axis=1)
    else:
        hinge_tgt = hinge_orig = sym
    pix_tgt = jnp.where(geo["core_tgt"], hinge_tgt, sym)
    pix_orig = jnp.where(geo["core_orig"], hinge_orig, sym)

    def weighted_mean(pix, w):
        num = (pix * w).sum(axis=(1, 2))
        den = w.sum(axis=(1, 2))
        return jnp.where(den > 0, num / jnp.maximum(den, 1e-12), 0.0)

    return {
        "tgt": weighted_mean(pix_tgt, geo["w_tgt"]),
        "orig": weighted_mean(pix_orig, geo["w_orig"]),
    }


def time_average(step_loss, alpha, has):
    a = alpha * has
    return (a * step_loss).sum() / jnp.maximum(a.sum(), 1e-12)


def combine_loss(step_losses, reg, geo, alpha_tgt, alpha_orig, w_t, w_o):
    terms = {
        "tgt": time_average(step_losses["tgt"], alpha_tgt, geo["has_tgt"]),
        "orig": time_average(step_losses["orig"], alpha_orig, geo["has_orig"]),
    }
    valid_tgt = jnp.any(geo["has_tgt"] > 0)
    valid_orig = jnp.any(geo["has_orig"] > 0)
    target_weight = valid_tgt.astype(jnp.float32) * w_t
    original_weight = valid_orig.astype(jnp.float32) * w_o
    denom = jnp.maximum(target_weight + original_weight, 1e-12)
    total = (target_weight * terms["tgt"] + original_weight * terms["orig"]) / denom + REG_LAMBDA * reg
    return total, terms


def input_reg_masked(delta, w_reg_hw, perturb_masks):
    values = []
    for var, d in delta.items():
        mask = perturb_masks[var].astype(d.dtype)
        weighted = (variable_distance_weight(var, d, w_reg_hw) * d) ** 2
        denom = jnp.maximum(mask.sum(), 1.0)
        values.append((weighted * mask).sum() / denom)
    return sum(values) / max(1, len(values))


def cosine_lr(step):
    c = 0.5 * (1.0 + math.cos(math.pi * step / max(1, END2END_STEPS - 1)))
    return LR_END + (LR_START - LR_END) * c


def update_delta_pgd(delta, grads, w_grad_hw, perturb_masks):
    """Paper-style projected sign descent; no momentum or gradient norm scaling.

    Optional Gaussian TI smoothing is applied only to selected channels, then
    sign() is applied. The *same* selectors govern the multiplicative distance
    weights here and the w_reg weights used in input_reg_masked().
    """
    updated, grad_mean_abs = {}, {}
    for var, old in delta.items():
        mask = perturb_masks[var].astype(old.dtype)
        g = gradient_with_optional_dilation(var, grads[var], mask)
        grad_mean_abs[var] = float(np.asarray(_masked_mean_abs(g, mask)))
        spatial = variable_distance_weight(var, old, w_grad_hw)
        updated[var] = (old - PGD_STEP_SIZE * spatial * jnp.sign(g)) * mask
    return updated, PGD_STEP_SIZE, grad_mean_abs


def adam_state_zero(delta):
    return {"m": {k: jnp.zeros_like(v) for k, v in delta.items()},
            "v": {k: jnp.zeros_like(v) for k, v in delta.items()}}


def update_delta_adam(delta, grads, state, step, w_grad_hw, perturb_masks):
    """TI-smoothed Projected Adam; apply spatial weighting AFTER Adam normalization."""
    lr = cosine_lr(step)
    new, m_new, v_new, grad_mean_abs = {}, {}, {}, {}
    for key, old in delta.items():
        mask = perturb_masks[key].astype(old.dtype)
        g = gradient_with_optional_dilation(key, grads[key], mask)
        grad_mean_abs[key] = float(np.asarray(_masked_mean_abs(g, mask)))
        m = (ADAM_BETA1 * state["m"][key] + (1 - ADAM_BETA1) * g) * mask
        v = (ADAM_BETA2 * state["v"][key] + (1 - ADAM_BETA2) * (g * g)) * mask
        m_hat = m / (1.0 - ADAM_BETA1 ** (step + 1))
        v_hat = v / (1.0 - ADAM_BETA2 ** (step + 1))
        adaptive = m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        new[key] = (old - lr * variable_distance_weight(key, old, w_grad_hw) * adaptive) * mask
        m_new[key], v_new[key] = m, v
    return new, {"m": m_new, "v": v_new}, lr, grad_mean_abs


# ===========================================================================
# Diagnostics (logged only, never part of the loss)
# ===========================================================================
def vortex_diagnostics(s_adv_np, s_clean_np, guide_np, geo, te_reference, sample, temperature=0.1):
    """Soft MSLP-minimum centre distance to the target, and core intensity deficits."""
    start = forecast_start_of(sample)
    dists = {}
    for item in changed_track_items(te_reference):
        if item["target_track"] is None:
            continue
        orig_steps = nodes_by_step(item["original_track"], start)
        for k, (tlon, tlat) in nodes_by_step(item["target_track"], start).items():
            window = grid_distance_deg(tlon, tlat) < REGION_RADIUS_DEG
            if k in orig_steps:
                window |= grid_distance_deg(*orig_steps[k]) < REGION_RADIUS_DEG
            p = s_adv_np[k, CH_MSLP][window]
            if p.size == 0:
                continue
            w = np.exp(-(p - p.min()) / temperature)
            w /= w.sum()
            lat = np.broadcast_to(_LAT_RAD_2D, window.shape)[window]
            lon = np.broadcast_to(_LON_RAD_2D, window.shape)[window]
            v = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 0) @ w
            v /= max(np.linalg.norm(v), 1e-12)
            clat, clon = np.rad2deg(np.arcsin(v[2])), np.rad2deg(np.arctan2(v[1], v[0])) % 360.0
            dists.setdefault(k, []).append(float(_haversine_deg(clon, clat, tlon, tlat)))

    out = {"center_dist_mean_deg": np.nan, "center_dist_first4_deg": np.nan, "center_dist_last_deg": np.nan}
    if dists:
        ks = sorted(dists)
        per = np.asarray([np.mean(dists[k]) for k in ks])
        out["center_dist_mean_deg"] = float(per.mean())
        out["center_dist_first4_deg"] = float(per[np.asarray(ks) < 4].mean()) if (np.asarray(ks) < 4).any() else np.nan
        out["center_dist_last_deg"] = float(per[-1])

    core = np.asarray(geo["core_tgt"])
    if core.any():
        e = (s_adv_np - s_clean_np) - guide_np
        std = DOWNSTREAM_STD.numpy()
        out["core_wind_deficit_ms"] = float(np.maximum(-e[:, CH_WIND], 0.0)[core].mean() * std[CH_WIND])
        out["core_mslp_deficit_hpa"] = float(np.maximum(e[:, CH_MSLP], 0.0)[core].mean() * std[CH_MSLP] / 100.0)
    else:
        out["core_wind_deficit_ms"] = np.nan
        out["core_mslp_deficit_hpa"] = np.nan
    return out


def standardized_to_raw(s_std):
    raw = (
        torch.from_numpy(np.asarray(s_std).copy()).float()
        * DOWNSTREAM_STD[None, :, None, None]
        + DOWNSTREAM_MEAN[None, :, None, None]
    )
    raw[:, CH_WIND].clamp_min_(0.0)
    return raw



def save_forecast_step_diagnostics(ctx, s_adv, out_dir, te_reference, pred_tracks=None):
    """RMS guided/induced changes by forecast step, in downstream standardized units."""
    out_dir = Path(out_dir)
    adv = np.asarray(s_adv, dtype=np.float32)
    clean = np.asarray(ctx["clean_s_np"], dtype=np.float32)
    guide = np.asarray(ctx["guide_np"], dtype=np.float32)
    delta = adv - clean
    geo = ctx["geo"]
    target_region = np.asarray(geo["w_tgt"]) > 0
    records = []
    names = [(CH_WIND, "wind"), (CH_THICK, "thickness"), (CH_MSLP, "mslp")]
    for t in range(delta.shape[0]):
        for ch, name in names:
            for region_name, mask in (("global", None), ("target_region", target_region[t])):
                if mask is not None and not mask.any():
                    continue
                x, g = delta[t, ch], guide[t, ch]
                if mask is not None:
                    x, g = x[mask], g[mask]
                records.append({"forecast_step": t + 1, "lead_hours": (t + 1) * 6,
                    "variable": name, "region": region_name,
                    "induced_rms_std": float(np.sqrt(np.mean(x**2))),
                    "guidance_rms_std": float(np.sqrt(np.mean(g**2))),
                    "residual_rms_std": float(np.sqrt(np.mean((x-g)**2)))})
    pd.DataFrame(records).to_csv(out_dir / "forecast_perturbation_propagation.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    frame = pd.DataFrame(records)
    for name in ("wind", "thickness", "mslp"):
        sub = frame[(frame.variable == name) & (frame.region == "target_region")]
        if not sub.empty:
            ax.plot(sub.lead_hours, sub.residual_rms_std, label=f"{name}: target residual")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Target-region standardized RMS residual")
    ax.set_title("GraphCast guidance matching across forecast horizon")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "forecast_perturbation_propagation.png")
    plt.close(fig)

    start = forecast_start_of(ctx["sample"])
    target_tracks = [item["target_track"] for item in changed_track_items(te_reference)
                     if item["target_track"] is not None]
    source_nodes = {}
    predicted_nodes = {}
    for track in target_tracks:
        for node in track.get("nodes", []):
            source_nodes.setdefault(node["datetime"], []).append(node)
    for track in pred_tracks or []:
        for node in track.get("nodes", []):
            predicted_nodes.setdefault(node["datetime"], []).append(node)
    rows = []
    for t in range(adv.shape[0]):
        time = start + timedelta(hours=6*t)
        targets, predictions = source_nodes.get(time, []), predicted_nodes.get(time, [])
        for target in targets:
            distances = [_haversine_deg(float(target["lon"]), float(target["lat"]),
                                       float(pred["lon"]), float(pred["lat"])) for pred in predictions]
            minimum = min(distances) if distances else np.nan
            rows.append({"forecast_step": t+1, "lead_hours": (t+1)*6, "datetime": time.isoformat(),
                "target_lon": float(target["lon"]), "target_lat": float(target["lat"]),
                "nearest_te_distance_deg": float(minimum),
                "is_temporal_hit": bool(np.isfinite(minimum) and minimum < TE_TOL_DEG)})
    pd.DataFrame(rows, columns=["forecast_step","lead_hours","datetime","target_lon","target_lat",
                                "nearest_te_distance_deg","is_temporal_hit"]).to_csv(
                                    out_dir / "trajectory_step_metrics.csv", index=False)
    stationarity = []
    for kind, tracks in (("target", target_tracks), ("te_prediction", pred_tracks or [])):
        for index, track in enumerate(tracks):
            points = [(round(float(n["lon"])) % 360, round(float(n["lat"]))) for n in track.get("nodes", [])]
            stationary = sum(a == b for a, b in zip(points[:-1], points[1:]))
            stationarity.append({"track_type": kind, "track_index": index, "nodes": len(points),
                "unique_rounded_locations": len(set(points)), "stationary_transitions": stationary,
                "stationary_transition_fraction": stationary/max(len(points)-1,1)})
    pd.DataFrame(stationarity).to_csv(out_dir / "trajectory_stationarity_metrics.csv", index=False)


def save_loss_components(history, out_dir, objective_best_step=None, te_best_step=None):
    """Save one CSV and one plot per raw and weighted loss, plus stage-wide source history."""
    out_dir = Path(out_dir)
    target = out_dir / "loss_components"
    target.mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(history)
    components = {
        "loss_total": "total", "loss_guide_tgt": "guide_tgt",
        "loss_guide_orig": "guide_orig", "loss_guide_core": "guide_core",
        "loss_traj_core": "traj_core", "loss_reg": "reg",
        "loss_unrelated": "unrelated", "loss_weighted_unrelated": "weighted_unrelated",
        "loss_weighted_guide": "weighted_guide", "loss_weighted_traj": "weighted_traj",
        "loss_weighted_reg": "weighted_reg",
        "loss_weighted_guide_tgt": "weighted_guide_tgt",
        "loss_weighted_guide_orig": "weighted_guide_orig",
    }
    for filename, column in components.items():
        if column not in h:
            continue
        data = h[["step", "epoch", column]].copy()
        data.rename(columns={column: "value"}, inplace=True)
        data.to_csv(target / f"{filename}.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 3.5), dpi=150)
        ax.plot(data["epoch"], data["value"], linewidth=1.2, label=column)
        if objective_best_step is not None and objective_best_step >= 0:
            ax.axvline(objective_best_step + 1, linestyle="--", alpha=0.7, label=f"Objective best: epoch {objective_best_step + 1}")
        if te_best_step is not None and te_best_step >= 0:
            ax.axvline(te_best_step + 1, linestyle=":", alpha=0.8, label=f"TE best: epoch {te_best_step + 1}")
        ax.set_xlabel("Optimization epoch (1-based)")
        ax.set_ylabel("Loss")
        ax.set_title(column)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(target / f"{filename}.png", bbox_inches="tight")
        plt.close(fig)
    active_terms = (h["weighted_guide"] + h["weighted_traj"] + h["weighted_reg"])
    if "weighted_unrelated" in h:
        active_terms = active_terms + h["weighted_unrelated"]
    identity_err = np.abs(h["total"] - active_terms)
    with open(target / "loss_identity_check.json", "w") as f:
        json.dump({"max_absolute_difference": float(identity_err.max()), "mean_absolute_difference": float(identity_err.mean()),
                   "definition": "total == weighted_guide + weighted_traj + weighted_reg + weighted_unrelated"}, f, indent=2)


def plot_target_focused_summary(history, clean_raw, guide_raw, final_raw, out_dir):
    out_dir = Path(out_dir)
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for col in ["total", "guide_core", "traj_core", "guide_tgt", "guide_orig", "reg", "unrelated"]:
        if col in h.columns:
            ax.plot(h["step"], h[col], label=col)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Combined target-focused loss history")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_history.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
    for j, (ch, name) in enumerate([(CH_WIND, "10 m wind speed"), (CH_THICK, "Thickness"), (CH_MSLP, "MSLP")]):
        a = np.abs(guide_raw[:, ch]).mean(0)
        b = np.abs(final_raw[:, ch] - clean_raw[:, ch]).mean(0)
        finite_a, finite_b = a[np.isfinite(a)], b[np.isfinite(b)]
        vmax = max(float(np.percentile(finite_a, 99.5)) if finite_a.size else 0.0,
                   float(np.percentile(finite_b, 99.5)) if finite_b.size else 0.0, 1e-12)
        im1 = axes[0, j].imshow(a, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"Guide |g| (raw units): {name}")
        fig.colorbar(im1, ax=axes[0, j], fraction=0.046, pad=0.04)
        im2 = axes[1, j].imshow(b, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        axes[1, j].set_title(f"Induced |Δ| (raw units): {name}")
        fig.colorbar(im2, ax=axes[1, j], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "guide_vs_induced.png", bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Per-case context shared by the attack and the diagnostic stage
# ===========================================================================
def prepare_case_context(sample, target_tag):
    tdir = sample_target_dir(sample, target_tag) / "target"
    target_track_path = tdir / f"target_tracks_{target_tag}.txt"
    target_mask_path = tdir / f"target_mask_{target_tag}.pt"
    metadata_path = tdir / "target_metadata.json"
    require_paths(target_track_path, target_mask_path, metadata_path)
    te_reference = build_te_reference(sample, target_track_path, metadata_path)

    _, eval_inputs, eval_targets, eval_forcings = prepare_graphcast_case(sample)
    base_ds = eval_inputs
    missing_vars = [v for v in PERTURB_INPUT_VARS if v not in base_ds]
    if missing_vars:
        raise KeyError(f"Attack variables missing from GraphCast inputs: {missing_vars}")

    lat = np.asarray(base_ds["lat"].values, dtype=float)
    lon = np.asarray(base_ds["lon"].values, dtype=float)
    if lat.size != GRID_H or not np.all(np.diff(lat) > 0) or abs(lat[0] + 90.0) > 1e-6:
        raise ValueError("Expected ascending 1-degree latitude from -90 to 90 in GraphCast inputs.")
    if lon.size != GRID_W or abs(lon[0]) > 1e-6:
        raise ValueError("Expected 1-degree longitude starting at 0 in GraphCast inputs.")

    def pipeline(ds):
        return run_pipeline(ds, eval_targets, eval_forcings)

    clean_s, _, clean_gc_output = pipeline(base_ds)
    clean_s_np = np.asarray(clean_s, dtype=np.float32)
    clean_gc_output_host = jax.device_get(clean_gc_output)

    if GUIDE_SOURCE == "relocation":
        guide_np, guide_info = build_relocation_guide(clean_s_np, te_reference, sample)
    else:
        guide_np, guide_info = build_saved_downstream_guide(sample, target_tag, clean_s_np)
    geo, geo_diag = build_loss_geometry(te_reference, sample)

    target_mask_torch = torch.load(target_mask_path, map_location="cpu").float()
    if tuple(target_mask_torch.shape) != (FORECAST_STEPS, GRID_H, GRID_W):
        raise ValueError(
            f"Unexpected target mask shape {tuple(target_mask_torch.shape)} at {target_mask_path}; "
            f"expected {(FORECAST_STEPS, GRID_H, GRID_W)}"
        )
    original_mask_np = rasterize_tracks(
        te_reference["original_tracks"], forecast_start_of(sample)
    ).astype(np.float32)
    target_mask_np = target_mask_torch.numpy().astype(np.float32)
    target_soft_torch = build_gaussian_heatmap_torch(
        target_mask_torch,
        TRAJECTORY_TARGET_RADIUS,
        TRAJECTORY_TARGET_SIGMA,
    )
    if TRAJECTORY_ZERO_ORIGINAL_ONLY_AFTER_DILATION:
        original_only_mask = torch.from_numpy(original_mask_np).bool() & (~target_mask_torch.bool())
        target_soft_torch = target_soft_torch.masked_fill(original_only_mask, 0.0)
    target_soft_np = target_soft_torch.numpy().astype(np.float32)
    target_mask_jax = jnp.asarray(target_mask_np)
    original_mask_jax = jnp.asarray(original_mask_np)
    clean_logits = surrogate_bridge(clean_s)
    clean_surrogate_mask = logits_to_mask_jax(clean_logits)
    calibration = (
        (target_mask_jax != original_mask_jax)
        & (clean_surrogate_mask == target_mask_jax)
    ).astype(jnp.float32)

    base_std, mean_arrays, std_arrays, perturb_masks = {}, {}, {}, {}
    for var in PERTURB_INPUT_VARS:
        da = base_ds[var]
        raw = jnp.asarray(np.asarray(da.values), dtype=jnp.float32)
        mean_v, std_v = stat_for_var(var, da)
        base_std[var] = (raw - mean_v) / std_v
        mean_arrays[var], std_arrays[var] = mean_v, std_v
        perturb_masks[var] = build_perturb_mask(var, da)
    # Strictly prevent optimization from modifying input grid cells outside the TC radius.
    # All input times / allowed levels share this spatial mask.
    center = geo["input_center"]
    input_distance = grid_distance_deg(center["lon"], center["lat"])
    input_inside = (input_distance <= INPUT_MASK_RADIUS_DEG) if INPUT_MASK_RADIUS_DEG > 0 else np.ones_like(input_distance, dtype=bool)
    for var in PERTURB_INPUT_VARS:
        arr = base_ds[var]
        axes = [arr.dims.index(name) for name in ("lat", "lon")]
        shape = [1] * arr.ndim
        shape[axes[0]], shape[axes[1]] = input_inside.shape
        spatial = np.reshape(input_inside.astype(np.float32), shape)
        perturb_masks[var] = perturb_masks[var] * jnp.asarray(spatial)
    mask_summary = validate_perturb_masks(base_ds, perturb_masks)
    for value in mask_summary.values():
        value["input_mask_radius_deg"] = float(INPUT_MASK_RADIUS_DEG)
        value["input_active_spatial_fraction"] = float(input_inside.mean())


    return {
        "sample": sample,
        "target_tag": target_tag,
        "te_reference": te_reference,
        "base_ds": base_ds,
        "pipeline": pipeline,
        "clean_s": clean_s,
        "clean_s_np": clean_s_np,
        "clean_gc_output": clean_gc_output_host,
        "guide": jnp.asarray(guide_np),
        "guide_np": guide_np,
        "guide_info": guide_info,
        "geo": geo,
        "geo_diag": geo_diag,
        "target_mask": target_mask_jax,
        "original_mask": original_mask_jax,
        "target_soft": jnp.asarray(target_soft_np),
        "trajectory_calibration": calibration,
        "clean_surrogate_mask": clean_surrogate_mask,
        "base_std": base_std,
        "mean_arrays": mean_arrays,
        "std_arrays": std_arrays,
        "perturb_masks": perturb_masks,
        "mask_summary": mask_summary,
    }


def build_adv_ds(ctx, delta):
    ds_new = ctx["base_ds"].copy(deep=True)
    for var in PERTURB_INPUT_VARS:
        d = delta[var] * ctx["perturb_masks"][var]
        raw = (ctx["base_std"][var] + d) * ctx["std_arrays"][var] + ctx["mean_arrays"][var]
        if var in NONNEGATIVE_INPUT_VARS:
            raw = jnp.maximum(raw, 0.0)
        ds_new[var] = ctx["base_ds"][var].copy(data=raw)
    return ds_new


def project_delta(ctx, delta):
    out = {}
    for var, d in delta.items():
        mask = ctx["perturb_masks"][var].astype(d.dtype)
        d = jnp.clip(d * mask, -END2END_EPSILON, END2END_EPSILON) * mask
        if var in NONNEGATIVE_INPUT_VARS:
            lower = (0.0 - ctx["mean_arrays"][var]) / ctx["std_arrays"][var] - ctx["base_std"][var]
            d = jnp.maximum(d, lower) * mask
            d = jnp.clip(d, -END2END_EPSILON, END2END_EPSILON) * mask
        out[var] = d
    return out


def combined_objective(ctx, delta):
    """Optimize one active branch, with independent input/forecast protection.

    guidance: L_guide + regularization + L_outside
    trajectory: L_traj + regularization + L_outside
    joint: old combined objective (for backward-compatible unit tests only).
    """
    adv_ds = build_adv_ds(ctx, delta)
    s_adv, downstream, gc_output = ctx["pipeline"](adv_ds)

    guide_steps = region_step_losses(s_adv, ctx["clean_s"], ctx["guide"], ctx["geo"])
    alpha = jnp.asarray(PREFIX_WEIGHTS)
    guide_terms = {
        "tgt": time_average(guide_steps["tgt"], alpha, ctx["geo"]["has_tgt"]),
        "orig": time_average(guide_steps["orig"], alpha, ctx["geo"]["has_orig"]),
    }
    valid_tgt = jnp.any(ctx["geo"]["has_tgt"] > 0)
    valid_orig = jnp.any(ctx["geo"]["has_orig"] > 0)
    target_weight = valid_tgt.astype(jnp.float32)
    original_weight = valid_orig.astype(jnp.float32) * ORIGINAL_WEIGHT
    denom = jnp.maximum(target_weight + original_weight, 1e-12)
    target_fraction = target_weight / denom
    original_fraction = original_weight / denom
    guide_core = target_fraction * guide_terms["tgt"] + original_fraction * guide_terms["orig"]

    if ACTIVE_OBJECTIVE_PHASE == "guidance":
        # True Guidance-only optimization: do not run the Torch surrogate in
        # every backward pass. Exact surrogate outputs are computed lazily on
        # archived TE/Objective Best checkpoints and in the final bundles.
        logits = None
        traj_core = jnp.asarray(0.0, dtype=guide_core.dtype)
    else:
        logits = surrogate_bridge(s_adv)
        traj_core = trajectory_focal_loss(
            logits, ctx["target_soft"], ctx["trajectory_calibration"]
        )

    reg = input_reg_masked(delta, ctx["geo"]["w_reg_family_hw"], ctx["perturb_masks"])
    # Standardized forecast-space collateral penalty, outside BOTH actual
    # original and desired target TC corridors. Average over ALL 38 steps,
    # pixels and selected downstream channels; unlike guidance no target mask.
    outside = ctx["geo"]["outside_mask"]
    field_difference = (s_adv - ctx["clean_s"])[:, jnp.asarray(GUIDE_CHANNELS)]
    unrelated = (jnp.square(field_difference) * outside[:, None]).sum() / (
        jnp.maximum(outside.sum(), 1.0) * field_difference.shape[1]
    )
    phase = ACTIVE_OBJECTIVE_PHASE
    active_guide = GUIDE_LOSS_WEIGHT if phase in {"joint", "guidance"} else 0.0
    active_traj = TRAJECTORY_LOSS_WEIGHT if phase in {"joint", "trajectory"} else 0.0
    total = (active_guide * guide_core + active_traj * traj_core
             + REG_LAMBDA * reg + UNRELATED_LOSS_WEIGHT * unrelated)
    parts = {
        "guide_tgt": guide_terms["tgt"],
        "guide_orig": guide_terms["orig"],
        "guide_core": guide_core,
        "guide_tgt_fraction": target_fraction,
        "guide_orig_fraction": original_fraction,
        "traj_core": traj_core,
        "reg": reg,
        "unrelated": unrelated,
    }
    return total, (parts, guide_steps, logits, s_adv, downstream, gc_output)


def active_loss_weights():
    """Actual objective coefficients in the current sequential phase."""
    return (GUIDE_LOSS_WEIGHT if ACTIVE_OBJECTIVE_PHASE in ("joint", "guidance") else 0.0,
            TRAJECTORY_LOSS_WEIGHT if ACTIVE_OBJECTIVE_PHASE in ("joint", "trajectory") else 0.0)


def save_unrelated_diagnostics(ctx, s_adv, delta_state, output_dir):
    """Independent outside-corridor diagnostics for a saved exact checkpoint."""
    outside = np.asarray(ctx["geo"]["outside_mask"], dtype=bool)
    difference = np.asarray(s_adv, dtype=np.float32) - np.asarray(ctx["clean_s_np"], dtype=np.float32)
    values = difference[:, list(GUIDE_CHANNELS)]
    sq = values ** 2
    rows = []
    for t in range(values.shape[0]):
        sub = values[t, :, outside[t]]
        rows.append({"forecast_step": t+1, "lead_hours": (t+1)*6,
                     "outside_pixels": int(outside[t].sum()),
                     "unrelated_rmse": float(np.sqrt(np.mean(sub*sub))) if sub.size else 0.0,
                     "unrelated_max_abs": float(np.max(np.abs(sub))) if sub.size else 0.0,
                     "unrelated_changed_fraction_abs_gt_0p1": float(np.mean(np.abs(sub)>0.1)) if sub.size else 0.0})
    pd.DataFrame(rows).to_csv(output_dir / "unrelated_region_by_step.csv", index=False)
    mask = np.broadcast_to(outside[:,None], values.shape)
    selected = values[mask]
    metrics = {"unrelated_rmse": float(np.sqrt(np.mean(selected*selected))) if selected.size else 0.0,
               "unrelated_max_abs": float(np.max(np.abs(selected))) if selected.size else 0.0,
               "unrelated_changed_fraction_abs_gt_0p1": float(np.mean(np.abs(selected)>0.1)) if selected.size else 0.0,
               "outside_pixels": int(outside.sum()),
               "input_mask_radius_deg": INPUT_MASK_RADIUS_DEG,
               "unrelated_exclusion_radius_deg": UNRELATED_EXCLUSION_RADIUS_DEG}
    for region_name, region_mask in (("target", np.asarray(ctx["geo"]["w_tgt"]) > 0),
                                    ("original", np.asarray(ctx["geo"]["w_orig"]) > 0)):
        subset_mask = np.broadcast_to(region_mask[:, None], values.shape)
        subset = values[subset_mask]
        metrics[region_name + "_region_rmse"] = float(np.sqrt(np.mean(subset ** 2))) if subset.size else 0.0
    # Input mask is enforced at projection; report exact realized violation.
    outside_change = []
    for var, d in delta_state.items():
        invalid = np.asarray(ctx["perturb_masks"][var]) == 0
        arr = np.asarray(d)
        if np.any(invalid):
            outside_change.append(np.max(np.abs(arr[invalid])))
    metrics["input_mask_violation_max_abs"] = float(max(outside_change, default=0.0))
    pd.DataFrame([metrics]).to_csv(output_dir / "unrelated_region_metrics.csv", index=False)
    return metrics

def delta_statistics(ctx, delta_state):
    per_var = {}
    for var, value in delta_state.items():
        arr = np.asarray(value)
        mask = np.asarray(ctx["perturb_masks"][var]) > 0.5
        active = np.abs(arr[mask])
        per_var[var] = {
            "linf": float(active.max()) if active.size else 0.0,
            "rms": float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0,
            "mean_abs": float(active.mean()) if active.size else 0.0,
        }
    return float(max((v["linf"] for v in per_var.values()), default=0.0)), per_var


def _te_row(te, step, kind):
    return {
        "step": int(step),
        "epoch": int(step) + 1,
        "kind": kind,
        "DR_changed": float(te.get("DR_changed", np.nan)),
        "mean_target_overlap": float(te.get("mean_target_overlap", np.nan)),
        "remove_success": float(te.get("remove_success", np.nan)),
        "mean_removed_original_overlap": float(te.get("mean_removed_original_overlap", np.nan)),
        "te_generation_score": float(te.get("te_generation_score", np.nan)),
        "te_removal_score": float(te.get("te_removal_score", np.nan)),
        "te_joint_score": float(te.get("te_joint_score", np.nan)),
        "te_joint_success": float(te.get("te_joint_success", np.nan)),
        "target_success": float(te.get("target_success", np.nan)),
        "primary_best_metric": float(te.get("primary_best_metric", -np.inf)),
        "secondary_best_metric": float(te.get("secondary_best_metric", -np.inf)),
        "n_detected_candidates": int(te.get("n_detected_candidates", 0)),
        "n_pred_tracks": int(te.get("n_pred_tracks", 0)),
        "target_hit_nodes": int(te.get("target_hit_nodes", 0)),
        "changed_target_nodes": int(te.get("changed_target_nodes", 0)),
        "residual_original_nodes": int(te.get("residual_original_nodes", 0)),
        "original_reference_nodes": int(te.get("original_reference_nodes", 0)),
        "fa_ratio": float(te.get("fa_ratio", 0.0)),
        "is_new_te_best": bool(te.get("is_new_te_best", False)),
        "artifacts_saved": bool(te.get("artifacts_saved", False)),
        "te_used_in_learning": False,
    }


# ===========================================================================
# Main attack for one (case, target) at the current upstream epsilon / guide delta
# ===========================================================================
def run_target_focused_case(sample, target_tag, warm_start_from=None, stage_index=1, stage_kind="final",
                            objective_phase="joint", output_dir=None, warm_start_checkpoint=None):
    global ACTIVE_OBJECTIVE_PHASE
    if objective_phase not in {"joint", "guidance", "trajectory"}:
        raise ValueError(f"Unknown objective phase {objective_phase!r}")
    ACTIVE_OBJECTIVE_PHASE = objective_phase
    out_dir = (Path(output_dir) if output_dir is not None else
               end2end_sample_dir(sample, target_tag, DOWNSTREAM_EPSILON, END2END_EPSILON))
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_meta = {"objective_phase": objective_phase, "stage_index": int(stage_index), "stage_kind": str(stage_kind),
                  "guide_delta": float(DOWNSTREAM_EPSILON), "input_epsilon": float(END2END_EPSILON)}
    ctx = prepare_case_context(sample, target_tag)
    geo, te_reference = ctx["geo"], ctx["te_reference"]
    perturb_masks = ctx["perturb_masks"]
    np.savez_compressed(out_dir / "guide_std.npz", guide=ctx["guide_np"])

    print(f"[{sample} {target_tag}] fixed guide delta={DOWNSTREAM_EPSILON:g}; upstream epsilon={END2END_EPSILON:g}")
    print(f"[{sample} {target_tag}] guide source={GUIDE_SOURCE}:", ctx["guide_info"])
    print(f"[{sample} {target_tag}] loss geometry:", ctx["geo_diag"])
    print(f"[{sample} {target_tag}] attack variables:", PERTURB_INPUT_VARS, "levels=", PERTURB_LEVELS)
    print(
        f"[{sample} {target_tag}] objective = {TRAJECTORY_LOSS_WEIGHT:g}*trajectory_focal + "
        f"{GUIDE_LOSS_WEIGHT:g}*weighted_mean(valid_guide_target, {ORIGINAL_WEIGHT:g}*valid_guide_original) + "
        f"{REG_LAMBDA:g}*input_reg"
    )
    print(f"[{sample} {target_tag}] SEQUENTIAL PHASE={objective_phase}; TE is evaluation-only.")

    delta = {k: jnp.zeros_like(v) for k, v in ctx["base_std"].items()}
    warm_start_description = None
    if warm_start_from is not None or warm_start_checkpoint is not None:
        prev_guide, prev_eps = ((float(warm_start_from[0]), float(warm_start_from[1]))
                                if warm_start_from is not None else (float(DOWNSTREAM_EPSILON), float(END2END_EPSILON)))
        previous_path = (Path(warm_start_checkpoint) if warm_start_checkpoint is not None else
                         end2end_sample_dir(sample, target_tag, prev_guide, prev_eps)
                         / "objective_best" / "best_delta_std.npz")
        require_paths(previous_path)
        previous = np.load(previous_path)
        missing = [k for k in ctx["base_std"] if k not in previous.files]
        if missing:
            raise KeyError(f"Missing warm-start variables in {previous_path}: {missing}")
        delta = {k: jnp.asarray(previous[k]) for k in ctx["base_std"]}
        warm_start_description = {"guide_delta": prev_guide, "input_epsilon": prev_eps}
        if warm_start_checkpoint is not None:
            warm_start_description["intra_epsilon_checkpoint"] = str(previous_path)
        print(f"[{sample} {target_tag}] warm-start from Objective best: {previous_path}")
    delta = project_delta(ctx, delta)
    if warm_start_from is not None or warm_start_checkpoint is not None:
        stage_path = out_dir / "warmup_source.json"
        same = all(np.array_equal(np.asarray(delta[k]), np.asarray(previous[k])) for k in delta)
        if not same:
            raise AssertionError("Warmup checkpoint was unexpectedly modified by projection")
        stage_path.write_text(json.dumps({"selection": "objective_best", "checkpoint_path": str(previous_path),
                                         "unchanged_after_projection": same}, indent=2))
    adam_state = adam_state_zero(delta) if OPTIMIZER == "adam" else None

    def snapshot(delta_state):
        return {k: jnp.array(v) for k, v in delta_state.items()}

    history, te_log = [], []
    best_objective = {
        "loss": np.inf,
        "step": -1,
        "delta": snapshot(delta),
    }

    init_s, _, init_gc_output = ctx["pipeline"](build_adv_ds(ctx, delta))
    initial_te = te_check(
        standardized_to_raw(init_s),
        sample,
        te_reference,
        out_dir / "te_checks" / "stage_initial",
        ctx=ctx,
        delta=delta,
        s_adv_std=init_s,
        gc_output=init_gc_output,
        best_result=None,
        step=-1,
        stage_meta=stage_meta,
    )
    best_te = {**initial_te, "delta": snapshot(delta), "step": -1}
    te_log.append(_te_row(initial_te, -1, "stage_initial"))

    for step in range(END2END_STEPS):
        (loss, (parts, guide_steps, logits, s_adv, downstream_adv, gc_output_adv)), grads = jax.value_and_grad(
            lambda d: combined_objective(ctx, d),
            has_aux=True,
        )(delta)
        loss_f = float(loss)
        objective_improved = loss_f < best_objective["loss"]
        if objective_improved:
            best_objective = {
                "loss": loss_f,
                "step": int(step),
                "delta": snapshot(delta),
            }

        row = {
            "step": int(step),
            "epoch": int(step) + 1,
            "phase": objective_phase,
            "is_new_objective_best": bool(objective_improved),
            "total": loss_f,
            "guide_tgt": float(parts["guide_tgt"]),
            "guide_orig": float(parts["guide_orig"]),
            "guide_core": float(parts["guide_core"]),
            "traj_core": float(parts["traj_core"]),
            "reg": float(parts["reg"]),
            "unrelated": float(parts["unrelated"]),
            "trajectory_term_evaluated": bool(objective_phase != "guidance"),
            "weighted_guide": float(active_loss_weights()[0] * parts["guide_core"]),
            "weighted_traj": float(active_loss_weights()[1] * parts["traj_core"]),
            "weighted_reg": float(REG_LAMBDA * parts["reg"]),
            "weighted_unrelated": float(UNRELATED_LOSS_WEIGHT * parts["unrelated"]),
            "weighted_guide_tgt": float(active_loss_weights()[0] * parts["guide_tgt_fraction"] * parts["guide_tgt"]),
            "weighted_guide_orig": float(active_loss_weights()[0] * parts["guide_orig_fraction"] * parts["guide_orig"]),
            "guide_loss_weight": float(active_loss_weights()[0]),
            "trajectory_loss_weight": float(active_loss_weights()[1]),
            "original_weight": float(ORIGINAL_WEIGHT),
            "guide_tgt_fraction": float(parts["guide_tgt_fraction"]),
            "guide_orig_fraction": float(parts["guide_orig_fraction"]),
            "te_used_in_learning": False,
        }

        if objective_improved:
            objective_history_dir = _unique_best_dir(out_dir, "objective", step)
            best_objective["history_dir"] = str(objective_history_dir)
            np.savez_compressed(objective_history_dir / "best_delta_std.npz",
                                **{k: np.asarray(jax.device_get(v), dtype=np.float32) for k, v in delta.items()})
            with open(objective_history_dir / "best_event.json", "w") as f:
                json.dump({"type": "objective", "epoch": int(step) + 1, "step": int(step),
                           "loss": loss_f, "loss_components": {k: float(v) for k, v in parts.items()},
                           "stage": stage_meta}, f, indent=2)
            # Full epoch-specific visualization is saved by the real TE check below,
            # after the prediction tracks for this exact state are available.
            _append_best_manifest(out_dir, {**stage_meta, "type": "objective", "epoch": int(step) + 1,
                                            "step": int(step), "loss": loss_f,
                                            "directory": str(objective_history_dir)})

        do_te = (((step + 1) % TE_EVERY == 0) or (step == END2END_STEPS - 1)
                 or (objective_improved and OBJECTIVE_BEST_TE_EVAL))
        if do_te or step % END2END_VERBOSE_EVERY == 0:
            row.update(vortex_diagnostics(
                np.asarray(s_adv), ctx["clean_s_np"], ctx["guide_np"], geo, te_reference, sample
            ))

        if do_te:
            te = te_check(
                standardized_to_raw(s_adv),
                sample,
                te_reference,
                out_dir / "te_checks" / f"step_{step:04d}",
                ctx=ctx,
                delta=delta,
                s_adv_std=s_adv,
                gc_output=gc_output_adv,
                best_result=best_te,
                step=step,
                logits=logits,
                stage_meta=stage_meta,
                objective_history_dir=objective_history_dir if objective_improved else None,
            )
            te_row = _te_row(te, step, "attack")
            te_log.append(te_row)
            row.update({f"te_{k}": v for k, v in te_row.items() if k not in ("step", "kind")})
            if te["is_new_te_best"]:
                best_te = {**te, "delta": snapshot(delta), "step": int(step)}
            print(
                f"[{sample} {target_tag}] step={step:04d} TE "
                f"DR_changed={te.get('DR_changed', np.nan):.3f} "
                f"mean_target_overlap={te.get('mean_target_overlap', np.nan):.3f} "
                f"remove_success={te.get('remove_success', np.nan):.3f} "
                f"joint_score={te.get('te_joint_score', np.nan):.3f} "
                f"tracks={te.get('n_pred_tracks', 0)} "
                f"(TE-best @ {best_te['step']}; objective-best @ {best_objective['step']})"
            )

        history.append(row)

        saved_optimizer_state = adam_state
        if OPTIMIZER == "adam":
            candidate, adam_state, lr, grad_mean_abs = update_delta_adam(
                delta, grads, adam_state, step, geo["w_grad_family_hw"], perturb_masks
            )
        else:
            candidate, lr, grad_mean_abs = update_delta_pgd(
                delta, grads, geo["w_grad_family_hw"], perturb_masks
            )
        delta_next = project_delta(ctx, candidate)
        accepted_scale = 1.0
        if ADAPTIVE_STEP_REJECTION:
            old_adam = saved_optimizer_state
            for attempt in range(5):
                candidate_loss = float(np.asarray(combined_objective(ctx, delta_next)[0]))
                if np.isfinite(candidate_loss) and candidate_loss <= loss_f * 1.05:
                    break
                accepted_scale *= 0.5
                delta_next = project_delta(ctx, {k: delta[k] + accepted_scale * (candidate[k] - delta[k]) for k in delta})
            else:
                delta_next = delta
                if OPTIMIZER == "adam":
                    adam_state = old_adam
                accepted_scale = 0.0
        history[-1]["accepted_step_scale"] = accepted_scale
        history[-1]["update_rms"] = float(np.sqrt(np.mean([float(np.asarray(jnp.mean((delta_next[k] - delta[k]) ** 2))) for k in delta])))
        history[-1]["grad_update_dot"] = float(sum(float(np.asarray(jnp.sum(grads[k] * (delta_next[k] - delta[k])))) for k in delta))
        delta = delta_next
        current_linf, current_stats = delta_statistics(ctx, delta)
        history[-1]["lr"] = float(lr)
        history[-1]["delta_linf"] = current_linf
        for var, value in grad_mean_abs.items():
            history[-1][f"grad_mean_abs__{var}"] = value
        for var, stats in current_stats.items():
            history[-1][f"delta_linf__{var}"] = stats["linf"]
            history[-1][f"delta_rms__{var}"] = stats["rms"]

        if step % END2END_VERBOSE_EVERY == 0:
            print(
                f"[{sample} {target_tag}] step={step:04d} lr={lr:.5f} "
                f"total={loss_f:.4e} guide={float(parts['guide_core']):.4e} "
                f"traj={float(parts['traj_core']):.4e} reg={float(parts['reg']):.4e} "
                f"unrelated={float(parts['unrelated']):.4e} phase={objective_phase} "
                f"ctr_dist={row.get('center_dist_mean_deg', np.nan):.2f}deg "
                f"delta_linf={current_linf:.3f}"
            )

    # The loop evaluates states before each optimizer update. Evaluate the last UPDATED
    # state explicitly so the 250th update is eligible for both best selections.
    final_step = END2END_STEPS
    final_loss, (final_parts, _, final_logits, final_s, _, final_gc) = combined_objective(ctx, delta)
    final_loss_f = float(np.asarray(final_loss))
    final_last_loss_row = {
        "sample": sample, "target_tag": target_tag, "stage_index": int(stage_index),
        "stage_kind": str(stage_kind), "input_epsilon": float(END2END_EPSILON),
        "guide_delta": float(DOWNSTREAM_EPSILON), "selection": "last",
        "selected_step": int(final_step), "selected_epoch": int(final_step) + 1,
        "updates_applied": int(END2END_STEPS), "total_loss": final_loss_f,
        **{k: float(np.asarray(v)) for k, v in final_parts.items()},
        "weighted_guide_tgt": float(active_loss_weights()[0] * final_parts["guide_tgt_fraction"] * final_parts["guide_tgt"]),
        "weighted_guide_orig": float(active_loss_weights()[0] * final_parts["guide_orig_fraction"] * final_parts["guide_orig"]),
        "weighted_guide": float(active_loss_weights()[0] * final_parts["guide_core"]),
        "weighted_traj": float(active_loss_weights()[1] * final_parts["traj_core"]),
        "weighted_reg": float(REG_LAMBDA * final_parts["reg"]),
        "weighted_unrelated": float(UNRELATED_LOSS_WEIGHT * final_parts["unrelated"]),
        "optimizer": OPTIMIZER, "objective_phase": objective_phase, "checkpoint_path": "",  # Last state is recorded by loss, not selected as a best checkpoint.
    }
    final_last_loss_row["loss_identity_error"] = abs(final_last_loss_row["total_loss"] -
        final_last_loss_row["weighted_guide"] - final_last_loss_row["weighted_traj"] - final_last_loss_row["weighted_reg"] - final_last_loss_row["weighted_unrelated"])
    final_delta_path = out_dir / "last_delta_std.npz"
    np.savez_compressed(final_delta_path, **{k: np.asarray(jax.device_get(v), dtype=np.float32) for k, v in delta.items()})
    final_last_loss_row["checkpoint_path"] = str(final_delta_path)
    pd.DataFrame([final_last_loss_row]).to_csv(out_dir / "last_checkpoint_losses.csv", index=False)
    best_objective_loss_before_final = best_objective["loss"]
    if final_loss_f < best_objective_loss_before_final:
        best_objective = {"loss": final_loss_f, "step": int(final_step), "delta": snapshot(delta)}
        event_dir = _unique_best_dir(out_dir, "objective", final_step)
        best_objective["history_dir"] = str(event_dir)
        np.savez_compressed(event_dir / "best_delta_std.npz", **{k: np.asarray(v) for k, v in delta.items()})
        (event_dir / "best_event.json").write_text(json.dumps({"type": "objective", "step": final_step,
            "epoch": final_step + 1, "loss": final_loss_f,
            "loss_components": {k: float(v) for k, v in final_parts.items()},
            "post_final_update": True, "stage": stage_meta}, indent=2))
        # Real TE post-final check below saves the full state and figures for
        # this Objective Best, just as for every intermediate Objective Best.
        _append_best_manifest(out_dir, {**stage_meta, "type": "objective", "epoch": final_step + 1,
                                        "step": final_step, "loss": final_loss_f, "directory": str(event_dir)})
    te = te_check(standardized_to_raw(final_s), sample, te_reference,
                  out_dir / "te_checks" / "post_final_update", ctx=ctx, delta=delta,
                  s_adv_std=final_s, gc_output=final_gc, best_result=best_te,
                  step=final_step, logits=final_logits, stage_meta=stage_meta,
                  objective_history_dir=event_dir if final_loss_f < best_objective_loss_before_final else None)
    te_log.append(_te_row(te, final_step, "post_final_update"))
    if te["is_new_te_best"]:
        best_te = {**te, "delta": snapshot(delta), "step": int(final_step)}

    for index, row in enumerate(history):
        after = history[index + 1] if index + 1 < len(history) else {
            "total": final_loss_f,
            "guide_tgt": float(final_parts["guide_tgt"]),
            "guide_orig": float(final_parts["guide_orig"]),
            "guide_core": float(final_parts["guide_core"]),
            "traj_core": float(final_parts["traj_core"]),
            "reg": float(final_parts["reg"]),
            "unrelated": float(final_parts["unrelated"]),
        }
        row["total_after_update"] = float(after["total"])
        row["loss_change"] = float(after["total"] - row["total"])
        for part in ("guide_tgt", "guide_orig", "guide_core", "traj_core", "reg", "unrelated"):
            row[f"{part}_after_update"] = float(after[part])

    pd.DataFrame(history).to_csv(out_dir / "loss_history.csv", index=False)
    pd.DataFrame(te_log).to_csv(out_dir / "te_check_log.csv", index=False)
    save_loss_components(history, out_dir, best_objective["step"], best_te["step"])
    pd.DataFrame(history)[[c for c in ("step", "epoch", "total", "lr", "delta_linf",
                                     "update_rms", "grad_update_dot", "accepted_step_scale",
                                     "total_after_update", "loss_change",
                                     "guide_tgt_after_update", "guide_orig_after_update",
                                     "guide_core_after_update", "traj_core_after_update", "reg_after_update", "unrelated_after_update") if c in pd.DataFrame(history)]].to_csv(
                                         out_dir / "loss_update_diagnostics.csv", index=False)
    with open(out_dir / "te_saved_check_manifest.json", "w") as f:
        json.dump({
            "policy": "persistent all TE improvements with compatibility alias for latest best",
            "TE_used_in_learning": False,
            "metric": (
                "remove_success then lower mean_removed_original_overlap"
                if te_reference.get("target_mode") == "remove_changed_original"
                else "weighted_mean(target_overlap, 1-original_overlap), tie: weighted_joint_success"
            ),
            "time_matching": "exact forecast time",
            "spatial_tolerance_deg": float(TE_TOL_DEG),
            "near_original_disambiguation": False,
            "best_step": int(best_te["step"]),
            "best_primary_metric": float(best_te["primary_best_metric"]),
            "best_secondary_metric": float(best_te["secondary_best_metric"]),
            "rolling_state_directory": "te_saved_states/te_best",
            "best_history_directory": "best_history/te",
            "rolling_visualization": "te_vis/te_best.png",
        }, f, indent=2)

    if not np.isfinite(best_objective["loss"]):
        raise RuntimeError("No differentiable objective checkpoint was recorded.")

    clean_raw = standardized_to_raw(ctx["clean_s_np"]).numpy()
    guide_raw = ctx["guide_np"] * DOWNSTREAM_STD.numpy()[None, :, None, None]

    def save_solution_bundle(selection_name, delta_state, selected_step, selection_metadata):
        bundle_dir = out_dir / selection_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        delta_state = project_delta(ctx, delta_state)
        adv_ds_selected = build_adv_ds(ctx, delta_state)
        adv_ds_selected.to_netcdf(bundle_dir / "adv_inputs_xr_attacked.nc")
        np.savez_compressed(
            bundle_dir / "best_delta_std.npz",
            **{k: np.asarray(jax.device_get(v), dtype=np.float32) for k, v in delta_state.items()},
        )

        selected_s, selected_downstream, selected_gc_output = ctx["pipeline"](adv_ds_selected)
        selected_tensor = torch.from_numpy(
            np.asarray(downstream_dict_to_tensor(selected_downstream)).copy()
        ).float()
        selected_tensor[:, CH_WIND].clamp_min_(0.0)
        torch.save(selected_tensor, bundle_dir / "final_downstream_input.pt")
        save_gc_output(selected_gc_output, sample, bundle_dir / "final_graphcast_output.nc")

        selected_logits = surrogate_bridge(selected_s)
        selected_pred = torch.from_numpy(np.asarray(logits_to_mask_jax(selected_logits))).to(torch.uint8)
        torch.save(selected_pred, bundle_dir / "final_surrogate_pred_mask.pt")
        np.save(bundle_dir / "final_surrogate_logits.npy", np.asarray(selected_logits))
        # Mandatory final, checkpoint-specific TE replay for BOTH best bundles.
        # Never copy a previous epoch's TE track map into a selected checkpoint.
        te_work = bundle_dir / "final_te_replay"
        te_work.mkdir(parents=True, exist_ok=True)
        replay_nc = te_work / "selected_downstream_te.nc"
        replay_nodes = bundle_dir / "te_nodes.txt"
        replay_tracks = bundle_dir / "te_tracks.txt"
        _te_downstream_to_nc(selected_tensor, sample, replay_nc)
        _te_run(replay_nc, replay_nodes, replay_tracks)
        replay_nc.unlink(missing_ok=True)
        selected_tracks = read_track_file(replay_tracks) if replay_tracks.exists() and replay_tracks.stat().st_size else []
        replay_start = forecast_start_of(sample)
        replay_candidates = _read_detected_nodes(replay_nodes, replay_start)
        replay_result = score_tracks_direct(selected_tracks, te_reference, candidates=replay_candidates)
        replay_result["n_detected_candidates"] = int(len(replay_candidates))
        replay_result["n_pred_tracks"] = int(len(selected_tracks))
        replay_metrics = {
            "selection": selection_name,
            "selected_step": int(selected_step),
            "selected_epoch": int(selected_step) + 1,
            "source_checkpoint": str(bundle_dir / "best_delta_std.npz"),
            "recomputed_from_exact_selected_checkpoint": True,
            **{k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
               for k, v in replay_result.items() if k not in ("matched_tracks",)},
        }
        (bundle_dir / "te_metrics.json").write_text(json.dumps(replay_metrics, indent=2, default=str))
        plot_te_check(te_reference, selected_tracks, replay_candidates, replay_result,
                      replay_start, bundle_dir / "te_track_map.png",
                      f"Final TE replay | {selection_name} | epoch {selected_step + 1} | {sample} | {target_tag}")
        (bundle_dir / "final_te_trajectory_times.txt").write_text(
            "\n".join(_end2end_track_time_lines(
                selected_tracks, replay_start, f"{selection_name}_final_replay")) + "\n"
        )
        shutil.rmtree(te_work, ignore_errors=True)
        print(f"[final_te_replay] {selection_name} epoch={selected_step + 1} "
              f"tracks={len(selected_tracks)} map={bundle_dir / 'te_track_map.png'}")
        _save_best_probability(selected_logits, te_reference, bundle_dir,
                               f"{selection_name} | epoch {selected_step + 1} | {sample} | {target_tag}",
                               pred_tracks=selected_tracks, full_array=True)

        # Recompute ALL differentiable terms on the exact selected checkpoint, including epoch 0.
        selected_loss, (selected_parts, _, _, _, _, _) = combined_objective(ctx, delta_state)
        selected_parts = {k: float(np.asarray(v)) for k, v in selected_parts.items()}
        selected_loss_row = {
            "sample": sample, "target_tag": target_tag, "stage_index": int(stage_index),
            "stage_kind": str(stage_kind), "input_epsilon": float(END2END_EPSILON),
            "guide_delta": float(DOWNSTREAM_EPSILON), "selection": selection_name,
            "selected_step": int(selected_step), "selected_epoch": int(selected_step) + 1,
            "total_loss": float(np.asarray(selected_loss)),
            **selected_parts,
            "weighted_guide_tgt": float(active_loss_weights()[0] * selected_parts["guide_tgt_fraction"] * selected_parts["guide_tgt"]),
            "weighted_guide_orig": float(active_loss_weights()[0] * selected_parts["guide_orig_fraction"] * selected_parts["guide_orig"]),
            "weighted_guide": float(active_loss_weights()[0] * selected_parts["guide_core"]),
            "weighted_traj": float(active_loss_weights()[1] * selected_parts["traj_core"]),
            "weighted_reg": float(REG_LAMBDA * selected_parts["reg"]),
            "weighted_unrelated": float(UNRELATED_LOSS_WEIGHT * selected_parts["unrelated"]),
            "optimizer": OPTIMIZER, "objective_phase": objective_phase, "checkpoint_path": str(bundle_dir / "best_delta_std.npz"),
        }
        selected_loss_row["loss_identity_error"] = abs(selected_loss_row["total_loss"] -
            selected_loss_row["weighted_guide"] - selected_loss_row["weighted_traj"] - selected_loss_row["weighted_reg"] - selected_loss_row["weighted_unrelated"])
        pd.DataFrame([selected_loss_row]).to_csv(bundle_dir / "selected_checkpoint_losses.csv", index=False)
        save_forecast_step_diagnostics(ctx, selected_s, bundle_dir, te_reference, pred_tracks=selected_tracks)
        save_unrelated_diagnostics(ctx, selected_s, delta_state, bundle_dir)
        selected_linf, selected_stats = delta_statistics(ctx, delta_state)
        plot_target_focused_summary(history, clean_raw, guide_raw, selected_tensor.numpy(), bundle_dir)
        if TE_SAVE_ATTACK_DIFF_VIS:
            try:
                plot_input_delta_check(
                    ctx, delta_state, te_reference,
                    bundle_dir / "adversarial_x_minus_original_x.png",
                    f"{sample} | {target_tag} | {selection_name}",
                )
                plot_downstream_guidance_difference(
                    ctx, selected_s, te_reference,
                    bundle_dir / "adversarial_downstream_vs_guidance.png",
                    f"{sample} | {target_tag} | {selection_name}",
                )
                plot_input_delta_check(
                    ctx, delta_state, te_reference,
                    bundle_dir / "adversarial_x_minus_original_x_std.png",
                    f"{sample} | {target_tag} | {selection_name}", standardized=True,
                )
                plot_downstream_guidance_difference(
                    ctx, selected_s, te_reference,
                    bundle_dir / "adversarial_downstream_vs_guidance_std.png",
                    f"{sample} | {target_tag} | {selection_name}", standardized=True,
                )
            except Exception as exc:
                print(f"[bundle_vis] warning: {selection_name}: {exc!r}")
                plt.close("all")
        with open(bundle_dir / "selection_summary.json", "w") as f:
            json.dump({
                "selection_name": selection_name,
                "selected_step": int(selected_step),
                "selected_epoch": int(selected_step) + 1,
                "selection_metadata": selection_metadata,
                "delta_linf": float(selected_linf),
                "delta_stats_by_variable": selected_stats,
                "wind_min_raw": float(selected_tensor[:, CH_WIND].min()),
                "TE_used_in_learning": False,
            }, f, indent=2)
        return {
            "dir": bundle_dir,
            "linf": float(selected_linf),
            "stats": selected_stats,
            "wind_min_raw": float(selected_tensor[:, CH_WIND].min()),
            "downstream_tensor": selected_tensor,
            "loss_row": selected_loss_row,
        }

    objective_bundle = save_solution_bundle(
        "objective_best",
        best_objective["delta"],
        best_objective["step"],
        {
            "policy": f"minimum {objective_phase}-only objective plus spatial protection terms",
            "loss": float(best_objective["loss"]),
        },
    )
    te_metadata = {
        "policy": (
            "direct downstream remove metric"
            if te_reference.get("target_mode") == "remove_changed_original"
            else "joint TE score: target overlap + original removal, equal default weight"
        ),
        "DR_changed": float(best_te.get("DR_changed", np.nan)),
        "mean_target_overlap": float(best_te.get("mean_target_overlap", np.nan)),
        "remove_success": float(best_te.get("remove_success", np.nan)),
        "mean_removed_original_overlap": float(best_te.get("mean_removed_original_overlap", np.nan)),
        "te_generation_score": float(best_te.get("te_generation_score", np.nan)),
        "te_removal_score": float(best_te.get("te_removal_score", np.nan)),
        "te_joint_score": float(best_te.get("te_joint_score", np.nan)),
        "te_joint_success": float(best_te.get("te_joint_success", np.nan)),
        "te_generation_weight": float(TE_GENERATION_WEIGHT),
        "target_success": float(best_te.get("target_success", np.nan)),
    }
    te_bundle = save_solution_bundle(
        "te_best", best_te["delta"], best_te["step"], te_metadata,
    )
    te_loss_row = {**te_bundle["loss_row"],
                   "DR_changed": float(best_te.get("DR_changed", np.nan)),
                   "mean_target_overlap": float(best_te.get("mean_target_overlap", np.nan)),
                   "remove_success": float(best_te.get("remove_success", np.nan)),
                   "mean_removed_original_overlap": float(best_te.get("mean_removed_original_overlap", np.nan)),
                   "te_generation_score": float(best_te.get("te_generation_score", np.nan)),
                   "te_removal_score": float(best_te.get("te_removal_score", np.nan)),
                   "te_joint_score": float(best_te.get("te_joint_score", np.nan)),
                   "te_joint_success": float(best_te.get("te_joint_success", np.nan))}
    pd.DataFrame([te_loss_row]).to_csv(out_dir / "te_best_loss_summary.csv", index=False)

    stage_colorbars = _render_stage_colorbars(out_dir, ctx, te_reference,
                                             objective_bundle["dir"], te_bundle["dir"])
    if best_te.get("history_dir"):
        _update_legacy_te_artifacts(out_dir, best_te["history_dir"])

    # Root outputs remain TE best for reporting; warmup continuation uses Objective best.
    for name in [
        "adv_inputs_xr_attacked.nc", "best_delta_std.npz", "final_downstream_input.pt",
        "final_graphcast_output.nc", "final_surrogate_pred_mask.pt", "final_surrogate_logits.npy",
        "surrogate_probability.npy", "surrogate_probability_heatmap.png",
        "loss_history.png", "guide_vs_induced.png",
        "adversarial_x_minus_original_x.png", "adversarial_downstream_vs_guidance.png",
        "adversarial_x_minus_original_x_std.png", "adversarial_downstream_vs_guidance_std.png",
    ]:
        src = te_bundle["dir"] / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    with open(out_dir / "selection_manifest.json", "w") as f:
        json.dump({
            "stage": stage_meta,
            "learning_checkpoint": "objective_best",
            "objective_phase": objective_phase,
            "warm_start_checkpoint": "objective_best",
            "root_output_checkpoint": "te_best",
            "TE_used_in_learning": False,
            "objective_best": {
                "directory": str(objective_bundle["dir"]),
                "selected_step": int(best_objective["step"]),
                "selected_epoch": int(best_objective["step"]) + 1,
                "loss": float(best_objective["loss"]),
            },
            "te_best": {
                "directory": str(te_bundle["dir"]),
                "selected_step": int(best_te["step"]),
                "selected_epoch": int(best_te["step"]) + 1,
                **te_metadata,
            },
        }, f, indent=2)

    with open(out_dir / "region_diagnostics.json", "w") as f:
        json.dump({
            "method_hyperparameters": METHOD_HYPERPARAMETERS,
            "guide_info": ctx["guide_info"],
            **ctx["geo_diag"],
            "trajectory_loss": {
                "target_radius_cells": int(TRAJECTORY_TARGET_RADIUS),
                "target_sigma_cells": float(TRAJECTORY_TARGET_SIGMA),
                "focal_gamma": float(TRAJECTORY_FOCAL_GAMMA),
                "zero_original_only_after_dilation": bool(TRAJECTORY_ZERO_ORIGINAL_ONLY_AFTER_DILATION),
                "surrogate_checkpoint": str(SURROGATE_CKPT),
            },
        }, f, indent=2)
    with open(out_dir / "attackable_input_mask_summary.json", "w") as f:
        json.dump(ctx["mask_summary"], f, indent=2)

    with open(out_dir / "attack_summary.json", "w") as f:
        json.dump({
            "sample": sample,
            "run_tag": RUN_TAG,
            "strategy": "sequential_guidance_then_trajectory",
            "objective_phase": objective_phase,
            "target_id": get_target_id(target_tag),
            "target_tag": target_tag,
            "experiment_tag": end2end_experiment_tag(target_tag, DOWNSTREAM_EPSILON, END2END_EPSILON),
            "guide_delta": float(DOWNSTREAM_EPSILON),
            "downstream_delta_tag": downstream_delta_tag(DOWNSTREAM_EPSILON),
            "end2end_epsilon": float(END2END_EPSILON),
            "end2end_epsilon_tag": end2end_epsilon_tag(END2END_EPSILON),
            "steps_this_stage": int(END2END_STEPS),
            "warm_start_from": warm_start_description,
            "warm_start_selection": "objective_best",
            "optimizer": OPTIMIZER,
            "guidance_time_weighting": "equal_over_each_component_valid_TC_steps",
            "guide_combine": "normalized_weighted_mean_target_1_original_original_weight",
            "root_output_selection": "te_best",
            "variant": END2END_VARIANT,
            "attack_config_fingerprint": ATTACK_CONFIG_FINGERPRINT,
            "method_hyperparameters": METHOD_HYPERPARAMETERS,
            "guide_source": GUIDE_SOURCE,
            "loss_design": {
                "guide_loss_weight": float(GUIDE_LOSS_WEIGHT),
                "trajectory_loss_weight": float(TRAJECTORY_LOSS_WEIGHT),
                "target_weight": 1.0,
                "original_weight": float(ORIGINAL_WEIGHT),
                "unrelated_loss": True,
                "unrelated_loss_weight": float(UNRELATED_LOSS_WEIGHT),
                "input_mask_radius_deg": float(INPUT_MASK_RADIUS_DEG),
                "unrelated_exclusion_radius_deg": float(UNRELATED_EXCLUSION_RADIUS_DEG),
                "reg_lambda": float(REG_LAMBDA),
                "channel_weights": dict(CHANNEL_WEIGHTS),
                "trajectory_loss": "exact downstream-style global dilated calibrated focal loss through pretrained surrogate",
                "full_forecast_horizon_from_first_step": True,
            },
            "TE_used_in_learning": False,
            "TE_role": "periodic real TempestExtremes evaluation and persistent best-history only",
            "stage": stage_meta,
            "best_history": {"te": "best_history/te", "objective": "best_history/objective",
                             "event_manifest": "best_manifest.csv",
                             "loss_components": "loss_components",
                             "fixed_colorbar_file": "downstream_colorbar_scale.json"},
            "te_metric_definition": {
                "matching": "direct downstream trajectory overlap",
                "time_matching": "exact forecast time",
                "spatial_tolerance_deg": float(TE_TOL_DEG),
                "target_track_success_threshold": 0.5,
                "near_target_or_original_disambiguation": False,
                "relocation_best_policy": "weighted_mean(target_overlap, 1-original_overlap)",
                "te_generation_weight": float(TE_GENERATION_WEIGHT),
                "objective_best_extra_te_evaluation": bool(OBJECTIVE_BEST_TE_EVAL),
            },
            "te_best": {
                "selected_step": int(best_te["step"]),
                "selected_epoch": int(best_te["step"]) + 1,
                **te_metadata,
            },
            "objective_best": {
                "selected_step": int(best_objective["step"]),
                "selected_epoch": int(best_objective["step"]) + 1,
                "loss": float(best_objective["loss"]),
            },
            "perturb_input_vars": list(PERTURB_INPUT_VARS),
            "perturb_levels": {k: list(v) for k, v in PERTURB_LEVELS.items()},
            "gradient_dilation_vars": list(GRADIENT_DILATION_VARS),
            "shared_distance_weighting_vars": list(GRADIENT_DILATION_VARS),
            "dilation_scale": dict(DILATION_SCALE_MAP),
            "pgd_step_size": float(PGD_STEP_SIZE),
            "optimizer_update": "sign_pgd_no_momentum" if OPTIMIZER == "pgd" else "projected_adam",
            "nonnegative_input_vars": sorted(NONNEGATIVE_INPUT_VARS),
            "te_visualizations": {
                "save_policy": "append each strict joint-score improvement; te_best is latest compatibility alias",
                "track_map": "te_vis/te_best.png",
                "upstream_input_difference": "te_vis/te_best_adversarial_x_minus_original_x.png",
                "downstream_guidance_difference": "te_vis/te_best_adversarial_downstream_vs_guidance.png",
            },
            "te_saved_state": "te_saved_states/te_best/",
            "objective_best_history_artifacts": "best_history/objective/epoch_*/ (full TE track map, original/std input and downstream figures, full state and probability)",
        }, f, indent=2)

    print(
        f"[{sample} {target_tag}] objective_best @ {best_objective['step']}; "
        f"TE-best @ {best_te['step']}; TE feedback into learning = OFF"
    )
    return te_bundle["downstream_tensor"]

