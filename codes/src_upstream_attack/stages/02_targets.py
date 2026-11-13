# Stage 02: Load and validate SAVED target artifacts only.
# This stage NEVER constructs, modifies, rewrites, or regenerates targets.

from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
import torch


# Hard policy: targets are frozen and must already exist on disk.
BUILD_TARGETS = False

LATS = np.linspace(-90.0, 90.0, GRID_H)
LONS = np.linspace(0.0, 360.0, GRID_W, endpoint=False)


def read_track_file(path):
    path = Path(path)
    lines = path.read_text().splitlines()

    tracks = []
    i = 0

    while i < len(lines):
        parts = lines[i].split()
        i += 1

        if not parts:
            continue

        if parts[0] != "start" or len(parts) < 6:
            raise ValueError(f"Malformed track header in {path}: {parts}")

        length = int(parts[1])
        start_dt = datetime(
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
        )

        nodes = []

        for _ in range(length):
            if i >= len(lines):
                raise ValueError(f"Unexpected end of file in {path}")

            vals = lines[i].split()
            i += 1

            if len(vals) < 11:
                raise ValueError(f"Malformed track node in {path}: {vals}")

            node = {
                "grid_x": int(float(vals[0])),
                "grid_y": int(float(vals[1])),
                "lon": float(vals[2]),
                "lat": float(vals[3]),
                "slp": float(vals[4]),
                "wind": float(vals[5]),
                "zs": float(vals[6]),
                "year": int(vals[7]),
                "month": int(vals[8]),
                "day": int(vals[9]),
                "hour": int(vals[10]),
            }

            node["datetime"] = datetime(
                node["year"],
                node["month"],
                node["day"],
                node["hour"],
            )

            nodes.append(node)

        tracks.append({
            "start": start_dt,
            "nodes": nodes,
        })

    return tracks


def tracks_to_mask_per_step(
    tracks,
    sample_start_iso,
    steps=FORECAST_STEPS,
    H=GRID_H,
    W=GRID_W,
):
    mask = torch.zeros(
        (steps, H, W),
        dtype=torch.uint8,
    )

    forecast_start = (
        datetime.strptime(sample_start_iso, "%Y-%m-%dT%H%M")
        + timedelta(hours=6)
    )

    for track in tracks:
        for node in track["nodes"]:
            delta_hours = (
                node["datetime"] - forecast_start
            ).total_seconds() / 3600.0

            k = int(round(delta_hours / 6.0))

            if (
                0 <= k < steps
                and abs(delta_hours - 6.0 * k) < 1e-4
            ):
                x = int(np.clip(node["grid_x"], 0, W - 1))
                y = int(np.clip(node["grid_y"], 0, H - 1))
                mask[k, y, x] = 1

    return mask


def _plot_lon(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def validate_saved_target_metadata(
    metadata,
    metadata_path,
    sample,
    target_tag,
    num_original_tracks,
    num_target_tracks,
):
    """
    Validate only fields needed by later attack/evaluation code.

    IMPORTANT:
    We intentionally do NOT compare:
      - construction_version
      - target generator version
      - smoothing version
      - historical target_mode naming such as distance vs euclidean
      - newly computed fingerprints
      - newly configured ratio definitions

    The existing frozen target artifacts are the source of truth because the
    saved downstream adversarial results were produced against those artifacts.
    """

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid metadata object in {metadata_path}: "
            f"expected dict, got {type(metadata)}"
        )

    # Identity checks only when those fields are present in historical metadata.
    if "sample" in metadata and str(metadata["sample"]) != str(sample):
        raise ValueError(
            f"Saved target belongs to a different sample in {metadata_path}: "
            f"{metadata['sample']!r} != {sample!r}"
        )

    if "target_tag" in metadata and str(metadata["target_tag"]) != str(target_tag):
        raise ValueError(
            f"Saved target tag mismatch in {metadata_path}: "
            f"{metadata['target_tag']!r} != {target_tag!r}"
        )

    # These fields are required later by 09_evaluate.py.
    required_runtime_fields = [
        "target_mode",
        "changed_track_indices",
    ]

    missing = [
        key
        for key in required_runtime_fields
        if key not in metadata
    ]

    if missing:
        raise ValueError(
            f"Saved target metadata is missing fields required by later "
            f"evaluation code in {metadata_path}: {missing}"
        )

    changed_indices = [
        int(x)
        for x in metadata.get("changed_track_indices", [])
    ]

    if not changed_indices:
        raise ValueError(
            f"No changed_track_indices recorded in {metadata_path}"
        )

    for idx in changed_indices:
        if not 0 <= idx < num_original_tracks:
            raise ValueError(
                f"Invalid changed original-track index {idx} in {metadata_path}; "
                f"there are {num_original_tracks} original tracks."
            )

    source_indices = [
        int(x)
        for x in metadata.get(
            "target_track_source_indices",
            range(num_target_tracks),
        )
    ]

    if len(source_indices) != num_target_tracks:
        raise ValueError(
            f"target_track_source_indices length mismatch in {metadata_path}: "
            f"{len(source_indices)} != {num_target_tracks}"
        )

    for idx in source_indices:
        if not 0 <= idx < num_original_tracks:
            raise ValueError(
                f"Invalid target_track_source_indices entry {idx} in "
                f"{metadata_path}; there are {num_original_tracks} "
                f"original tracks."
            )

    removed_indices = [
        int(x)
        for x in metadata.get(
            "removed_original_track_indices",
            [],
        )
    ]

    for idx in removed_indices:
        if not 0 <= idx < num_original_tracks:
            raise ValueError(
                f"Invalid removed original-track index {idx} in "
                f"{metadata_path}; there are {num_original_tracks} "
                f"original tracks."
            )

    return {
        "target_mode": metadata["target_mode"],
        "changed_track_indices": changed_indices,
        "target_track_source_indices": source_indices,
        "removed_original_track_indices": removed_indices,
        "construction_version": metadata.get("construction_version"),
        "ratio": metadata.get("ratio"),
    }


print(
    "Loading SAVED frozen targets only. "
    "No target construction or target rewriting will be performed."
)

SAVED_TARGET_TRACKS = {}
SAVED_TARGET_MASKS = {}
SAVED_TARGET_METADATA = {}
SAVED_TARGET_MANIFEST = []

for sample in RUN_CASES:
    original_track_path = (
        DATA_DIR
        / sample
        / f"{PREFIX}_cyclone_tracks_{sample}.txt"
    )

    require_paths(original_track_path)

    original_tracks = read_track_file(original_track_path)

    for target_tag in TARGET_SCENARIOS:
        target_dir = (
            sample_target_dir(sample, target_tag)
            / "target"
        )

        track_path = (
            target_dir
            / f"target_tracks_{target_tag}.txt"
        )

        mask_path = (
            target_dir
            / f"target_mask_{target_tag}.pt"
        )

        metadata_path = (
            target_dir
            / "target_metadata.json"
        )

        # Strictly require the already-saved artifacts.
        # Missing artifacts are fatal; this stage never regenerates them.
        require_paths(
            track_path,
            mask_path,
            metadata_path,
        )

        target_tracks = read_track_file(track_path)

        target_mask = torch.load(
            mask_path,
            map_location="cpu",
        ).to(torch.uint8)

        if tuple(target_mask.shape) != (
            FORECAST_STEPS,
            GRID_H,
            GRID_W,
        ):
            raise ValueError(
                f"Unexpected saved target-mask shape in {mask_path}: "
                f"{tuple(target_mask.shape)}; expected "
                f"{(FORECAST_STEPS, GRID_H, GRID_W)}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata_summary = validate_saved_target_metadata(
            metadata=metadata,
            metadata_path=metadata_path,
            sample=sample,
            target_tag=target_tag,
            num_original_tracks=len(original_tracks),
            num_target_tracks=len(target_tracks),
        )

        key = (sample, target_tag)

        SAVED_TARGET_TRACKS[key] = target_tracks
        SAVED_TARGET_MASKS[key] = target_mask
        SAVED_TARGET_METADATA[key] = metadata

        SAVED_TARGET_MANIFEST.append({
            "sample": sample,
            "target_tag": target_tag,
            "target_mode_saved": metadata_summary["target_mode"],
            "ratio_saved": metadata_summary["ratio"],
            "construction_version_saved": (
                metadata_summary["construction_version"]
            ),
            "num_original_tracks": len(original_tracks),
            "num_target_tracks": len(target_tracks),
            "target_points": int(target_mask.sum()),
            "changed_track_indices": (
                metadata_summary["changed_track_indices"]
            ),
            "removed_original_track_indices": (
                metadata_summary["removed_original_track_indices"]
            ),
            "track_path": str(track_path),
            "mask_path": str(mask_path),
            "metadata_path": str(metadata_path),
        })

        print(
            f"  OK: {sample} / {target_tag} | "
            f"saved_mode={metadata_summary['target_mode']!r} | "
            f"saved_version="
            f"{metadata_summary['construction_version']!r} | "
            f"tracks={len(target_tracks)} | "
            f"target_points={int(target_mask.sum())}"
        )

print(
    f"Loaded {len(SAVED_TARGET_MANIFEST)} saved target sets. "
    "No target files were created, modified, or overwritten."
)
