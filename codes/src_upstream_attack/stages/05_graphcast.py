# Stage 05: Load GraphCast and the differentiable GraphCast -> TempestExtremes-channel pipeline

from google.cloud import storage
from graphcast import autoregressive, casting, checkpoint, data_utils, graphcast, normalization, xarray_jax, xarray_tree

GRAPHCAST_PARAMS_FILE = "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
GRAPHCAST_LOCAL_FILES = {
    f"params/{GRAPHCAST_PARAMS_FILE}": GRAPHCAST_ASSET_ROOT / "params" / GRAPHCAST_PARAMS_FILE,
    "stats/diffs_stddev_by_level.nc": GRAPHCAST_ASSET_ROOT / "stats" / "diffs_stddev_by_level.nc",
    "stats/mean_by_level.nc": GRAPHCAST_ASSET_ROOT / "stats" / "mean_by_level.nc",
    "stats/stddev_by_level.nc": GRAPHCAST_ASSET_ROOT / "stats" / "stddev_by_level.nc",
}


def ensure_graphcast_asset(blob_name, local_path):
    local_path = Path(local_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        print("Using cached GraphCast asset:", local_path)
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading GraphCast asset to permanent HOME cache:", blob_name, "->", local_path)
    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket("dm_graphcast")
    bucket.blob(blob_name).download_to_filename(str(local_path))
    return local_path


for blob_name, local_path in GRAPHCAST_LOCAL_FILES.items():
    ensure_graphcast_asset(blob_name, local_path)

params_path = GRAPHCAST_LOCAL_FILES[f"params/{GRAPHCAST_PARAMS_FILE}"]
print("Loading GraphCast checkpoint from HOME software cache:", params_path)
with open(params_path, "rb") as f:
    GC_CKPT = checkpoint.load(f, graphcast.CheckPoint)
GC_PARAMS, GC_STATE = GC_CKPT.params, {}
GC_MODEL_CONFIG, GC_TASK_CONFIG = GC_CKPT.model_config, GC_CKPT.task_config
GC_DIFF_STD = xr.load_dataset(GRAPHCAST_LOCAL_FILES["stats/diffs_stddev_by_level.nc"]).compute()
GC_MEAN = xr.load_dataset(GRAPHCAST_LOCAL_FILES["stats/mean_by_level.nc"]).compute()
GC_STD = xr.load_dataset(GRAPHCAST_LOCAL_FILES["stats/stddev_by_level.nc"]).compute()


def construct_wrapped_graphcast(model_config, task_config):
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor, diffs_stddev_by_level=GC_DIFF_STD,
        mean_by_level=GC_MEAN, stddev_by_level=GC_STD,
    )
    return autoregressive.Predictor(predictor, gradient_checkpointing=True)


@hk.transform_with_state
def gc_forward(model_config, task_config, inputs, targets_template, forcings):
    return construct_wrapped_graphcast(model_config, task_config)(
        inputs, targets_template=targets_template, forcings=forcings
    )


def _with_configs(fn):
    return functools.partial(fn, model_config=GC_MODEL_CONFIG, task_config=GC_TASK_CONFIG)


def _with_params(fn):
    return functools.partial(fn, params=GC_PARAMS, state=GC_STATE)


def _drop_state(fn):
    return lambda **kw: fn(**kw)[0]


GC_FORWARD = _drop_state(_with_params(jax.jit(_with_configs(gc_forward.apply))))


_GC_FORWARD_FP32 = None


def gc_forward_fp32_fn():
    """Optional float32 GraphCast forward (no bfloat16 cast) for cleaner gradients."""
    global _GC_FORWARD_FP32
    if _GC_FORWARD_FP32 is None:
        def construct(model_config, task_config):
            predictor = graphcast.GraphCast(model_config, task_config)
            predictor = normalization.InputsAndResiduals(
                predictor, diffs_stddev_by_level=GC_DIFF_STD,
                mean_by_level=GC_MEAN, stddev_by_level=GC_STD,
            )
            return autoregressive.Predictor(predictor, gradient_checkpointing=True)

        @hk.transform_with_state
        def fwd(model_config, task_config, inputs, targets_template, forcings):
            return construct(model_config, task_config)(
                inputs, targets_template=targets_template, forcings=forcings
            )

        _GC_FORWARD_FP32 = _drop_state(_with_params(jax.jit(_with_configs(fwd.apply))))
    return _GC_FORWARD_FP32


ATTACK_GC_FORWARD = gc_forward_fp32_fn() if FP32_GRAPHCAST else GC_FORWARD

# ---------------------------------------------------------------------------
# Differentiable TempestExtremes surrogate used only in the training loss.
# The real TempestExtremes binary remains outside the gradient path.
# ---------------------------------------------------------------------------
if str(SURROGATE_REPO) not in sys.path:
    sys.path.insert(0, str(SURROGATE_REPO))
import network.modeling


def fetch_surrogate_model():
    key = f"{SURROGATE_MODEL_NAME}_{SURROGATE_BACKBONE_NAME}"
    if key not in network.modeling.__dict__:
        raise KeyError(f"Surrogate architecture {key!r} is not available in {SURROGATE_REPO}")
    model = network.modeling.__dict__[key](
        num_classes=2, output_stride=SURROGATE_OUTPUT_STRIDE, pretrained_backbone=False
    )
    if hasattr(model.backbone, "conv1"):
        old = model.backbone.conv1
        if int(old.in_channels) != 4:
            new = nn.Conv2d(
                4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, dilation=old.dilation, groups=old.groups,
                bias=old.bias is not None,
            )
            with torch.no_grad():
                copy_channels = min(3, int(old.in_channels))
                new.weight[:, :copy_channels] = old.weight[:, :copy_channels]
                new.weight[:, 3:4] = old.weight[:, :1]
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            model.backbone.conv1 = new
    state = torch.load(SURROGATE_CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("backbone.") for k in state):
        state = state["state_dict"]
    if isinstance(state, dict) and state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


SURROGATE = fetch_surrogate_model()
print("Loaded downstream trajectory surrogate:", SURROGATE_CKPT)


def _torch_from_jax(x):
    return torch.utils.dlpack.from_dlpack(x)


def _jax_from_torch(x):
    return jax.dlpack.from_dlpack(x)


@jax.custom_vjp
def surrogate_bridge(x_jax):
    x_torch = _torch_from_jax(x_jax).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        y = SURROGATE(x_torch)
    return _jax_from_torch(y.detach())


def surrogate_bridge_fwd(x_jax):
    x_torch = _torch_from_jax(x_jax).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        y = SURROGATE(x_torch)
    return _jax_from_torch(y.detach()), x_jax


def surrogate_bridge_bwd(x_jax, g_jax):
    x_torch = _torch_from_jax(x_jax).to(DEVICE, dtype=torch.float32).detach().requires_grad_(True)
    y = SURROGATE(x_torch)
    g_torch = _torch_from_jax(g_jax).to(y.device, dtype=y.dtype)
    grad_x, = torch.autograd.grad(
        y, x_torch, grad_outputs=g_torch, retain_graph=False, create_graph=False
    )
    return (_jax_from_torch(grad_x.detach()),)


surrogate_bridge.defvjp(surrogate_bridge_fwd, surrogate_bridge_bwd)


ELEVATION_DS = xr.open_dataset(ELEVATION_NC)


def prepare_graphcast_case(sample):
    input_path = DATA_DIR / sample / f"graphcast_input_range_{sample}.nc"
    require_paths(input_path)
    full_ds = xr.open_dataset(input_path)
    ds = full_ds.copy()
    dt_vals = np.tile(ds.time.values[None, :], (ds.sizes["batch"], 1))
    ds = ds.assign_coords(datetime=(("batch", "time"), dt_vals))
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("6h", f"{FORECAST_STEPS * 6}h"),
        **dataclasses.asdict(GC_TASK_CONFIG),
    )
    return full_ds, eval_inputs, eval_targets, eval_forcings


def graphcast_output_to_downstream(graphcast_output):
    out = graphcast_output.squeeze("batch", drop=True)
    z300 = xarray_jax.unwrap_data(out["geopotential"].sel(level=300).transpose("time", "lat", "lon"))
    z500 = xarray_jax.unwrap_data(out["geopotential"].sel(level=500).transpose("time", "lat", "lon"))
    u10 = xarray_jax.unwrap_data(out["10m_u_component_of_wind"].transpose("time", "lat", "lon"))
    v10 = xarray_jax.unwrap_data(out["10m_v_component_of_wind"].transpose("time", "lat", "lon"))
    mslp = xarray_jax.unwrap_data(out["mean_sea_level_pressure"].transpose("time", "lat", "lon"))
    elev2d = ELEVATION_DS["geopotential_at_surface"].squeeze(
        dim=[d for d in ["batch", "time"] if d in ELEVATION_DS["geopotential_at_surface"].dims], drop=True
    ).transpose("lat", "lon")
    elev2d = elev2d.sel(lat=out["lat"].values, lon=out["lon"].values)
    wind = jnp.sqrt(u10**2 + v10**2)
    elevation = jnp.broadcast_to(jnp.asarray(elev2d.values.astype(np.float32))[None], wind.shape)
    return {
        "wind_speed_10m": wind, "elevation": elevation,
        "thickness": z300 - z500, "mean_sea_level_pressure": mslp,
    }


def downstream_dict_to_tensor(d):
    return jnp.stack([d[v] for v in DOWNSTREAM_VARS], axis=1)


def downstream_to_standardized(d):
    """[T, 4, H, W] TempestExtremes channels standardized with DOWNSTREAM_MEAN/STD."""
    x = downstream_dict_to_tensor(d)
    mean = jnp.asarray(DOWNSTREAM_MEAN.numpy()).reshape(1, 4, 1, 1)
    std = jnp.asarray(DOWNSTREAM_STD.numpy()).reshape(1, 4, 1, 1)
    return (x - mean) / std


def run_pipeline(inputs_xr, eval_targets, eval_forcings):
    gc_output = ATTACK_GC_FORWARD(
        rng=jax.random.PRNGKey(0), inputs=inputs_xr,
        targets_template=eval_targets * jnp.nan, forcings=eval_forcings,
    )
    downstream = graphcast_output_to_downstream(gc_output)
    return downstream_to_standardized(downstream), downstream, gc_output
