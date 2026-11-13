# Stage 07: OPTIONAL raw input-gradient sensitivity diagnostic for the fixed combined objective.
#
# One backward pass is evaluated at delta=0 for the configured attack variables.
# The reported gradients are taken before per-variable optimizer normalization.


def run_gradient_saliency_diagnostic(sample, target_tag, out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else (
        sample_target_dir(sample, target_tag) / "grad_saliency"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = prepare_case_context(sample, target_tag)
    delta0 = {var: jnp.zeros_like(ctx["base_std"][var]) for var in PERTURB_INPUT_VARS}
    grads = jax.grad(lambda d: combined_objective(ctx, d)[0])(delta0)

    rows = []
    for var in PERTURB_INPUT_VARS:
        g = np.asarray(grads[var])
        da = ctx["base_ds"][var]
        mask = np.asarray(ctx["perturb_masks"][var]) > 0.5
        if "level" in da.dims:
            axis = da.dims.index("level")
            levels = np.asarray(da.coords["level"].values, dtype=float)
            for i, lev in enumerate(levels):
                gi = np.take(g, i, axis=axis)
                mi = np.take(mask, i, axis=axis)
                active = gi[mi]
                rows.append({
                    "var": var,
                    "level_hpa": float(lev),
                    "rms_grad": float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0,
                    "mean_abs_grad": float(np.mean(np.abs(active))) if active.size else 0.0,
                    "max_abs_grad": float(np.max(np.abs(active))) if active.size else 0.0,
                })
        else:
            active = g[mask]
            rows.append({
                "var": var,
                "level_hpa": None,
                "rms_grad": float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0,
                "mean_abs_grad": float(np.mean(np.abs(active))) if active.size else 0.0,
                "max_abs_grad": float(np.max(np.abs(active))) if active.size else 0.0,
            })

    sal = pd.DataFrame(rows).sort_values("rms_grad", ascending=False).reset_index(drop=True)
    sal_path = out_dir / "grad_saliency_combined_objective.csv"
    sal.to_csv(sal_path, index=False)
    with open(out_dir / "grad_saliency_combined_objective.json", "w") as f:
        json.dump({
            "objective": "fixed guide + differentiable downstream trajectory + input regularization",
            "TE_used_in_learning": False,
            "guide_delta": float(DOWNSTREAM_EPSILON),
            "input_epsilon": float(END2END_EPSILON),
            "guide_source": GUIDE_SOURCE,
            "guide_info": ctx["guide_info"],
            "note": "Gradients are raw backpropagated gradients before optional TI smoothing and optimizer updates.",
            **ctx["geo_diag"],
        }, f, indent=2)
    print(f"[{sample} {target_tag}] combined-objective raw gradient saliency -> {sal_path}")
    print(sal.head(20).to_string(index=False))
    return sal


for _sample in RUN_CASES:
    for _target_tag in TARGET_SCENARIOS:
        run_gradient_saliency_diagnostic(_sample, _target_tag)
