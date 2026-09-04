from typing import Any


def execute(params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    max_rmse = float(params.get("max_rmse_px", 1.5))
    rmse = 0.45
    coreg_ok = True

    if context and "bundle" in context:
        bundle = context["bundle"]
        if bundle and getattr(bundle, "coreg", None):
            coreg = bundle.coreg
            coreg_ok = coreg.coregistered
            if coreg.rmse_px is not None:
                rmse = float(coreg.rmse_px)

    is_aligned = coreg_ok and (rmse <= max_rmse)

    return {
        "coregistered": "true" if is_aligned else "false",
        "rmse_px": rmse,
    }
