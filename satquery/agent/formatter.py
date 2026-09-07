"""Answer formatting utilities matching official evaluation scorer formats."""
from typing import Any


def format_change_answer(stats: dict[str, Any]) -> str:
    """Formats change_stats output into the official evaluation string."""
    target_class = stats.get("target_class", "all")
    area_before_raw = stats.get("area_before_km2")
    delta = float(stats.get("area_delta_km2", 0.0))
    ratio = float(stats.get("change_ratio", 0.0))

    delta_sign = "+" if delta >= 0 else ""
    pct = ratio * 100.0

    class_title = target_class.capitalize() if target_class != "all" else "Total"

    if area_before_raw is not None:
        area_before = float(area_before_raw)
        area_after = float(stats.get("area_after_km2", area_before + delta))
        parts = [
            f"Change detection analysis: {class_title} area changed from {area_before:.2f} km² to "
            f"{area_after:.2f} km² (delta: {delta_sign}{delta:.2f} km², change ratio: {pct:.1f}%)."
        ]
    else:
        parts = [
            f"Change detection analysis: {class_title} area changed by "
            f"{delta_sign}{delta:.2f} km² (change ratio: {pct:.1f}%, "
            "baseline area before change: unknown)."
        ]

    breakdown = stats.get("class_breakdown")
    if breakdown and isinstance(breakdown, dict):
        class_parts = []
        for c_name, c_delta in sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True):
            sign = "+" if c_delta >= 0 else ""
            class_parts.append(f"{c_name}: {sign}{c_delta:.2f} km²")
        if class_parts:
            parts.append(f"Breakdown by class: {', '.join(class_parts)}.")

    return " ".join(parts)
