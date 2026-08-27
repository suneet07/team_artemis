from typing import Any


def execute(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": f"dummy_tool[{params['index']}] processed at scale={params['scale']}",
        "area_km2": round(float(params["scale"]) * 10.0, 4),
    }
