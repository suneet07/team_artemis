from satquery.agent.bundle import ImageRef


def assemble_prompt(
    images: list[ImageRef],
    roles: list[str],  # ["t0","t1"] | ["opt","sar"] | ["single"]
    modalities: list[str],
    effective_gsd_m: list[float],
    question: str,
    point_prior: tuple[float, float] | None = None,
) -> str:
    """Assembles a model prompt strictly following the train/serve parity contract (§13).

    Byte-identical between training collation and inference prompt preparation.
    Format:
    <image>{role0}</image><image>{role1}</image>
    [sensor: mod0 | GSD: X.X m] [sensor: mod1 | GSD: Y.Y m]
    {question}
    """
    image_tags = "".join(f"<image>{{{role}}}</image>" for role in roles)
    sensor_tags = " ".join(
        f"[sensor: {mod} | GSD: {gsd:.1f} m]"
        for mod, gsd in zip(modalities, effective_gsd_m, strict=False)
    )

    if point_prior is not None:
        px, py = point_prior
        q_segment = f"[prior: ({px:.3f}, {py:.3f})] {question}"
    else:
        q_segment = question

    return f"{image_tags}\n{sensor_tags}\n{q_segment}"
