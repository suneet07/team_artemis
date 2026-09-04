from satquery.agent.bundle import ImageRef
from satquery.agent.prompt import assemble_prompt
from training.collator import QGPCollator


def test_prompt_parity_bitemporal_fixture():
    img0 = ImageRef(scene_id="s0", path="p0.tif", modality="optical", native_gsd_m=4.0)
    img1 = ImageRef(scene_id="s1", path="p1.tif", modality="optical", native_gsd_m=4.0)

    prompt = assemble_prompt(
        images=[img0, img1],
        roles=["t0", "t1"],
        modalities=["optical", "optical"],
        effective_gsd_m=[4.0, 4.0],
        question="What changed between these two acquisitions?",
    )

    expected = (
        "<image>{t0}</image><image>{t1}</image>\n"
        "[sensor: optical | GSD: 4.0 m] [sensor: optical | GSD: 4.0 m]\n"
        "What changed between these two acquisitions?"
    )

    assert prompt == expected

    # Parity verification with training collator
    collator = QGPCollator()
    example = {
        "images": [
            {"scene_id": "s0", "path": "p0.tif", "modality": "optical", "native_gsd_m": 4.0},
            {"scene_id": "s1", "path": "p1.tif", "modality": "optical", "native_gsd_m": 4.0},
        ],
        "roles": ["t0", "t1"],
        "modalities": ["optical", "optical"],
        "effective_gsd_m": [4.0, 4.0],
        "question": "What changed between these two acquisitions?",
    }
    collated = collator.collate_example(example)
    assert collated["prompt"] == prompt
    assert collated["prompt"] == expected


def test_prompt_parity_with_point_prior():
    img0 = ImageRef(scene_id="s0", path="p0.tif", modality="optical", native_gsd_m=0.5)

    prompt = assemble_prompt(
        images=[img0],
        roles=["single"],
        modalities=["optical"],
        effective_gsd_m=[0.5],
        question="Where is the aircraft?",
        point_prior=(0.450, 0.625),
    )

    expected = (
        "<image>{single}</image>\n"
        "[sensor: optical | GSD: 0.5 m]\n"
        "[prior: (0.450, 0.625)] Where is the aircraft?"
    )

    assert prompt == expected

    # Parity verification with training collator
    collator = QGPCollator()
    example = {
        "images": [
            {"scene_id": "s0", "path": "p0.tif", "modality": "optical", "native_gsd_m": 0.5}
        ],
        "roles": ["single"],
        "modalities": ["optical"],
        "effective_gsd_m": [0.5],
        "question": "Where is the aircraft?",
        "point_prior": (0.450, 0.625),
    }
    collated = collator.collate_example(example)
    assert collated["prompt"] == prompt
    assert collated["prompt"] == expected
