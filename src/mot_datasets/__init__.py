"""
mot_datasets -- Lop mo ta dataset (Giai doan 2).

Ten package la `mot_datasets` (khong phai `datasets`) de khong che khuat package
`datasets` cua HuggingFace neu duoc cai trong cung moi truong.

    from mot_datasets import get_dataset
    ds = get_dataset("detrac")        # UA-DETRAC (bao config.py cu, khong doi hanh vi)
    ds = get_dataset("bdd100k")       # BDD100K-MOT box_track_20
"""
from __future__ import annotations

from .base import DatasetSpec, FrameRef  # noqa: F401


def get_dataset(name: str, **kwargs) -> DatasetSpec:
    name = name.lower()
    if name in ("detrac", "ua-detrac", "ua_detrac"):
        from .detrac import DetracDataset
        return DetracDataset(**kwargs)
    if name in ("bdd100k", "bdd100k_mot", "bdd"):
        from .bdd100k_mot import BDD100KMOTDataset
        return BDD100KMOTDataset(**kwargs)
    if name in ("bdd100k_coco", "bdd_coco"):
        # Nhan BDD100K-MOT da CHUYEN DOI sang COCO-video (khong phai box_track_20 native).
        # Can `labels_json`; xem src/mot_datasets/bdd100k_coco.py.
        from .bdd100k_coco import BDD100KCocoDataset
        return BDD100KCocoDataset(**kwargs)
    raise ValueError(f"Dataset khong ho tro: {name!r} (detrac | bdd100k | bdd100k_coco)")
