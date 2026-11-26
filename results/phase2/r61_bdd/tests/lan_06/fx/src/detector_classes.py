"""
detector_classes.py -- Thu tu 5 lop detector da KHOA va anh xa lop giua checkpoint <-> dataset.

    0: car   1: truck   2: bus   3: motorcycle   4: bicycle

Hai loai checkpoint dung trong de tai co ID lop KHAC NHAU:
    - YOLOv8n COCO (baseline): 80 lop, car=2, truck=7, bus=5, motorcycle=3, bicycle=1.
    - Model fine-tune 5 lop:   ID 0..4 theo dung thu tu tren.
Vi vay KHONG bao gio so ID truc tiep: moi anh xa di qua TEN LOP doc tu checkpoint (`model.names`).
Checkpoint thieu mot trong 5 ten -> loi (khong doan). Anh xa duoc ghi vao ket qua danh gia / metadata cache.

Cache detection + tracker hien tai loc theo ID COCO (`--classes 1 2 3 5 7`). Khi tao cache tu model 5 lop,
chuyen ID ve COCO bang `sang_id_coco()` de duong tracking/ten lop phia sau giu nguyen quy uoc.
"""
from __future__ import annotations

LOP_5: tuple[str, ...] = ("car", "truck", "bus", "motorcycle", "bicycle")
ID_LOP_5: dict[str, int] = {c: i for i, c in enumerate(LOP_5)}

#: ID COCO-80 cua 5 lop (thu tu ten cua ultralytics / COCO 2017)
ID_COCO: dict[str, int] = {"bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


class LoiAnhXaLop(ValueError):
    """Checkpoint/dataset khong co du 5 lop hoac ten lop mo ho."""


def chuan_hoa_names(names) -> dict[int, str]:
    """`model.names` co the la dict {id: ten} hoac list -> dict {int: str}."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def loai_checkpoint(names) -> str:
    """'lop5' (dung 5 lop, dung thu tu), 'coco80' (80 lop COCO) hoac 'khac'."""
    n = chuan_hoa_names(names)
    if [n.get(i) for i in range(len(n))] == list(LOP_5) and len(n) == 5:
        return "lop5"
    if len(n) == 80 and all(n.get(i) == c for c, i in ID_COCO.items()):
        return "coco80"
    return "khac"


def anh_xa_model_sang_lop5(names) -> dict[int, int]:
    """{ID lop cua model: ID trong LOP_5}, theo TEN. Lop khac cua model bi bo (khong co trong dict)."""
    n = chuan_hoa_names(names)
    theo_ten: dict[str, list[int]] = {}
    for i, ten in n.items():
        theo_ten.setdefault(ten, []).append(i)
    trung = {t: ids for t, ids in theo_ten.items() if t in ID_LOP_5 and len(ids) > 1}
    if trung:
        raise LoiAnhXaLop(f"ten lop trung trong checkpoint: {trung}")
    thieu = [c for c in LOP_5 if c not in theo_ten]
    if thieu:
        raise LoiAnhXaLop(f"checkpoint thieu lop {thieu} (co {len(n)} lop: {sorted(theo_ten)[:12]}...)")
    return {theo_ten[c][0]: ID_LOP_5[c] for c in LOP_5}


def mo_ta_anh_xa(names) -> dict:
    """Ban ghi anh xa de luu cung ket qua (JSON-safe)."""
    m = anh_xa_model_sang_lop5(names)
    n = chuan_hoa_names(names)
    return {"loai_checkpoint": loai_checkpoint(names), "so_lop_model": len(n),
            "model_id_sang_lop5": {str(k): v for k, v in sorted(m.items())},
            "ten": {str(k): n[k] for k in sorted(m)},
            "lop5": list(LOP_5)}


def sang_id_coco(id_lop5: int) -> int:
    """ID trong LOP_5 -> ID COCO tuong ung (de cache/tracker giu quy uoc `--classes 1 2 3 5 7`)."""
    return ID_COCO[LOP_5[int(id_lop5)]]


def kiem_names_dataset(names) -> None:
    """names trong YAML dataset phai DUNG 5 lop theo dung thu tu da khoa."""
    n = chuan_hoa_names(names)
    if [n.get(i) for i in range(len(n))] != list(LOP_5) or len(n) != 5:
        raise LoiAnhXaLop(f"names cua dataset {n} khac thu tu lop da khoa {list(LOP_5)}")
