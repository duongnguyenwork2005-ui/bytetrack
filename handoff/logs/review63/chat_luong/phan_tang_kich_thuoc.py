"""R63 - recall/precision detection theo KICH THUOC xe x MUC CHE KHUAT tren UA-DETRAC (MOT GT), tu detection
cache cua pipeline (doc qua det_cache.open_cache, cung bo loc conf/lop/max_det voi tracking).

Muc dich: khi doi kich thuoc dau vao detector (vd 640 -> 512/448), do rieng xe nho va xe bi che - nhung nhom de
mat nhat - thay vi chi nhin mAP/HOTA tong. LUA CHON ung vien chi tren DEV (results/phase2/manifests/detrac_dev.json);
test40 chi bao cao sau khi da chot.

    python phan_tang_kich_thuoc.py --cache-id <id> --manifest results/phase2/manifests/detrac_dev.json --ra <dir>
    python phan_tang_kich_thuoc.py --cache-id <ung_vien> --so-voi <baseline> --manifest ... --ra <dir>

Quy uoc (dang ky truoc, doi phai ghi lai):
- GT: data/processed/<split>/<video>/gt/gt.txt, chi dong class == 1 (xe). Kich thuoc = sqrt(w*h) theo pixel
  anh GOC; nhom: nho <24, vua 24-48, lon 48-96, rat_lon >=96. Che khuat theo cot visibility (= 1 - occlusion):
  khong_che >=0.9, mot_phan 0.5-0.9, nang <0.5.
- Detection: conf > --conf (mac dinh 0.25 = dieu kien c25), lop --classes (mac dinh 2,3,5,7 nhu thi nghiem
  UA-DETRAC), max_det 300. Ghep 1-1 theo frame bang Hungarian tren IoU, khong phan biet lop, IoU >= 0.5.
- Precision theo nhom kich thuoc tinh theo kich thuoc cua CHINH detection; khong co precision theo che khuat.
- So cap (--so-voi): hieu recall theo video (ung vien - baseline), bootstrap theo video 5.000 lan, seed 0,
  CI 95 % percentile, gop vi mo (tong TP / tong GT trong mau). CI rieng le, chua hieu chinh so sanh boi.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))

NHOM_KT = [("nho", 0.0, 24.0), ("vua", 24.0, 48.0), ("lon", 48.0, 96.0), ("rat_lon", 96.0, float("inf"))]
NHOM_CHE = [("nang", 0.0, 0.5), ("mot_phan", 0.5, 0.9), ("khong_che", 0.9, 1.01)]
#: --nguon bdd (split BDDDEV-train, g4_bdd_dev_split.py): BDD chi co CO NHI PHAN `occluded`; adapter bdd100k_mot ghi
#: visibility 0,5 neu occluded, 1,0 neu khong -> hai nhom theo co, KHONG phai muc do che (khong co nhom "nang").
NHOM_CHE_BDD = [("occluded", 0.0, 0.9), ("khong_occluded", 0.9, 1.01)]
IOU_GHEP = 0.5


def nhom(gia_tri: np.ndarray, bang) -> np.ndarray:
    out = np.empty(len(gia_tri), dtype=object)
    for ten, lo, hi in bang:
        out[(gia_tri >= lo) & (gia_tri < hi)] = ten
    return out


def iou_ma_tran(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,4), b (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def ghep(gt: np.ndarray, det: np.ndarray, nguong: float = IOU_GHEP) -> tuple[np.ndarray, np.ndarray]:
    """Tra ve (gt_trung (n,) bool, det_trung (m,) bool). Tai dung hop dong `occlusion_eval.match_frame`
    (toi da SO CAP IoU >= nguong truoc, roi toi da tong IoU). Ban dau (30/10) dung chi phi cap khong hop le = 1
    nen co the chon 4 cap IoU=1 thay vi 5 cap IoU=0,5 (GPT review R63 Q7) - da sua, co regression."""
    from occlusion_eval import match_frame
    g_ok = np.zeros(len(gt), bool); d_ok = np.zeros(len(det), bool)
    if len(gt) == 0 or len(det) == 0:
        return g_ok, d_ok
    for r, c in match_frame(iou_ma_tran(gt, det), nguong):
        g_ok[r] = True; d_ok[c] = True
    return g_ok, d_ok


def doc_gt(duong: Path) -> dict[int, np.ndarray]:
    """frame -> (n, 5) [x1, y1, x2, y2, visibility] cua dong class == 1."""
    a = np.loadtxt(duong, delimiter=",", ndmin=2)
    a = a[a[:, 7] == 1]
    out = {}
    for f in np.unique(a[:, 0]).astype(int):
        s = a[a[:, 0] == f]
        out[f] = np.column_stack([s[:, 2], s[:, 3], s[:, 2] + s[:, 4], s[:, 3] + s[:, 5], s[:, 8]])
    return out


def dem_video(gt_frames: dict[int, np.ndarray], det_frames: dict[int, np.ndarray]) -> dict:
    """Dem TP/GT theo (kich_thuoc, che) va TP/det theo kich_thuoc detection cho mot video."""
    dem = {}
    def cong(k, n=1):
        dem[k] = dem.get(k, 0) + n
    for f in sorted(set(gt_frames) | set(det_frames)):
        g = gt_frames.get(f, np.zeros((0, 5))); d = det_frames.get(f, np.zeros((0, 6)))
        g_ok, d_ok = ghep(g[:, :4], d[:, :4])
        if len(g):
            kt = nhom(np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])), NHOM_KT)
            ch = nhom(g[:, 4], NHOM_CHE)
            for k1, k2, ok in zip(kt, ch, g_ok):
                cong(("gt", k1, k2)); cong(("gt", k1, "*")); cong(("gt", "*", k2)); cong(("gt", "*", "*"))
                if ok:
                    cong(("tp", k1, k2)); cong(("tp", k1, "*")); cong(("tp", "*", k2)); cong(("tp", "*", "*"))
        if len(d):
            kt = nhom(np.sqrt(np.clip(d[:, 2] - d[:, 0], 0, None) * np.clip(d[:, 3] - d[:, 1], 0, None)), NHOM_KT)
            for k1, ok in zip(kt, d_ok):
                cong(("det", k1)); cong(("det", "*"))
                if ok:
                    cong(("dtp", k1)); cong(("dtp", "*"))
    return dem


def doc_det_cache(cache_id: str, video: str, conf: float, classes: list[int], max_det: int,
                  gt_frames: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict]:
    """Doc cache; KHONG am tham bo frame: frame unreadable hoac frame GT khong co trong cache -> loi (ValueError)."""
    from det_cache import open_cache
    vd = open_cache(cache_id, video, conf, classes, max_det)
    out, unreadable = {}, []
    for fi, _t, unr, arr in vd.iter_frames():
        if unr:
            unreadable.append(int(fi))
        out[int(fi)] = arr
    thieu = sorted(set(gt_frames) - set(out))
    if unreadable or thieu:
        raise ValueError(f"{cache_id}/{video}: unreadable={unreadable[:10]} (n={len(unreadable)}), "
                         f"frame GT khong co trong cache={thieu[:10]} (n={len(thieu)})")
    return out, {"n_frame_cache": len(out), "n_frame_gt": len(gt_frames)}


def bang_tong(dem_theo_video: dict[str, dict]) -> list[dict]:
    tong = {}
    for d in dem_theo_video.values():
        for k, v in d.items():
            tong[k] = tong.get(k, 0) + v
    hang = []
    for k1 in [x[0] for x in NHOM_KT] + ["*"]:
        for k2 in [x[0] for x in NHOM_CHE] + ["*"]:
            n = tong.get(("gt", k1, k2), 0); tp = tong.get(("tp", k1, k2), 0)
            r = {"kich_thuoc": k1, "che": k2, "n_gt": n, "tp": tp, "recall": tp / n if n else None}
            if k2 == "*":
                nd = tong.get(("det", k1), 0); dtp = tong.get(("dtp", k1), 0)
                r.update({"n_det": nd, "precision": dtp / nd if nd else None})
            hang.append(r)
    return hang


def so_cap(a: dict[str, dict], b: dict[str, dict], n_boot: int = 5000, seed: int = 0) -> list[dict]:
    """Hieu recall vi mo (a - b) theo nhom, bootstrap theo video."""
    if set(a) != set(b):
        raise ValueError(f"hai cache khac tap video: chi a={sorted(set(a) - set(b))}, chi b={sorted(set(b) - set(a))}")
    vids = sorted(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(n_boot, len(vids)))
    out = []
    for k1 in [x[0] for x in NHOM_KT] + ["*"]:
        for k2 in [x[0] for x in NHOM_CHE] + ["*"]:
            def mang(d, loai):
                return np.array([d[v].get((loai, k1, k2), 0) for v in vids], dtype=float)
            ga, ta, gb, tb = mang(a, "gt"), mang(a, "tp"), mang(b, "gt"), mang(b, "tp")
            if ga.sum() == 0 or not np.array_equal(ga, gb):
                out.append({"kich_thuoc": k1, "che": k2, "n_gt": int(ga.sum()), "hieu_recall": None,
                            "ghi_chu": "khong co GT" if ga.sum() == 0 else "GT khac nhau giua hai cache"})
                continue
            diem = ta.sum() / ga.sum() - tb.sum() / gb.sum()
            g = ga[idx].sum(1)
            hop = g > 0
            boot = (ta[idx].sum(1)[hop] - tb[idx].sum(1)[hop]) / g[hop]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append({"kich_thuoc": k1, "che": k2, "n_gt": int(ga.sum()), "hieu_recall": float(diem),
                        "ci95": [float(lo), float(hi)], "ci_chua_0": bool(lo <= 0 <= hi), "n_video": len(vids),
                        "n_video_co_gt": int((ga > 0).sum()), "n_boot_hop_le": int(hop.sum()), "n_boot": n_boot})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-id", required=True)
    ap.add_argument("--so-voi", default=None, help="cache-id baseline de so cap")
    ap.add_argument("--manifest", default=None, help="json co khoa 'videos' (vd detrac_dev.json)")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--split", default="DETRAC-all", help="thu muc data/processed/<split>")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="2,3,5,7")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--nguon", choices=["detrac", "bdd"], default="detrac",
                    help="bdd: nhom che theo co occluded (visibility 0,5/1,0 cua adapter); precision BDD CHUA tru vung "
                         "bo qua lop 13 (chi recall dung cho cong)")
    a = ap.parse_args(argv)
    global NHOM_CHE
    if a.nguon == "bdd":
        NHOM_CHE = NHOM_CHE_BDD
    vids = list(a.videos or [])
    if a.manifest:
        vids += json.loads(Path(a.manifest).read_text(encoding="utf-8"))["videos"]
    vids = [v if isinstance(v, str) else v.get("name", v.get("video")) for v in vids]
    if not vids:
        print("[LOI] can --manifest hoac --videos", file=sys.stderr)
        return 2
    cls = [int(x) for x in a.classes.split(",")]
    ra = Path(a.ra)
    if ra.exists() and any(ra.iterdir()):
        print(f"[LOI] {ra} khong rong (moi luot mot thu muc moi)", file=sys.stderr)
        return 2
    ra.mkdir(parents=True, exist_ok=True)
    kq, timeline = {}, {}
    for cid in [a.cache_id] + ([a.so_voi] if a.so_voi else []):
        dem = {}
        for v in vids:
            gt = doc_gt(GOC / "data" / "processed" / a.split / v / "gt" / "gt.txt")
            if a.nguon == "bdd":
                vis = np.unique(np.concatenate([g[:, 4] for g in gt.values()])) if gt else np.zeros(0)
                if not set(np.round(vis, 3)) <= {0.5, 1.0}:
                    raise ValueError(f"{v}: visibility BDD ngoai {{0.5, 1.0}}: {vis[:10]}")
            det, tl = doc_det_cache(cid, v, a.conf, cls, a.max_det, gt)
            timeline[f"{cid}/{v}"] = tl
            dem[v] = dem_video(gt, det)
        kq[cid] = dem
        hang = bang_tong(dem)
        with open(ra / f"bang_{cid}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kich_thuoc", "che", "n_gt", "tp", "recall", "n_det", "precision"])
            w.writeheader(); w.writerows(hang)
        print(f"[{cid}] tong: " + ", ".join(f"{h['kich_thuoc']}/{h['che']} R={h['recall']:.4f}"
                                           for h in hang if h["che"] == "*" and h["recall"] is not None))
    cap = so_cap(kq[a.cache_id], kq[a.so_voi]) if a.so_voi else None
    (ra / "ket_qua.json").write_text(json.dumps({
        "cache_id": a.cache_id, "so_voi": a.so_voi, "videos": vids, "conf": a.conf, "classes": cls,
        "max_det": a.max_det, "iou_ghep": IOU_GHEP, "nhom_kich_thuoc": NHOM_KT[:-1] + [("rat_lon", 96.0, None)],
        "nhom_che": NHOM_CHE, "timeline": timeline, "so_cap": cap, "nguon": a.nguon,
        "pham_vi": ("UA-DETRAC (class==1, visibility = 1 - occlusion)" if a.nguon == "detrac" else
                    "BDD dev (class==1 = car/truck/bus/motorcycle/bicycle khong crowd; lop 13 bo qua khong vao GT; "
                    "nhom che theo co occluded nhi phan; precision CHUA tru vung bo qua lop 13)")},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
