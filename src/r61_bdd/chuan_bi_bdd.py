"""
chuan_bi_bdd.py -- Entry point CHUAN BI R61 (chi thu vien chuan; KHONG import thu vien ben thu ba).

    --buoc chup-dau   : trang thai Git, PLAN nguyen byte, hash nguon nho duoc bao ve, liet ke vung co nguy co ghi
    --buoc kiem-ke    : kiem ke 200 video BDD COCO-video val tu manifest + nhan nguon + GT/seqinfo/seqmap hien co
                        (khong mo anh, khong inference, khong tao interim/GT/cache)
    --buoc chup-cuoi  : chup lai va doi chieu voi chup-dau (PLAN ngoai muc 0, nguon bao ve, vung nguy co, pyc, Git)

Ghi DUY NHAT duoi --phien (phai nam trong results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/).
Exit: 0 dat, 2 loi hop dong / doi chieu lech.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop_dong_bdd as H  # noqa: E402

VUNG = [H.ROOT / "results" / "phase2" / "r61_bdd" / "chuan_bi" / "chuan_bi_cpu"]
PLAN = H.ROOT / "handoff" / "PLAN_DA_CHOT.md"
DAU_MUC0 = "## 0. TRẠNG THÁI HIỆN TẠI".encode("utf-8")
DAU_SAU_MUC0 = "## Thứ tự các mốc".encode("utf-8")

#: Nguon nho duoc bao ve (bam day du o dau va cuoi luot).
NGUON_BAO_VE = [
    "configs/phase2/r54_tracking/giao_thuc_test40.json",
    "configs/phase2/experiments/bdd_transfer_cocovid_val.yaml",
    "configs/phase2/LOCK_bdd_transfer.md",
    "results/phase2/lock/lock_bdd_transfer.json",
    "results/phase2/manifests/bdd100k_transfer_cocovid_val.json",
    "results/phase2/bdd_cocovid_check/val_report.json",
    "results/phase2/bdd_transfer_cocovid_val/bao_cao/toc_do.md",
    "handoff/logs/review60/01_ho_so_quyet_dinh_moc5.md",
    "handoff/logs/review56/03_thuc_thi_tracking_test40.md",
    "handoff/logs/review53/04_ket_qua_danh_gia_day_du.md",
    "handoff/logs/review57/01_bien_thien_test40.md",
    "handoff/logs/review59/03_ket_qua_sort_test40.md",
    "handoff/logs/review48/09_thiet_ke_phan_tich_deta_pf_ekf.md",
    "AGENTS.md", "CLAUDE.md", ".gitignore",
    "src/config.py", "src/det_cache.py", "src/detector_classes.py", "src/run_tracking.py", "src/run_trackeval.py",
    "src/trackeval_ext.py", "src/mot_datasets/bdd100k_coco.py", "src/mot_datasets/bdd100k_mot.py",
    "src/mot_datasets/base.py", "src/mot_datasets/__init__.py", "src/time_units.py", "src/boot_stats.py",
    "src/kc48/boot_seed.py", "src/tracker_ctrv.py", "src/pf_ctrv.py", "src/ekf_ctrv.py", "src/ukf_ctrv.py",
    "src/baseline_track.py", "src/ul_backend.py", "src/run_state.py",
    "configs/bytetrack_ekf_ctrv.yaml", "configs/bytetrack_ukf_ctrv.yaml",
    "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml",
    "models/yolov8n.pt",
    "models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt",
    "models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt",
    "data/processed/seqmaps/BDDCOCOVID-val.txt",
]

#: Vung co nguy co phat sinh ghi -> chi liet ke (size, mtime), khong bam du lieu lon.
VUNG_NGUY_CO = [("data/interim", False), ("data/interim/det_cache", False), ("data/processed", False),
                ("data/processed/trackers", False), ("data/processed/seqmaps", False),
                ("data/processed/BDDCOCOVID-val", True), ("results", False), ("results/phase2", False),
                ("configs/phase2", False), ("src", False), ("src/mot_datasets", False), ("src/kc48", False),
                ("handoff/logs", False), (".", False)]


def git(*args) -> bytes:
    """Git CHI DOC (--no-optional-locks: khong lam moi index)."""
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=H.ROOT, capture_output=True, check=True).stdout


def tach_plan(b: bytes) -> dict:
    i, j = b.find(DAU_MUC0), b.find(DAU_SAU_MUC0)
    if i < 0 or j < 0 or j < i:
        raise H.LoiHopDong("khong tach duoc muc 0 cua PLAN")
    return {"sha256_toan_bo": H.sha256_bytes(b), "sha256_truoc_muc0": H.sha256_bytes(b[:i]),
            "sha256_sau_muc0": H.sha256_bytes(b[j:]), "byte_muc0": [i, j], "so_byte": len(b)}


def appdata_ultralytics() -> dict:
    p = Path.home() / "AppData" / "Roaming" / "Ultralytics" / "settings.json"
    return {"duong": str(p), "sha256": H.sha256_file(p) if p.is_file() else None}


def chup(thu_muc: Path) -> dict:
    H.tao_thu_muc(thu_muc, VUNG)
    plan_b = PLAN.read_bytes()
    H.ghi_bytes(thu_muc / "PLAN_DA_CHOT.md.bytes", plan_b, VUNG)
    st = git("status", "--short", "--untracked-files=all")
    H.ghi_bytes(thu_muc / "git_status_short.txt", st, VUNG)
    H.ghi_bytes(thu_muc / "git_diff.patch", git("diff"), VUNG)
    nguon = {}
    for r in NGUON_BAO_VE:
        p = H.ROOT / r
        nguon[r] = H.sha256_file(p) if p.is_file() else None
    nguy_co = {d: H.liet_ke_nhe(H.ROOT / d, de_quy) for d, de_quy in VUNG_NGUY_CO}
    pyc = sorted(H.rel(p) for p in (H.ROOT / "src").rglob("*.pyc"))
    out = {"head": git("rev-parse", "HEAD").decode().strip(),
           "nhanh": git("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
           "git_status_sha256": H.sha256_bytes(st), "plan": tach_plan(plan_b), "nguon_bao_ve": nguon,
           "appdata_ultralytics": appdata_ultralytics(), "pyc_trong_src": pyc,
           "so_muc_nguy_co": {k: len(v) for k, v in nguy_co.items()}}
    H.ghi_json(thu_muc / "vung_nguy_co.json", nguy_co, VUNG)
    H.ghi_json(thu_muc / "chup.json", out, VUNG)
    return out


def buoc_chup_dau(ns) -> int:
    phien = Path(ns.phien)
    if (phien / "chup_dau").exists():
        raise H.LoiHopDong("chup_dau da ton tai -- khong chup de")
    out = chup(phien / "chup_dau")
    print(json.dumps({"head": out["head"], "nhanh": out["nhanh"], "plan": out["plan"]}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Kiem ke
# ---------------------------------------------------------------------------
def doc_gt(p: Path) -> list[tuple[int, int]]:
    """[(frame, class)] cua gt.txt (MOTChallenge 9 cot)."""
    out = []
    for ln in p.read_bytes().decode("utf-8").splitlines():
        if ln.strip():
            c = ln.split(",")
            out.append((int(c[0]), int(float(c[7]))))
    return out


def doc_seqinfo(p: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read_string(p.read_bytes().decode("utf-8"))
    s = cp["Sequence"]
    return {"seqLength": int(s["seqLength"]), "frameRate": float(s["frameRate"]), "name": s["name"]}


def vi_tri_khoang(frames: list[int], dau: int, cuoi_ann: int | None) -> dict:
    if cuoi_ann is None:
        return {"truoc_annotation_dau": 0, "giua": 0, "sau_annotation_cuoi": len(frames)}
    return {"truoc_annotation_dau": sum(1 for f in frames if f < dau),
            "giua": sum(1 for f in frames if dau <= f <= cuoi_ann),
            "sau_annotation_cuoi": sum(1 for f in frames if f > cuoi_ann)}


def nen_day(fs: list[int]) -> str:
    """[3,4,5,9] -> '3-5;9'."""
    out, i = [], 0
    while i < len(fs):
        j = i
        while j + 1 < len(fs) and fs[j + 1] == fs[j] + 1:
            j += 1
        out.append(str(fs[i]) if i == j else f"{fs[i]}-{fs[j]}")
        i = j + 1
    return ";".join(out)


def buoc_kiem_ke(ns) -> int:
    phien = Path(ns.phien)
    gt_proto = H.doc_json(ns.giao_thuc)
    ds = gt_proto["tap_du_lieu"]
    out_dir = phien / "kiem_ke"
    if out_dir.exists():
        raise H.LoiHopDong("kiem_ke da ton tai trong phien -- dung phien moi")
    man_p, lab_p = H.ROOT / ds["manifest"]["rel"], H.ROOT / ds["nhan"]["rel"]
    gt_root, split = H.ROOT / ds["gt_folder"], ds["split"]
    seqmap_p = gt_root / "seqmaps" / f"{split}.txt"
    nguon = {"manifest": {"rel": H.rel(man_p), "sha256": H.sha256_file(man_p)},
             "nhan": {"rel": H.rel(lab_p), "sha256": H.sha256_file(lab_p), "so_byte": lab_p.stat().st_size},
             "seqmap": {"rel": H.rel(seqmap_p), "sha256": H.sha256_file(seqmap_p)},
             "val_report": {"rel": ds["bao_cao_kiem_ke_cu"]["rel"],
                            "sha256": H.sha256_file(H.ROOT / ds["bao_cao_kiem_ke_cu"]["rel"])}}
    loi = []
    for k in ("manifest", "nhan"):
        if nguon[k]["sha256"] != ds[k]["sha256"]:
            loi.append(f"{k}: sha256 {nguon[k]['sha256']} != giao thuc {ds[k]['sha256']}")
    man = H.doc_json(man_p)
    videos = list(man["videos"])
    d = json.loads(lab_p.read_bytes().decode("utf-8"))
    vid2name = {v["id"]: v["name"] for v in d["videos"]}
    cat = {c["id"]: c["name"] for c in d["categories"]}
    eval_cat = set(ds["gt_vehicle_lop_nguon"])
    ign_cat = set(ds["gt_ignore_lop_nguon_ngoai_crowd"])
    anh = {}          # image_id -> (video, frame 1-based)
    theo_video: dict[str, dict] = {v: {"frames": [], "ann_frames": set(), "n_ann": 0, "n_eval": 0, "n_ign": 0,
                                       "eval_frames": set()} for v in videos}
    la = set()
    for im in d["images"]:
        v = vid2name[im["video_id"]]
        if v not in theo_video:
            la.add(v)
            continue
        f = int(im["frame_id"]) + 1
        anh[im["id"]] = (v, f)
        theo_video[v]["frames"].append(f)
    for a in d["annotations"]:
        if a["image_id"] not in anh:
            continue
        v, f = anh[a["image_id"]]
        t = theo_video[v]
        t["n_ann"] += 1
        t["ann_frames"].add(f)
        c = cat[int(a["category_id"])]
        crowd = bool(a.get("iscrowd", 0)) or bool(a.get("ignore", 0))
        if crowd or c in ign_cat:
            t["n_ign"] += 1
        elif c in eval_cat:
            t["n_eval"] += 1
            t["eval_frames"].add(f)
    if la:
        loi.append(f"{len(la)} video trong nhan khong co trong manifest")
    seqmap = [ln.strip() for ln in seqmap_p.read_bytes().decode("utf-8").splitlines() if ln.strip()][1:]
    if sorted(seqmap) != sorted(videos):
        loi.append("seqmap khac danh sach manifest")
    rows = []
    for v in videos:
        t = theo_video[v]
        fs = sorted(t["frames"])
        if not fs:
            loi.append(f"{v}: khong co anh trong nhan")
            continue
        khong_ann = sorted(set(fs) - t["ann_frames"])
        gap = sorted(set(range(fs[0], fs[-1] + 1)) - set(fs))
        si = doc_seqinfo(gt_root / split / v / "seqinfo.ini")
        g = doc_gt(gt_root / split / v / "gt" / "gt.txt")
        f1 = {f for f, c in g if c == 1}
        f13 = {f for f, c in g if c == 13}
        n1, n13 = sum(1 for _, c in g if c == 1), sum(1 for _, c in g if c == 13)
        lop_la = sorted({c for _, c in g if c not in (1, 13)})
        xe_trong = sorted(set(range(1, si["seqLength"] + 1)) - f1)
        cuoi_ann = max(t["ann_frames"]) if t["ann_frames"] else None
        dau_ann = min(t["ann_frames"]) if t["ann_frames"] else None
        vt = vi_tri_khoang(khong_ann, dau_ann or 0, cuoi_ann)
        kiem = {"seqLength_bang_frame_cuoi": si["seqLength"] == fs[-1],
                "frame_dau_1": fs[0] == 1, "khong_khoang_thieu": not gap,
                "frameRate_5": si["frameRate"] == 5.0,
                "gt_lop1_bang_ann_vehicle": n1 == t["n_eval"], "gt_lop13_bang_ann_ignore": n13 == t["n_ign"],
                "khong_ann_la_tap_con_xe_trong": set(khong_ann) <= set(xe_trong),
                "gt_khong_lop_la": not lop_la, "gt_frame_trong_timeline": all(1 <= f <= si["seqLength"] for f, _ in g),
                "frame_co_xe_bang_nguon": f1 == t["eval_frames"]}
        for k, ok in kiem.items():
            if not ok:
                loi.append(f"{v}: {k} sai")
        rows.append({"video": v, "da_xem_truoc": v in man.get("sequence_da_xem_truoc", []),
                     "n_anh_nhan": len(fs), "frame_dau": fs[0], "frame_cuoi": fs[-1], "n_khoang_thieu": len(gap),
                     "n_anh_khong_annotation": len(khong_ann),
                     "khong_ann_truoc_dau": vt["truoc_annotation_dau"], "khong_ann_giua": vt["giua"],
                     "khong_ann_sau_cuoi": vt["sau_annotation_cuoi"], "frame_khong_annotation": nen_day(khong_ann),
                     "frame_annotation_cuoi": cuoi_ann or 0, "n_annotation": t["n_ann"],
                     "n_ann_vehicle_cham": t["n_eval"], "n_ann_vung_bo_qua": t["n_ign"],
                     "gt_seqLength": si["seqLength"], "gt_frameRate": si["frameRate"], "gt_dong_lop1": n1,
                     "gt_dong_lop13": n13, "gt_frame_co_xe": len(f1), "gt_frame_xe_trong": len(xe_trong),
                     "gt_frame_xe_trong_nhung_co_annotation_nguon": len(set(xe_trong) - set(khong_ann)),
                     "gt_frame_chi_lop13": len(f13 - f1), "trong_seqmap": v in seqmap,
                     "kiem_dat": all(kiem.values())})
    tong = {"so_video": len(rows), "so_anh": sum(r["n_anh_nhan"] for r in rows),
            "so_anh_khong_annotation": sum(r["n_anh_khong_annotation"] for r in rows),
            "so_video_co_anh_khong_annotation": sum(1 for r in rows if r["n_anh_khong_annotation"]),
            "khong_ann_truoc_dau": sum(r["khong_ann_truoc_dau"] for r in rows),
            "khong_ann_giua": sum(r["khong_ann_giua"] for r in rows),
            "khong_ann_sau_cuoi": sum(r["khong_ann_sau_cuoi"] for r in rows),
            "so_video_khong_annotation_nao": sum(1 for r in rows if r["n_annotation"] == 0),
            "gt_frame_xe_trong": sum(r["gt_frame_xe_trong"] for r in rows),
            "gt_frame_xe_trong_nhung_co_annotation_nguon": sum(r["gt_frame_xe_trong_nhung_co_annotation_nguon"]
                                                              for r in rows),
            "gt_frame_chi_lop13": sum(r["gt_frame_chi_lop13"] for r in rows),
            "gt_dong_lop1": sum(r["gt_dong_lop1"] for r in rows), "gt_dong_lop13": sum(r["gt_dong_lop13"] for r in rows),
            "so_annotation": sum(r["n_annotation"] for r in rows), "so_video_kiem_dat": sum(r["kiem_dat"] for r in rows)}
    ky_vong = ds["ky_vong"]
    for k_tong, k_ky in (("so_video", "so_video"), ("so_anh", "so_anh"),
                         ("so_anh_khong_annotation", "so_anh_khong_annotation"), ("so_annotation", "so_annotation")):
        if tong[k_tong] != ky_vong[k_ky]:
            loi.append(f"{k_tong} {tong[k_tong]} != ky vong {ky_vong[k_ky]}")
    vr = H.doc_json(H.ROOT / ds["bao_cao_kiem_ke_cu"]["rel"])
    ket = {"tong": tong, "ky_vong": ky_vong, "loi": loi, "dat": not loi,
           "anh_thieu_tren_dia": {"gia_tri": vr.get("n_missing_images_on_disk"),
                                  "nguon": "TRICH bao cao cu " + ds["bao_cao_kiem_ke_cu"]["rel"] +
                                           "; luot nay KHONG kiem lai anh tren dia (khong mo anh)",
                                  "khong_thay_the_so_anh_khong_annotation": True},
           "dinh_nghia": {"anh_khong_annotation": "anh co trong images[] cua nhan nguon nhung khong co annotation nao "
                                                  "(moi lop, ke ca crowd/ignore)",
                          "frame_gt_xe_trong": "frame 1..seqLength khong co dong lop 1 trong gt.txt hien co "
                                               "(gom ca anh khong annotation LAN anh chi co lop khong cham / ignore)",
                          "chinh_sach": "giu trong timeline va cham nhu frame khong co GT (hanh vi evaluator hien hanh); "
                                        "khong dien giai la am tinh that; nguyen nhan chua xac minh"},
           "khong_lam": ["khong mo anh", "khong inference", "khong tao interim/GT/cache"]}
    H.tao_thu_muc(out_dir, VUNG)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    nguon["kiem_ke_theo_video.csv"] = H.ghi_bytes(out_dir / "kiem_ke_theo_video.csv", buf.getvalue().encode("utf-8"),
                                                  VUNG)
    ket["kiem_ke_theo_video_sha256"] = nguon["kiem_ke_theo_video.csv"]
    H.ghi_json(out_dir / "kiem_ke.json", ket, VUNG)
    H.ghi_json(out_dir / "nguon.json", nguon, VUNG)
    print(json.dumps({"tong": tong, "loi": loi[:20], "so_loi": len(loi)}, ensure_ascii=False))
    return 0 if not loi else H.EXIT_HOP_DONG


# ---------------------------------------------------------------------------
# Chup cuoi + doi chieu
# ---------------------------------------------------------------------------
def duoc_phep(r: str, write_paths: list[str]) -> bool:
    return any(r == w.rstrip("/") or r.startswith(w if w.endswith("/") else w + "/") for w in write_paths)


def buoc_chup_cuoi(ns) -> int:
    phien = Path(ns.phien)
    # --phien-dau: doi chieu voi baseline GOC cua phien khac (CHI DOC), ghi chup_cuoi vao phien moi -> khong ghi de
    goc_dau = Path(os.path.abspath(H.ROOT / ns.phien_dau)) if ns.phien_dau else Path(os.path.abspath(phien))
    if not H._trong(goc_dau, H.ROOT / "results" / "phase2" / "r61_bdd" / "chuan_bi" / "chuan_bi_cpu"):
        raise H.LoiHopDong("--phien-dau phai nam duoi results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/")
    dau = H.doc_json(goc_dau / "chup_dau" / "chup.json")
    if (phien / "chup_cuoi").exists():
        raise H.LoiHopDong("chup_cuoi da ton tai -- dung phien moi")
    cuoi = chup(phien / "chup_cuoi")
    wp = ["src/r61_bdd/", "configs/phase2/r61_bdd/", "results/phase2/r61_bdd/tests/",
          "results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/", "handoff/logs/review61/01_quyet_dinh_va_giao_thuc.md",
          "handoff/logs/review61/02_ket_qua_chuan_bi.md", "handoff/PLAN_DA_CHOT.md"]
    loi = []
    for k in ("sha256_truoc_muc0", "sha256_sau_muc0"):
        if dau["plan"][k] != cuoi["plan"][k]:
            loi.append(f"PLAN {k} doi")
    nguon_doi = [r for r in dau["nguon_bao_ve"] if dau["nguon_bao_ve"][r] != cuoi["nguon_bao_ve"].get(r)]
    loi += [f"nguon bao ve doi: {r}" for r in nguon_doi]
    if dau["appdata_ultralytics"] != cuoi["appdata_ultralytics"]:
        loi.append("settings.json Ultralytics trong AppData doi")
    pyc_moi = sorted(set(cuoi["pyc_trong_src"]) - set(dau["pyc_trong_src"]))
    loi += [f"pyc moi: {p}" for p in pyc_moi]
    nc_dau = H.doc_json(goc_dau / "chup_dau" / "vung_nguy_co.json")
    nc_cuoi = H.doc_json(phien / "chup_cuoi" / "vung_nguy_co.json")
    doi_nc = []
    for d in nc_cuoi:
        a, b = nc_dau.get(d, {}), nc_cuoi[d]
        for r in sorted(set(a) | set(b)):
            if a.get(r) != b.get(r) and not duoc_phep(r, wp) and r not in ("results/phase2/r61_bdd", "src/r61_bdd",
                                                                            "configs/phase2/r61_bdd",
                                                                            "handoff/logs/review61", "handoff"):
                doi_nc.append({"vung": d, "muc": r, "dau": a.get(r), "cuoi": b.get(r)})
    loi += [f"vung nguy co doi ngoai write_paths: {x['muc']}" for x in doi_nc]
    st_dau = (goc_dau / "chup_dau" / "git_status_short.txt").read_bytes().decode("utf-8").splitlines()
    st_cuoi = (phien / "chup_cuoi" / "git_status_short.txt").read_bytes().decode("utf-8").splitlines()
    moi = sorted(set(st_cuoi) - set(st_dau))
    ngoai = [ln for ln in moi if not duoc_phep(ln[3:].strip().strip('"'), wp)]
    loi += [f"git status moi ngoai write_paths: {ln}" for ln in ngoai]
    if dau["head"] != cuoi["head"] or dau["nhanh"] != cuoi["nhanh"]:
        loi.append("HEAD/nhanh doi")
    ss = {"dat": not loi, "loi": loi, "phien_dau": H.rel(goc_dau),
          "chup_dau_sha256": H.sha256_file(goc_dau / "chup_dau" / "chup.json"), "head": cuoi["head"], "nhanh": cuoi["nhanh"],
          "plan_dau": dau["plan"], "plan_cuoi": cuoi["plan"], "git_status_moi": moi,
          "vung_nguy_co_doi_ngoai_write_paths": doi_nc, "so_nguon_bao_ve": len(dau["nguon_bao_ve"])}
    H.ghi_json(phien / "chup_cuoi" / "so_sanh.json", ss, VUNG)
    print(json.dumps({"dat": ss["dat"], "loi": loi, "git_status_moi": moi}, ensure_ascii=False))
    return 0 if not loi else H.EXIT_HOP_DONG


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buoc", required=True, choices=["chup-dau", "kiem-ke", "chup-cuoi"])
    ap.add_argument("--phien", required=True)
    ap.add_argument("--giao-thuc", default=None)
    ap.add_argument("--phien-dau", default=None, help="chup-cuoi: phien chua chup_dau goc (chi doc)")
    ns = ap.parse_args(argv)
    if ns.buoc == "kiem-ke" and not ns.giao_thuc:
        raise H.LoiHopDong("kiem-ke can --giao-thuc")
    return {"chup-dau": buoc_chup_dau, "kiem-ke": buoc_kiem_ke, "chup-cuoi": buoc_chup_cuoi}[ns.buoc](ns)


if __name__ == "__main__":
    argv = sys.argv[1:]
    phien_arg = argv[argv.index("--phien") + 1] if "--phien" in argv else None
    if not phien_arg:
        print("thieu --phien", file=sys.stderr)
        sys.exit(H.EXIT_HOP_DONG)
    phien = Path(os.path.abspath(H.ROOT / phien_arg if not Path(phien_arg).is_absolute() else phien_arg))
    H.tao_thu_muc(phien, VUNG)
    H.chuan_bi_moi_truong(phien, VUNG)
    sys.exit(H.chay_va_ghi_log(main, argv, phien, VUNG, "chuan_bi_bdd"))
