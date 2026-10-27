"""
test_chuan_bi_bdd.py -- Fixture CPU cua R61_PREP. Du lieu TONG HOP rieng trong --phien (duoi results/phase2/r61_bdd/tests/);
detector MO PHONG (khong nap checkpoint that, CUDA_VISIBLE_DEVICES=""), con lai goi DUNG ham wrapper + helper that:
BDD100KCocoDataset.frames, parse_coco/export_motchallenge (GT fixture), det_cache.build_video_yolo/detect_frame/save_video/
write_meta/VideoDetections/filter_dets, detector_classes, run_tracking.track_video, TrackEval + MotChallengeCrowdIgnore,
boot_stats. Luong A/B: MOI lenh chay trong TIEN TRINH CON rieng (Python khoaluan-mot) qua dieu_phoi_bdd.py trong phien
gia (A, B deu duoi tests/); lenh `cache` can detector mo phong -> tien trinh con `test_chuan_bi_bdd.py --con-dieu-phoi`
gan NAP_MODEL roi chay DUNG khoi __main__ cua dieu_phoi_bdd. Log tung lenh con (argv/Python/exit/stdout/stderr) ghi
vao <phien>/lenh_con/ ke ca khi con loi/crash (segfault native lan_03, lan_05..08: tach tien trinh, nguyen nhan CHUA RO).
Kiem am (review_003): exit 2 CHUA DU -- dong [LOI HOP DONG] trong stderr phai dung nguyen nhan du kien (dung_ly_do), danh
sach loi khoa/hau-kiem phai bang tung muc; loi git_dirty_sha256 (vd PLAN bi sua giua phien) lam test that bai.

Exit 0: moi kiem dat; 2: co kiem khong dat (chi tiet trong ket_qua_test.json).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # fixture KHONG dung GPU ("" van de is_available() True o lan_01)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop_dong_bdd as H  # noqa: E402

KQ: list[dict] = []
VIDEOS = ["fxv-a", "fxv-b", "fxv-c"]
N_FRAME = 10
KHONG_ANN = {"fxv-b": {8, 9, 10}, "fxv-c": {5}}
NAMES_LOP5 = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}
NAMES_COCO = {i: f"lop{i}" for i in range(80)}
NAMES_COCO.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
CROWD = (900.0, 200.0, 200.0, 100.0)
CON_DIEU_PHOI = "--con-dieu-phoi"                # che do tien trinh con: dieu_phoi_bdd + detector mo phong
SCRIPT_DP = H.SRC / "r61_bdd" / "dieu_phoi_bdd.py"
DAU_NAP = "[NAP_DETECTOR_MO_PHONG] "             # dong stdout moi lan nap detector mo phong (dem cho F2)
TIMEOUT_CON = 1800
LENH_CON: list[dict] = []
DAU_LOI = "[LOI HOP DONG] "                      # chay_va_ghi_log in LoiHopDong ra stderr voi tien to nay
DS_LOI = ("trang thai hien tai khac khoa: ", "hau-kiem: ")   # thong diep mang danh sach loi (so khop TUNG muc)


def kiem(ten: str, ok: bool, chi_tiet=None):
    KQ.append({"kiem": ten, "dat": bool(ok), "chi_tiet": chi_tiet})
    print(("[DAT] " if ok else "[KHONG DAT] ") + ten + ("" if ok else f" -- {chi_tiet}"))


def bat_loi(fn, loai=Exception) -> str | None:
    try:
        fn()
    except loai as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"[:300]
    return None


def chay_con(phien_test: Path, argv: list[str], ten: str) -> dict:
    """Chay MOT lenh dieu_phoi_bdd trong tien trinh con rieng; ghi <phien_test>/lenh_con/NNN_<ten>.json NGAY sau khi con
    ket thuc (ca khi exit bat thuong / het gio). `cache` can detector mo phong -> qua che do --con-dieu-phoi cua file nay."""
    import subprocess
    can_det = "cache" in argv[argv.index("--buoc") + 1:argv.index("--buoc") + 2]
    script = Path(__file__).resolve() if can_det else SCRIPT_DP
    cmd = [H.PYTHON, str(script), *([CON_DIEU_PHOI] if can_det else []), *argv]
    env = dict(os.environ)
    env.update({"PYTHONFAULTHANDLER": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_VISIBLE_DEVICES": "-1"})
    t0 = time.time()
    het_gio = False
    try:
        r = subprocess.run(cmd, cwd=str(H.ROOT), env=env, capture_output=True, timeout=TIMEOUT_CON)
        ma, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        ma, out, err, het_gio = None, e.stdout or b"", e.stderr or b"", True
    out_s, err_s = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    muc = {"stt": len(LENH_CON) + 1, "ten": ten, "argv": cmd, "python": H.PYTHON, "python_cha": sys.executable,
           "cwd": str(H.ROOT), "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
           "giay": round(time.time() - t0, 3), "exit": ma, "exit_hex": None if ma is None else hex(ma & 0xFFFFFFFF),
           "het_gio": het_gio, "script_sha256": H.sha256_file(script),
           "so_lan_nap_detector": sum(ln.startswith(DAU_NAP) for ln in out_s.splitlines()),
           "stdout": out_s, "stderr": err_s}
    H.ghi_json(phien_test / "lenh_con" / f"{muc['stt']:03d}_{ten}.json", muc, [phien_test])
    LENH_CON.append({**{k: muc[k] for k in ("stt", "ten", "exit", "giay", "het_gio", "so_lan_nap_detector")},
                     "ly_do": ly_do(muc), "loi_git_dirty": "git_dirty_sha256" in err_s})
    if ma not in (0, 2, 10):
        print(f"[TIEN TRINH CON BAT THUONG] {ten}: exit {ma} ({muc['exit_hex']}), het_gio={het_gio}\n{err_s[-2000:]}")
    return muc


def ly_do(m: dict) -> list[str]:
    """Cac dong `[LOI HOP DONG] ...` trong stderr cua lenh con (chay_va_ghi_log in LoiHopDong -> exit 2)."""
    return [ln[len(DAU_LOI):].strip() for ln in m["stderr"].splitlines() if ln.startswith(DAU_LOI)]


def dung_ly_do(m: dict, *can: str, ds: list[str] | None = None) -> tuple[bool, dict]:
    """Kiem am DUNG NGUYEN NHAN (review_003): exit 2 VA dung MOT dong [LOI HOP DONG] chua moi chuoi `can`, khong co
    git_dirty_sha256 (vd PLAN doi giua phien). Neu `ds`: dong loi phai la '<tien to DS_LOI><list>' va list parse duoc
    phai BANG `ds` (thua muc nao, ke ca loi khoa PLAN ngoai du kien -> khong dat)."""
    ld = ly_do(m)
    ok = m["exit"] == 2 and len(ld) == 1 and all(c in ld[0] for c in can) and "git_dirty_sha256" not in ld[0]
    ds_thuc = None
    if ok and ds is not None:
        tien_to = next((t for t in DS_LOI if ld[0].startswith(t)), None)
        try:
            ds_thuc = ast.literal_eval(ld[0][len(tien_to):]) if tien_to else None
        except (ValueError, SyntaxError):
            ds_thuc = None
        ok = ds_thuc == list(ds)
    return ok, {"exit": m["exit"], "ly_do": ld, "ky_vong_chua": list(can), "ky_vong_ds": ds, "ds_thuc": ds_thuc}


def main_con(argv: list[str]) -> int:
    """Tien trinh con cho `cache`: gan detector MO PHONG (D.NAP_MODEL, D chi chap nhan voi giao thuc fixture) roi chay
    DUNG khoi __main__ cua dieu_phoi_bdd.py (Ctx -> vung_phien -> tao_thu_muc -> chuan_bi_moi_truong -> chay_va_ghi_log)."""
    import dieu_phoi_bdd as D

    def nap(det, cfg):
        print(DAU_NAP + det, flush=True)
        return ModelGia(cfg["loai_checkpoint"])
    D.NAP_MODEL = nap
    ctx0 = D.Ctx(Path(argv[argv.index("--giao-thuc") + 1]) if "--giao-thuc" in argv else D.GIAO_THUC)
    cong = argv[argv.index("--cong") + 1] if "--cong" in argv else "?"
    vung0 = D.vung_phien(ctx0, argv[argv.index("--phien") + 1], cong)
    H.tao_thu_muc(vung0[0], vung0)
    H.chuan_bi_moi_truong(vung0[0], vung0)
    return H.chay_va_ghi_log(D.main, argv, vung0[0], vung0, "dieu_phoi_bdd")


# ---------------------------------------------------------------------------
# The gioi tong hop
# ---------------------------------------------------------------------------
def doi_tuong(f: int):
    """(ten, bbox xywh, ten lop, conf detector) cua frame f (1-based)."""
    return [("A", (100 + 30 * f, 300.0, 80.0, 50.0), "car", 0.9), ("B", (600 - 20 * f, 320.0, 120.0, 80.0), "truck", 0.6),
            ("C", (300.0, 400 + 5 * f, 30.0, 40.0), "bicycle", 0.18)]


def tao_the_gioi(fx: Path, vung) -> dict:
    import numpy as np
    import cv2
    cats = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    cid = {c: i + 1 for i, c in enumerate(cats)}
    d = {"videos": [], "images": [], "annotations": [], "categories": [{"id": cid[c], "name": c} for c in cats]}
    img_id = ann_id = 0
    for vi, v in enumerate(VIDEOS + ["fxv-thieu"], 1):
        d["videos"].append({"id": vi, "name": v})
        for f in range(1, N_FRAME + 1):
            img_id += 1
            fn = f"{v}/{v}-{f:07d}.png"
            d["images"].append({"id": img_id, "video_id": vi, "frame_id": f - 1, "file_name": fn, "width": 1280, "height": 720})
            if v != "fxv-thieu":
                p = fx / "data/raw/bdd100k/images/track/val" / fn
                H.tao_thu_muc(p.parent, vung)
                ok, buf = cv2.imencode(".png", np.full((18, 32, 3), (vi * 20, f * 10, 0), np.uint8))
                H.ghi_bytes(p, buf.tobytes(), vung)
            if f in KHONG_ANN.get(v, set()):
                continue
            for k, (ten, bb, lop, _) in enumerate(doi_tuong(f)):
                ann_id += 1
                d["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cid[lop], "instance_id": k + 1,
                                         "bbox": list(bb), "iscrowd": 0, "ignore": 0})
            ann_id += 1
            d["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cid["car"], "instance_id": 9,
                                     "bbox": list(CROWD), "iscrowd": 1, "ignore": 0})
    sha_nhan = H.ghi_bytes(fx / "labels.json", json.dumps(d).encode("utf-8"), vung)
    sha_man = H.ghi_json(fx / "manifest.json", {"name": "fixture", "videos": VIDEOS}, vung)
    from mot_datasets.bdd100k_coco import df_for_gt_export, parse_coco, seq_lengths_from_images
    from mot_datasets.bdd100k_mot import export_motchallenge
    out_root = H.kiem_duong_ghi(fx / "data/processed", vung)
    H.tao_thu_muc(out_root, vung)
    export_motchallenge(df_for_gt_export(parse_coco(d, "val", VIDEOS)), "val", "vehicle",
                        seq_lengths_from_images(d, VIDEOS), out_root=out_root, videos=VIDEOS, benchmark="BDDCOCOVID")
    ck = {}
    for det in ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"):
        ck[det] = H.ghi_bytes(fx / f"models/fake_{det}.pt", f"checkpoint gia {det}".encode(), vung)
    return {"nhan": sha_nhan, "manifest": sha_man, "checkpoint": ck,
            "seqmap": H.sha256_file(fx / "data/processed/seqmaps/BDDCOCOVID-val.txt")}


def giao_thuc_fixture(goc_gt: dict, fx: Path, sha: dict, vung) -> Path:
    g = json.loads(json.dumps(goc_gt))
    g["che_do"], g["goc"] = "fixture", str(fx)
    for det in sha["checkpoint"]:
        g["detector"][det]["checkpoint"] = f"models/fake_{det}.pt"
        g["detector"][det]["sha256"] = sha["checkpoint"][det]
    ds = g["tap_du_lieu"]
    ds["manifest"] = {"rel": "manifest.json", "sha256": sha["manifest"]}
    ds["nhan"] = {"rel": "labels.json", "sha256": sha["nhan"]}
    ds["seqmap"] = {"rel": "data/processed/seqmaps/BDDCOCOVID-val.txt", "sha256": sha["seqmap"]}
    g["smoke_cong_A"]["videos"] = ["fxv-a"]
    g["tai_lap_cong_B"]["video"] = "fxv-b"
    g["thong_ke"]["bootstrap"].update({"so_mau": 300, "so_video": len(VIDEOS)})    # ngoai le so mau CHI qua giao thuc fixture
    # YAML: ban sao TRUNG BYTE trong fx (de thu sua YAML sau mo-phien ma khong cham file that); sha256 giu nguyen giao thuc
    import run_tracking as rt
    for bl in g["tracker"]["bo_loc"]:
        src = Path(rt.resolve_tracker_yaml(bl["yaml"]))
        dst = fx / "yaml" / src.name
        H.ghi_bytes(dst, src.read_bytes(), vung)
        bl["yaml"] = str(dst)
    # Ma nguon: ban sao TRUNG BYTE dong import tinh cua src/ (khoa fixture bam ban sao nay -> thu sua ma duoc)
    import dieu_phoi_bdd as D
    for p in D.dong_import_tinh(H.SRC):
        H.ghi_bytes(fx / "src" / p.relative_to(H.SRC), p.read_bytes(), vung)
    g["ma_nguon_goc"] = "src"
    p = fx / "giao_thuc_fixture.json"
    H.ghi_json(p, g, vung)
    return p


class _T:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _R:
    def __init__(self, a):
        self.boxes = type("B", (), {"data": _T(a)})()


class ModelGia:
    """Detector mo phong: doc (video, frame) tu mau anh tong hop; tra ket qua theo thu tu NMS (conf giam dan)."""

    def __init__(self, loai: str, tran: bool = False):
        self.names = dict(NAMES_COCO if loai == "coco80" else NAMES_LOP5)
        self.loai, self.tran = loai, tran
        self.id_lop = {n: i for i, n in self.names.items()}

    def predict(self, img, conf, iou, imgsz, classes, max_det, agnostic_nms, augment, device, verbose, **kw):
        import numpy as np
        vi, f = int(round(float(img[..., 0].mean()) / 20)), int(round(float(img[..., 1].mean()) / 10))
        rows = [[x, y, x + w, y + h, c, self.id_lop[lop]] for _, (x, y, w, h), lop, c in doi_tuong(f)]
        rows.append([CROWD[0], CROWD[1], CROWD[0] + CROWD[2], CROWD[1] + CROWD[3], 0.7, self.id_lop["car"]])
        if self.loai == "coco80":
            rows.append([50, 50, 90, 150, 0.8, 0])                     # person: bi loai luc load
        if vi == 1 and f == 1:                                         # canh conf quanh 0,10 / 0,25
            for k, t in enumerate((0.10, 0.25)):
                t32 = np.float32(t)
                for j, c in enumerate((np.nextafter(t32, np.float32(0)), t32, np.nextafter(t32, np.float32(1)))):
                    rows.append([1000 + 40 * j, 500 + 60 * k, 1030 + 40 * j, 540 + 60 * k, float(c), self.id_lop["car"]])
        if self.tran and f == 1:
            rows += [[0, 0, 10 + i % 50, 10, 0.5, self.id_lop["car"]] for i in range(3000)]
        a = np.array(rows, dtype=np.float32).reshape(-1, 6)
        a = a[np.argsort(-a[:, 4], kind="stable")]
        return [_R(a[a[:, 4] > conf][:max_det])]


# ---------------------------------------------------------------------------
# Kiem don vi
# ---------------------------------------------------------------------------
def kiem_loc_va_anh_xa(D):
    import numpy as np
    import det_cache
    import detector_classes as dc
    kiem("anh_xa_lop5_sang_coco", D.bang_anh_xa_coco(NAMES_LOP5, "lop5") == {0: 2, 1: 7, 2: 5, 3: 3, 4: 1})
    kiem("anh_xa_coco80_giu_nguyen", D.bang_anh_xa_coco(NAMES_COCO, "coco80") is None)
    thieu = dict(NAMES_LOP5)
    thieu[4] = "xe_dap"
    kiem("anh_xa_thieu_lop_bi_chan", bat_loi(lambda: D.bang_anh_xa_coco(thieu, "lop5"), dc.LoiAnhXaLop) is not None)
    kiem("anh_xa_sai_loai_bi_chan", bat_loi(lambda: D.bang_anh_xa_coco(NAMES_LOP5, "coco80"), H.LoiHopDong) is not None)
    # loc conf -> lop -> [:300]: 200 person diem cao dung TRUOC 350 car
    a = np.array([[0, 0, 1, 1, 0.99 - i * 1e-4, 0] for i in range(200)] +
                 [[0, 0, 1, 1, 0.9 - i * 1e-3, 2] for i in range(350)], dtype=np.float32)
    out = det_cache.filter_dets(a, 0.25, np.array([1, 2, 3, 5, 7]), 300, None)
    ky = a[200:][a[200:, 4] > 0.25][:300]
    kiem("loc_conf_roi_lop_roi_cat_300", np.array_equal(out, ky) and len(out) == 300,
         {"so_giu": len(out), "neu_cat_truoc": int((a[:300, 5] == 2).sum())})


def kiem_cache_va_timeline(D, ctx, A: Path, vung):
    import numpy as np
    import run_tracking as rt
    cd = A / "cache" / "yolov8n_mixed5"
    with np.load(cd / "fxv-a.npz") as z:
        data, cls_goc, fi, ts = z["data"], z["cls_goc"], z["frame_idx"], z["time_s"]
    kiem("cls_goc_lop5_va_cls_coco", set(cls_goc.tolist()) <= {0, 1, 4} and set(data[:, 5].astype(int).tolist()) <= {2, 7, 1}
         and all({0: 2, 1: 7, 4: 1}[int(g)] == int(c) for g, c in zip(cls_goc, data[:, 5])))
    with np.load(A / "cache" / "yolov8n_coco" / "fxv-a.npz") as z:
        kiem("coco80_cls_goc_bang_cls", np.array_equal(z["cls_goc"].astype(np.float32), z["data"][:, 5]))
    kiem("timeline_bdd_lien_tuc", fi.tolist() == list(range(1, N_FRAME + 1)) and ts.tolist() == [k / 5 for k in range(N_FRAME)])
    for t, ten in ((0.25, "c25"), (0.10, "c10")):
        vd = D.mo_cache(ctx, cd, "fxv-a", t)
        c = vd.frame(0)[:, 4]
        t32 = np.float32(t)
        kiem(f"conf_ngat_{ten}", (t32 not in c) and (np.nextafter(t32, np.float32(1)) in c)
             and (np.nextafter(t32, np.float32(0)) not in c) and bool((c > t).all()), sorted(c.tolist()))
    reg = rt.tracker_registry()
    bl = next(b for b in ctx.gt["tracker"]["bo_loc"] if b["ten"] == "ekf")
    cfg = D.cau_hinh_bo_loc(ctx, bl)
    vd = D.mo_cache(ctx, cd, "fxv-a", 0.1)
    d1, _, t1 = rt.track_video(vd, cfg, 0.2, "steps", None, "fxv-a", reg)
    d2, _, t2 = rt.track_video(D.mo_cache(ctx, cd, "fxv-a", 0.1), cfg, 0.2, "skip", None, "fxv-a", reg)
    kiem("steps_bang_skip_khi_n_steps_1", d1.equals(d2) and set(t1["n_steps"]) == {1} and set(t2["n_steps"]) == {1},
         {"n_steps": sorted(set(t1["n_steps"]))})
    kiem("track_buffer_6_buoc", int(cfg.track_buffer) == 6)


def kiem_cache_hong(D, ctx, A: Path, phien: Path, vung):
    import det_cache
    det = "yolov8n_mixed5"
    goc = A / "cache" / det
    spec = D.spec_cho(ctx, det, "cpu").fill_env()
    # truncated / unreadable bi chan khi TAO cache
    d_tran = phien / "don_vi" / "tran"
    H.tao_thu_muc(d_tran, vung)
    ng = D.kiem_nguon(ctx)
    ng["gt"]["fxv-thieu"] = ng["seqinfo"]["fxv-thieu"] = None
    e = bat_loi(lambda: D.tao_cache_video(ctx, det, ModelGia("lop5", tran=True), spec, d_tran, "fxv-a", vung, ng, "0" * 64),
                H.LoiHopDong)
    kiem("truncated_bi_chan_khi_tao", e is not None and "truncated 1" in e, e)
    e = bat_loi(lambda: D.tao_cache_video(ctx, det, ModelGia("lop5"), spec, d_tran, "fxv-thieu", vung, ng, "0" * 64),
                H.LoiHopDong)
    kiem("unreadable_bi_chan_khi_tao", e is not None and "unreadable 10" in e, e)
    kiem("khong_ghi_npz_khi_bi_chan", not any(d_tran.glob("*.npz")))

    def ban_sao(ten: str, sua=None, sua_marker=True):
        d = phien / "don_vi" / "cache_hong" / ten
        for f in ("meta.json", "fxv-a.npz", "fxv-a.json"):
            H.ghi_bytes(d / f, (goc / f).read_bytes(), vung)
        if sua:
            sua(d)
        if sua_marker:
            H.ghi_marker(d / "fxv-a.xong.json", [d / "fxv-a.npz", d / "fxv-a.json", d / "meta.json"], vung,
                         {"detector": det, "video": "fxv-a"})
        else:
            H.ghi_bytes(d / "fxv-a.xong.json", (goc / "fxv-a.xong.json").read_bytes(), vung)
        return d

    def sua_json(k, v):
        def f(d):
            j = H.doc_json(d / "fxv-a.json")
            if v is None:
                j.pop(k)
            else:
                j[k] = v
            H.ghi_json(d / "fxv-a.json", j, vung, ghi_de=True)
        return f
    kiem("ban_sao_nguyen_hop_le", D.kiem_cache_video(ctx, det, ban_sao("nguyen"), "fxv-a") == [])
    d = ban_sao("legacy", sua_json("spec_sha256", None))
    loi = D.kiem_cache_video(ctx, det, d, "fxv-a")
    kiem("legacy_bi_chan", any("legacy" in x for x in loi) and
         bat_loi(lambda: D.mo_cache(ctx, d, "fxv-a", 0.25), H.LoiHopDong) is not None, loi)
    loi = D.kiem_cache_video(ctx, det, ban_sao("nhan_sai", sua_json("nhan_sha256", "0" * 64)), "fxv-a")
    kiem("provenance_nhan_sai_bi_chan", any("nhan_sha256" in x for x in loi), loi)
    loi = D.kiem_cache_video(ctx, det, ban_sao("ckpt_sai", sua_json("checkpoint_sha256", "1" * 64)), "fxv-a")
    kiem("provenance_checkpoint_sai_bi_chan", any("checkpoint_sha256" in x for x in loi), loi)

    def sua_meta(d):
        m = H.doc_json(d / "meta.json")
        m["weights_sha256"] = "2" * 64
        H.ghi_json(d / "meta.json", m, vung, ghi_de=True)
    loi = D.kiem_cache_video(ctx, det, ban_sao("meta_sai", sua_meta), "fxv-a")
    kiem("meta_cau_hinh_sinh_sai_bi_chan", any("gen_spec" in x for x in loi), loi)

    def sua_npz(d):
        b = bytearray((d / "fxv-a.npz").read_bytes())
        b[-30] ^= 1
        H.ghi_bytes(d / "fxv-a.npz", bytes(b), vung, ghi_de=True)
    loi = D.kiem_cache_video(ctx, det, ban_sao("npz_doi", sua_npz, sua_marker=False), "fxv-a")
    kiem("hash_npz_doi_bi_chan", bool(loi) and "marker hong" in loi[0], loi)
    import numpy as np
    z = dict(np.load(goc / "fxv-a.npz"))
    z["truncated"] = z["truncated"].copy()
    z["truncated"][0] = True
    dt = phien / "don_vi" / "cache_hong" / "tran_npz"
    H.tao_thu_muc(dt, vung)
    H.kiem_duong_ghi(dt / "x.npz", vung)
    np.savez_compressed(dt / "x.npz", **z)
    kiem("truncated_bi_chan_khi_nap", bat_loi(lambda: det_cache.VideoDetections(dt / "x.npz", 0.25, [1, 2, 3, 5, 7], 300))
         is not None)


def don_vi_gia(D, run_dir: Path, run: str, video: str, mot: bytes, vung):
    """Don vi tracking gia DUNG hop dong marker (3 artifact + danh tinh run/video)."""
    data, dcls, tim = D.file_don_vi(run_dir, video)
    H.ghi_bytes(data, mot, vung)
    H.ghi_bytes(dcls, b"frame,id,cls\n", vung)
    H.ghi_bytes(tim, b"frame,n_steps\n", vung)
    H.ghi_marker(run_dir / "xong" / f"{video}.xong.json", [data, dcls, tim], vung, {"run": run, "video": video})


def kiem_evaluator_duong_ghi(D, ctx, phien: Path, vung):
    """R61-F3: moi duong ghi that cua cham_run (TrackEval summary/detailed, log loi, pandas, raw_fields) qua guard."""
    import _winapi
    gt = (ctx.gt_root / ctx.split / "fxv-b" / "gt" / "gt.txt").read_text().splitlines()
    rows = [",".join(x.split(",")[:6]) + ",1.0" for x in gt if x.split(",")[7] == "1"]
    goc = phien / "don_vi" / "f3"
    root, out_root = goc / "tracking", goc / "cham"
    don_vi_gia(D, root / ctx.split / "run_f3", "run_f3", "fxv-b", ("\n".join(rows) + "\n").encode(), vung)
    ngoai = H.tao_thu_muc(goc / "ngoai", vung)
    H.tao_thu_muc(out_root / "run_f3", vung)
    H.tao_thu_muc(out_root / "_trackeval", vung)
    ket = {}
    for ten, lk in (("trackeval_summary_dir", out_root / "_trackeval" / "run_f3"),
                    ("log_loi", out_root / "run_f3" / "trackeval_error.log"),
                    ("raw_fields", out_root / "run_f3" / "raw_fields.json"),
                    ("per_video_pandas", out_root / "run_f3" / "per_video_metrics.csv")):
        _winapi.CreateJunction(str(ngoai), str(lk))
        try:
            e = bat_loi(lambda: D.cham_run(ctx, "run_f3", ["fxv-b"], root, out_root, vung), H.LoiHopDong)
        finally:
            os.rmdir(lk)
        ket[ten] = e
        kiem(f"cham_run_junction_{ten}_bi_chan_truoc_ghi", e is not None and "reparse" in e and not any(ngoai.iterdir())
             and not (out_root / "run_f3" / "cham.xong.json").exists(), e)
    # nhanh log loi: file MOT hong -> TrackEval nem loi; log + stdout nam TRONG phien, loi khong bi nuot
    don_vi_gia(D, root / ctx.split / "run_loi", "run_loi", "fxv-b", b"x,y,z\n", vung)
    e = bat_loi(lambda: D.cham_run(ctx, "run_loi", ["fxv-b"], root, out_root, vung))
    log = out_root / "run_loi" / "trackeval_error.log"
    kiem("cham_run_nhanh_loi_log_trong_phien_khong_nuot",
         e is not None and not e.startswith("LoiHopDong") and log.is_file() and log.stat().st_size > 0
         and (out_root / "run_loi" / "trackeval_stdout.txt").is_file() and not (out_root / "run_loi" / "cham.xong.json").exists(),
         {"loi": e, "log": H.rel(log)})
    c = D.cham_run(ctx, "run_f3", ["fxv-b"], root, out_root, vung)      # sau khi go junction: cham duoc, dung vung
    tep = sorted(H.rel(p) for p in H.liet_ke_file(out_root) if p.is_file())
    kiem("cham_run_chi_ghi_trong_vung_sau_go_junction", int(c["IDSW"]) == 0 and not any(ngoai.iterdir())
         and all(t.startswith(H.rel(out_root)) for t in tep), tep)
    return ket


def kiem_ma_nguon(D, ctx, fx: Path):
    """R61-F5: danh muc ma nguon XAC DINH (ast), doc lap lich su import; ban sao fixture trung byte."""
    tinh = {p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)}
    can = {"r61_bdd/thong_ke_bdd.py", "r61_bdd/dieu_phoi_bdd.py", "r61_bdd/hop_dong_bdd.py", "boot_stats.py", "det_cache.py",
           "detector_classes.py", "run_tracking.py", "run_trackeval.py", "trackeval_ext.py", "tracker_ctrv.py", "ekf_ctrv.py",
           "ukf_ctrv.py", "pf_ctrv.py", "config.py", "time_units.py", "mot_datasets/bdd100k_coco.py"}
    kiem("ma_nguon_dong_import_tinh_du", can <= set(tinh), sorted(can - set(tinh)))
    tam = {m: sys.modules.pop(m) for m in ("thong_ke_bdd", "boot_stats") if m in sys.modules}
    try:
        khong_nap = {p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)}
    finally:
        sys.modules.update(tam)
    kiem("ma_nguon_doc_lap_lich_su_import", khong_nap == tinh and len(tam) == 2, {"so_file": len(tinh), "go": sorted(tam)})
    kiem("ma_nguon_ban_sao_fixture_trung_byte", D.ma_nguon(ctx) == tinh and ctx.src_goc == fx / "src")
    return tinh


def kiem_so_mau_san_xuat(D, gt_that: Path, phien: Path, vung):
    """R61-F6: ngoai le so mau chi cho fixture; giao thuc san xuat != 5000 -> 2."""
    g = H.doc_json(gt_that)
    g["thong_ke"]["bootstrap"]["so_mau"] = 300
    p = phien / "don_vi" / "giao_thuc_sx_300.json"
    H.ghi_json(p, g, vung)
    e = bat_loi(lambda: D.Ctx(p), H.LoiHopDong)
    kiem("so_mau_san_xuat_khac_5000_bi_chan", e is not None and D.Ctx(gt_that).so_mau == 5000, e)


def kiem_evaluator_crowd(D, ctx, phien: Path, vung):
    import trackeval
    import run_trackeval as rte
    gt = (ctx.gt_root / ctx.split / "fxv-b" / "gt" / "gt.txt").read_text().splitlines()
    lop1 = [ln.split(",") for ln in gt if ln.split(",")[7] == "1"]
    frame_trong = sorted(set(range(1, N_FRAME + 1)) - {int(x[0]) for x in lop1})
    kiem("gt_frame_xe_trong_trong_timeline", frame_trong == [8, 9, 10] and any(ln.split(",")[7] == "13" for ln in gt), frame_trong)
    rows = [f"{x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]},1.0" for x in lop1]
    rows += [f"2,99,{CROWD[0]},{CROWD[1]},{CROWD[2]},{CROWD[3]},0.9", "9,98,100,300,80,50,0.9"]
    root = phien / "don_vi" / "crowd" / "tracking"
    run_dir = root / ctx.split / "run_crowd"
    don_vi_gia(D, run_dir, "run_crowd", "fxv-b", ("\n".join(rows) + "\n").encode(), vung)
    c = D.cham_run(ctx, "run_crowd", ["fxv-b"], root, phien / "don_vi" / "crowd" / "cham", vung)
    eval_cfg, ds_cfg, met_cfg = rte.build_configs(ctx.split, "run_crowd", None)
    out = H.kiem_duong_ghi(phien / "don_vi" / "crowd" / "cham_mot", vung)
    ds_cfg.update({"GT_FOLDER": str(ctx.gt_root), "TRACKERS_FOLDER": str(root), "SEQ_INFO": {"fxv-b": N_FRAME},
                   "OUTPUT_FOLDER": str(out)})
    eval_cfg.update({"LOG_ON_ERROR": str(out / "loi.log"), "PRINT_RESULTS": False})
    res, _ = trackeval.Evaluator(eval_cfg).evaluate([rte.dataset_class("mot")(ds_cfg)],
                                                    [trackeval.metrics.CLEAR(met_cfg)])
    fp_mot = int(next(iter(res.values()))["run_crowd"]["fxv-b"]["pedestrian"]["CLEAR"]["CLR_FP"])
    kiem("mot_crowd_lop13_loai_fp_va_frame_trong_tinh_fp", int(c["FP"]) == 1 and fp_mot == 2 and int(c["IDSW"]) == 0,
         {"fp_mot_crowd": int(c["FP"]), "fp_mot": fp_mot})


def raw_gia(rng, videos, thang):
    import numpy as np
    out = {}
    for v in videos:
        tp = rng.integers(5, 50, 19).astype(float) * thang
        out[v] = {"HOTA_TP": tp.tolist(), "HOTA_FN": (rng.integers(0, 30, 19) * thang).tolist(),
                  "HOTA_FP": (rng.integers(0, 30, 19) * thang).tolist(), "AssA": rng.uniform(0.2, 0.9, 19).tolist(),
                  "AssRe": rng.uniform(0.2, 0.9, 19).tolist(), "AssPr": rng.uniform(0.2, 0.9, 19).tolist(),
                  "IDTP": int(rng.integers(10, 90) * thang), "IDFN": int(rng.integers(0, 40)), "IDFP": int(rng.integers(0, 40)),
                  "CLR_TP": int(rng.integers(10, 90)), "CLR_FN": int(rng.integers(0, 40)), "CLR_FP": int(rng.integers(0, 40)),
                  "IDSW": int(rng.integers(0, 9))}
        _ = np
    return out


def kiem_thong_ke(T, phien: Path, vung) -> dict:
    import io
    import numpy as np
    import boot_stats as bs
    rng = np.random.default_rng(123)
    vids = [f"v{i}" for i in range(7)]
    raws = {"cv": [raw_gia(rng, vids, 1)], "ekf": [raw_gia(rng, vids, 2)], "ukf": [raw_gia(rng, vids, 1)],
            "pf500": [raw_gia(rng, vids, s) for s in (1, 3, 7)]}
    pf = raws["pf500"]
    a = T.metric_bo_loc([T.xep(r, vids) for r in pf])["HOTA"]
    b = bs.combine(bs.average_seeds(pf), vids)["HOTA"]
    c = float(np.mean([np.mean([bs.combine(r, [v])["HOTA"] for v in vids]) for r in pf]))
    kiem("pf_trung_binh_combined_khac_gop_raw_va_tb_video", abs(a - b) > 1e-6 and abs(a - c) > 1e-6,
         {"tb_combined_seed": a, "combined_raw_gop": b, "tb_metric_video": c})
    so_mau = 300
    cs = T.tao_chi_so(len(vids), so_mau, 0)
    r2 = np.random.default_rng(0)
    kiem("chi_so_cung_day_boot_stats", np.array_equal(cs, np.stack([r2.integers(0, 7, size=7) for _ in range(so_mau)])))
    r = T.bootstrap_nhom(raws, vids, cs)
    ref = bs.paired_bootstrap(raws["ekf"][0], raws["cv"][0], vids, so_mau, 0)
    x = r[("ekf", "cv")]
    kiem("ghep_cap_khop_boot_stats_paired_bootstrap",
         all(abs(x[m]["ci_low"] - ref[m]["ci_low"] * T.HE_SO[m]) < 1e-9 and abs(x[m]["ci_high"] - ref[m]["ci_high"] * T.HE_SO[m])
             < 1e-9 and abs(x[m]["diff"] - ref[m]["diff"] * T.HE_SO[m]) < 1e-9 for m in T.METRIC))
    bio = io.BytesIO()
    np.save(bio, cs)
    p = phien / "bang_chung" / "chi_so_bootstrap_mau.npy"
    H.ghi_bytes(p, bio.getvalue(), vung)
    r_lai = T.bootstrap_nhom(raws, vids, np.load(p))
    kiem("tai_tinh_tu_chi_so_luu_trung", all(np.array_equal(r[k][m]["mau"], r_lai[k][m]["mau"]) for k in r for m in T.METRIC))
    kiem("don_vi_pp_count", all(r[k]["HOTA"]["don_vi"] == "pp" and r[k]["IDF1"]["don_vi"] == "pp" and
                                r[k]["IDSW"]["don_vi"] == "count" for k in r))
    mau = {f"{k[0]}-{k[1]}": {m: {kk: vv for kk, vv in r[k][m].items() if kk != "mau"} for m in T.METRIC} for k in r}
    H.ghi_json(phien / "bang_chung" / "bootstrap_mau.json", {"so_mau": so_mau, "videos": vids, "ket_qua": mau}, vung)
    return mau


def kiem_duong_dan(phien: Path, vung):
    import _winapi
    e = bat_loi(lambda: H.ghi_bytes(phien / "a" / ".." / ".." / "thoat.txt", b"x", vung), H.LoiHopDong)
    kiem("thoat_vung_bi_chan", e is not None and not (phien.parent / "thoat.txt").exists(), e)
    ngoai = H.tao_thu_muc(phien / "dd" / "ngoai", vung)
    vj = H.tao_thu_muc(phien / "dd" / "vung_j", vung)
    lk = vj / "lien_ket"
    _winapi.CreateJunction(str(ngoai), str(lk))
    e1 = bat_loi(lambda: H.ghi_bytes(lk / "a.txt", b"x", [vj]), H.LoiHopDong)
    cha = phien / "dd" / "cha_j"
    _winapi.CreateJunction(str(ngoai), str(cha))
    e2 = bat_loi(lambda: H.tao_thu_muc(cha / "con" / "chau", [cha / "con"]), H.LoiHopDong)
    e3 = bat_loi(lambda: H.ghi_bytes(cha / "b.txt", b"x", [phien]), H.LoiHopDong)
    rong = not any(ngoai.iterdir())
    os.rmdir(lk)
    os.rmdir(cha)
    kiem("junction_bi_chan_truoc_ghi_ke_ca_thu_muc_cha", None not in (e1, e2, e3) and rong, [e1, e2, e3])


# ---------------------------------------------------------------------------
# Luong A/B
# ---------------------------------------------------------------------------
def luong_ab(D, gtp: Path, phien: Path, vung) -> dict:
    A, B = phien / "A" / "phien_A_01", phien / "B" / "phien_B_01"
    exits = []

    def chay_m(cong, buoc, *them, ph=None) -> dict:
        p = ph or (A if cong == "A" else B)
        argv = ["--cong", cong, "--buoc", buoc, "--giao-thuc", str(gtp), "--phien", str(p), *them]
        m = chay_con(phien, argv, f"{cong}_{buoc}")
        exits.append({"stt": m["stt"], "argv": m["argv"], "exit": m["exit"]})
        return m

    def chay(cong, buoc, *them, ph=None):
        return chay_m(cong, buoc, *them, ph=ph)["exit"]

    def kiem_am(ten, m, *can, ds=None, them_ok=True, them_ct=None):
        ok, ct = dung_ly_do(m, *can, ds=ds)
        kiem(ten, ok and them_ok, {**ct, "stt": m["stt"], **(them_ct or {})})

    kiem("A_tien_kiem", chay("A", "tien-kiem") == 0)
    det0 = "yolov8n_coco"
    m1 = chay("A", "cache", "--detector", det0, "--device", "cpu", "--gioi-han-don-vi", "1")
    u = A / "cache" / det0 / "fxv-a.npz"
    truoc = (H.sha256_file(u), os.stat(u).st_mtime_ns)
    m2 = chay("A", "cache", "--detector", det0, "--device", "cpu", "--gioi-han-don-vi", "1")
    m3 = chay("A", "cache", "--detector", det0, "--device", "cpu")
    kiem("cache_ngat_tiep_tuc_khong_ghi_de", (m1, m2, m3) == (10, 10, 0) and (H.sha256_file(u), os.stat(u).st_mtime_ns) == truoc,
         [m1, m2, m3])
    for det in ("yolov8n_mixed5", "yolo11n_mixed5"):
        kiem(f"A_cache_{det}", chay("A", "cache", "--detector", det, "--device", "cpu") == 0)
    kiem("A_kiem_cache", chay("A", "kiem-cache") == 0)
    kiem_am("A_cong_sai_bi_chan", chay_m("B", "smoke", ph=phien / "B" / "cong_sai"), "buoc smoke khong thuoc cong B")
    kiem("A_smoke", chay("A", "smoke") == 0)
    kiem("A_khoa", chay("A", "khoa") == 0)
    kiem("A_kiem_khoa", chay("A", "kiem-khoa") == 0)
    kiem("A_dong_goi", chay("A", "dong-goi-a") == 0)
    snap_A = H.bang_hash_thu_muc(A)
    goi = A / "goi_a.json"
    sha_goi, sha_khoa = H.sha256_file(goi), H.sha256_file(A / "khoa" / "khoa.json")
    vao = phien / "B" / "dau_vao"
    H.ghi_json(vao / "review_revise.json", {"verdict": "revise", "goi_a_sha256": sha_goi}, vung)
    H.ghi_json(vao / "review_a.json", {"verdict": "approve", "goi_a_sha256": sha_goi}, vung)
    H.ghi_json(vao / "plan_b.json", {"duyet": True, "goi_a_sha256": sha_goi, "khoa_sha256": sha_khoa}, vung)
    kiem_am("B_mo_phien_review_chua_approve_bi_chan",
            chay_m("B", "mo-phien", "--goi-a", str(goi), "--review-a", str(vao / "review_revise.json"), "--plan-b",
                   str(vao / "plan_b.json"), ph=phien / "B" / "phien_tu_choi"),
            "review A khong approve hoac khong rang buoc dung goi A")
    kiem_am("B_chua_mo_phien_bi_chan", chay_m("B", "tracking", ph=phien / "B" / "chua_mo"), "chua mo-phien B")
    kiem("B_mo_phien", chay("B", "mo-phien", "--goi-a", str(goi), "--review-a", str(vao / "review_a.json"), "--plan-b",
                            str(vao / "plan_b.json")) == 0)
    t1 = chay("B", "tracking", "--gioi-han-don-vi", "20")
    u = B / "tracking" / "BDDCOCOVID-val" / "r61-yolov8n_coco-c25-cv" / "data" / "fxv-a.txt"
    truoc = (H.sha256_file(u), os.stat(u).st_mtime_ns)
    c1 = chay_m("B", "cham")
    c1_chua_cham = not (B / "cham" / "r61-yolov8n_coco-c25-cv" / "cham.xong.json").exists()
    t2 = chay("B", "tracking", "--so-doan", "2", "--gioi-han-don-vi", "30")
    t3 = chay("B", "tracking")
    kiem("B_tracking_ngat_tiep_tuc_khong_ghi_de", (t1, t2, t3) == (10, 10, 0) and
         (H.sha256_file(u), os.stat(u).st_mtime_ns) == truoc, [t1, t2, t3])
    kiem_am("B_thieu_don_vi_khong_cham_full", c1, "don vi tracking", "-- khong cham", them_ok=c1_chua_cham,
            them_ct={"chua_co_cham_xong": c1_chua_cham})
    kiem("B_cham", chay("B", "cham") == 0)
    kiem("B_lap_tracker_trung_byte", chay("B", "lap-tracker") == 0)
    kiem("B_cham_lai_trung_byte", chay("B", "cham-lai") == 0)
    kiem("B_tong_hop", chay("B", "tong-hop") == 0)
    kiem("B_kiem_khoa", chay("B", "kiem-khoa") == 0)
    kiem("B_hau_kiem", chay("B", "hau-kiem") == 0)
    kiem("goi_A_bat_bien_khi_B_chay", H.bang_hash_thu_muc(A) == snap_A)
    n_kk_A = len(list((A / "kiem_khoa").glob("*/ket_qua.json")))
    kiem("kiem_khoa_B_ghi_trong_B_khong_ghi_A", n_kk_A == 1 and len(list((B / "kiem_khoa").glob("*/ket_qua.json"))) == 1)

    def thu_sua(p: Path, buoc: str, ten: str, *can: str):
        goc = p.read_bytes()
        p.write_bytes(goc + b" ")
        try:
            m = chay_m("B", buoc)
        finally:
            p.write_bytes(goc)
        kiem_am(ten, m, *can)
    npz_b = H.rel(A / "cache" / det0 / "fxv-b.npz")
    thu_sua(A / "khoa" / "khoa.json", "cham", "sua_khoa_bi_chan", "khoa doi sau mo-phien")
    thu_sua(goi, "tong-hop", "sua_goi_a_bi_chan", "goi_a doi sau mo-phien")
    thu_sua(gtp, "cham", "sua_giao_thuc_bi_chan", "giao thuc doi sau mo-phien")
    thu_sua(A / "cache" / det0 / "fxv-b.npz", "kiem-khoa", "sua_cache_A_kiem_khoa_B_bi_chan", f"goi A bi doi: lech ['{npz_b}']")
    thu_sua(A / "cache" / det0 / "fxv-b.npz", "lap-tracker", "sua_cache_A_tracking_B_bi_chan", f"goi A bi doi: lech ['{npz_b}']")
    thu_sua(u, "tracking", "sua_don_vi_hoan_tat_bi_chan_khong_chay_de",
            f"marker {H.rel(u.parent.parent / 'xong' / 'fxv-a.xong.json')} hong", "bam lech ../data/fxv-a.txt")
    kiem("goi_A_khoi_phuc_nguyen_byte", H.bang_hash_thu_muc(A) == snap_A)
    pf = [H.doc_json(B / "tracking" / "BDDCOCOVID-val" / f"r61-yolo11n_mixed5-c10-pf500-s{s}" / "xong" / "fxv-c.xong.json")
          for s in (0, 1, 2)]
    kiem("pf500_ba_seed_hieu_luc", [x["seed_thuc_dung"] for x in pf] == [0, 1, 2] and
         all((x["num_particles"], x["optimal_proposal"], x["roughening"]) == (500, False, 0.0) for x in pf), pf)
    import csv
    ci = list(csv.DictReader((B / "tong_hop" / "ci_108.csv").read_text().splitlines()))
    kiem("cau_truc_108_ci", len(ci) == 108 and len({(r["detector"], r["dieu_kien"]) for r in ci}) == 6 and
         all(float(r["ci_low"]) <= float(r["ci_high"]) for r in ci) and
         all(r["dau"] == ("+" if float(r["diff"]) > 0 else "-" if float(r["diff"]) < 0 else "0") for r in ci) and
         {r["don_vi"] for r in ci if r["metric"] == "IDSW"} == {"count"})
    kc = H.doc_json(B / "tong_hop" / "kiem_chung.json")
    kiem("tai_to_hop_khop_evaluator_trong_dung_sai", all(kc["sai_so_tai_to_hop_max"][m] <= kc["dung_sai"][m]
                                                        for m in ("HOTA", "IDF1", "IDSW")), kc["sai_so_tai_to_hop_max"])
    kiem("tong_hop_so_mau_tu_giao_thuc_fixture", kc["so_mau"] == 300 and kc["chi_so_shape"] == [300, 3]
         and kc["chi_so_dtype"] == "int64")

    # --- R61-F7: tham so bo loc hieu luc tren instance that == giao thuc --------------------------------------
    gt_fx = H.doc_json(gtp)["tracker"]["tham_so_bo_loc_du_kien"]
    tk = H.doc_json(A / "tien_kiem" / "tien_kiem.json")["cau_hinh_hieu_luc"]
    kiem("tham_so_bo_loc_tien_kiem_khop_giao_thuc", all(tk[b]["tham_so_bo_loc"] == gt_fx[b] for b in ("cv", "ekf", "ukf", "pf500")),
         {b: tk[b]["tham_so_bo_loc"] for b in tk})
    kiem("tham_so_bo_loc_don_vi_pf_khop_giao_thuc", all(x["tham_so_bo_loc"] == gt_fx["pf500"] for x in pf))

    # --- snapshot B (byte + mtime, bo cli_log/_tam/_cau_hinh_thu_vien) ----------------------------------------
    def snap_B():
        out = {}
        for p in H.liet_ke_file(B):
            r = p.relative_to(B).as_posix()
            if p.is_file() and p.name != "cli_log.json" and not r.startswith(("_tam/", "_cau_hinh_thu_vien/")):
                out[r] = (H.sha256_file(p), os.stat(p).st_mtime_ns)
        return out

    def thu_doi(p: Path, buoc: str, ten: str, *can: str, bien=lambda b: b + b" ", ds=None):
        """Sua p -> chup B -> chay buoc (exit 2 DUNG nguyen nhan `can`/`ds`) -> chup lai (B khong doi byte/mtime, ke ca
        file bi sua) -> khoi phuc."""
        goc = p.read_bytes()
        p.write_bytes(bien(goc))
        try:
            s0 = snap_B()
            m = chay_m("B", buoc)
            s1 = snap_B()
        finally:
            p.write_bytes(goc)
        doi = sorted(k for k in set(s0) | set(s1) if s0.get(k) != s1.get(k))
        kiem_am(ten, m, *can, ds=ds, them_ok=not doi, them_ct={"doi": doi[:5]})

    # --- R61-F1: doi nguon/YAML/artifact goi A SAU mo-phien -> 2 truoc moi ket qua phu thuoc ------------------
    fx = gtp.parent
    thu_doi(fx / "data/processed/BDDCOCOVID-val/fxv-a/gt/gt.txt", "cham", "F1_doi_gt_sau_mo_phien_bi_chan",
            ds=["khoa lech: nguon"])
    thu_doi(fx / "data/processed/BDDCOCOVID-val/fxv-b/seqinfo.ini", "tracking", "F1_doi_seqinfo_sau_mo_phien_bi_chan",
            ds=["khoa lech: nguon"])
    thu_doi(fx / "yaml" / "bytetrack_ekf_ctrv.yaml", "lap-tracker", "F1_doi_yaml_sau_mo_phien_bi_chan",
            f"ekf: yaml {H.rel(fx / 'yaml' / 'bytetrack_ekf_ctrv.yaml')} sha256 lech giao thuc")
    thu_doi(fx / "labels.json", "tracking", "F1_doi_nhan_sau_mo_phien_bi_chan", "nhan sha256 lech")
    thu_doi(fx / "manifest.json", "cham-lai", "F1_doi_manifest_sau_mo_phien_bi_chan", "manifest sha256 lech")
    sm = A / "smoke" / "cham" / "r61-yolov8n_coco-c25-cv" / "combined_metrics.csv"
    thu_doi(sm, "tong-hop", "F1_doi_artifact_smoke_goi_A_bi_chan", f"goi A bi doi: lech ['{H.rel(sm)}'] thua []")
    # --- R61-F5: doi ma nguon da khoa (thong ke / helper) -> 2, khong phu thuoc thu tu import ----------------
    thu_doi(fx / "src" / "r61_bdd" / "thong_ke_bdd.py", "tong-hop", "F5_doi_ma_thong_ke_bi_chan",
            bien=lambda b: b + b"\n# doi\n", ds=["ma_nguon: r61_bdd/thong_ke_bdd.py doi/thieu/thua"])
    thu_doi(fx / "src" / "boot_stats.py", "cham-lai", "F5_doi_ma_boot_stats_bi_chan",
            bien=lambda b: b + b"\n# doi\n", ds=["ma_nguon: boot_stats.py doi/thieu/thua"])
    k = H.doc_json(A / "khoa" / "khoa.json")
    kiem("F5_khoa_co_ma_nguon_thong_ke_va_git_dirty", "r61_bdd/thong_ke_bdd.py" in k["ma_nguon"] and "ma_da_nap" not in k
         and isinstance(k["git_dirty_sha256"], dict) and len(k["ma_thu_vien"]) > 50,
         {"so_ma_nguon": len(k["ma_nguon"]), "so_ma_thu_vien": len(k["ma_thu_vien"]), "dirty": sorted(k["git_dirty_sha256"])})

    # --- R61-F4: marker hong / ket qua hoan tat bi doi -> 2, khong ghi de (byte + mtime) -----------------------
    run0, run_cl = "r61-yolov8n_coco-c25-cv", H.doc_json(gtp)["tai_lap_cong_B"]["cham_lai"][0]
    lap_mk = B / "lap" / "BDDCOCOVID-val" / run0 / "xong" / "fxv-b.xong.json"

    def sua_mk(f):
        return lambda b: (json.dumps(f(json.loads(b.decode("utf-8")))) + "\n").encode()
    hong_lap = f"marker {H.rel(lap_mk)} hong (khong chay de/ghi de)"
    cham_mk = B / "cham" / run0 / "cham.xong.json"
    hong_cham = f"marker {H.rel(cham_mk)} hong (khong chay de/ghi de)"
    thu_doi(lap_mk, "lap-tracker", "F4_marker_rong_bi_chan", hong_lap, "schema marker sai / artifact rong",
            bien=lambda b: b"{}\n")
    thu_doi(lap_mk, "lap-tracker", "F4_marker_artifact_rong_bi_chan", hong_lap, "schema marker sai / artifact rong",
            bien=sua_mk(lambda d: {**d, "artifact": {}}))
    thu_doi(lap_mk, "lap-tracker", "F4_marker_thieu_artifact_bi_chan", hong_lap,
            "tap artifact ['../data/fxv-b.txt', '../data_cls/fxv-b.csv'] != ky vong",
            bien=sua_mk(lambda d: {**d, "artifact": {r: h for r, h in d["artifact"].items() if "timing" not in r}}))
    thu_doi(lap_mk, "lap-tracker", "F4_marker_sai_danh_tinh_bi_chan", hong_lap, "danh tinh video='fxv-a' != 'fxv-b'",
            bien=sua_mk(lambda d: {**d, "video": "fxv-a"}))
    thu_doi(B / "lap" / "BDDCOCOVID-val" / run0 / "data" / "fxv-b.txt", "lap-tracker", "F4_mot_lap_bi_doi_khong_ghi_de",
            hong_lap, "bam lech ../data/fxv-b.txt")
    thu_doi(B / "cham" / run0 / "combined_metrics.csv", "cham", "F4_metric_cham_bi_doi_khong_ghi_de",
            hong_cham, "bam lech combined_metrics.csv")
    thu_doi(cham_mk, "cham", "F4_marker_cham_rong_bi_chan", hong_cham, "schema marker sai / artifact rong",
            bien=lambda b: b"{}\n")
    thu_doi(B / "cham_lai" / run_cl / "raw_fields.json", "cham-lai", "F4_metric_cham_lai_bi_doi_khong_ghi_de",
            f"marker {H.rel(B / 'cham_lai' / run_cl / 'cham.xong.json')} hong (khong chay de/ghi de)",
            "bam lech raw_fields.json")

    # --- R61-F6: chi so bootstrap sai / CSV tong hop bi sua -> tong-hop 2 truoc ghi, hau-kiem 2 ---------------
    import io
    import numpy as np
    import thong_ke_bdd as T
    cs_p = B / "tong_hop" / "chi_so_bootstrap.npy"

    def npy(a):
        b = io.BytesIO()
        np.save(b, a)
        return lambda _: b.getvalue()
    cs = np.load(cs_p)
    am, doi1 = cs.copy(), cs.copy()
    am[0, 0] = -1
    doi1[0, 0] = (doi1[0, 0] + 1) % 3
    hk_p = B / "hau_kiem.json"

    def kiem_hau_kiem(ten, ds_ky_vong, them_ok=True):
        """hau-kiem: stderr 'hau-kiem: [...]' == ds_ky_vong (tung muc) VA artifact hau_kiem.json vua ghi cung loi do."""
        m = chay_m("B", "hau-kiem")
        hk = H.doc_json(hk_p) if hk_p.is_file() else {}
        kiem_am(ten, m, ds=ds_ky_vong, them_ok=them_ok and hk.get("dat") is False and hk.get("loi") == ds_ky_vong,
                them_ct={"hau_kiem_json_loi": hk.get("loi")})
    loi_cs = {"so_mau_299": "chi so bootstrap dtype/shape int64/(299, 3) != int64/(300, 3)",
              "chi_so_am": "chi so bootstrap ngoai mien [0, so_video)",
              "doi_noi_dung": "chi so bootstrap khac day numpy.random.default_rng(0)",
              "dtype_int32": "chi so bootstrap dtype/shape int32/(300, 3) != int64/(300, 3)"}
    for ten, bien in (("so_mau_299", npy(T.tao_chi_so(3, 299, 0))), ("chi_so_am", npy(am)), ("doi_noi_dung", npy(doi1)),
                      ("dtype_int32", npy(cs.astype(np.int32)))):
        thu_doi(cs_p, "tong-hop", f"F6_chi_so_{ten}_tong_hop_bi_chan", loi_cs[ten], bien=bien)
        goc = cs_p.read_bytes()
        cs_p.write_bytes(bien(goc))
        try:
            kiem_hau_kiem(f"F6_chi_so_{ten}_hau_kiem_bi_chan", [f"thong ke: {loi_cs[ten]}"])
        finally:
            cs_p.write_bytes(goc)
    loi_f = {"ci_108.csv": "ci_108.csv khac tinh lai tu chi so luu", "bang_chinh.csv": "bang_chinh.csv khac tinh lai tu chi so luu",
             "kiem_chung.json": "kiem_chung so_mau lech"}
    for f in ("ci_108.csv", "bang_chinh.csv", "kiem_chung.json"):
        p = B / "tong_hop" / f
        goc = p.read_bytes()
        p.write_bytes(goc.replace(b"0", b"1", 1) if f != "kiem_chung.json" else goc.replace(b'"so_mau": 300', b'"so_mau": 299'))
        try:
            da_sua = p.read_bytes() != goc
            kiem_hau_kiem(f"F6_sua_{f}_sau_tong_hop_hau_kiem_bi_chan", [loi_f[f]], them_ok=da_sua)
        finally:
            p.write_bytes(goc)
    m = chay_m("B", "hau-kiem")
    hk = H.doc_json(hk_p) if hk_p.is_file() else {}
    kiem("F6_hau_kiem_dat_lai_sau_khoi_phuc", m["exit"] == 0 and not ly_do(m) and hk.get("dat") is True and hk.get("loi") == [],
         {"exit": m["exit"], "ly_do": ly_do(m), "hau_kiem_json": {k: hk.get(k) for k in ("dat", "loi", "dem")}})
    kiem("goi_A_nguyen_byte_sau_moi_thu_sua", H.bang_hash_thu_muc(A) == snap_A)
    return {"A": H.rel(A), "B": H.rel(B), "exits": exits, "snapshot_A": snap_A, "kiem_chung_tong_hop": kc}


def luong_cache_f2(gtp: Path, phien: Path) -> None:
    """R61-F2: cache chi chay khi co tien-kiem va nguon thuc te khop; chan TRUOC khi nap detector/ghi cache.
    So lan nap detector mo phong dem tu stdout cua tien trinh con (DAU_NAP)."""
    dem_nap: list[int] = []

    def chay_m(ph, buoc, *them) -> dict:
        argv = ["--cong", "A", "--buoc", buoc, "--giao-thuc", str(gtp), "--phien", str(ph), *them]
        m = chay_con(phien, argv, f"F2_A_{buoc}")
        dem_nap.extend([1] * m["so_lan_nap_detector"])
        return m

    def chay(ph, buoc, *them):
        return chay_m(ph, buoc, *them)["exit"]

    def npz(d):
        return [p for p in H.liet_ke_file(d) if p.suffix == ".npz"]
    lenh = ("--detector", "yolov8n_coco", "--device", "cpu", "--gioi-han-don-vi", "1")
    ak = phien / "A" / "phien_A_khong_tien_kiem"
    n0 = len(dem_nap)
    m = chay_m(ak, "cache", *lenh)
    ok, ct = dung_ly_do(m, "chua co tien-kiem hop le trong phien A")
    kiem("F2_cache_khong_tien_kiem_bi_chan", ok and len(dem_nap) == n0 and not npz(ak) and not (ak / "cache").exists(), ct)
    an = phien / "A" / "phien_A_nguon_doi"
    kiem("F2_tien_kiem_phien_moi", chay(an, "tien-kiem") == 0)
    fx = gtp.parent
    for ten, p, can in (("nhan", fx / "labels.json", "nhan sha256 lech"),
                        ("manifest", fx / "manifest.json", "manifest sha256 lech"),
                        ("gt", fx / "data/processed/BDDCOCOVID-val/fxv-c/gt/gt.txt", "nguon thuc te khac tien-kiem: ['gt']")):
        goc = p.read_bytes()
        p.write_bytes(goc + b" ")
        try:
            n0 = len(dem_nap)
            m = chay_m(an, "cache", *lenh)
        finally:
            p.write_bytes(goc)
        ok, ct = dung_ly_do(m, can)
        kiem(f"F2_cache_{ten}_doi_sau_tien_kiem_bi_chan", ok and len(dem_nap) == n0 and not npz(an)
             and not (an / "cache").exists(), ct)
    n0 = len(dem_nap)
    kiem("F2_cache_chay_khi_nguon_khop_tien_kiem", chay(an, "cache", *lenh) == 10 and len(dem_nap) == n0 + 1
         and len(list((an / "cache" / "yolov8n_coco").glob("*.npz"))) == 1)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phien", required=True)
    ap.add_argument("--giao-thuc", required=True)
    ns = ap.parse_args(argv)
    phien = Path(os.path.abspath(H.ROOT / ns.phien))
    vung = [phien]
    appdata = Path.home() / "AppData" / "Roaming" / "Ultralytics" / "settings.json"
    truoc = {"appdata": H.sha256_file(appdata) if appdata.is_file() else None, "pyc": H.kiem_khong_pyc(H.SRC),
             "interim": H.liet_ke_nhe(H.ROOT / "data" / "interim"), "processed": H.liet_ke_nhe(H.ROOT / "data" / "processed"),
             "trackers": H.liet_ke_nhe(H.ROOT / "data" / "processed" / "trackers"), "results": H.liet_ke_nhe(H.ROOT / "results")}
    import numpy  # noqa: F401
    import dieu_phoi_bdd as D
    import thong_ke_bdd as T
    D.NAP_MODEL = lambda det, cfg: ModelGia(cfg["loai_checkpoint"])   # chan nap that neu ham trong tien trinh goi toi
    ma_truoc = {H.rel(p): H.sha256_file(p) for p in sorted((H.SRC / "r61_bdd").glob("*.py"))}
    fx = phien / "fx"
    sha = tao_the_gioi(fx, vung)
    gtp = giao_thuc_fixture(H.doc_json(ns.giao_thuc), fx, sha, vung)
    kiem("giao_thuc_that_36_run_200_video", len(D.Ctx(Path(ns.giao_thuc)).runs()) == 36)
    ctx = D.Ctx(gtp)
    kiem_loc_va_anh_xa(D)
    kiem_duong_dan(phien, vung)
    tinh = kiem_ma_nguon(D, ctx, fx)
    kiem_so_mau_san_xuat(D, Path(ns.giao_thuc), phien, vung)
    ab = luong_ab(D, gtp, phien, vung)
    A = H.ROOT / ab["A"]
    luong_cache_f2(gtp, phien)
    kiem_cache_va_timeline(D, ctx, A, vung)
    kiem_cache_hong(D, ctx, A, phien, vung)
    kiem_evaluator_crowd(D, ctx, phien, vung)
    kiem_evaluator_duong_ghi(D, ctx, phien, vung)
    mau = kiem_thong_ke(T, phien, vung)
    sau = {"appdata": H.sha256_file(appdata) if appdata.is_file() else None, "pyc": H.kiem_khong_pyc(H.SRC),
           "interim": H.liet_ke_nhe(H.ROOT / "data" / "interim"), "processed": H.liet_ke_nhe(H.ROOT / "data" / "processed"),
           "trackers": H.liet_ke_nhe(H.ROOT / "data" / "processed" / "trackers")}
    kiem("khong_pyc_settings_interim_ngoai_pham_vi", all(truoc[k] == sau[k] for k in sau),
         {k: truoc[k] == sau[k] for k in sau})
    kiem("khong_trackeval_error_log_mac_dinh", not (H.ROOT / "results" / "trackeval_error.log").exists() or
         "trackeval_error.log" in json.dumps(truoc["results"]))
    tep = [p for p in H.liet_ke_file(phien) if p.is_file()]
    kiem("khong_file_tam_do_dang", not any(p.name.endswith(".dang_ghi") for p in tep))
    kiem("cau_hinh_thu_vien_trong_phien", (phien / "_cau_hinh_thu_vien" / "ultralytics").is_dir())
    kiem("moi_lenh_con_exit_0_2_10_khong_het_gio", all(m["exit"] in (0, 2, 10) and not m["het_gio"] for m in LENH_CON)
         and len(list((phien / "lenh_con").glob("*.json"))) == len(LENH_CON),
         [m for m in LENH_CON if m["exit"] not in (0, 2, 10) or m["het_gio"]])
    # review_003: loi khoa git_dirty (vd PLAN bi sua giua phien) KHONG nam trong kiem am nao -> moi lan xuat hien la loi
    kiem("khong_loi_khoa_git_dirty_ngoai_du_kien", not any(m["loi_git_dirty"] for m in LENH_CON),
         [(m["stt"], m["ten"], m["ly_do"]) for m in LENH_CON if m["loi_git_dirty"]])
    ma_sau = {H.rel(p): H.sha256_file(p) for p in sorted((H.SRC / "r61_bdd").glob("*.py"))}
    kiem("ma_r61_khong_doi_trong_luc_test", ma_truoc == ma_sau)
    dat = all(k["dat"] for k in KQ)
    H.ghi_json(phien / "bang_chung" / "luong_ab.json", ab, vung)
    H.ghi_json(phien / "bang_chung" / "hash_phien.json", H.bang_hash_thu_muc(phien), vung)
    H.ghi_json(phien / "ket_qua_test.json", {"dat": dat, "so_kiem": len(KQ), "so_dat": sum(k["dat"] for k in KQ),
                                             "kiem": KQ, "bootstrap_mau_ekf_cv": mau.get("ekf-cv"),
                                             "file_phat_sinh": len(tep), "truoc_sau": {k: truoc[k] == sau[k] for k in sau},
                                             "python": sys.executable, "argv": [sys.executable, __file__, *argv],
                                             "ma_sha256": ma_sau, "ma_dong_import_tinh_sha256": tinh,
                                             "lenh_con": LENH_CON, "so_lenh_con": len(LENH_CON),
                                             "exit_lenh_con": {str(x): sum(m["exit"] == x for m in LENH_CON)
                                                               for x in sorted({m["exit"] for m in LENH_CON}, key=str)}},
               vung)
    print(f"KET QUA: {sum(k['dat'] for k in KQ)}/{len(KQ)} dat")
    return 0 if dat else H.EXIT_HOP_DONG


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == [CON_DIEU_PHOI]:              # tien trinh con do chay_con tao (chi lenh cache, giao thuc fixture)
        sys.exit(main_con(argv[1:]))
    ph = Path(os.path.abspath(H.ROOT / argv[argv.index("--phien") + 1]))
    if not H._trong(ph, H.VUNG_TESTS) or ph.exists():
        print("--phien phai la thu muc MOI duoi results/phase2/r61_bdd/tests/", file=sys.stderr)
        sys.exit(H.EXIT_HOP_DONG)
    H.tao_thu_muc(ph, [ph])
    H.chuan_bi_moi_truong(ph, [ph])
    import faulthandler                          # segfault goc (lan_03/05/06): stack Python moi luong -> file trong phien
    _fh = open(H.kiem_duong_ghi(ph / "faulthandler.log", [ph]), "w", encoding="utf-8")
    faulthandler.enable(file=_fh, all_threads=True)
    shutil.disk_usage(ph)
    sys.exit(H.chay_va_ghi_log(main, argv, ph, [ph], "test_chuan_bi_bdd"))
