"""
dieu_phoi_bdd.py -- Bo dieu phoi R61 cho cong A (chuan bi/cache/smoke/khoa) va cong B (thuc thi/tong hop).

LUOT R61_PREP CHI KIEM CAC LENH NAY BANG FIXTURE CPU (giao thuc fixture nam duoi results/phase2/r61_bdd/tests/).
Goi tren du lieu that can plan cong A / cong B duoc duyet rieng (--cong A|B + phien dung vung).

Cong A (ghi DUY NHAT duoi results/phase2/r61_bdd/chuan_bi/<phien_A>/):
    tien-kiem -> cache (moi detector, tiep tuc theo detector-video) -> kiem-cache -> smoke -> khoa -> kiem-khoa
    -> dong-goi-a
Cong B (chi DOC goi A; ghi DUY NHAT duoi results/phase2/r61_bdd/thuc_thi/<phien_B>/):
    mo-phien -> tracking (doan 420 s, tiep tuc theo run-video) -> cham -> lap-tracker -> cham-lai -> tong-hop
    -> kiem-khoa -> hau-kiem

Ma thoat: 0 hoan tat; 10 chua hoan tat (chay lai cung lenh de tiep tuc); 2 loi hop dong/dau vao; 1 loi bat ngo.
Chi goi helper hien co (det_cache, detector_classes, run_tracking, run_trackeval, trackeval_ext, boot_stats,
mot_datasets.bdd100k_coco); khong sua thuat toan/metric; khong goi gt_dataframe/build_interim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop_dong_bdd as H  # noqa: E402

sys.path.insert(0, str(H.SRC))
GIAO_THUC = H.ROOT / "configs" / "phase2" / "r61_bdd" / "giao_thuc.json"
LOP_COCO_LOP5 = {1, 2, 3, 5, 7}
KHOA_NPZ = {"frame_idx": "int32", "n_per_frame": "int32", "unreadable": "bool", "truncated": "bool",
            "time_s": "float64", "data": "float32", "orig_shape": "int32", "cls_goc": "int16"}
THU_VIEN = ("ultralytics", "torch", "numpy", "pandas", "scipy", "trackeval", "opencv-python", "lap")
#: Goi ben thu ba co MA NGUON duoc bam toan bo (.py) vao khoa (khong phu thuoc lich su import).
GOI_MA_KHOA = ("ultralytics", "trackeval")
#: Diem vao cua dong import tinh (tuong doi goc src); moi file src/ ma chung import (ke ca import trong ham) duoc khoa.
#: Plan A0 (24/10): them lop van hanh cong A (van_hanh_cong_a.py) -> khoa bao ma runner + moi file no import.
DIEM_VAO_MA = ("r61_bdd/__init__.py", "r61_bdd/hop_dong_bdd.py", "r61_bdd/dieu_phoi_bdd.py", "r61_bdd/thong_ke_bdd.py",
               "r61_bdd/van_hanh_cong_a.py")
#: So mau bootstrap san xuat da duyet (M5-Q6); chi giao thuc fixture duoc khac.
SO_MAU_SAN_XUAT = 5000
#: Hook nap detector cho FIXTURE (chi khi giao thuc che_do == fixture); None -> ultralytics.YOLO(checkpoint).
NAP_MODEL = None


# ---------------------------------------------------------------------------
# Ngu canh
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, giao_thuc: Path):
        self.gt_path = Path(os.path.abspath(giao_thuc))
        self.gt = H.doc_json(self.gt_path)
        self.fixture = self.gt.get("che_do") == "fixture"
        if self.fixture and not H._trong(self.gt_path, H.VUNG_TESTS):
            raise H.LoiHopDong("giao thuc fixture phai nam duoi results/phase2/r61_bdd/tests/")
        self.goc = Path(self.gt["goc"]) if self.fixture else H.ROOT
        ds = self.gt["tap_du_lieu"]
        self.split = ds["split"]
        self.gt_root = self.duong(ds["gt_folder"])
        man = H.doc_json(self.duong(ds["manifest"]["rel"]))
        self.videos = list(man["videos"])
        self.fps = float(self.gt["tracker"]["fps"])
        self.dt = 1.0 / self.fps
        self._ds = None
        # Goc ma nguon duoc khoa: repo src/ (san xuat); fixture co the tro ban sao trung byte de thu sua ma.
        self.src_goc = self.duong(self.gt["ma_nguon_goc"]) if self.fixture and self.gt.get("ma_nguon_goc") else H.SRC
        bt = self.gt["thong_ke"]["bootstrap"]
        if int(bt["so_video"]) != len(self.videos):
            raise H.LoiHopDong(f"bootstrap so_video {bt['so_video']} != {len(self.videos)} video manifest")
        if not self.fixture and int(bt["so_mau"]) != SO_MAU_SAN_XUAT:
            raise H.LoiHopDong(f"so_mau bootstrap {bt['so_mau']} != {SO_MAU_SAN_XUAT} (ngoai le chi cho fixture)")
        self.so_mau = int(bt["so_mau"])

    def duong(self, r: str) -> Path:
        return self.goc / r

    def dataset(self):
        if self._ds is None:
            from mot_datasets.bdd100k_coco import BDD100KCocoDataset
            ds = self.gt["tap_du_lieu"]
            self._ds = BDD100KCocoDataset(self.duong(ds["nhan"]["rel"]), root=self.duong(ds["goc_anh_bdd"]))
        return self._ds

    def runs(self) -> list[dict]:
        out = []
        tai = self.gt["tai"]
        for det in self.gt["detector"]:
            if det == "nguon_danh_tinh":
                continue
            for dk, conf in tai["dieu_kien"].items():
                for bl in self.gt["tracker"]["bo_loc"]:
                    for s in (bl["seeds"] or [None]):
                        ten = f"r61-{det}-{dk}-{bl['ten']}" + ("" if s is None else f"-s{s}")
                        out.append({"run": ten, "detector": det, "dieu_kien": dk, "conf": float(conf),
                                    "bo_loc": bl["ten"], "yaml": bl["yaml"], "seed": s})
        if [r["run"] for r in out] != self.gt["ma_tran"]["danh_sach"]:
            raise H.LoiHopDong("danh sach run sinh ra khac giao thuc")
        return out

    def detectors(self) -> list[str]:
        return [d for d in self.gt["detector"] if d != "nguon_danh_tinh"]


def vung_phien(ctx: Ctx, phien, cong: str) -> list[Path]:
    p = Path(os.path.abspath(H.ROOT / phien if not Path(phien).is_absolute() else phien))
    goc = H.VUNG_TESTS if ctx.fixture else (H.VUNG_CHUAN_BI if cong == "A" else H.VUNG_THUC_THI)
    if not H._trong(p, goc) or p == goc:
        raise H.LoiHopDong(f"phien cong {cong} phai nam duoi {H.rel(goc)}/<phien>")
    return [p]


# ---------------------------------------------------------------------------
# Nguon + cau hinh hieu luc
# ---------------------------------------------------------------------------
def doc_seqinfo(p: Path) -> int:
    import configparser
    cp = configparser.ConfigParser()
    cp.read_string(p.read_bytes().decode("utf-8"))
    return int(cp["Sequence"]["seqLength"])


def kiem_nguon(ctx: Ctx) -> dict:
    ds = ctx.gt["tap_du_lieu"]
    out = {"manifest": H.sha256_file(ctx.duong(ds["manifest"]["rel"])), "nhan": H.sha256_file(ctx.duong(ds["nhan"]["rel"])),
           "seqmap": H.sha256_file(ctx.duong(ds["seqmap"]["rel"])), "checkpoint": {}, "gt": {}, "seqinfo": {}}
    loi = [f"{k} sha256 lech" for k in ("manifest", "nhan", "seqmap") if out[k] != ds[k]["sha256"]]
    for det in ctx.detectors():
        c = ctx.gt["detector"][det]
        p = ctx.duong(c["checkpoint"])
        out["checkpoint"][det] = H.sha256_file(p) if p.is_file() else None
        if out["checkpoint"][det] != c["sha256"]:
            loi.append(f"checkpoint {det} sha256 lech")
    for v in ctx.videos:
        d = ctx.gt_root / ctx.split / v
        out["gt"][v] = H.sha256_file(d / "gt" / "gt.txt")
        out["seqinfo"][v] = H.sha256_file(d / "seqinfo.ini")
    if loi:
        raise H.LoiHopDong("; ".join(loi))
    return out


def cau_hinh_bo_loc(ctx: Ctx, bl: dict):
    import run_tracking as rt
    from time_units import track_buffer_frames
    tb = track_buffer_frames(float(ctx.gt["tracker"]["track_buffer_sec"]), ctx.fps)
    if tb != int(ctx.gt["tracker"]["track_buffer_buoc"]):
        raise H.LoiHopDong(f"track_buffer {tb} != giao thuc")
    cfg = rt.load_tracker_cfg(bl["yaml"], {"track_buffer": tb})
    if cfg.tracker_type != bl["tracker_type"]:
        raise H.LoiHopDong(f"{bl['ten']}: tracker_type {cfg.tracker_type}")
    return cfg


def doc_tham_so(kf, ky_vong: dict) -> dict:
    """Doc tham so HIEU LUC tren instance bo loc that (ca thuoc tinh late-binding doc config.py) theo khoa ky vong."""
    out = {}
    for k, v in ky_vong.items():
        x = getattr(kf, k, None)
        out[k] = None if x is None else (bool(x) if isinstance(v, bool) else int(x) if isinstance(v, int)
                                         else str(x) if isinstance(v, str) else float(x))
    return out


def kiem_tham_so(ctx: Ctx, ten_bl: str, kf) -> dict:
    ky = ctx.gt["tracker"]["tham_so_bo_loc_du_kien"][ten_bl]
    ts = doc_tham_so(kf, ky)
    if H.chuan_json(ts) != H.chuan_json(ky):
        lech = {k: [ts[k], ky[k]] for k in ky if ts[k] != ky[k]}
        raise H.LoiHopDong(f"{ten_bl}: tham so bo loc hieu luc khac giao thuc {lech}")
    return ts


def cau_hinh_hieu_luc(ctx: Ctx) -> dict:
    import run_tracking as rt
    reg = rt.tracker_registry()
    tsk = ctx.gt["tracker"]["tham_so_bo_loc_du_kien"]
    for r, h in tsk["nguon_sha256"].items():          # nguon mac dinh (config.py ...) cua cac tham so
        if H.sha256_file(H.ROOT / r) != h:
            raise H.LoiHopDong(f"nguon tham so {r} sha256 lech giao thuc")
    out = {}
    for bl in ctx.gt["tracker"]["bo_loc"]:
        cfg = cau_hinh_bo_loc(ctx, bl)
        tr = rt.make_tracker(cfg, reg, assoc_log=False)
        kf = getattr(tr, "kalman_filter", None)
        p_yaml = Path(rt.resolve_tracker_yaml(bl["yaml"]))
        if H.sha256_file(p_yaml) != bl["yaml_sha256"]:
            raise H.LoiHopDong(f"{bl['ten']}: yaml {H.rel(p_yaml)} sha256 lech giao thuc")
        d = {"yaml_resolve": H.rel(p_yaml), "yaml_sha256": H.sha256_file(p_yaml),
             "cfg": rt.cfg_to_dict(cfg), "lop_tracker": type(tr).__name__, "lop_bo_loc": type(kf).__name__,
             "tham_so_bo_loc": kiem_tham_so(ctx, bl["ten"], kf),
             "assoc_log": False, "pf_diag": False, "elapsed_mode": ctx.gt["tracker"]["elapsed_mode"]}
        if hasattr(kf, "num_particles"):
            d.update({"num_particles": int(kf.num_particles), "optimal_proposal": bool(kf.optimal_proposal),
                      "roughening": float(kf.roughening)})
            if (d["num_particles"], d["optimal_proposal"], d["roughening"]) != \
                    (bl["pf_num_particles"], bl["pf_optimal_proposal"], bl["pf_roughening"]):
                raise H.LoiHopDong(f"PF hieu luc {d} khac giao thuc")
        out[bl["ten"]] = json.loads(H.chuan_json(d))
    return out


def thu_vien() -> dict:
    from importlib import metadata
    out = {}
    for n in THU_VIEN:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    out["python"] = sys.version
    out["BYTETRACK_BACKEND"] = os.environ.get("BYTETRACK_BACKEND", "")
    if out["BYTETRACK_BACKEND"] not in ("", "ultralytics"):
        raise H.LoiHopDong("BYTETRACK_BACKEND phai trong hoac ultralytics")
    return out


def _file_mo_dun(ten: str, tim: list[Path]) -> list[Path]:
    """'a.b.c' -> file .py trong cac goc tim (goi __init__.py + module); [] neu khong thuoc goc (thu vien ngoai)."""
    phan = ten.split(".")
    for goc in tim:
        out, cur, ok = [], goc, True
        for i, p in enumerate(phan):
            if (cur / p / "__init__.py").is_file():
                cur = cur / p
                out.append(cur / "__init__.py")
            elif (cur / f"{p}.py").is_file():
                out.append(cur / f"{p}.py")
                break                          # module la file: phan con lai la thuoc tinh
            else:
                ok = i > 0                     # ten goc khong co o goc nay -> thu goc khac
                break
        if ok and out:
            return out
    return []


def dong_import_tinh(src_goc: Path, diem_vao=DIEM_VAO_MA) -> list[Path]:
    """Dong import TINH (ast, ke ca import trong ham) tu cac diem vao, gioi han trong src_goc.
    Thu tu tim giong sys.path cua entry point: src/ roi src/r61_bdd/. KHONG doc sys.modules -> doc lap lich su import."""
    import ast
    tim = [src_goc, src_goc / "r61_bdd"]
    da, hang = set(), [src_goc / d for d in diem_vao]
    while hang:
        f = hang.pop()
        if f in da:
            continue
        if not f.is_file():
            raise H.LoiHopDong(f"diem vao/module {f} khong ton tai")
        da.add(f)
        cay = ast.parse(f.read_bytes(), filename=str(f))
        for n in ast.walk(cay):
            ten = []
            if isinstance(n, ast.Import):
                ten = [(a.name, tim) for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    goc = f.parent
                    for _ in range(n.level - 1):
                        goc = goc.parent
                    if n.module:
                        ten = [(n.module, [goc])] + [(f"{n.module}.{a.name}", [goc]) for a in n.names]
                    else:
                        ten = [(a.name, [goc]) for a in n.names]
                else:
                    ten = [(n.module, tim)] + [(f"{n.module}.{a.name}", tim) for a in n.names]
            for t, g in ten:
                hang += [p for p in _file_mo_dun(t, g) if p not in da]
    return sorted(da)


def ma_nguon(ctx: Ctx) -> dict:
    """{duong tuong doi src_goc: sha256 byte tho} cua dong import tinh -- danh muc XAC DINH, khong phu thuoc import."""
    goc = ctx.src_goc
    return {p.relative_to(goc).as_posix(): H.sha256_file(p) for p in dong_import_tinh(goc)}


def ma_thu_vien() -> dict:
    """sha256 MOI file .py cua ultralytics/trackeval da cai (tim qua find_spec, khong import goi)."""
    from importlib.util import find_spec
    out = {}
    for g in GOI_MA_KHOA:
        sp = find_spec(g)
        if sp is None or not sp.submodule_search_locations:
            raise H.LoiHopDong(f"khong tim thay goi {g}")
        for d in sp.submodule_search_locations:
            for p in (x for x in H.liet_ke_file(d) if x.suffix == ".py"):
                out[f"{g}/{p.relative_to(d).as_posix()}"] = H.sha256_file(p)
    return dict(sorted(out.items()))


def git_ban(ctx: Ctx) -> dict:
    """HEAD + sha256 trang thai Git + sha256 BYTE moi file tracked dang dirty (khong chi trang thai)."""
    st = git_doc("status", "--porcelain", "--untracked-files=no")
    dirty = {}
    for ln in st.splitlines():
        r = ln[3:].strip().strip('"')
        if " -> " in r:
            r = r.split(" -> ")[-1]
        p = H.ROOT / r
        dirty[r] = H.sha256_file(p) if p.is_file() else None
    return {"head": git_doc("rev-parse", "HEAD").strip(), "git_status_tracked_sha256": H.sha256_bytes(st.encode()),
            "git_dirty_sha256": dict(sorted(dirty.items()))}


# ---------------------------------------------------------------------------
# Cache (cong A)
# ---------------------------------------------------------------------------
def bang_anh_xa_coco(names, loai_ky_vong: str) -> dict | None:
    import detector_classes as dc
    m = dc.anh_xa_model_sang_lop5(names)
    loai = dc.loai_checkpoint(names)
    if loai != loai_ky_vong:
        raise H.LoiHopDong(f"loai checkpoint {loai!r} != {loai_ky_vong!r}")
    if loai == "coco80":
        return None
    return {int(k): dc.sang_id_coco(v) for k, v in m.items()}


def doi_id_sang_coco(data, bang: dict | None):
    import numpy as np
    a = np.array(data, dtype=np.float32, copy=True).reshape(-1, 6)
    if bang is None or not len(a):
        return a
    goc = a[:, 5].astype(np.int64)
    thieu = sorted(set(goc.tolist()) - set(bang))
    if thieu or not np.all(a[:, 5] == goc):
        raise H.LoiHopDong(f"cls {thieu} khong co trong bang anh xa / khong nguyen")
    a[:, 5] = np.array([bang[int(c)] for c in goc], dtype=np.float32)
    return a


def spec_cho(ctx: Ctx, det: str, device: str):
    import det_cache
    c, cc = ctx.gt["detector"][det], ctx.gt["cache"]
    return det_cache.DetectorSpec(backend=cc["backend"], weights=str(ctx.duong(c["checkpoint"])), weights_sha256=c["sha256"],
                                  imgsz=cc["imgsz"], conf=cc["conf"], iou=cc["iou"], classes=cc["classes"],
                                  max_det=cc["max_det"], half=cc["half"], agnostic_nms=cc["agnostic_nms"],
                                  augment=cc["augment"], batch=cc["batch"], device=str(device),
                                  mask_ignored_regions=cc["mask_ignored_regions"], dataset=ctx.gt["tap_du_lieu"]["dataset"],
                                  part=ctx.gt["tap_du_lieu"]["part"])


def nap_model(ctx: Ctx, det: str):
    if NAP_MODEL is not None:
        if not ctx.fixture:
            raise H.LoiHopDong("NAP_MODEL mo phong chi dung voi giao thuc fixture")
        return NAP_MODEL(det, ctx.gt["detector"][det])
    from ultralytics import YOLO
    return YOLO(str(ctx.duong(ctx.gt["detector"][det]["checkpoint"])))


def provenance(ctx: Ctx, det: str, video: str, frames, names, bang, nguon: dict | None = None) -> dict:
    """nguon = ket qua kiem_nguon THUC TE luc tao cache (bam lai file); None -> gia tri giao thuc (de doi chieu)."""
    ds = ctx.gt["tap_du_lieu"]
    ck = nguon["checkpoint"][det] if nguon else ctx.gt["detector"][det]["sha256"]
    return {"r61": True, "detector": det, "checkpoint_sha256": ck,
            "dataset": ds["dataset"], "split": ctx.split, "nhan_sha256": nguon["nhan"] if nguon else ds["nhan"]["sha256"],
            "manifest_sha256": nguon["manifest"] if nguon else ds["manifest"]["sha256"],
            "names": {str(k): str(n) for k, n in sorted(names.items())},
            "anh_xa_coco": None if bang is None else {str(k): v for k, v in sorted(bang.items())},
            "timeline": {"n": len(frames), "frame_dau": frames[0].index, "frame_cuoi": frames[-1].index}}


def file_marker_cache(cache_dir: Path, video: str) -> list[Path]:
    return [cache_dir / f"{video}.npz", cache_dir / f"{video}.json", cache_dir / "meta.json"]


def xong_cache(cache_dir: Path, det: str, video: str) -> bool:
    return H.da_xong(cache_dir / f"{video}.xong.json", file_marker_cache(cache_dir, video), {"detector": det, "video": video})


def nap_tien_kiem(ctx: Ctx, phien_A: Path) -> tuple[dict, str]:
    """Tien kiem cong A PHAI co va nguon THUC TE (bam lai ngay luc goi) phai trung tien kiem + giao thuc."""
    p = phien_A / "tien_kiem" / "tien_kiem.json"
    if not p.is_file() or H.la_reparse(p):
        raise H.LoiHopDong("chua co tien-kiem hop le trong phien A -- khong tao/tiep tuc cache/khoa")
    tk = H.doc_json(p)
    if tk.get("giao_thuc_sha256") != H.sha256_file(ctx.gt_path):
        raise H.LoiHopDong("giao thuc doi sau tien-kiem")
    nguon = kiem_nguon(ctx)                       # nem LoiHopDong neu nhan/manifest/seqmap/checkpoint lech giao thuc
    if H.chuan_json(nguon) != H.chuan_json(tk.get("nguon")):
        lech = [k for k in nguon if H.chuan_json(nguon[k]) != H.chuan_json(tk["nguon"].get(k))]
        raise H.LoiHopDong(f"nguon thuc te khac tien-kiem: {lech}")
    return nguon, H.sha256_file(p)


def tao_cache_video(ctx: Ctx, det: str, model, spec, cache_dir: Path, video: str, vung, nguon: dict, sha_tk: str) -> dict:
    import numpy as np
    import det_cache
    if (cache_dir / f"{video}.xong.json").exists():
        raise H.LoiHopDong(f"cache {det}/{video} da co marker -- khong ghi de")
    frames = ctx.dataset().frames(video, ctx.gt["tap_du_lieu"]["part"])
    regions = ctx.dataset().ignored_regions(video)
    bang = bang_anh_xa_coco(model.names, ctx.gt["detector"][det]["loai_checkpoint"])
    for f in (f"{video}.npz", f"{video}.json"):
        H.kiem_duong_ghi(cache_dir / f, vung)
    t0 = time.time()
    arrays = det_cache.build_video_yolo(model, frames, regions, spec, tuple(ctx.gt["cache"]["orig_shape"]))
    giay = time.time() - t0
    if arrays["unreadable"].any() or arrays["truncated"].any():
        raise H.LoiHopDong(f"{det}/{video}: unreadable {int(arrays['unreadable'].sum())} truncated "
                           f"{int(arrays['truncated'].sum())} -- chan, khong bo frame")
    goc = arrays["data"]
    moi = doi_id_sang_coco(goc, bang)
    if moi.shape != goc.shape or not np.array_equal(moi[:, :5], goc[:, :5]):
        raise H.LoiHopDong("doi ID lam doi box/conf/thu tu")
    arrays["data"] = moi
    arrays["cls_goc"] = goc[:, 5].astype(np.int16)
    extra = provenance(ctx, det, video, frames, {int(k): v for k, v in dict(model.names).items()}, bang, nguon)
    for f in file_marker_cache(cache_dir, video):
        H.kiem_duong_ghi(f, vung)
    det_cache.save_video(cache_dir, video, arrays, extra=extra, spec=spec)
    H.ghi_marker(cache_dir / f"{video}.xong.json", file_marker_cache(cache_dir, video), vung,
                 {"detector": det, "video": video, "tien_kiem_sha256": sha_tk, "gt_sha256": nguon["gt"][video],
                  "seqinfo_sha256": nguon["seqinfo"][video], "giay_suy_luan": round(giay, 3), "n_frames": len(frames),
                  "n_boxes": int(len(moi))})
    return {"n_boxes": int(len(moi)), "giay": giay}


def buoc_cache(ctx: Ctx, phien: Path, det: str, vung, device: str, gioi_han: int | None, ngan_sach: float) -> int:
    import det_cache
    nguon, sha_tk = nap_tien_kiem(ctx, phien)     # TRUOC moi ghi / nap detector
    cache_dir = phien / "cache" / det
    for v in ctx.videos:                          # marker hong -> 2 truoc moi ghi
        if (cache_dir / f"{v}.xong.json").exists():
            xong_cache(cache_dir, det, v)
    H.tao_thu_muc(cache_dir, vung)
    spec = spec_cho(ctx, det, device).fill_env()
    H.kiem_duong_ghi(cache_dir / "meta.json", vung)
    det_cache.write_meta(cache_dir, spec, mode="append" if (cache_dir / "meta.json").exists() else "create")
    model = nap_model(ctx, det)
    t0, lam = time.time(), 0
    for v in ctx.videos:
        if xong_cache(cache_dir, det, v):
            continue
        if (gioi_han is not None and lam >= gioi_han) or time.time() - t0 > ngan_sach:
            raise H.ChuaXong(f"cache {det}: dung giua don vi sau {lam} video")
        tao_cache_video(ctx, det, model, spec, cache_dir, v, vung, nguon, sha_tk)
        lam += 1
    print(f"cache {det}: xong ({lam} video moi)")
    return 0


def kiem_cache_video(ctx: Ctx, det: str, cache_dir: Path, video: str) -> list[str]:
    import numpy as np
    import det_cache
    loi = []
    st, why = H.kiem_marker(cache_dir / f"{video}.xong.json", file_marker_cache(cache_dir, video),
                            {"detector": det, "video": video})
    if st != "xong":
        return [f"{video}: marker {st} {why}"]
    frames = ctx.dataset().frames(video, ctx.gt["tap_du_lieu"]["part"])
    n = len(frames)
    with np.load(cache_dir / f"{video}.npz") as z:
        if set(z.files) != set(KHOA_NPZ):
            return [f"{video}: khoa npz {sorted(z.files)}"]
        a = {k: z[k] for k in KHOA_NPZ}
    loi += [f"{video}: {k} dtype {a[k].dtype}" for k, t in KHOA_NPZ.items() if str(a[k].dtype) != t]
    N = len(a["data"])
    if a["data"].ndim != 2 or a["data"].shape[1] != 6 or a["cls_goc"].shape != (N,):
        loi.append(f"{video}: shape data/cls_goc")
    if a["n_per_frame"].shape != (n,) or int(a["n_per_frame"].sum()) != N or (a["n_per_frame"] < 0).any():
        loi.append(f"{video}: n_per_frame")
    if a["frame_idx"].tolist() != [f.index for f in frames]:
        loi.append(f"{video}: frame_idx khac timeline nhan")
    if a["time_s"].tolist() != [float(f.time_s) for f in frames]:
        loi.append(f"{video}: time_s khac timeline nhan")
    if a["unreadable"].any() or a["truncated"].any() or a["unreadable"].shape != (n,) or a["truncated"].shape != (n,):
        loi.append(f"{video}: unreadable/truncated")
    if a["orig_shape"].tolist() != list(ctx.gt["cache"]["orig_shape"]):
        loi.append(f"{video}: orig_shape")
    info = H.doc_json(cache_dir / f"{video}.json")
    meta = det_cache.load_meta(cache_dir)
    try:
        names = {int(k): s for k, s in info.get("names", {}).items()}
        bang = bang_anh_xa_coco(names, ctx.gt["detector"][det]["loai_checkpoint"])
        goc = a["data"].copy()
        goc[:, 5] = a["cls_goc"]
        if not np.array_equal(doi_id_sang_coco(goc, bang), a["data"]):
            loi.append(f"{video}: cls != anh xa(cls_goc)")
        frames_info = provenance(ctx, det, video, frames, names, bang)
        for k, v in frames_info.items():
            if H.chuan_json(info.get(k)) != H.chuan_json(v):
                loi.append(f"{video}: provenance {k} lech")
    except Exception as e:  # noqa: BLE001 -- moi loi anh xa la loi hop dong cache
        loi.append(f"{video}: anh xa/provenance loi {e}")
    if ctx.gt["detector"][det]["loai_checkpoint"] == "lop5" and not set(int(c) for c in a["data"][:, 5]) <= LOP_COCO_LOP5:
        loi.append(f"{video}: lop ngoai {sorted(LOP_COCO_LOP5)}")
    st2, why2 = det_cache.sequence_spec_status(cache_dir, video, meta)
    if st2 != "ok":
        loi.append(f"{video}: spec {st2} {why2}")
    ky_vong = spec_cho(ctx, det, meta.get("device"))
    ky_vong.versions = meta.get("versions", {})
    if det_cache.meta_gen_spec(meta) != ky_vong.gen_spec() or meta.get("spec_sha256") != ky_vong.spec_sha256():
        loi.append(f"{video}: meta gen_spec khac giao thuc (checkpoint/cau hinh sinh)")
    if (info.get("n_frames"), info.get("n_boxes"), info.get("n_unreadable"), info.get("n_truncated")) != (n, N, 0, 0):
        loi.append(f"{video}: json n_frames/n_boxes/unreadable/truncated")
    return loi


def mo_cache(ctx: Ctx, cache_dir: Path, video: str, conf: float):
    import det_cache
    meta = det_cache.load_meta(cache_dir)
    tai = ctx.gt["tai"]
    errs = det_cache.check_compatible(meta, conf, tai["classes"], tai["max_det"])
    st, why = det_cache.sequence_spec_status(cache_dir, video, meta)
    if errs or st != "ok":
        raise H.LoiHopDong(f"cache {cache_dir.name}/{video}: {errs} {st} {why}")
    return det_cache.VideoDetections(cache_dir / f"{video}.npz", conf, tai["classes"], tai["max_det"], None,
                                     allow_truncated=False)


# ---------------------------------------------------------------------------
# Tracking / cham (dung chung smoke A va thuc thi B)
# ---------------------------------------------------------------------------
def file_don_vi(run_dir: Path, video: str) -> list[Path]:
    return [run_dir / "data" / f"{video}.txt", run_dir / "data_cls" / f"{video}.csv", run_dir / "timing" / f"{video}.csv"]


def xong_don_vi(run_dir: Path, run: str, video: str) -> bool:
    return H.da_xong(run_dir / "xong" / f"{video}.xong.json", file_don_vi(run_dir, video), {"run": run, "video": video})


def file_cham(out: Path) -> list[Path]:
    return [out / "per_video_metrics.csv", out / "combined_metrics.csv", out / "raw_fields.json"]


def xong_cham(out: Path, run: str) -> bool:
    return H.da_xong(out / "cham.xong.json", file_cham(out), {"run": run})


def chay_don_vi(ctx: Ctx, r: dict, video: str, cache_dir: Path, run_dir: Path, registry, vung) -> dict:
    import run_tracking as rt
    if (run_dir / "xong" / f"{video}.xong.json").exists():
        raise H.LoiHopDong(f"{r['run']}/{video} da co marker -- khong ghi de")
    bl = next(b for b in ctx.gt["tracker"]["bo_loc"] if b["ten"] == r["bo_loc"])
    cfg = cau_hinh_bo_loc(ctx, bl)
    vd = mo_cache(ctx, cache_dir, video, r["conf"])
    outs = file_don_vi(run_dir, video)
    for p in outs:
        H.tao_thu_muc(p.parent, vung)
        H.kiem_duong_ghi(p, vung)
    sink = {"assoc_log": bool(ctx.gt["tracker"]["assoc_log"]), "pf_diag": bool(ctx.gt["tracker"]["pf_diag"])}
    t0 = time.time()
    df, df_cls, df_t = rt.track_video(vd, cfg, ctx.dt, ctx.gt["tracker"]["elapsed_mode"], r["seed"], video, registry,
                                      log_sink=sink)
    giay = time.time() - t0
    df.to_csv(outs[0], header=False, index=False)          # Y HET run_tracking.run_video_to_dir
    df_cls.to_csv(outs[1], index=False)
    df_t.to_csv(outs[2], index=False)
    kf = getattr(sink["tracker"], "kalman_filter", None)
    them = {"run": r["run"], "video": video, "giay": round(giay, 3), "n_frames": len(vd),
            "n_steps": sorted(set(int(x) for x in df_t["n_steps"])) if len(df_t) else [],
            "seed_truyen": r["seed"], "seed_thuc_dung": getattr(kf, "seed", None) if r["seed"] is not None else None,
            "npz_sha256": H.sha256_file(cache_dir / f"{video}.npz"),
            "tham_so_bo_loc": kiem_tham_so(ctx, r["bo_loc"], kf)}     # instance THAT da chay don vi nay
    if hasattr(kf, "num_particles"):
        them.update({"num_particles": int(kf.num_particles), "optimal_proposal": bool(kf.optimal_proposal),
                     "roughening": float(kf.roughening)})
    H.ghi_marker(run_dir / "xong" / f"{video}.xong.json", outs, vung, them)
    return them


def lap_don_vi(ctx: Ctx, runs: list[dict], videos: list[str], cache_of, root: Path, vung, gioi_han, ngan_sach,
               kiem_truoc=None) -> int:
    """Lap tuan tu cac don vi run-video; bo qua don vi co marker hop le; dung giua don vi -> ChuaXong."""
    import run_tracking as rt
    reg = rt.tracker_registry()
    for r in runs:                                       # quet truoc: marker hong o bat ky dau -> 2 truoc moi ghi
        for v in videos:
            if (root / ctx.split / r["run"] / "xong" / f"{v}.xong.json").exists():
                xong_don_vi(root / ctx.split / r["run"], r["run"], v)
    t0, lam = time.time(), 0
    for r in runs:
        run_dir = root / ctx.split / r["run"]
        for v in videos:
            if xong_don_vi(run_dir, r["run"], v):          # marker hong -> LoiHopDong (khong chay de)
                continue
            if (gioi_han is not None and lam >= gioi_han) or time.time() - t0 > ngan_sach:
                raise H.ChuaXong(f"tracking dung giua don vi sau {lam} don vi moi")
            if kiem_truoc is not None:
                kiem_truoc(r["detector"], v)
            chay_don_vi(ctx, r, v, cache_of(r["detector"]), run_dir, reg, vung)
            lam += 1
    print(f"tracking: xong ({lam} don vi moi)")
    return 0


def cham_run(ctx: Ctx, run: str, videos: list[str], tracking_root: Path, out_root: Path, vung) -> dict:
    """TrackEval mot_crowd tren DU don vi; thieu don vi -> LoiHopDong (khong cham full)."""
    import contextlib
    import io
    import trackeval
    import run_trackeval as rte
    from boot_stats import save_raw_fields
    run_dir = tracking_root / ctx.split / run
    thieu = [v for v in videos if not xong_don_vi(run_dir, run, v)]       # marker hong -> LoiHopDong
    if thieu:
        raise H.LoiHopDong(f"{run}: thieu {len(thieu)} don vi hop le (vd {thieu[:3]}) -- khong cham")
    out = out_root / run
    if (out / "cham.xong.json").exists() or H.la_reparse(out / "cham.xong.json"):
        raise H.LoiHopDong(f"{run}: da co cham.xong.json -- khong cham/ghi de")
    eval_cfg, ds_cfg, met_cfg = rte.build_configs(ctx.split, run, None)
    te_dir = out_root / "_trackeval" / run                   # TrackEval: OUTPUT_FOLDER/<tracker>/<cls>_*.{txt,csv}
    # MOI duong ghi thuc te (TrackEval summary/detailed, log loi, pandas, save_raw_fields, stdout, marker) kiem
    # lexical/realpath/reparse TRUOC khi goi helper; thu muc tao truoc qua tao_thu_muc (TrackEval khong tu mkdir).
    H.tao_thu_muc(out, vung)
    H.tao_thu_muc(te_dir, vung)
    ghi = [te_dir / "pedestrian_summary.txt", te_dir / "pedestrian_detailed.csv", out / "trackeval_error.log",
           out / "trackeval_stdout.txt", out / "cham.xong.json", out / "cham.xong.json.dang_ghi", *file_cham(out)]
    for p in ghi:
        H.kiem_duong_ghi(p, vung)
    seq_info = {v: doc_seqinfo(ctx.gt_root / ctx.split / v / "seqinfo.ini") for v in videos}
    ds_cfg.update({"GT_FOLDER": str(ctx.gt_root), "TRACKERS_FOLDER": str(tracking_root),
                   "SEQMAP_FOLDER": str(ctx.gt_root / "seqmaps"), "SEQ_INFO": seq_info,
                   "OUTPUT_FOLDER": str(out_root / "_trackeval"), "OUTPUT_SUB_FOLDER": ""})
    eval_cfg["LOG_ON_ERROR"] = str(out / "trackeval_error.log")
    eval_cfg["PLOT_CURVES"] = False
    ten = rte.dataset_class_for(ctx.gt["tap_du_lieu"]["dataset"], ctx.split)
    lop = rte.dataset_class(ten)
    from trackeval_ext import MotChallengeCrowdIgnore
    if ten != ctx.gt["danh_gia"]["evaluator"] or lop is not MotChallengeCrowdIgnore:
        raise H.LoiHopDong(f"evaluator {ten}/{lop} khac giao thuc")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res, msg = trackeval.Evaluator(eval_cfg).evaluate(
                [lop(ds_cfg)], [trackeval.metrics.HOTA(met_cfg), trackeval.metrics.CLEAR(met_cfg),
                                trackeval.metrics.Identity(met_cfg)])
    finally:                                                 # ca nhanh loi: stdout luu trong phien, loi nem lai
        H.ghi_bytes(out / "trackeval_stdout.txt", buf.getvalue().encode("utf-8"), vung, ghi_de=True)
    m = next(iter(msg.values())).get(run)
    if m != "Success":
        raise H.LoiHopDong(f"TrackEval {run}: {m}")
    df = rte.flatten_results(res, run)
    pv, cb, rf = file_cham(out)
    for p in (pv, cb, rf):                                   # kiem lai ngay truoc ghi (pandas/save_raw_fields ghi thang)
        H.kiem_duong_ghi(p, vung)
    df[df.video != "COMBINED_SEQ"].sort_values("video").to_csv(pv, index=False)
    df[df.video == "COMBINED_SEQ"].to_csv(cb, index=False)
    save_raw_fields(res, run, rf)
    c = df[df.video == "COMBINED_SEQ"].iloc[0].to_dict()
    H.ghi_marker(out / "cham.xong.json", [pv, cb, rf], vung,
                 {"run": run, "so_video": len(videos), "evaluator": ten,
                  "combined": {k: v for k, v in c.items() if k != "video"}})
    return c


# ---------------------------------------------------------------------------
# Khoa / goi A / phien B
# ---------------------------------------------------------------------------
def git_doc(*args) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=H.ROOT, capture_output=True, text=True,
                          check=True).stdout


KHOA_KHOA = ("giao_thuc_sha256", "tien_kiem_sha256", "nguon", "cau_hinh_hieu_luc", "thu_vien", "ma_nguon",
             "ma_thu_vien", "head", "git_status_tracked_sha256", "git_dirty_sha256", "cache")


def anh_khoa(ctx: Ctx, phien_A: Path, cache: dict | None = None) -> dict:
    """Anh khoa XAC DINH (khong phu thuoc module da import): nguon du lieu, cau hinh + tham so bo loc hieu luc, ma
    nguon dong import tinh, ma ultralytics/trackeval, Git (HEAD + byte file dirty), tien kiem, cache.
    cache=None -> bam lai phien_A/cache; truyen vao khi vua bam lai toan bo goi A (tranh bam 2 lan)."""
    tk = phien_A / "tien_kiem" / "tien_kiem.json"
    d = {"giao_thuc_sha256": H.sha256_file(ctx.gt_path), "tien_kiem_sha256": H.sha256_file(tk) if tk.is_file() else None,
         "nguon": kiem_nguon(ctx), "cau_hinh_hieu_luc": cau_hinh_hieu_luc(ctx), "thu_vien": thu_vien(),
         "ma_nguon": ma_nguon(ctx), "ma_thu_vien": ma_thu_vien(), **git_ban(ctx),
         "cache": H.bang_hash_thu_muc(phien_A / "cache") if cache is None else cache}
    return d


def so_khoa(cu: dict, moi: dict) -> list[str]:
    loi = [f"khoa thieu truong {k}" for k in KHOA_KHOA if k not in cu]
    for k in KHOA_KHOA:
        a, b = cu.get(k), moi.get(k)
        if isinstance(a, dict) and isinstance(b, dict) and k in ("ma_nguon", "ma_thu_vien", "cache", "git_dirty_sha256"):
            loi += [f"{k}: {f} doi/thieu/thua" for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
        elif H.chuan_json(a) != H.chuan_json(b):
            loi.append(f"khoa lech: {k}")
    return loi


def buoc_khoa(ctx: Ctx, phien_A: Path, vung) -> int:
    nguon, sha_tk = nap_tien_kiem(ctx, phien_A)
    for det in ctx.detectors():
        for v in ctx.videos:
            loi = kiem_cache_video(ctx, det, phien_A / "cache" / det, v)
            if loi:
                raise H.LoiHopDong(f"khong khoa: {loi[:3]}")
            mk = H.doc_json(phien_A / "cache" / det / f"{v}.xong.json")
            if (mk.get("tien_kiem_sha256"), mk.get("gt_sha256"), mk.get("seqinfo_sha256")) != \
                    (sha_tk, nguon["gt"][v], nguon["seqinfo"][v]):
                raise H.LoiHopDong(f"khong khoa: cache {det}/{v} khong rang buoc tien kiem/nguon hien tai")
    k = anh_khoa(ctx, phien_A)
    H.ghi_json(phien_A / "khoa" / "khoa.json", k, vung)
    print(f"khoa: {len(k['cache'])} file cache, {len(k['ma_nguon'])} file ma src, {len(k['ma_thu_vien'])} file thu vien")
    return 0


def buoc_kiem_khoa(ctx: Ctx, khoa: Path, phien_ghi: Path, vung, phien_A: Path) -> int:
    """Doc khoa (A), bam lai, GHI ket qua DUY NHAT vao phien_ghi/kiem_khoa/ (B khong ghi A)."""
    cu = H.doc_json(khoa)
    moi = anh_khoa(ctx, phien_A)
    loi = so_khoa(cu, moi)
    d = phien_ghi / "kiem_khoa" / time.strftime("%Y%m%d_%H%M%S")
    i = 0
    while d.exists():
        i += 1
        d = d.with_name(d.name.split("~")[0] + f"~{i}")
    H.ghi_json(d / "ket_qua.json", {"khoa": H.rel(khoa), "khoa_sha256": H.sha256_file(khoa), "dat": not loi, "loi": loi},
               vung)
    print(json.dumps({"dat": not loi, "loi": loi[:10]}, ensure_ascii=False))
    if loi:
        raise H.LoiHopDong(f"kiem-khoa lech: {loi[:5]}")
    return 0


def buoc_dong_goi_a(ctx: Ctx, phien_A: Path, vung) -> int:
    khoa = phien_A / "khoa" / "khoa.json"
    if not khoa.is_file():
        raise H.LoiHopDong("chua co khoa")
    loi = so_khoa(H.doc_json(khoa), anh_khoa(ctx, phien_A))
    if loi:
        raise H.LoiHopDong(f"goi A: khoa lech {loi[:5]}")
    tep = {r: h for r, h in H.bang_hash_thu_muc(phien_A).items()
           if "/_tam/" not in r and "/_cau_hinh_thu_vien/" not in r and "/kiem_khoa/" not in r}
    goi = {"phien_A": H.rel(phien_A), "khoa": H.rel(khoa), "khoa_sha256": H.sha256_file(khoa),
           "giao_thuc_sha256": H.sha256_file(ctx.gt_path), "so_file": len(tep), "file": tep}
    H.ghi_json(phien_A / "goi_a.json", goi, vung)
    sums = "".join(f"{h}  {r}\n" for r, h in sorted(tep.items()))
    H.ghi_bytes(phien_A / "SHA256SUMS", sums.encode("utf-8"), vung)
    print(f"goi A: {len(tep)} file")
    return 0


def kiem_goi_a(goi_a: Path) -> dict:
    """Bam lai TOAN BO file goi A (cache, smoke, tien kiem, khoa ...) + file thua trong phien A."""
    g = H.doc_json(goi_a)
    lech = [r for r, h in g["file"].items()
            if not (H.ROOT / r).is_file() or H.la_reparse(H.ROOT / r) or H.sha256_file(H.ROOT / r) != h]
    pa = H.ROOT / g["phien_A"]
    thua = [r for r in H.bang_hash_thu_muc(pa / "cache") if r not in g["file"]] + \
           [r for r in H.bang_hash_thu_muc(pa / "smoke") if r not in g["file"]]
    if lech or thua:
        raise H.LoiHopDong(f"goi A bi doi: lech {lech[:5]} thua {thua[:5]}")
    return g


def kiem_toan_ven_b(ctx: Ctx, g: dict) -> None:
    """Truoc MOI thao tac B: goi A nguyen byte (kiem_goi_a da goi) + snapshot khoa == trang thai hien tai
    (nguon du lieu, YAML/tham so, ma nguon, thu vien, Git dirty, tien kiem, cache)."""
    phien_A = H.ROOT / g["phien_A"]
    cache = {r: h for r, h in g["file"].items() if r.startswith(f"{g['phien_A']}/cache/")}
    loi = so_khoa(H.doc_json(H.ROOT / g["khoa"]), anh_khoa(ctx, phien_A, cache))
    if loi:
        raise H.LoiHopDong(f"trang thai hien tai khac khoa: {loi[:6]}")


def buoc_mo_phien(ctx: Ctx, phien_B: Path, goi_a: Path, review_a: Path, plan_b: Path, vung) -> int:
    if (phien_B / "phien.json").exists():
        raise H.LoiHopDong("phien B da mo")
    g = kiem_goi_a(goi_a)
    kiem_toan_ven_b(ctx, g)
    sha_goi = H.sha256_file(goi_a)
    rv, pb = H.doc_json(review_a), H.doc_json(plan_b)
    if rv.get("verdict") != "approve" or rv.get("goi_a_sha256") != sha_goi:
        raise H.LoiHopDong("review A khong approve hoac khong rang buoc dung goi A")
    if pb.get("duyet") is not True or pb.get("goi_a_sha256") != sha_goi or pb.get("khoa_sha256") != g["khoa_sha256"]:
        raise H.LoiHopDong("plan B chua duoc duyet hoac khong rang buoc dung goi A/khoa")
    if g["giao_thuc_sha256"] != H.sha256_file(ctx.gt_path):
        raise H.LoiHopDong("giao thuc doi sau goi A")
    H.ghi_json(phien_B / "phien.json", {"goi_a": H.rel(goi_a), "goi_a_sha256": sha_goi, "khoa": g["khoa"],
                                         "khoa_sha256": g["khoa_sha256"], "phien_A": g["phien_A"],
                                         "review_a": H.rel(review_a), "review_a_sha256": H.sha256_file(review_a),
                                         "plan_b": H.rel(plan_b), "plan_b_sha256": H.sha256_file(plan_b),
                                         "giao_thuc_sha256": g["giao_thuc_sha256"]}, vung)
    print("mo-phien B: ok")
    return 0


def nap_phien_b(ctx: Ctx, phien_B: Path) -> dict:
    """Moi thao tac B: kiem lai NOI DUNG goi A/khoa/review/plan, khong chi marker ngoai."""
    p = phien_B / "phien.json"
    if not p.is_file():
        raise H.LoiHopDong("chua mo-phien B")
    ph = H.doc_json(p)
    for k in ("goi_a", "khoa", "review_a", "plan_b"):
        if H.sha256_file(H.ROOT / ph[k]) != ph[f"{k}_sha256"]:
            raise H.LoiHopDong(f"{k} doi sau mo-phien")
    if H.sha256_file(ctx.gt_path) != ph["giao_thuc_sha256"]:
        raise H.LoiHopDong("giao thuc doi sau mo-phien")
    ph["goi"] = kiem_goi_a(H.ROOT / ph["goi_a"])          # NOI DUNG goi A (moi file), khong chi file mo ta
    if ph["goi"]["khoa"] != ph["khoa"] or ph["goi"]["phien_A"] != ph["phien_A"]:
        raise H.LoiHopDong("phien.json khong khop goi A")
    kiem_toan_ven_b(ctx, ph["goi"])                         # nguon/ma/cau hinh hien tai == khoa
    return ph


def kiem_cache_tu_goi(ph: dict):
    """Truoc moi don vi B: bam lai npz/json/meta cua detector-video dung voi goi A."""
    def kiem(det: str, v: str):
        for f in (f"{v}.npz", f"{v}.json", "meta.json"):
            r = f"{ph['phien_A']}/cache/{det}/{f}"
            if ph["goi"]["file"].get(r) is None or H.sha256_file(H.ROOT / r) != ph["goi"]["file"][r]:
                raise H.LoiHopDong(f"cache {det}/{f} khac goi A")
    return kiem


# ---------------------------------------------------------------------------
# B: lap / cham lai / tong hop / hau kiem
# ---------------------------------------------------------------------------
def so_byte(a: Path, b: Path) -> bool:
    return a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()


def buoc_lap_tracker(ctx: Ctx, phien_B: Path, ph: dict, video: str, vung) -> int:
    import run_tracking as rt
    reg = rt.tracker_registry()
    cache_A = H.ROOT / ph["phien_A"] / "cache"
    kiem = kiem_cache_tu_goi(ph)
    ket = []
    for r in ctx.runs():                                  # quet truoc: marker hong (lap hoac goc) -> 2 truoc moi ghi
        xong_don_vi(phien_B / "lap" / ctx.split / r["run"], r["run"], video)
        if not xong_don_vi(phien_B / "tracking" / ctx.split / r["run"], r["run"], video):
            raise H.LoiHopDong(f"lap-tracker: tracking goc {r['run']}/{video} chua xong")
    for r in ctx.runs():
        kiem(r["detector"], video)
        lap_dir = phien_B / "lap" / ctx.split / r["run"]
        if not xong_don_vi(lap_dir, r["run"], video):
            chay_don_vi(ctx, r, video, cache_A / r["detector"], lap_dir, reg, vung)
        goc = phien_B / "tracking" / ctx.split / r["run"]
        ket.append({"run": r["run"], "trung_byte": all(so_byte(goc / s / f"{video}{e}", lap_dir / s / f"{video}{e}")
                                                       for s, e in (("data", ".txt"), ("data_cls", ".csv")))})
    H.ghi_json(phien_B / "lap" / f"ket_qua_{video}.json", ket, vung, ghi_de=True)
    sai = [k["run"] for k in ket if not k["trung_byte"]]
    if sai:
        raise H.LoiHopDong(f"tai lap tracker lech: {sai}")
    print(f"lap-tracker {video}: {len(ket)} don vi trung byte")
    return 0


def buoc_cham_lai(ctx: Ctx, phien_B: Path, runs: list[str], vung) -> int:
    ket = []
    for run in runs:                                      # quet truoc: marker hong (cham goc/cham lai) -> 2
        xong_cham(phien_B / "cham_lai" / run, run)
        if not xong_cham(phien_B / "cham" / run, run):
            raise H.LoiHopDong(f"cham-lai: cham goc {run} chua xong")
    for run in runs:
        if not xong_cham(phien_B / "cham_lai" / run, run):
            cham_run(ctx, run, ctx.videos, phien_B / "tracking", phien_B / "cham_lai", vung)
        ket.append({"run": run, "trung_byte": all(so_byte(phien_B / "cham" / run / f, phien_B / "cham_lai" / run / f)
                                                  for f in ("per_video_metrics.csv", "combined_metrics.csv",
                                                            "raw_fields.json"))})
    H.ghi_json(phien_B / "cham_lai" / "ket_qua.json", ket, vung, ghi_de=True)
    sai = [k["run"] for k in ket if not k["trung_byte"]]
    if sai:
        raise H.LoiHopDong(f"cham lai lech: {sai}")
    return 0


def doc_chi_so(ctx: Ctx, p_cs: Path):
    """Chi so bootstrap da luu PHAI dung giao thuc: int64, (so_mau, so_video), mien [0, n), TRUNG default_rng(0)."""
    import numpy as np
    import thong_ke_bdd as T
    if not p_cs.is_file() or H.la_reparse(p_cs):
        raise H.LoiHopDong(f"thieu chi so bootstrap {H.rel(p_cs)}")
    a = np.load(p_cs, allow_pickle=False)
    n = len(ctx.videos)
    if a.dtype != np.int64 or a.shape != (ctx.so_mau, n):
        raise H.LoiHopDong(f"chi so bootstrap dtype/shape {a.dtype}/{a.shape} != int64/({ctx.so_mau}, {n})")
    if a.size and (int(a.min()) < 0 or int(a.max()) >= n):
        raise H.LoiHopDong("chi so bootstrap ngoai mien [0, so_video)")
    if not np.array_equal(a, T.tao_chi_so(n, ctx.so_mau, 0)):
        raise H.LoiHopDong("chi so bootstrap khac day numpy.random.default_rng(0)")
    return a


def csv_bytes(rs: list[dict]) -> bytes:
    import csv
    import io
    b = io.StringIO(newline="")
    cot = list(dict.fromkeys(k for r in rs for k in r))       # PF co them cot seed/min/max
    w = csv.DictWriter(b, fieldnames=cot, restval="", lineterminator="\n")
    w.writeheader()
    w.writerows(rs)
    return b.getvalue().encode("utf-8")


def doc_cham(ctx: Ctx, phien_B: Path) -> tuple[dict, dict, dict]:
    """raw_fields + COMBINED cua 36 run (marker cham hop le, hong -> 2) + sha256 dau vao."""
    import pandas as pd
    import boot_stats as bs
    runs = ctx.runs()
    thieu = [r["run"] for r in runs if not xong_cham(phien_B / "cham" / r["run"], r["run"])]
    if thieu:
        raise H.LoiHopDong(f"tong-hop: thieu cham {thieu[:5]}")
    raw, comb, dau_vao = {}, {}, {}
    for r in runs:
        d = phien_B / "cham" / r["run"]
        raw[r["run"]] = bs.load_raw_fields(d / "raw_fields.json")
        comb[r["run"]] = pd.read_csv(d / "combined_metrics.csv").iloc[0].to_dict()
        dau_vao[r["run"]] = {f: H.sha256_file(d / f) for f in ("raw_fields.json", "combined_metrics.csv")}
    return raw, comb, dau_vao


def tinh_ci(ctx: Ctx, raw: dict, comb: dict, chi_so) -> tuple[list[dict], list[dict]]:
    import thong_ke_bdd as T
    nhom = {}
    for r in ctx.runs():
        nhom.setdefault((r["detector"], r["dieu_kien"]), {}).setdefault(r["bo_loc"], []).append(raw[r["run"]])
    rows = T.bang_ci(nhom, ctx.videos, chi_so)
    if len(rows) != ctx.gt["thong_ke"]["so_ci"]:
        raise H.LoiHopDong(f"so CI {len(rows)} != {ctx.gt['thong_ke']['so_ci']}")
    return rows, T.bang_chinh(comb)


def buoc_tong_hop(ctx: Ctx, phien_B: Path, vung) -> int:
    import io
    import numpy as np
    import thong_ke_bdd as T
    raw, comb, dau_vao = doc_cham(ctx, phien_B)
    out = phien_B / "tong_hop"
    p_cs = out / "chi_so_bootstrap.npy"
    for f in ("chi_so_bootstrap.npy", "ci_108.csv", "bang_chinh.csv", "kiem_chung.json"):
        H.kiem_duong_ghi(out / f, vung)
    sai = {run: T.sai_so_tai_to_hop(raw[run], ctx.videos, comb[run]) for run in raw}
    dung = ctx.gt["thong_ke"]["dung_sai_tai_to_hop"]
    vuot = {run: s for run, s in sai.items() if any(s[m] > dung[m] for m in T.METRIC)}
    if vuot:
        raise H.LoiHopDong(f"tai to hop lech evaluator vuot dung sai: {list(vuot)[:3]}")
    if p_cs.exists() or H.la_reparse(p_cs):
        chi_so = doc_chi_so(ctx, p_cs)                       # tai dung: kiem so mau/dtype/mien/day rng TRUOC moi ghi
    else:
        chi_so = T.tao_chi_so(len(ctx.videos), ctx.so_mau, 0)
        bio = io.BytesIO()
        np.save(bio, chi_so)
        H.ghi_bytes(p_cs, bio.getvalue(), vung)
    rows, bang = tinh_ci(ctx, raw, comb, chi_so)
    rows2, _ = tinh_ci(ctx, raw, comb, doc_chi_so(ctx, p_cs))       # tai tinh tu chi so DA LUU
    if rows2 != rows:
        raise H.LoiHopDong("tai tinh tu chi so luu khong trung")
    b_ci, b_bang = csv_bytes(rows), csv_bytes(bang)
    H.ghi_bytes(out / "ci_108.csv", b_ci, vung, ghi_de=True)
    H.ghi_bytes(out / "bang_chinh.csv", b_bang, vung, ghi_de=True)
    H.ghi_json(out / "kiem_chung.json", {
        "so_ci": len(rows), "so_mau": ctx.so_mau, "so_video": len(ctx.videos), "rng": "numpy.random.default_rng(0)",
        "chi_so_sha256": H.sha256_file(p_cs), "chi_so_shape": list(chi_so.shape), "chi_so_dtype": str(chi_so.dtype),
        "tai_tinh_tu_chi_so_luu_trung": True, "ci_108_sha256": H.sha256_bytes(b_ci),
        "bang_chinh_sha256": H.sha256_bytes(b_bang), "dau_vao_sha256": dau_vao,
        "sai_so_tai_to_hop_max": {m: max(s[m] for s in sai.values()) for m in T.METRIC},
        "dung_sai": dung}, vung, ghi_de=True)
    print(f"tong-hop: {len(rows)} CI, {ctx.so_mau} mau")
    return 0


def kiem_lai_tong_hop(ctx: Ctx, phien_B: Path) -> list[str]:
    """Hau kiem KHONG tin kiem_chung.json: kiem lai chi so (so mau/dtype/mien/day rng), tinh lai 108 CI + bang chinh
    tu raw/COMBINED da cham va so BYTE voi CSV tren dia; so hash dau vao/dau ra voi kiem_chung."""
    out = phien_B / "tong_hop"
    kc_p = out / "kiem_chung.json"
    if not kc_p.is_file():
        return ["thong ke chua tong hop"]
    try:
        kc = H.doc_json(kc_p)
        raw, comb, dau_vao = doc_cham(ctx, phien_B)
        chi_so = doc_chi_so(ctx, out / "chi_so_bootstrap.npy")
        rows, bang = tinh_ci(ctx, raw, comb, chi_so)
    except H.LoiHopDong as e:
        return [f"thong ke: {e}"]
    loi = []
    for f, b in (("ci_108.csv", csv_bytes(rows)), ("bang_chinh.csv", csv_bytes(bang))):
        p = out / f
        if not p.is_file() or p.read_bytes() != b:
            loi.append(f"{f} khac tinh lai tu chi so luu")
    ky = {"so_ci": ctx.gt["thong_ke"]["so_ci"], "so_mau": ctx.so_mau, "so_video": len(ctx.videos),
          "chi_so_sha256": H.sha256_file(out / "chi_so_bootstrap.npy"), "ci_108_sha256": H.sha256_bytes(csv_bytes(rows)),
          "bang_chinh_sha256": H.sha256_bytes(csv_bytes(bang)), "dau_vao_sha256": dau_vao,
          "tai_tinh_tu_chi_so_luu_trung": True}
    loi += [f"kiem_chung {k} lech" for k, v in ky.items() if H.chuan_json(kc.get(k)) != H.chuan_json(v)]
    return loi


def buoc_hau_kiem(ctx: Ctx, phien_B: Path, ph: dict, vung) -> int:
    loi = []
    try:
        kiem_goi_a(H.ROOT / ph["goi_a"])
    except H.LoiHopDong as e:
        loi.append(str(e))
    runs = ctx.runs()
    ca = H.ROOT / ph["phien_A"] / "cache"
    n_cache = sum(1 for d in ctx.detectors() for v in ctx.videos
                  if H.kiem_marker(ca / d / f"{v}.xong.json", file_marker_cache(ca / d, v),
                                   {"detector": d, "video": v})[0] == "xong")
    n_trk = sum(1 for r in runs for v in ctx.videos
                if H.kiem_marker(phien_B / "tracking" / ctx.split / r["run"] / "xong" / f"{v}.xong.json",
                                 file_don_vi(phien_B / "tracking" / ctx.split / r["run"], v),
                                 {"run": r["run"], "video": v})[0] == "xong")
    n_cham = sum(1 for r in runs if H.kiem_marker(phien_B / "cham" / r["run"] / "cham.xong.json",
                                                  file_cham(phien_B / "cham" / r["run"]), {"run": r["run"]})[0] == "xong")
    ky = {"cache": len(ctx.detectors()) * len(ctx.videos), "tracking": len(runs) * len(ctx.videos), "cham": len(runs)}
    for k, n in (("cache", n_cache), ("tracking", n_trk), ("cham", n_cham)):
        if n != ky[k]:
            loi.append(f"{k}: {n} != {ky[k]}")
    loi += kiem_lai_tong_hop(ctx, phien_B)
    tl = ctx.gt["tai_lap_cong_B"]
    lp = phien_B / "lap" / f"ket_qua_{tl['video']}.json"
    if not lp.is_file() or not all(x["trung_byte"] for x in H.doc_json(lp)) or len(H.doc_json(lp)) != len(runs):
        loi.append("tai lap tracker chua du / lech")
    cl = phien_B / "cham_lai" / "ket_qua.json"
    if not cl.is_file() or [x["run"] for x in H.doc_json(cl)] != tl["cham_lai"] or \
            not all(x["trung_byte"] for x in H.doc_json(cl)):
        loi.append("cham lai chua du / lech")
    kk = sorted((phien_B / "kiem_khoa").glob("*/ket_qua.json")) if (phien_B / "kiem_khoa").exists() else []
    if not kk or not H.doc_json(kk[-1]).get("dat"):
        loi.append("chua co kiem-khoa dat trong phien B")
    d = {"dat": not loi, "loi": loi, "dem": {"cache": n_cache, "tracking": n_trk, "cham": n_cham}, "ky_vong": ky}
    H.ghi_json(phien_B / "hau_kiem.json", d, vung, ghi_de=True)
    print(json.dumps(d, ensure_ascii=False))
    if loi:
        raise H.LoiHopDong(f"hau-kiem: {loi}")
    return 0


# ---------------------------------------------------------------------------
# A: tien-kiem / smoke
# ---------------------------------------------------------------------------
def buoc_tien_kiem(ctx: Ctx, phien_A: Path, vung) -> int:
    d = {"giao_thuc_sha256": H.sha256_file(ctx.gt_path), "nguon": kiem_nguon(ctx),
         "cau_hinh_hieu_luc": cau_hinh_hieu_luc(ctx), "thu_vien": thu_vien(),
         "head": git_doc("rev-parse", "HEAD").strip(), "so_run": len(ctx.runs()), "so_video": len(ctx.videos)}
    H.ghi_json(phien_A / "tien_kiem" / "tien_kiem.json", d, vung)
    print(f"tien-kiem: {d['so_run']} run, {d['so_video']} video")
    return 0


def buoc_smoke(ctx: Ctx, phien_A: Path, vung, gioi_han, ngan_sach) -> int:
    vids = ctx.gt["smoke_cong_A"]["videos"]
    for det in ctx.detectors():
        for v in vids:
            loi = kiem_cache_video(ctx, det, phien_A / "cache" / det, v)
            if loi:
                raise H.LoiHopDong(f"smoke: cache {det}/{v} {loi[:3]}")
    root = phien_A / "smoke"
    for r in ctx.runs():                                  # marker cham hong -> 2 truoc moi ghi
        xong_cham(root / "cham" / r["run"], r["run"])
    lap_don_vi(ctx, ctx.runs(), vids, lambda d: phien_A / "cache" / d, root / "tracking", vung, gioi_han, ngan_sach)
    for r in ctx.runs():
        if not xong_cham(root / "cham" / r["run"], r["run"]):
            cham_run(ctx, r["run"], vids, root / "tracking", root / "cham", vung)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
BUOC_A = ("tien-kiem", "cache", "kiem-cache", "smoke", "khoa", "dong-goi-a")
BUOC_B = ("mo-phien", "tracking", "cham", "lap-tracker", "cham-lai", "tong-hop", "hau-kiem")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buoc", required=True, choices=BUOC_A + BUOC_B + ("kiem-khoa",))
    ap.add_argument("--cong", required=True, choices=["A", "B"])
    ap.add_argument("--giao-thuc", default=str(GIAO_THUC))
    ap.add_argument("--phien", required=True, help="phien A (chuan_bi/<phien_A>) hoac B (thuc_thi/<phien_B>)")
    ap.add_argument("--phien-a", default=None, help="kiem-khoa cong B: phien A (chi doc)")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--goi-a"); ap.add_argument("--review-a"); ap.add_argument("--plan-b")
    ap.add_argument("--gioi-han-don-vi", type=int, default=None)
    ap.add_argument("--ngan-sach-giay", type=float, default=420.0)
    ap.add_argument("--so-doan", type=int, default=1, help="lap toi da N doan foreground trong lenh nay")
    ns = ap.parse_args(argv)                 # so mau bootstrap CHI tu giao thuc (5000; fixture khai bao rieng)
    ctx = Ctx(Path(ns.giao_thuc))
    if (ns.buoc in BUOC_A and ns.cong != "A") or (ns.buoc in BUOC_B and ns.cong != "B"):
        raise H.LoiHopDong(f"buoc {ns.buoc} khong thuoc cong {ns.cong}")
    vung = vung_phien(ctx, ns.phien, ns.cong)
    phien = vung[0]
    H.tao_thu_muc(phien, vung)
    H.chuan_bi_moi_truong(phien, vung)

    def theo_doan(fn):
        for i in range(max(1, ns.so_doan)):
            try:
                return fn()
            except H.ChuaXong:
                if i + 1 >= ns.so_doan:
                    raise
                print(f"doan {i + 1}: chua xong, chay doan tiep")
        return 0

    b = ns.buoc
    if b == "tien-kiem":
        return buoc_tien_kiem(ctx, phien, vung)
    if b == "cache":
        if ns.detector not in ctx.detectors():
            raise H.LoiHopDong("--detector khong thuoc giao thuc")
        return theo_doan(lambda: buoc_cache(ctx, phien, ns.detector, vung, ns.device, ns.gioi_han_don_vi,
                                            ns.ngan_sach_giay))
    if b == "kiem-cache":
        loi = [e for d in ctx.detectors() for v in ctx.videos for e in kiem_cache_video(ctx, d, phien / "cache" / d, v)]
        H.ghi_json(phien / "kiem_cache" / f"ket_qua_{time.strftime('%Y%m%d_%H%M%S')}.json", {"dat": not loi, "loi": loi},
                   vung)
        if loi:
            raise H.LoiHopDong(f"kiem-cache: {loi[:5]}")
        return 0
    if b == "smoke":
        return theo_doan(lambda: buoc_smoke(ctx, phien, vung, ns.gioi_han_don_vi, ns.ngan_sach_giay))
    if b == "khoa":
        return buoc_khoa(ctx, phien, vung)
    if b == "dong-goi-a":
        return buoc_dong_goi_a(ctx, phien, vung)
    if b == "kiem-khoa":
        if ns.cong == "A":
            return buoc_kiem_khoa(ctx, phien / "khoa" / "khoa.json", phien, vung, phien)
        ph = nap_phien_b(ctx, phien)
        return buoc_kiem_khoa(ctx, H.ROOT / ph["khoa"], phien, vung, H.ROOT / ph["phien_A"])
    if b == "mo-phien":
        return buoc_mo_phien(ctx, phien, Path(os.path.abspath(ns.goi_a)), Path(os.path.abspath(ns.review_a)),
                             Path(os.path.abspath(ns.plan_b)), vung)
    ph = nap_phien_b(ctx, phien)
    cache_A = H.ROOT / ph["phien_A"] / "cache"
    if b == "tracking":
        return theo_doan(lambda: lap_don_vi(ctx, ctx.runs(), ctx.videos, lambda d: cache_A / d, phien / "tracking", vung,
                                            ns.gioi_han_don_vi, ns.ngan_sach_giay, kiem_cache_tu_goi(ph)))
    if b == "cham":
        thieu = [(r["run"], v) for r in ctx.runs() for v in ctx.videos
                 if not xong_don_vi(phien / "tracking" / ctx.split / r["run"], r["run"], v)]   # hong -> 2
        if thieu:                                  # chua du 7.200 don vi -> khong cham run nao
            raise H.LoiHopDong(f"thieu {len(thieu)} don vi tracking (vd {thieu[:3]}) -- khong cham")
        chua = [r for r in ctx.runs() if not xong_cham(phien / "cham" / r["run"], r["run"])]   # hong -> 2 truoc ghi
        for r in chua:
            cham_run(ctx, r["run"], ctx.videos, phien / "tracking", phien / "cham", vung)
        return 0
    if b == "lap-tracker":
        return buoc_lap_tracker(ctx, phien, ph, ctx.gt["tai_lap_cong_B"]["video"], vung)
    if b == "cham-lai":
        return buoc_cham_lai(ctx, phien, ctx.gt["tai_lap_cong_B"]["cham_lai"], vung)
    if b == "tong-hop":
        return buoc_tong_hop(ctx, phien, vung)
    if b == "hau-kiem":
        return buoc_hau_kiem(ctx, phien, ph, vung)
    raise H.LoiHopDong(f"buoc {b} chua ho tro")


if __name__ == "__main__":
    argv = sys.argv[1:]
    ctx0 = Ctx(Path(argv[argv.index("--giao-thuc") + 1]) if "--giao-thuc" in argv else GIAO_THUC)
    cong = argv[argv.index("--cong") + 1] if "--cong" in argv else "?"
    vung0 = vung_phien(ctx0, argv[argv.index("--phien") + 1], cong)
    H.tao_thu_muc(vung0[0], vung0)
    H.chuan_bi_moi_truong(vung0[0], vung0)
    sys.exit(H.chay_va_ghi_log(main, argv, vung0[0], vung0, "dieu_phoi_bdd"))
