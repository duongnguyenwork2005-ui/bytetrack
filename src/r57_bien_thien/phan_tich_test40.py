"""
phan_tich_test40.py -- R57A: bootstrap ghep cap theo video cho chenh lech CV / EKF / UKF / PF500 tren test40 da nghiem thu
(phien thuc thi R56 phien_20261021_161832_735800_pid11640). Chi phan tich hau nghiem tu metric da luu: khong tracking,
evaluator, cache, GPU.

Hai buoc, chay rieng, foreground:
    --buoc chuan-bi  --giao-thuc configs/phase2/r57_bien_thien/giao_thuc.json
        doc DUNG 36 lan cham trong nguon.json R56, kiem marker HOAN_TAT.json, lien ket khoa / giao thuc R54 / hop dong
        R55 / run / cache, hash file metric, tap video va seed; tao phien moi results/phase2/r57_bien_thien/phien_<ts>_pid<p>/
        voi ban giao thuc bat bien + nguon.json (duong dan / SHA256) + nhom_video.csv. Thieu / trung / sai hash / NaN /
        sai schema -> exit 2, khong ghi gi; khong tu tim lan moi nhat, khong bo video.
    --buoc phan-tich --phien phien_<ts>_pid<p>
        kiem lai hash phien + ma + dau vao; tai to hop COMBINED (boot_stats._stack / combine_stacked) tung run, doi chieu
        combined_metrics.csv (1e-9) va bang R56 (lam tron); bootstrap hai thiet ke (video 40 / nhom 39) voi MOT bo chi so
        moi thiet ke dung chung cho moi run va seed; PF500 = trung binh COMBINED 3 seed (boot_seed._tb) roi moi tru.
        Nhanh video doi chieu boot_seed.paired_bootstrap_seed (trung khop ==); tai tinh tu chi so da luu (CSV trong bo
        nho) trung khop ==; hash dau vao truoc / sau khong doi. Moi kiem dat moi ghi dau ra (tao file doc quyen, khong
        ghi de).

Exit: 0 dat; 2 loi dau vao / kiem khong dat; 1 loi khac.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import boot_stats as bs  # noqa: E402
import boot_seed as bse  # noqa: E402

VUNG_REL = "results/phase2/r57_bien_thien"
FILE_MA = {"phan_tich_test40.py": Path(__file__).resolve(), "boot_stats.py": ROOT / "src" / "boot_stats.py",
           "boot_seed.py": ROOT / "src" / "kc48" / "boot_seed.py"}
TEN_PHIEN = re.compile(r"^phien_\d{8}_\d{6}_\d{6}_pid\d+$")
DAU_RA_PHAN_TICH = ["chi_so_video.csv", "chi_so_nhom.csv", "diem_uoc_luong.csv", "doi_chieu_video.csv",
                    "doi_chieu_nhom.csv", "kiem_chung.json", "SHA256SUMS"]
DAU_RA_CHUAN_BI = ["giao_thuc.json", "nguon.json", "nhom_video.csv"]
PP = {"HOTA", "IDF1"}      # metric trinh bay x100 (diem %)


class LoiDauVao(Exception):
    pass


# ----------------------------------------------------------------------------------------------- tien ich duong dan / hash
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(1 << 20), b""):
            h.update(k)
    return h.hexdigest()


def la_reparse(p: Path) -> bool:
    """symlink hoac junction / reparse point (Windows)."""
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _kiem_khong_reparse(goc: Path, p: Path) -> None:
    goc = goc.resolve()
    rel = Path(os.path.abspath(p)).relative_to(Path(os.path.abspath(goc)))
    cur = Path(os.path.abspath(goc))
    for part in rel.parts:
        cur = cur / part
        if la_reparse(cur):
            raise LoiDauVao(f"duong dan qua symlink/junction: {cur}")


def duong_doc(goc: Path, rel: str) -> Path:
    """File doc phai la duong tuong doi trong goc, khong '..', khong qua symlink/junction, ton tai."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts or ":" in rel:
        raise LoiDauVao(f"duong dan doc khong hop le: {rel!r}")
    p = goc / rel
    if not Path(os.path.abspath(p)).is_relative_to(Path(os.path.abspath(goc))):
        raise LoiDauVao(f"duong dan doc thoat goc: {rel}")
    _kiem_khong_reparse(goc, p)
    if not p.is_file():
        raise LoiDauVao(f"thieu file: {rel}")
    return p


def kiem_vung_ghi(goc: Path, p: Path, vung_rel: str = VUNG_REL) -> Path:
    """Moi ghi phai nam trong goc/vung_rel (duong tuyet doi chuan hoa), khong qua symlink/junction."""
    vung = Path(os.path.abspath(goc / vung_rel))
    pa = Path(os.path.abspath(p))
    if pa != vung and not pa.is_relative_to(vung):
        raise LoiDauVao(f"ghi ngoai vung {vung_rel}: {p}")
    _kiem_khong_reparse(goc, pa)
    if pa.exists() and os.path.normcase(str(pa.resolve())) != os.path.normcase(str(pa)):
        raise LoiDauVao(f"duong dan ghi phan giai khac: {p}")
    return pa


def ghi_moi(goc: Path, p: Path, data: bytes, vung_rel: str = VUNG_REL) -> str:
    """Tao file moi (doc quyen 'xb'); file da ton tai -> loi, khong ghi de."""
    kiem_vung_ghi(goc, p, vung_rel)
    try:
        with open(p, "xb") as f:
            f.write(data)
    except FileExistsError as e:
        raise LoiDauVao(f"khong ghi de: {p}") from e
    return sha256_bytes(data)


def _json_khong_trung(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise LoiDauVao(f"khoa JSON trung: {k}")
        d[k] = v
    return d


def doc_json(p: Path):
    try:
        return json.loads(p.read_bytes().decode("utf-8"), object_pairs_hook=_json_khong_trung)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LoiDauVao(f"JSON hong {p}: {e}") from e


def json_bytes(o) -> bytes:
    return (json.dumps(o, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def csv_bytes(header: list[str], rows: list[list]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _so(x, ten: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as e:
        raise LoiDauVao(f"{ten}: khong phai so ({x!r})") from e
    if not math.isfinite(v):
        raise LoiDauVao(f"{ten}: khong huu han ({x!r})")
    return v


# ------------------------------------------------------------------------------------------------------------ giao thuc
def ten_run(det: str, dk: str, bl: str, seed=None) -> str:
    return f"r54-{det}-{dk}-{bl}" + ("" if seed is None else f"-s{seed}")


def danh_sach_run(gt: dict) -> list[dict]:
    pf, seeds = gt["pf"]["bo_loc"], list(gt["pf"]["seeds"])
    out = []
    for det in gt["detector"]:
        for dk in gt["dieu_kien"]:
            for bl in gt["bo_loc"]:
                for s in (seeds if bl == pf else [None]):
                    out.append({"run": ten_run(det, dk, bl, s), "detector": det, "dieu_kien": dk, "bo_loc": bl, "seed": s})
    return out


def kiem_giao_thuc(gt: dict) -> None:
    can = ["phien_ban", "nguon", "so_video", "video", "detector", "dieu_kien", "bo_loc", "pf", "so_run", "metric",
           "hieu_so", "bootstrap", "thiet_ke", "dung_sai"]
    for k in can:
        if k not in gt:
            raise LoiDauVao(f"giao thuc thieu '{k}'")
    v = gt["video"]
    if len(v) != gt["so_video"] or len(set(v)) != len(v):
        raise LoiDauVao("giao thuc: danh sach video sai so luong hoac trung")
    if gt["pf"]["bo_loc"] not in gt["bo_loc"] or sorted(gt["pf"]["seeds"]) != list(gt["pf"]["seeds"]) or \
            len(set(gt["pf"]["seeds"])) != len(gt["pf"]["seeds"]) or len(gt["pf"]["seeds"]) < 1:
        raise LoiDauVao("giao thuc: pf sai")
    runs = danh_sach_run(gt)
    if len(runs) != gt["so_run"] or len({r["run"] for r in runs}) != len(runs):
        raise LoiDauVao(f"giao thuc: so_run {gt['so_run']} != {len(runs)}")
    ms = gt["metric"]["danh_sach"]
    if ms[0] != gt["metric"]["chinh"] or ms[1:] != gt["metric"]["bo_sung"] or any(m not in bs.METRICS for m in ms):
        raise LoiDauVao("giao thuc: metric sai")
    for a, b in gt["hieu_so"]:
        if a not in gt["bo_loc"] or b not in gt["bo_loc"] or a == b:
            raise LoiDauVao(f"giao thuc: hieu so sai {a}-{b}")
    if len({tuple(x) for x in gt["hieu_so"]}) != len(gt["hieu_so"]):
        raise LoiDauVao("giao thuc: hieu so trung")
    bt = gt["bootstrap"]
    if not (isinstance(bt["B"], int) and bt["B"] > 0 and isinstance(bt["rng_seed"], int)):
        raise LoiDauVao("giao thuc: B / rng_seed sai")
    if bt["phan_vi"] != [2.5, 97.5] or bt["method"] != "linear":
        raise LoiDauVao("giao thuc: phan vi / method sai")
    if gt["thiet_ke"]["video"]["K"] != len(v):
        raise LoiDauVao("giao thuc: K video sai")
    nhom = gt["thiet_ke"]["nhom"]["nhom"]
    if gt["thiet_ke"]["nhom"]["K"] != len(nhom) or len({n["id"] for n in nhom}) != len(nhom):
        raise LoiDauVao("giao thuc: K nhom sai hoac id trung")
    phang = [x for n in nhom for x in n["video"]]
    if sorted(phang) != sorted(v) or len(phang) != len(v) or any(len(n["video"]) < 1 for n in nhom):
        raise LoiDauVao("giao thuc: nhom khong phan hoach dung tap video")
    vt = {x: i for i, x in enumerate(v)}
    dau = [min(vt[x] for x in n["video"]) for n in nhom]
    if dau != sorted(dau):
        raise LoiDauVao("giao thuc: thu tu nhom khong theo video dau tien trong manifest")
    tb = gt["dung_sai"].get("bang_r56", {})
    eps = tb.get("eps_dau_phay_dong")
    if not (isinstance(eps, float) and 0 < eps < 1e-6):
        raise LoiDauVao("giao thuc: dung_sai.bang_r56.eps_dau_phay_dong sai")
    for bang in ("tong_hop_36_run", "pf500_trung_binh"):
        d = tb.get(bang, {})
        if any(not isinstance(d.get(m), (int, float)) or isinstance(d.get(m), bool) or not 0 <= d[m] < 1 for m in ms):
            raise LoiDauVao(f"giao thuc: dung_sai.bang_r56.{bang} thieu / sai metric")
    if tb["tong_hop_36_run"]["IDSW"] != 0:
        raise LoiDauVao("giao thuc: IDSW bang tong_hop phai khop chinh xac")
    t = gt["dung_sai"].get("tai_to_hop_vs_combined_metrics", {})
    if t.get("IDSW") != 0 or any(not isinstance(t.get(m), (int, float)) or t[m] > 1e-9 for m in ms):
        raise LoiDauVao("giao thuc: dung sai tai to hop vs combined_metrics phai <= 1e-9, IDSW = 0")


def thanh_vien_nhom(gt: dict) -> list[np.ndarray]:
    vt = {x: i for i, x in enumerate(gt["video"])}
    return [np.array([vt[x] for x in n["video"]], dtype=np.int64) for n in gt["thiet_ke"]["nhom"]["nhom"]]


# -------------------------------------------------------------------------------------------------- doc / kiem dau vao
def doc_raw(p: Path, videos: list[str], ten: str) -> dict:
    raw = doc_json(p)
    if not isinstance(raw, dict) or set(raw) != set(videos) or len(raw) != len(videos):
        thieu = sorted(set(videos) - set(raw)) if isinstance(raw, dict) else videos
        raise LoiDauVao(f"{ten}: raw_fields sai tap video (thieu {thieu[:5]})")
    n_alpha = None
    for v in videos:
        rec = raw[v]
        if not isinstance(rec, dict) or set(rec) != set(bs.ARRAY_FIELDS) | set(bs.SCALAR_FIELDS):
            raise LoiDauVao(f"{ten}/{v}: raw_fields sai schema")
        for f in bs.ARRAY_FIELDS:
            a = rec[f]
            if not isinstance(a, list) or (n_alpha is not None and len(a) != n_alpha) or len(a) == 0:
                raise LoiDauVao(f"{ten}/{v}/{f}: mang sai do dai")
            n_alpha = len(a)
            for x in a:
                _so(x, f"{ten}/{v}/{f}")
        for f in bs.SCALAR_FIELDS:
            x = _so(rec[f], f"{ten}/{v}/{f}")
            if x != int(x) or x < 0:
                raise LoiDauVao(f"{ten}/{v}/{f}: khong phai so dem")
    if n_alpha != 19:
        raise LoiDauVao(f"{ten}: so alpha {n_alpha} != 19")
    return raw


def doc_csv(p: Path, cot: list[str], ten: str) -> list[dict]:
    """CSV chat: header khong trung / khong rong, du cac cot 'cot'; moi dong dung so cot cua header (khong thua /
    thieu, khong dong trong). Khong dung csv.DictReader vi no am tham gop header trung va gom cot thua vao khoa None."""
    try:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f, strict=True))
    except (UnicodeDecodeError, csv.Error) as e:
        raise LoiDauVao(f"{ten}: CSV hong ({e})") from e
    if not rows:
        raise LoiDauVao(f"{ten}: CSV rong")
    h = rows[0]
    if any(not c for c in h) or len(set(h)) != len(h):
        raise LoiDauVao(f"{ten}: header trung hoac rong {h}")
    thieu = [c for c in cot if c not in h]
    if thieu:
        raise LoiDauVao(f"{ten}: header thieu cot {thieu}")
    out = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(h):
            raise LoiDauVao(f"{ten}: dong {i} co {len(r)} cot, header {len(h)} cot")
        out.append(dict(zip(h, r)))
    return out


def _theo_khoa(rows: list[dict], khoa, ten: str) -> dict:
    """dict theo khoa; khoa trung -> loi (khong de dict comprehension gop am tham)."""
    out = {}
    for r in rows:
        k = khoa(r)
        if k in out:
            raise LoiDauVao(f"{ten}: dong trung khoa {k}")
        out[k] = r
    return out


def doc_combined_csv(p: Path, ten: str, metrics: list[str]) -> dict:
    rows = doc_csv(p, ["video", *metrics], f"{ten}/combined_metrics.csv")
    if len(rows) != 1 or rows[0]["video"] != "COMBINED_SEQ":
        raise LoiDauVao(f"{ten}: combined_metrics.csv sai schema")
    return {m: _so(rows[0][m], f"{ten}/combined/{m}") for m in metrics}


def doc_per_video_csv(p: Path, videos: list[str], ten: str, metrics: list[str]) -> dict:
    rows = doc_csv(p, ["video", *metrics], f"{ten}/per_video_metrics.csv")
    by = _theo_khoa(rows, lambda r: r["video"], f"{ten}/per_video_metrics.csv")
    if set(by) != set(videos) or len(rows) != len(videos):
        raise LoiDauVao(f"{ten}: per_video_metrics.csv sai tap video")
    return {v: {m: _so(r[m], f"{ten}/per_video/{v}/{m}") for m in metrics} for v, r in by.items()}


def doc_bang_r56(goc: Path, gt: dict) -> tuple[dict, dict]:
    """Bang R56 (chi de doi chieu lam tron): kiem schema, khoa duy nhat, dung tap run / nhom, so huu han. Tra ve
    (tong_hop theo run -> {metric: float}, pf500 theo (detector, dieu_kien) -> {metric: float})."""
    metrics = gt["metric"]["danh_sach"]
    pf, seeds = gt["pf"]["bo_loc"], list(gt["pf"]["seeds"])
    runs = {r["run"]: r for r in danh_sach_run(gt)}
    ten = "bang R56 tong_hop"
    rows = doc_csv(duong_doc(goc, gt["nguon"]["bang_r56"]["tong_hop"]["rel"]),
                   ["run", "detector", "dieu_kien", "bo_loc", "seed", *metrics], ten)
    by = _theo_khoa(rows, lambda r: r["run"], ten)
    if set(by) != set(runs) or len(rows) != len(runs):
        raise LoiDauVao(f"{ten}: sai tap run")
    th = {}
    for run, r in by.items():
        mong = runs[run]
        if (r["detector"], r["dieu_kien"], r["bo_loc"], r["seed"]) != (
                mong["detector"], mong["dieu_kien"], mong["bo_loc"], "" if mong["seed"] is None else str(mong["seed"])):
            raise LoiDauVao(f"{ten}/{run}: detector / dieu_kien / bo_loc / seed khong khop ten run")
        th[run] = {m: _so(r[m], f"{ten}/{run}/{m}") for m in metrics}
        if th[run]["IDSW"] != int(th[run]["IDSW"]):
            raise LoiDauVao(f"{ten}/{run}: IDSW khong nguyen")
    ten = "bang R56 pf500"
    rows = doc_csv(duong_doc(goc, gt["nguon"]["bang_r56"]["pf500"]["rel"]),
                   ["detector", "dieu_kien", "so_seed", *[f"{m}_tb" for m in metrics]], ten)
    by = _theo_khoa(rows, lambda r: (r["detector"], r["dieu_kien"]), ten)
    nhom = {(d, c) for d in gt["detector"] for c in gt["dieu_kien"]}
    if set(by) != nhom or len(rows) != len(nhom):
        raise LoiDauVao(f"{ten}: sai tap nhom")
    pfr = {}
    for key, r in by.items():
        if r["so_seed"] != str(len(seeds)):
            raise LoiDauVao(f"{ten}/{key}: so_seed {r['so_seed']!r} != {len(seeds)} ({pf})")
        pfr[key] = {m: _so(r[f"{m}_tb"], f"{ten}/{key}/{m}_tb") for m in metrics}
    return th, pfr


def doi_chieu_bang_r56(c_full: dict, diem: dict, th: dict, pfr: dict, gt: dict) -> tuple[list, dict]:
    """|gia tri tai to hop (x100 voi HOTA/IDF1) - bang R56| <= dung sai khai bao + eps dau phay dong. Dung sai lay
    TRUC TIEP tu giao thuc cho tung bang / metric, khong nang ngam. Tra ve (danh sach lech, sai so lon nhat)."""
    metrics = gt["metric"]["danh_sach"]
    tb = gt["dung_sai"]["bang_r56"]
    eps = tb["eps_dau_phay_dong"]
    pf = gt["pf"]["bo_loc"]
    lech, sai = [], {"tong_hop_36_run": {m: 0.0 for m in metrics}, "pf500_trung_binh": {m: 0.0 for m in metrics}}
    for run, ref in th.items():
        for m in metrics:
            d = abs(c_full[run][m] * (100.0 if m in PP else 1.0) - ref[m])
            sai["tong_hop_36_run"][m] = max(sai["tong_hop_36_run"][m], d)
            if not d <= tb["tong_hop_36_run"][m] + eps:
                lech.append((run, m, d))
    for key, ref in pfr.items():
        for m in metrics:
            d = abs(diem[key][pf][m] * (100.0 if m in PP else 1.0) - ref[m])
            sai["pf500_trung_binh"][m] = max(sai["pf500_trung_binh"][m], d)
            if not d <= tb["pf500_trung_binh"][m] + eps:
                lech.append((f"{key[0]}-{key[1]}-{pf}_tb", m, d))
    return lech, sai


def _rel(goc: Path, p: Path) -> str:
    return Path(os.path.abspath(p)).relative_to(Path(os.path.abspath(goc))).as_posix()


def kiem_nguon(goc: Path, gt: dict) -> dict:
    """Kiem toan bo dau vao theo giao thuc; tra ve manifest nguon (chua ghi). Loi -> LoiDauVao."""
    ng = gt["nguon"]
    videos = gt["video"]
    metrics = gt["metric"]["danh_sach"]
    file_vao: list[dict] = []

    def them(rel: str, vai_tro: str, sha_mong: str | None = None) -> Path:
        p = duong_doc(goc, rel)
        h = sha256_file(p)
        if sha_mong is not None and h != sha_mong:
            raise LoiDauVao(f"sai hash {rel}: {h} != {sha_mong}")
        file_vao.append({"rel": rel, "vai_tro": vai_tro, "sha256": h})
        return p

    p_khoa = them(ng["khoa"]["rel"], "khoa", ng["khoa"]["sha256"])
    p_gt54 = them(ng["giao_thuc_r54"]["rel"], "giao_thuc_r54", ng["giao_thuc_r54"]["sha256"])
    them(ng["hop_dong_r55"]["rel"], "hop_dong_r55", ng["hop_dong_r55"]["sha256"])
    p_man = them(ng["manifest_video"]["rel"], "manifest_video", ng["manifest_video"]["sha256"])
    p_n56 = them(ng["nguon_r56"]["rel"], "nguon_r56", ng["nguon_r56"]["sha256"])
    them(ng["bang_r56"]["tong_hop"]["rel"], "bang_r56_tong_hop", ng["bang_r56"]["tong_hop"]["sha256"])
    them(ng["bang_r56"]["pf500"]["rel"], "bang_r56_pf500", ng["bang_r56"]["pf500"]["sha256"])
    doc_bang_r56(goc, gt)   # schema / khoa duy nhat / so huu han; dung tap run va nhom

    man = doc_json(p_man)
    if man.get("videos") != videos:
        raise LoiDauVao("thu tu / tap video giao thuc khac manifest da khoa")
    runs = danh_sach_run(gt)
    ten_runs = [r["run"] for r in runs]
    gt54 = doc_json(p_gt54)
    if sorted(gt54["ma_tran"]["danh_sach"]) != sorted(ten_runs) or gt54["tap_du_lieu"]["so_video"] != len(videos):
        raise LoiDauVao("giao thuc R54: ma tran / so video khac giao thuc R57")
    khoa = doc_json(p_khoa)
    sha_khoa = ng["khoa"]["sha256"]
    for k, mong in (("giao_thuc_sha256", ng["giao_thuc_r54"]["sha256"]), ("hop_dong_sha256", ng["hop_dong_r55"]["sha256"])):
        if khoa.get(k) != mong:
            raise LoiDauVao(f"khoa.json {k} khong khop")
    if sorted(khoa.get("ma_tran", [])) != sorted(ten_runs):
        raise LoiDauVao("khoa.json ma_tran khac 36 run")
    if not any(f.get("rel") == ng["manifest_video"]["rel"] and f.get("sha256") == ng["manifest_video"]["sha256"]
               for f in khoa.get("file", [])):
        raise LoiDauVao("manifest video khong nam trong khoa voi dung hash")

    phien_tt = ng["phien_thuc_thi"]
    p_phien = them(phien_tt + "/phien.json", "phien_thuc_thi")
    ph = doc_json(p_phien)
    for k, mong in (("khoa", ng["khoa"]["ten"]), ("khoa_sha256", sha_khoa),
                    ("hop_dong_sha256", ng["hop_dong_r55"]["sha256"]), ("giao_thuc_sha256", ng["giao_thuc_r54"]["sha256"]),
                    ("so_video", len(videos))):
        if ph.get(k) != mong:
            raise LoiDauVao(f"phien.json {k} khong khop")
    if sorted(ph.get("ma_tran", [])) != sorted(ten_runs) or len(ph["ma_tran"]) != len(ten_runs):
        raise LoiDauVao("phien.json ma_tran khac 36 run")

    n56 = doc_json(p_n56)
    if n56.get("phien") != Path(phien_tt).name or n56.get("so_run") != gt["so_run"]:
        raise LoiDauVao("nguon R56: phien / so_run khong khop")
    sm = n56.get("summary", {})
    if set(sm) != set(ten_runs) or len(sm) != len(ten_runs):
        raise LoiDauVao(f"nguon R56: tap run sai (thieu {sorted(set(ten_runs) - set(sm))[:3]}, "
                        f"thua {sorted(set(sm) - set(ten_runs))[:3]})")

    goc_tt = Path(os.path.abspath(goc / phien_tt))
    out_runs = {}
    lan_da_dung = set()
    for r in runs:
        run = r["run"]
        rel_sum = sm[run]["file"]
        p_sum = duong_doc(goc, rel_sum)
        p_sum_a = Path(os.path.abspath(p_sum))
        tev = p_sum_a.parent
        lan = tev.parents[1]
        if (p_sum_a.name != "pedestrian_summary.txt" or tev.name != run or tev.parent.name != "trackeval"
                or lan.parent.name != run or lan.parent.parent.name != "cham" or lan.parent.parent.parent != goc_tt):
            raise LoiDauVao(f"{run}: duong summary sai cau truc {rel_sum}")
        if lan in lan_da_dung:
            raise LoiDauVao(f"{run}: lan cham trung")
        lan_da_dung.add(lan)
        h_sum = sha256_file(p_sum)
        if h_sum != sm[run]["sha256"]:
            raise LoiDauVao(f"{run}: sai hash pedestrian_summary.txt")
        file_vao.append({"rel": rel_sum, "vai_tro": f"summary:{run}", "sha256": h_sum})
        rel_lan = _rel(goc, lan)
        p_mk = them(rel_lan + "/HOAN_TAT.json", f"marker:{run}")
        h_mk = file_vao[-1]["sha256"]
        mk = doc_json(p_mk)
        rb = mk.get("rang_buoc", {})
        for k, mong in (("loai", "cham"), ("khoa", ng["khoa"]["ten"]), ("khoa_sha256", sha_khoa),
                        ("hop_dong_sha256", ng["hop_dong_r55"]["sha256"]),
                        ("giao_thuc_sha256", ng["giao_thuc_r54"]["sha256"]), ("run", run)):
            if rb.get(k) != mong:
                raise LoiDauVao(f"{run}: marker rang_buoc.{k} khong khop")
        si = rb.get("seq_info", {})
        if set(si) != set(videos) or len(si) != len(videos):
            raise LoiDauVao(f"{run}: marker seq_info sai tap video")
        mc = rb.get("manifest_cache_sha256")
        if not (isinstance(mc, str) and re.fullmatch(r"[0-9a-f]{64}", mc)):
            raise LoiDauVao(f"{run}: marker thieu manifest_cache_sha256")
        mf = mk.get("file", {})
        if mf.get(f"trackeval/{run}/pedestrian_summary.txt") != h_sum:
            raise LoiDauVao(f"{run}: marker khong khop hash summary")
        fm = mk.get("file_metric", {})
        if sorted(fm) != sorted(ng["file_metric"]):
            raise LoiDauVao(f"{run}: marker file_metric sai schema")
        files = {}
        for f in ng["file_metric"]:
            rel_f = _rel(goc, tev / f)
            p_f = duong_doc(goc, rel_f)
            h = sha256_file(p_f)
            if h != fm[f] or h != mf.get(f"trackeval/{run}/{f}"):
                raise LoiDauVao(f"{run}: sai hash {f} so voi marker")
            file_vao.append({"rel": rel_f, "vai_tro": f"metric:{run}:{f}", "sha256": h})
            files[f] = {"rel": rel_f, "sha256": h}
        doc_raw(duong_doc(goc, files["raw_fields.json"]["rel"]), videos, run)
        doc_per_video_csv(duong_doc(goc, files["per_video_metrics.csv"]["rel"]), videos, run, metrics)
        comb = doc_combined_csv(duong_doc(goc, files["combined_metrics.csv"]["rel"]), run, metrics)
        mkc = mk.get("combined", {})
        for m in metrics:
            if abs(_so(mkc.get(m), f"{run}/marker.combined.{m}") - comb[m]) > 1e-12:
                raise LoiDauVao(f"{run}: combined_metrics.csv khac marker.combined ({m})")
        out_runs[run] = {**r, "lan": rel_lan, "marker": {"rel": rel_lan + "/HOAN_TAT.json", "sha256": h_mk},
                         "summary": {"rel": rel_sum, "sha256": h_sum}, "file": files, "manifest_cache_sha256": mc,
                         "combined_csv": comb}

    cache = {}
    for r in out_runs.values():
        cache.setdefault(r["detector"], set()).add(r["manifest_cache_sha256"])
    if any(len(s) != 1 for s in cache.values()) or len({next(iter(s)) for s in cache.values()}) != len(cache):
        raise LoiDauVao(f"lien ket cache sai: moi detector phai co dung 1 manifest cache rieng ({ {k: sorted(v) for k, v in cache.items()} })")
    for det in gt["detector"]:
        for dk in gt["dieu_kien"]:
            s = sorted(r["seed"] for r in out_runs.values()
                       if r["detector"] == det and r["dieu_kien"] == dk and r["bo_loc"] == gt["pf"]["bo_loc"])
            if s != list(gt["pf"]["seeds"]):
                raise LoiDauVao(f"{det}-{dk}: seed PF {s}")
    rels = [f["rel"] for f in file_vao]
    if len(rels) != len(set(rels)):
        raise LoiDauVao("file dau vao trung")
    return {"file_vao": file_vao, "run": out_runs,
            "cache_theo_detector": {k: next(iter(v)) for k, v in cache.items()}}


# ------------------------------------------------------------------------------------------------------------- chuan-bi
def tao_ten_phien() -> str:
    return "phien_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"


def nhom_video_bytes(gt: dict) -> bytes:
    rows = []
    vt = {x: i for i, x in enumerate(gt["video"])}
    for i, n in enumerate(gt["thiet_ke"]["nhom"]["nhom"]):
        for v in n["video"]:
            rows.append([i, n["id"], v, vt[v], len(n["video"])])
    return csv_bytes(["vi_tri_nhom", "nhom", "video", "vi_tri_video", "so_video_nhom"], rows)


def sha_ma() -> dict:
    return {k: sha256_file(p) for k, p in FILE_MA.items()}


def chuan_bi(goc: Path, gt_rel: str, vung_rel: str = VUNG_REL) -> Path:
    goc = Path(os.path.abspath(goc))
    p_gt = duong_doc(goc, gt_rel)
    gt_b = p_gt.read_bytes()
    gt = doc_json(p_gt)
    kiem_giao_thuc(gt)
    ng = kiem_nguon(goc, gt)
    vung = kiem_vung_ghi(goc, goc / vung_rel, vung_rel)
    vung.mkdir(parents=True, exist_ok=True)
    kiem_vung_ghi(goc, vung, vung_rel)
    phien = kiem_vung_ghi(goc, vung / tao_ten_phien(), vung_rel)
    phien.mkdir(exist_ok=False)
    h_gt = ghi_moi(goc, phien / "giao_thuc.json", gt_b, vung_rel)
    h_nhom = ghi_moi(goc, phien / "nhom_video.csv", nhom_video_bytes(gt), vung_rel)
    nguon = {"buoc": "chuan-bi", "phien": phien.name, "tao_luc": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "giao_thuc_rel": gt_rel, "giao_thuc_sha256": h_gt, "nhom_video_sha256": h_nhom, "ma_sha256": sha_ma(),
             "so_run": len(ng["run"]), "so_video": len(gt["video"]), "so_file_vao": len(ng["file_vao"]),
             "cache_theo_detector": ng["cache_theo_detector"], "file_vao": ng["file_vao"], "run": ng["run"],
             "ghi_chu": "chi hash file nho duoc nguon tro toi (marker, metric, summary, khoa/giao thuc/hop dong/manifest, bang "
                        "R56); khong bam lai anh/video/NPZ/MOT/cache; khong phai nghiem thu lai toan bo R56"}
    ghi_moi(goc, phien / "nguon.json", json_bytes(nguon), vung_rel)
    return phien


# ------------------------------------------------------------------------------------------------------------ phan-tich
def rut_chi_so(B: int, K: int, seed: int) -> np.ndarray:
    """Cung day so voi boot_stats.paired_bootstrap / boot_seed.paired_bootstrap_seed."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, K, size=K) for _ in range(B)]).astype(np.int64)


def mo_rong_nhom(gidx: np.ndarray, tv: list[np.ndarray]) -> np.ndarray:
    """Chi so nhom -> chi so video: moi lan nhom xuat hien dua TOAN BO video cua nhom vao mau."""
    return np.concatenate([tv[g] for g in gidx])


def chi_so_csv(idx: np.ndarray, cot: str) -> bytes:
    return csv_bytes(["mau"] + [f"{cot}{j}" for j in range(idx.shape[1])], [[i, *row.tolist()] for i, row in enumerate(idx)])


def doc_chi_so_csv(b: bytes) -> np.ndarray:
    rows = list(csv.reader(io.StringIO(b.decode("utf-8"))))
    arr = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=np.int64)
    if [int(r[0]) for r in rows[1:]] != list(range(len(arr))):
        raise LoiDauVao("chi so CSV sai thu tu mau")
    return arr


def gia_tri_nhom(c: dict, gt: dict, det: str, dk: str, metrics: list[str]) -> dict:
    """c: {run: dict COMBINED}. Bo loc don -> COMBINED cua run; PF -> trung binh COMBINED tung seed (boot_seed._tb)."""
    out = {}
    pf = gt["pf"]["bo_loc"]
    for bl in gt["bo_loc"]:
        if bl == pf:
            out[bl] = bse._tb([c[ten_run(det, dk, bl, s)] for s in gt["pf"]["seeds"]], metrics)
        else:
            out[bl] = {m: c[ten_run(det, dk, bl)][m] for m in metrics}
    return out


def bootstrap(stacks: dict, mau: list[np.ndarray], gt: dict, metrics: list[str]) -> dict:
    """Moi mau: COMBINED rieng tung run voi CUNG chi so video -> hieu so co huong theo nhom detector x dieu kien.
    Tra ve {(det, dk, a, b, m): mang B}."""
    B = len(mau)
    out = {(det, dk, a, b, m): np.empty(B) for det in gt["detector"] for dk in gt["dieu_kien"]
           for a, b in gt["hieu_so"] for m in metrics}
    for i, idx in enumerate(mau):
        c = {run: bs.combine_stacked({f: s[f][idx] for f in s}) for run, s in stacks.items()}
        for det in gt["detector"]:
            for dk in gt["dieu_kien"]:
                g = gia_tri_nhom(c, gt, det, dk, metrics)
                for a, b in gt["hieu_so"]:
                    for m in metrics:
                        out[(det, dk, a, b, m)][i] = g[a][m] - g[b][m]
    return out


def bang_ci(thiet_ke: str, boots: dict, diem: dict, gt: dict, metrics: list[str], so_don_vi: int) -> list[list]:
    rows = []
    pv = gt["bootstrap"]["phan_vi"]
    for det in gt["detector"]:
        for dk in gt["dieu_kien"]:
            for a, b in gt["hieu_so"]:
                for m in metrics:
                    arr = boots[(det, dk, a, b, m)]
                    if not np.all(np.isfinite(arr)):
                        raise LoiDauVao(f"bootstrap khong huu han {det}-{dk}-{a}-{b}-{m}")
                    lo, hi = np.percentile(arr, pv, method=gt["bootstrap"]["method"])
                    va, vb = diem[(det, dk)][a][m], diem[(det, dk)][b][m]
                    k = 100.0 if m in PP else 1.0
                    rows.append([thiet_ke, det, dk, f"{a}-{b}", a, b, m, repr(va), repr(vb), repr(va - vb), repr(float(lo)),
                                 repr(float(hi)), "pp" if m in PP else "so_dem", f"{(va - vb) * k:.4f}",
                                 f"{lo * k:.4f}", f"{hi * k:.4f}", len(arr), so_don_vi])
    return rows


HEADER_CI = ["thiet_ke", "detector", "dieu_kien", "hieu_so", "a", "b", "metric", "gia_tri_a", "gia_tri_b", "chenh",
             "ci_thap", "ci_cao", "don_vi_hien_thi", "chenh_hien_thi", "ci_thap_hien_thi", "ci_cao_hien_thi", "B", "so_don_vi_lay_mau"]


def kiem_phien(goc: Path, phien_ten: str, vung_rel: str) -> Path:
    if not TEN_PHIEN.match(phien_ten or ""):
        raise LoiDauVao(f"ten phien khong hop le: {phien_ten!r}")
    phien = kiem_vung_ghi(goc, goc / vung_rel / phien_ten, vung_rel)
    if not phien.is_dir():
        raise LoiDauVao(f"khong co phien {phien_ten}")
    co = sorted(x.name for x in phien.iterdir())
    if co != sorted(DAU_RA_CHUAN_BI):
        raise LoiDauVao(f"phien da co dau ra / file la, khong ghi de: {co}")
    return phien


def phan_tich(goc: Path, phien_ten: str, vung_rel: str = VUNG_REL, kiem_boot_seed: bool = True) -> dict:
    goc = Path(os.path.abspath(goc))
    phien = kiem_phien(goc, phien_ten, vung_rel)
    nguon = doc_json(phien / "nguon.json")
    gt_b = (phien / "giao_thuc.json").read_bytes()
    if sha256_bytes(gt_b) != nguon["giao_thuc_sha256"]:
        raise LoiDauVao("giao_thuc.json cua phien bi doi")
    gt = json.loads(gt_b.decode("utf-8"), object_pairs_hook=_json_khong_trung)
    kiem_giao_thuc(gt)
    if sha256_file(duong_doc(goc, nguon["giao_thuc_rel"])) != nguon["giao_thuc_sha256"]:
        raise LoiDauVao("giao thuc trong configs da doi sau chuan-bi -> tao phien moi")
    if sha256_bytes((phien / "nhom_video.csv").read_bytes()) != nguon["nhom_video_sha256"] or \
            (phien / "nhom_video.csv").read_bytes() != nhom_video_bytes(gt):
        raise LoiDauVao("nhom_video.csv cua phien bi doi")
    ma = sha_ma()
    if ma != nguon["ma_sha256"]:
        raise LoiDauVao(f"ma da doi sau chuan-bi -> tao phien moi: {ma}")

    def bam_vao() -> list[str]:
        lech = []
        for f in nguon["file_vao"]:
            if sha256_file(duong_doc(goc, f["rel"])) != f["sha256"]:
                lech.append(f["rel"])
        return lech

    lech_truoc = bam_vao()
    if lech_truoc:
        raise LoiDauVao(f"hash dau vao lech truoc phan tich: {lech_truoc[:5]}")

    videos = gt["video"]
    metrics = gt["metric"]["danh_sach"]
    runs = nguon["run"]
    if sorted(runs) != sorted(r["run"] for r in danh_sach_run(gt)):
        raise LoiDauVao("nguon.json: tap run khac giao thuc")
    raws = {run: doc_raw(duong_doc(goc, r["file"]["raw_fields.json"]["rel"]), videos, run) for run, r in runs.items()}
    stacks = {run: bs._stack(raws[run], videos) for run in runs}
    tol = gt["dung_sai"]

    # 1) tai to hop toan bo 40 video vs combined_metrics.csv + per_video_metrics.csv
    c_full = {run: bs.combine_stacked(stacks[run]) for run in runs}
    sai = {m: 0.0 for m in metrics}
    sai_video = {m: 0.0 for m in metrics}
    diem_rows = []
    for run, r in runs.items():
        comb = doc_combined_csv(duong_doc(goc, r["file"]["combined_metrics.csv"]["rel"]), run, metrics)
        pv = doc_per_video_csv(duong_doc(goc, r["file"]["per_video_metrics.csv"]["rel"]), videos, run, metrics)
        for m in metrics:
            sai[m] = max(sai[m], abs(c_full[run][m] - comb[m]))
        for v in videos:
            cv1 = bs.combine(raws[run], [v])
            for m in metrics:
                sai_video[m] = max(sai_video[m], abs(cv1[m] - pv[v][m]))
        diem_rows.append([r["detector"], r["dieu_kien"], r["bo_loc"], "" if r["seed"] is None else r["seed"], run,
                          *[repr(c_full[run][m]) for m in metrics], *[repr(comb[m]) for m in metrics],
                          *[repr(c_full[run][m] - comb[m]) for m in metrics]])
    t = tol["tai_to_hop_vs_combined_metrics"]
    dat_to_hop = all(sai[m] <= t[m] for m in metrics) and all(sai_video[m] <= t[m] for m in metrics)
    if not dat_to_hop:
        raise LoiDauVao(f"tai to hop khong khop combined/per_video: {sai} / {sai_video}")
    diem = {(det, dk): gia_tri_nhom(c_full, gt, det, dk, metrics) for det in gt["detector"] for dk in gt["dieu_kien"]}
    for (det, dk), g in diem.items():
        pf = gt["pf"]["bo_loc"]
        diem_rows.append([det, dk, pf, "tb", "", *[repr(g[pf][m]) for m in metrics], *[""] * (2 * len(metrics))])

    # 2) bang R56: chi doi chieu theo do lam tron
    tb = tol["bang_r56"]
    th, pfr = doc_bang_r56(goc, gt)
    if set(th) != set(runs) or set(pfr) != set(diem):
        raise LoiDauVao("bang R56 sai tap run / nhom")
    lech56, sai56 = doi_chieu_bang_r56(c_full, diem, th, pfr, gt)
    if lech56:
        raise LoiDauVao(f"lech bang R56 vuot do lam tron: {lech56[:5]}")

    # 3) chi so lay mau -> CSV trong bo nho -> doc lai
    B, seed = gt["bootstrap"]["B"], gt["bootstrap"]["rng_seed"]
    tv = thanh_vien_nhom(gt)
    idx_v = rut_chi_so(B, len(videos), seed)
    idx_n = rut_chi_so(B, len(tv), seed)
    b_v, b_n = chi_so_csv(idx_v, "v"), chi_so_csv(idx_n, "n")
    idx_v2, idx_n2 = doc_chi_so_csv(b_v), doc_chi_so_csv(b_n)
    if not (np.array_equal(idx_v, idx_v2) and np.array_equal(idx_n, idx_n2)):
        raise LoiDauVao("chi so doc lai khac")
    mau_v = list(idx_v)
    mau_n = [mo_rong_nhom(g, tv) for g in idx_n]

    # kiem nhom: cap luon di cung nhau, dung so lan
    nhieu = [i for i, x in enumerate(tv) if len(x) > 1]
    for g, vv in zip(idx_n, mau_n):
        if len(vv) != sum(len(tv[x]) for x in g):
            raise LoiDauVao("mo rong nhom sai do dai")
        dem = np.bincount(vv, minlength=len(videos))
        dem_g = np.bincount(g, minlength=len(tv))
        for gi, x in enumerate(tv):
            if not np.all(dem[x] == dem_g[gi]):
                raise LoiDauVao("video trong nhom khong di cung nhau")
    do_dai = np.array([len(x) for x in mau_n])

    # 4) bootstrap hai thiet ke + tai tinh tu chi so da doc lai
    boot_v = bootstrap(stacks, mau_v, gt, metrics)
    boot_n = bootstrap(stacks, mau_n, gt, metrics)
    boot_v2 = bootstrap(stacks, list(idx_v2), gt, metrics)
    boot_n2 = bootstrap(stacks, [mo_rong_nhom(g, tv) for g in idx_n2], gt, metrics)
    tai_tinh = all(np.array_equal(boot_v[k], boot_v2[k]) for k in boot_v) and \
        all(np.array_equal(boot_n[k], boot_n2[k]) for k in boot_n)
    if not tai_tinh:
        raise LoiDauVao("tai tinh tu chi so da luu khong trung")

    # 5) nhanh video vs boot_seed.paired_bootstrap_seed (helper hien co)
    doi_bse = {"so_so_sanh": 0, "trung": 0, "lech": []}
    if kiem_boot_seed:
        def raws_bl(det, dk, bl):
            if bl == gt["pf"]["bo_loc"]:
                return [raws[ten_run(det, dk, bl, s)] for s in gt["pf"]["seeds"]]
            return [raws[ten_run(det, dk, bl)]]
        for det in gt["detector"]:
            for dk in gt["dieu_kien"]:
                for a, b in gt["hieu_so"]:
                    ref = bse.paired_bootstrap_seed(raws_bl(det, dk, a), raws_bl(det, dk, b), videos, n_boot=B, seed=seed,
                                                    metrics=metrics, tra_mau=True)
                    for m in metrics:
                        doi_bse["so_so_sanh"] += 1
                        arr = boot_v[(det, dk, a, b, m)]
                        lo, hi = np.percentile(arr, gt["bootstrap"]["phan_vi"], method=gt["bootstrap"]["method"])
                        d = diem[(det, dk)][a][m] - diem[(det, dk)][b][m]
                        ok = (np.array_equal(ref["_mau"][m], arr) and ref[m]["diff"] == d
                              and ref[m]["ci_low"] == float(lo) and ref[m]["ci_high"] == float(hi))
                        if ok:
                            doi_bse["trung"] += 1
                        else:
                            doi_bse["lech"].append(f"{det}-{dk}-{a}-{b}-{m}")
        if doi_bse["lech"]:
            raise LoiDauVao(f"nhanh video khac boot_seed.paired_bootstrap_seed: {doi_bse['lech'][:5]}")

    rows_v = bang_ci("video", boot_v, diem, gt, metrics, len(videos))
    rows_n = bang_ci("nhom", boot_n, diem, gt, metrics, len(tv))
    n_mong = len(gt["detector"]) * len(gt["dieu_kien"]) * len(gt["hieu_so"]) * len(metrics)
    if len(rows_v) != n_mong or len(rows_n) != n_mong:
        raise LoiDauVao(f"so dong CI {len(rows_v)} / {len(rows_n)} != {n_mong}")

    # so sanh dien giai giua hai thiet ke (chi bao, khong chon)
    khac = []
    ti_le_rong = []
    for rv, rn in zip(rows_v, rows_n):
        lo_v, hi_v, lo_n, hi_n = float(rv[10]), float(rv[11]), float(rn[10]), float(rn[11])
        loai = lambda lo, hi: "chua_0" if lo <= 0 <= hi else ("duong" if lo > 0 else "am")  # noqa: E731
        if loai(lo_v, hi_v) != loai(lo_n, hi_n):
            khac.append({"dong": "-".join(rv[1:4]) + "-" + rv[6], "video": loai(lo_v, hi_v), "nhom": loai(lo_n, hi_n)})
        if hi_v > lo_v:
            ti_le_rong.append((hi_n - lo_n) / (hi_v - lo_v))

    lech_sau = bam_vao()
    if lech_sau:
        raise LoiDauVao(f"hash dau vao lech sau phan tich: {lech_sau[:5]}")

    # 6) ghi dau ra (moi kiem da dat)
    hdr_diem = ["detector", "dieu_kien", "bo_loc", "seed", "run", *[f"{m}" for m in metrics],
                *[f"{m}_combined_csv" for m in metrics], *[f"sai_so_{m}" for m in metrics]]
    kc = {"buoc": "phan-tich", "phien": phien.name, "luc": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
          "giao_thuc_sha256": nguon["giao_thuc_sha256"], "ma_sha256": ma,
          "dau_vao": {"so_run": len(runs), "so_video": len(videos), "so_file_vao": len(nguon["file_vao"]),
                      "hash_truoc_lech": lech_truoc, "hash_sau_lech": lech_sau,
                      "seed_pf_moi_nhom": {f"{d}-{k}": list(gt["pf"]["seeds"]) for d, k in diem}},
          "tai_to_hop": {"sai_so_max_combined": sai, "sai_so_max_per_video": sai_video, "dung_sai": t, "dat": dat_to_hop},
          "bang_r56": {"dat": True, "dung_sai_ap_dung": tb, "sai_so_max": sai56,
                       "ghi_chu": "chi doi chieu lam tron, khong la nguon bootstrap; dung sai rieng tung bang, "
                                  "khong nang ngam (sua finding R57 review 1)"},
          "chi_so": {"video": {"B": B, "K": len(videos), "rng": f"default_rng({seed})"},
                     "nhom": {"B": B, "K": len(tv), "rng": f"default_rng({seed})", "nhom_nhieu_video": [
                         gt["thiet_ke"]["nhom"]["nhom"][i]["id"] for i in nhieu],
                              "so_video_moi_mau": {"min": int(do_dai.min()), "max": int(do_dai.max()),
                                                   "tb": float(do_dai.mean())}},
                     "doc_lai_trung": True, "tai_tinh_trung": tai_tinh, "cap_di_cung": True},
          "doi_chieu_boot_seed": {**doi_bse, "chay": kiem_boot_seed},
          "so_dong_ci": {"video": len(rows_v), "nhom": len(rows_n), "mong": n_mong},
          "khac_dien_giai_hai_thiet_ke": khac,
          "ti_le_do_rong_ci_nhom_tren_video": {"min": min(ti_le_rong), "max": max(ti_le_rong),
                                              "tb": float(np.mean(ti_le_rong))} if ti_le_rong else None,
          "ghi_chu": "CI phan vi 95% rieng le, tham do, chua dieu chinh da so sanh; gia dinh video (hoac nhom) doc lap; "
                     "CI chua 0 khong co nghia tuong duong"}
    ghi = [("chi_so_video.csv", b_v), ("chi_so_nhom.csv", b_n),
           ("diem_uoc_luong.csv", csv_bytes(hdr_diem, diem_rows)),
           ("doi_chieu_video.csv", csv_bytes(HEADER_CI, rows_v)), ("doi_chieu_nhom.csv", csv_bytes(HEADER_CI, rows_n)),
           ("kiem_chung.json", json_bytes(kc))]
    hs = {}
    for ten, data in ghi:
        hs[ten] = ghi_moi(goc, phien / ten, data, vung_rel)
    for ten in DAU_RA_CHUAN_BI:
        hs[ten] = sha256_file(phien / ten)
    ghi_moi(goc, phien / "SHA256SUMS", "".join(f"{hs[k]}  {k}\n" for k in sorted(hs)).encode("utf-8"), vung_rel)
    return {"phien": phien, "kiem_chung": kc, "boot_video": boot_v, "boot_nhom": boot_n, "diem": diem,
            "rows_video": rows_v, "rows_nhom": rows_n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R57A bootstrap ghep cap test40 (chi phan tich hau nghiem)")
    ap.add_argument("--buoc", required=True, choices=["chuan-bi", "phan-tich"])
    ap.add_argument("--giao-thuc")
    ap.add_argument("--phien")
    ns = ap.parse_args(argv)
    try:
        if ns.buoc == "chuan-bi":
            if not ns.giao_thuc or ns.phien:
                raise LoiDauVao("chuan-bi can --giao-thuc, khong nhan --phien")
            phien = chuan_bi(ROOT, ns.giao_thuc)
            print(f"PHIEN {phien.name}")
        else:
            if not ns.phien or ns.giao_thuc:
                raise LoiDauVao("phan-tich can --phien, khong nhan --giao-thuc")
            kq = phan_tich(ROOT, ns.phien)
            kc = kq["kiem_chung"]
            print(f"PHIEN {kq['phien'].name} dong CI video {kc['so_dong_ci']['video']} nhom {kc['so_dong_ci']['nhom']}; "
                  f"tai to hop {kc['tai_to_hop']['sai_so_max_combined']}; boot_seed {kc['doi_chieu_boot_seed']['trung']}/"
                  f"{kc['doi_chieu_boot_seed']['so_so_sanh']}; khac dien giai {len(kc['khac_dien_giai_hai_thiet_ke'])}")
    except LoiDauVao as e:
        print(f"LOI: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
