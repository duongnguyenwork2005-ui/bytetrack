"""
run_fingerprint.py -- (sau review Codex) DAU VAN TAY cua mot lan chay de quyet dinh co duoc TAI SU DUNG
ket qua danh gia da co hay khong.

VAN DE DUOC SUA: `run_experiment.stage_eval` truoc day bo qua TrackEval chi vi ba file CSV/JSON da ton tai
-> neu tracking duoc chay lai (mã khac, cau hinh khac, cache khac) thi metric CU van duoc giu, tao ra bang
ket qua "tracking moi + metric cu" ma khong co canh bao nao.

DAU VAN TAY gom (tat ca deu anh huong ket qua danh gia):
    tracking  : sha256 cua TUNG file `data/<video>.txt` cua lan chay (theo dung danh sach sequence)
    config    : tracker_cfg + overrides + conf_min + classes + max_det + elapsed_mode + fps + seed
    cache     : meta.json cua cache + sha256 cua tung <video>.npz duoc dung
    sequences : danh sach video (co thu tu)
    code      : git commit + sha256 cua cac module anh huong ket qua
                (run_tracking, tracker_ctrv, sort_tracker, ekf_ctrv, ukf_ctrv, pf_ctrv, lost_recovery,
                 lost_association, time_units, det_cache, occlusion_eval, boot_stats, run_trackeval,
                 trackeval_ext)
    evaluator : phien ban trackeval + lop dataset dung de danh gia
    gt        : (review ca93c1e, P1) NOI DUNG that cua dau vao danh gia -- `gt/gt.txt` va `seqinfo.ini`
                cua TUNG sequence (seqinfo chua `seqLength` = do dai timeline), cong seqmap KHI
                duong chay thuc su doc no. Truoc day thieu han => doi GT tai cung duong dan van
                bi cham bang metric CU (xem `handoff/logs/review7/repro_F1_truoc_khi_sua.log`).
Ghi ra `eval_meta.json` (canh ket qua TrackEval) va `occl_meta.json` (canh ket qua occlusion_eval).
`compare(old, new)` tra ve danh sach truong LECH -> runner in ly do chay lai.

QUY UOC HASH (mot quy tac duy nhat, cong bo o day va o `HASH_CONVENTION`):
    - file VAN BAN theo `TEXT_SUFFIXES` (.txt / .ini / .csv): sha256 cua noi dung da CHUAN HOA
      xuong dong ve LF. Nho vay hash GT trung khop quy uoc cua ban khoa (`src/make_lock.py`)
      nen doi chieu duoc truc tiep, va phan NAY cua dau van tay on dinh giua Windows va Nano.
    - MOI file khac (.npz, VA CA .py/.json/.yaml): sha256 cua BYTE NGUYEN BAN.

PHAM VI ON DINH GIUA MAY (review b623465 luu y dung): chi cac duoi trong `TEXT_SUFFIXES` moi duoc
chuan hoa. `code_fingerprint()` hash module `.py` theo BYTE THO, ma cay nguon hien co xuong dong
LAN LON (vd `run_fingerprint.py` toan CRLF, `bytetrack_np.py` toan LF, do `core.autocrlf=true` +
`* text=auto`). Vi vay truong `code` CO THE lech giua Windows va Jetson Nano du cung commit.
=> KHONG duoc phat bieu "dau van tay khong doi giua hai may" mot cach chung chung.
Xu ly viec nay thuoc moc Nano (xem `handoff/NANO_STATUS.md`); doi `TEXT_SUFFIXES` bay gio se lam
12 `eval_meta.json` vua duoc review thanh cu, nen hoan lai den khi bat dau chay tren thiet bi.
File THIEU tra ve sentinel `MISSING` -- KHAC voi file RONG HOP LE (rong co hash sha256 that su cua
chuoi rong), nen hai tinh huong nay khong bao gio bi lan.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

#: Module anh huong truc tiep ket qua tracking/danh gia (doi -> phai danh gia lai)
CODE_FILES = ("run_tracking.py", "run_experiment.py", "tracker_ctrv.py", "sort_tracker.py", "ekf_ctrv.py",
              "ekf_ctrv_clamped.py", "ukf_ctrv.py", "pf_ctrv.py", "lost_recovery.py", "lost_association.py",
              "time_units.py", "det_cache.py", "occlusion_eval.py", "boot_stats.py", "run_trackeval.py",
              "trackeval_ext.py", "assoc_log.py",
              # backend ByteTrack (ultralytics da cai / ban vendor khong phu thuoc torch)
              "ul_backend.py", "bytetrack_np.py",
              # review ca93c1e: `run_state.py` quyet dinh chinh sach tai su dung cho CA CLI va runner
              # nhung truoc day khong nam trong dau van tay -> sua module nay khong lam mat hieu luc.
              "run_state.py")

#: Cong bo quy uoc hash de tai lieu/ban khoa doi chieu duoc (xem docstring dau file)
HASH_CONVENTION = ("van ban (.txt/.ini/.csv): sha256 cua noi dung chuan hoa xuong dong ve LF; "
                   "nhi phan (.npz): sha256 byte nguyen ban; file thieu: sentinel 'MISSING'")

#: Duoi file duoc coi la VAN BAN -> hash sau khi chuan hoa xuong dong
TEXT_SUFFIXES = (".txt", ".ini", ".csv")


def sha256_file(p: Path, missing: str = "MISSING") -> str:
    """Hash mot file theo QUY UOC DUY NHAT o docstring dau file.

    File THIEU -> `missing` (mac dinh 'MISSING'). File RONG HOP LE -> hash that cua chuoi rong
    (`e3b0c442...`), KHAC sentinel => phan biet duoc "GT rong" voi "GT bi xoa".
    """
    if not p.exists():
        return missing
    if p.suffix.lower() in TEXT_SUFFIXES:
        return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_fingerprint() -> dict:
    """Dau van tay cua MA sinh ra ket qua.

    Gom ca DANH TINH BACKEND: doi `BYTETRACK_BACKEND` (ultralytics <-> np), doi ban vendor, hay doi
    phien ban numpy/scipy/lap deu lam fingerprint doi => runner KHONG duoc tai su dung ket qua cu.
    """
    import ul_backend                                       # noqa: PLC0415

    src = Path(__file__).resolve().parent
    return {"git_commit": _git_commit(),
            "modules": {f: sha256_file(src / f)[:16] for f in CODE_FILES},
            "backend": ul_backend.fingerprint()}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def evaluator_fingerprint(dataset_cls: str) -> dict:
    try:
        import trackeval
        ver = getattr(trackeval, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        ver = f"? ({e})"
    return {"trackeval": ver, "dataset_class": dataset_cls}


def gt_paths(split_name: str, video: str) -> tuple[Path, Path]:
    """Hai dau vao danh gia cua mot sequence: `gt/gt.txt` va `seqinfo.ini`."""
    d = config.PROCESSED_DIR / split_name / video
    return d / "gt" / "gt.txt", d / "seqinfo.ini"


def seq_length_of(ini: Path) -> int | None:
    """Doc `seqLength` tu seqinfo.ini (None neu khong doc duoc)."""
    if not ini.exists():
        return None
    import configparser
    cp = configparser.ConfigParser()
    try:
        cp.read(ini)
        return int(cp["Sequence"]["seqLength"])
    except Exception:  # noqa: BLE001
        return None


def gt_fingerprint(split_name: str, videos: list[str], seqmap_used: bool = False) -> dict:
    """Dau van tay cua DAU VAO DANH GIA -- thu quyet dinh metric cung voi tracking.

    Gom: noi dung `gt.txt`, noi dung `seqinfo.ini` va `seqLength` da phan tich cua TUNG sequence.
    `seqLength` duoc ghi TACH RIENG (ngoai hash seqinfo) de khi no doi thi `compare()` chi ra duoc
    ngay, khong chi bao "seqinfo doi".

    `seqmap_used`: duong chay TrackEval chi doc seqmap khi KHONG truyen danh sach video
    (`SEQ_INFO` ghi de). Chi dua seqmap vao dau van tay khi no that su duoc dung.
    """
    gt, ini, lens = {}, {}, {}
    for v in videos:
        g, i = gt_paths(split_name, v)
        gt[v] = sha256_file(g)
        ini[v] = sha256_file(i)
        lens[v] = seq_length_of(i)
    seqmap = config.PROCESSED_DIR / "seqmaps" / f"{split_name}.txt"
    return {"n_videos": len(videos),
            "n_missing_gt": sum(1 for h in gt.values() if h == "MISSING"),
            "n_missing_seqinfo": sum(1 for h in ini.values() if h == "MISSING"),
            "gt_sha256": sha256_obj(gt), "seqinfo_sha256": sha256_obj(ini),
            "seq_lengths": sha256_obj(lens),
            "gt_per_video": {v: h[:16] for v, h in gt.items()},
            "seqmap_used": bool(seqmap_used),
            "seqmap_sha256": (sha256_file(seqmap) if seqmap_used else None)}


class ThieuDauVaoDanhGia(FileNotFoundError):
    """Thieu dau vao BAT BUOC cua buoc danh gia -> phai bao loi, KHONG duoc tra metric cu."""


def check_eval_inputs(split_name: str, videos: list[str]) -> None:
    """Kiem dau vao BAT BUOC TRUOC khi quyet dinh tai su dung metric.

    Thieu `gt.txt` hoac `seqinfo.ini` => nem `ThieuDauVaoDanhGia`. GT RONG van hop le va di tiep:
    "khong co doi tuong nao trong sequence" la mot tinh huong that, khac han "GT bi mat".
    """
    thieu = []
    for v in videos:
        g, i = gt_paths(split_name, v)
        if not g.exists():
            thieu.append(f"{v}: thieu {g}")
        if not i.exists():
            thieu.append(f"{v}: thieu {i}")
    if thieu:
        raise ThieuDauVaoDanhGia(
            f"thieu {len(thieu)} dau vao danh gia bat buoc cua split {split_name!r}:" + "\n  "
            + "\n  ".join(thieu[:20]) + ("\n  ..." if len(thieu) > 20 else ""))


def tracking_fingerprint(tracker_dir: Path, videos: list[str]) -> dict:
    """sha256 cua tung file ket qua tracking + so file thieu."""
    files = {v: sha256_file(tracker_dir / "data" / f"{v}.txt") for v in videos}
    return {"n_videos": len(videos), "n_missing": sum(1 for h in files.values() if h == "MISSING"),
            "sha256": sha256_obj(files), "per_video": {v: h[:16] for v, h in files.items()}}


def cache_fingerprint(cache_id: str, videos: list[str]) -> dict:
    import det_cache
    cdir = det_cache.CACHE_ROOT / cache_id
    try:
        meta = det_cache.load_meta(cdir)
    except FileNotFoundError:
        meta = {"MISSING": str(cdir)}
    npz = {v: sha256_file(cdir / f"{v}.npz")[:16] for v in videos}
    return {"cache_id": cache_id, "meta_sha256": sha256_obj(meta), "npz_sha256": sha256_obj(npz),
            "weights_sha256": meta.get("weights_sha256"), "conf": meta.get("conf"),
            "classes": meta.get("classes"), "max_det": meta.get("max_det")}


def run_fingerprint(exp: dict, r: dict, videos: list[str], cfg, dataset_cls: str = "mot",
                    seqmap_used: bool = False) -> dict:
    """Dau van tay day du cua mot lan chay (dung cho buoc danh gia TrackEval)."""
    tracker_dir = config.PROCESSED_DIR / "trackers" / exp["split_name"] / r["dir"]
    cfg_d = vars(cfg) if not isinstance(cfg, dict) else cfg
    return {
        "dir": r["dir"], "split_name": exp["split_name"],
        "tracking": tracking_fingerprint(tracker_dir, videos),
        "config": sha256_obj({"tracker_cfg": cfg_d, "overrides": r.get("set"), "conf_min": r.get("conf_min"),
                              "classes": exp["classes"], "max_det": exp["max_det"],
                              "elapsed_mode": exp["elapsed_mode"], "fps": exp["fps"], "seed": r.get("seed")}),
        "cache": cache_fingerprint(exp["cache_id"], videos),
        "sequences": sha256_obj(list(videos)),
        # P1: NOI DUNG ground truth + do dai timeline; thieu cai nay thi doi GT khong lam gi ca
        "gt": gt_fingerprint(exp["split_name"], videos, seqmap_used),
        "code": code_fingerprint(),
        "evaluator": evaluator_fingerprint(dataset_cls),
    }


def occl_source_files(occ: dict) -> dict:
    """Cac file annotation che khuat ma `occlusion_eval.py` THUC SU doc.

    Giong het cach `occlusion_eval.main()` phan giai duong dan: co `tag` thi lay
    `<INTERIM>/<tag>_<ten>`, khong thi theo quy uoc `config.part_file(ten, part)`.
    """
    tag, part = occ.get("tag"), occ.get("part", "train")
    names = ["occlusion_segments.csv", "full_occlusion_segments.csv"]
    if occ.get("gaps"):
        names.append("track_gaps.csv")
    out = {}
    for n in names:
        f = (config.INTERIM_DIR / f"{tag}_{n}") if tag else config.part_file(n, part)
        out[n] = sha256_file(Path(f))
    return out


def occl_fingerprint(exp: dict, cond: str, dirs: list[str], videos: list[str], occ: dict) -> dict:
    """Dau van tay cho buoc occlusion_eval cua mot dieu kien (phu thuoc MOI tracker tham gia)."""
    split = exp["split_name"]
    tr = {d: tracking_fingerprint(config.PROCESSED_DIR / "trackers" / split / d, videos)["sha256"] for d in dirs}
    src = occl_source_files(occ)
    return {"cond": cond, "split_name": split, "trackers": list(dirs), "tracking": sha256_obj(tr),
            "sequences": sha256_obj(list(videos)), "settings": sha256_obj(occ),
            # P1: occlusion_eval doc GT cua split VA cac file segment trong data/interim.
            # Doi bat ky nguon nao trong so do deu phai lam mat hieu luc ket qua cu.
            "gt": gt_fingerprint(split, videos), "annotation_source": sha256_obj(src),
            "annotation_files": {k: v[:16] for k, v in src.items()},
            "code": code_fingerprint()}


def compare(old: dict | None, new: dict) -> list[str]:
    """Danh sach truong LECH giua dau van tay cu va moi ('' = khong co ban cu)."""
    if not old:
        return ["(chua co dau van tay)"]
    diff = []
    for k, v in new.items():
        if old.get(k) != v:
            diff.append(k)
    return diff


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def save(path: Path, fp: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dict(fp)
    if extra:
        d.update(extra)
    path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
