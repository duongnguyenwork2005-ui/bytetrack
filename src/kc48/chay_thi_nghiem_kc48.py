"""
chay_thi_nghiem_kc48.py -- R48C / R48D: chay thi nghiem kc48_* bang DUNG ham src/run_experiment.py:main (buoc track,
eval) nhung KHONG ghi vao data/processed/ (ngoai write_paths): doi goc config.PROCESSED_DIR sang
results/phase2/<ten>/_processed/ trong cung tien trinh. Moi duong khac giu nguyen nhu runner goc:
results/phase2/<ten>/ (experiment.yaml, videos.json, runs.csv), results/trackeval/<split>/p2-<ten>-*/ (metric +
eval_meta.json), cache detection doc tu data/interim/det_cache (chi doc).

Truoc buoc track / eval: sao `gt/gt.txt` + `seqinfo.ini` cua tung video tu data/processed/<split>/<video>/ sang goc moi,
kiem sha256 ban sao trung ban goc va doi chieu sha256[:16] cua gt.txt voi eval_meta.json tham chieu cua moc 2 (lech ->
dung, exit 2). Goc moi co .gitignore '*' (GT la nhan dataset, repo public). Ket qua kiem: results/phase2/<ten>/gt_sao_kiem.json.

Gioi han (giao thuc handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md muc 5.1): buoc occlusion_eval / aggregate
cua runner chay TIEN TRINH CON doc duong co dinh -> wrapper tu choi --stage all / aggregate va tu choi thi nghiem co
khai bao `occlusion`.

Thoi gian: moi lan chay buoc track / eval ghi mot dong vao results/phase2/<ten>/nhat_ky_wrapper.jsonl (buoc, bat dau,
ket thuc, giay, exit). `--stage thoi-gian` tong hop summary.csv cua tung thu muc tracker (giay / video va ms / frame do
runner ghi) thanh results/phase2/<ten>/thoi_gian.csv + thoi_gian.json, kem ngoai suy tuyen tinh theo so frame
(seqLength) cho cac video cua --manifest-ngoai-suy (mac dinh dev12).

    python src/kc48/chay_thi_nghiem_kc48.py --exp configs/phase2/experiments/kc48_pf_smoke.yaml --stage track
    python src/kc48/chay_thi_nghiem_kc48.py --exp configs/phase2/experiments/kc48_pf_smoke.yaml --stage eval
    python src/kc48/chay_thi_nghiem_kc48.py --exp configs/phase2/experiments/kc48_pf_smoke.yaml --stage thoi-gian
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

GOC_PROCESSED = Path(config.PROCESSED_DIR)          # goc that: chi DOC GT tu day
THAM_CHIEU_GT = ROOT / "results/trackeval/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/eval_meta.json"
GT_FILES = ("gt/gt.txt", "seqinfo.ini")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tim_khoa(d, khoa):
    """Tim de quy gia tri dau tien cua `khoa` trong dict/list long nhau."""
    if isinstance(d, dict):
        if khoa in d:
            return d[khoa]
        d = list(d.values())
    if isinstance(d, list):
        for x in d:
            r = _tim_khoa(x, khoa)
            if r is not None:
                return r
    return None


def doc_exp(path: str) -> dict:
    exp = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    if not str(exp.get("name", "")).startswith("kc48_"):
        raise SystemExit(f"tu choi: ten thi nghiem phai bat dau bang 'kc48_' (dang la {exp.get('name')!r})")
    if exp.get("occlusion"):
        raise SystemExit("tu choi: thi nghiem khai bao `occlusion` -> occlusion_eval chay tien trinh con doc "
                         "data/processed co dinh; can wrapper rieng (giao thuc 04 muc 5.1)")
    return exp


def videos_of(exp: dict) -> list[str]:
    if exp.get("manifest"):
        return json.loads((ROOT / exp["manifest"]).read_text(encoding="utf-8"))["videos"]
    return list(exp["videos"])


def seq_length(ini: Path) -> int:
    cp = configparser.ConfigParser()
    cp.read(ini)
    return int(cp["Sequence"]["seqLength"])


def chuan_bi_gt(exp: dict, videos: list[str], goc_moi: Path, out_dir: Path) -> None:
    split = exp["split_name"]
    ref = {}
    if THAM_CHIEU_GT.exists():
        ref = _tim_khoa(json.loads(THAM_CHIEU_GT.read_text(encoding="utf-8")), "gt_per_video") or {}
    goc_moi.mkdir(parents=True, exist_ok=True)
    gi = goc_moi / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")
    rows, loi = [], []
    for v in videos:
        for rel in GT_FILES:
            src, dst = GOC_PROCESSED / split / v / rel, goc_moi / split / v / rel
            if not src.is_file():
                loi.append(f"thieu ban goc {src}")
                continue
            h = sha256_file(src)
            hanh_dong = "da_co"
            if dst.exists():
                if sha256_file(dst) != h:
                    loi.append(f"ban sao lech ban goc: {dst}")
                    continue
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                hanh_dong = "sao_moi"
            ok = sha256_file(dst) == h
            khop_moc2 = (h[:16] == ref[v]) if (rel == "gt/gt.txt" and v in ref) else None
            if not ok:
                loi.append(f"sha256 ban sao khac ban goc: {dst}")
            if khop_moc2 is False:
                loi.append(f"gt.txt {v} khac sha256[:16] trong {THAM_CHIEU_GT.relative_to(ROOT).as_posix()}")
            rows.append({"video": v, "file": rel, "sha256": h, "byte": src.stat().st_size, "hanh_dong": hanh_dong,
                         "ban_sao_trung_goc": ok, "gt_khop_moc2": khop_moc2})
    (out_dir / "gt_sao_kiem.json").write_text(json.dumps(
        {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "goc": GOC_PROCESSED.relative_to(ROOT).as_posix() + f"/{split}",
         "dich": goc_moi.relative_to(ROOT).as_posix() + f"/{split}",
         "tham_chieu_moc2": THAM_CHIEU_GT.relative_to(ROOT).as_posix(),
         "tham_chieu_co_gt_per_video": bool(ref), "files": rows, "loi": loi},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if loi:
        print("GT: LOI\n  " + "\n  ".join(loi))
        raise SystemExit(2)
    n_khop = sum(r["gt_khop_moc2"] is True for r in rows)
    print(f"GT: {len(rows)} file sao/kiem trung sha256 ban goc; gt.txt khop moc 2: {n_khop}/{len(videos)}"
          + ("" if ref else " (tham chieu KHONG co truong gt_per_video -> khong doi chieu duoc, ghi null)"))


def dir_tracker_trong_goc_that(exp: dict) -> list[str]:
    d = GOC_PROCESSED / "trackers" / exp["split_name"]
    return sorted(p.name for p in d.glob(f"p2-{exp['name']}-*")) if d.exists() else []


def chay_runner(exp_path: str, exp: dict, stage: str, them: list[str], out_dir: Path, goc_moi: Path) -> int:
    truoc = dir_tracker_trong_goc_that(exp)
    config.PROCESSED_DIR = goc_moi                    # doi goc TRUOC khi nap runner; moi ham doc thuoc tinh luc goi
    import run_experiment  # noqa: E402
    assert run_experiment.config is config and config.PROCESSED_DIR == goc_moi
    argv_cu = sys.argv
    sys.argv = ["run_experiment.py", "--exp", exp_path, "--stage", stage, *them]
    t0, bat_dau = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        rc = int(run_experiment.main() or 0)
    finally:
        sys.argv = argv_cu
    giay = time.perf_counter() - t0
    sau = dir_tracker_trong_goc_that(exp)
    import det_cache  # noqa: E402
    dong = {"buoc": stage, "them": them, "bat_dau": bat_dau, "ket_thuc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "giay": round(giay, 2), "exit": rc, "goc_processed": goc_moi.relative_to(ROOT).as_posix(),
            "git_head": det_cache.git_commit(), "dir_p2_ten_nay_trong_data_processed_truoc_sau": [truoc, sau]}
    with (out_dir / "nhat_ky_wrapper.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dong, ensure_ascii=False) + "\n")
    print(f"[wrapper] buoc {stage}: exit {rc}, {giay:.1f} s; thu muc p2-{exp['name']}-* trong data/processed: "
          f"truoc {len(truoc)}, sau {len(sau)}")
    if sau != truoc:
        print("[wrapper] CANH BAO: co thu muc moi trong data/processed/trackers")
        return rc or 3
    return rc


def tong_hop_thoi_gian(exp: dict, out_dir: Path, goc_moi: Path, manifest_ngoai_suy: str) -> int:
    split = exp["split_name"]
    runs = pd.read_csv(out_dir / "runs.csv")
    rows = []
    for r in runs.itertuples():
        s = pd.read_csv(goc_moi / "trackers" / split / r.dir / "summary.csv")
        nf = float(s.n_frames.sum())
        rows.append({"tracker": r.tracker, "seed": None if pd.isna(r.seed) else int(r.seed), "dir": r.dir,
                     "n_video": len(s), "n_frames": int(nf), "giay_track": round(float(s.seconds.sum()), 2),
                     "ms_moi_frame_tong": round(1000 * float(s.seconds.sum()) / nf, 3),
                     **{c: round(float((s[c] * s.n_frames).sum() / nf), 4)
                        for c in ("predict_ms_per_frame", "assoc_ms_per_frame", "update_ms_per_frame")},
                     "n_boxes": int(s.n_boxes.sum()), "n_tracks": int(s.n_tracks.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "thoi_gian.csv", index=False)
    vids = json.loads((ROOT / manifest_ngoai_suy).read_text(encoding="utf-8"))["videos"]
    f_ns = sum(seq_length(GOC_PROCESSED / split / v / "seqinfo.ini") for v in vids)
    f_smoke_seq = sum(seq_length(GOC_PROCESSED / split / v / "seqinfo.ini") for v in videos_of(exp))
    nk = [json.loads(x) for x in (out_dir / "nhat_ky_wrapper.jsonl").read_text(encoding="utf-8").splitlines() if x]
    ev = [x for x in nk if x["buoc"] == "eval" and x["exit"] == 0]
    giay_eval_moi_run = (ev[-1]["giay"] / len(runs)) if ev else None
    bien_the = {}
    for name, g in df.groupby("tracker", sort=False):
        ms = float(g.ms_moi_frame_tong.mean())
        bien_the[name] = {"so_lan_chay_smoke": len(g), "ms_moi_frame_tong_TB_theo_seed": round(ms, 3),
                          "ms_moi_frame_min_max": [float(g.ms_moi_frame_tong.min()), float(g.ms_moi_frame_tong.max())],
                          "uoc_giay_track_ngoai_suy_moi_seed": round(ms / 1000 * f_ns, 1)}
    kq = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "thi_nghiem": exp["name"],
          "video_smoke": videos_of(exp), "frame_smoke_runner": int(df.n_frames.iloc[0]) if len(df) else None,
          "frame_smoke_seqLength": f_smoke_seq, "manifest_ngoai_suy": manifest_ngoai_suy,
          "n_video_ngoai_suy": len(vids), "frame_ngoai_suy_seqLength": f_ns,
          "giay_eval_moi_run_smoke_xap_xi": None if giay_eval_moi_run is None else round(giay_eval_moi_run, 2),
          "uoc_giay_eval_ngoai_suy_moi_run": None if giay_eval_moi_run is None
          else round(giay_eval_moi_run * f_ns / f_smoke_seq, 1),
          "bien_the": bien_the,
          "gia_dinh": ["thoi gian track ti le tuyen tinh voi so frame (bo qua khac biet mat do giua video)",
                       "giay_eval_moi_run = giay buoc eval cuoi cung (exit 0) / so lan chay, gom ca nap TrackEval",
                       "mot tien trinh, cung may, khong do lap lai: so lieu la uoc luong de dinh ngan sach, "
                       "khong phai benchmark toc do"]}
    (out_dir / "thoi_gian.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(kq, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="duong tuong doi tu goc repo: configs/phase2/experiments/kc48_*.yaml")
    ap.add_argument("--stage", choices=["track", "eval", "thoi-gian"], required=True)
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest-ngoai-suy", default="results/phase2/manifests/detrac_dev.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    exp_path = Path(a.exp).as_posix()
    if not Path(exp_path).name.startswith("kc48_"):
        raise SystemExit("tu choi: chi chay configs/phase2/experiments/kc48_*.yaml")
    exp = doc_exp(exp_path)
    out_dir = ROOT / "results" / "phase2" / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    goc_moi = out_dir / "_processed"
    if a.stage == "thoi-gian":
        return tong_hop_thoi_gian(exp, out_dir, goc_moi, a.manifest_ngoai_suy)
    chuan_bi_gt(exp, videos_of(exp), goc_moi, out_dir)
    them = (["--force-eval"] if a.force_eval else []) + (["--dry-run"] if a.dry_run else [])
    return chay_runner(exp_path, exp, a.stage, them, out_dir, goc_moi)


if __name__ == "__main__":
    raise SystemExit(main())
