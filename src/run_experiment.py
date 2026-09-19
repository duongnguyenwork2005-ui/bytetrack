"""
run_experiment.py -- Chay MOT thi nghiem so sanh (Giai doan 2) tu file YAML:
    tracking (tu cache) x dieu kien x tracker x seed  ->  TrackEval  ->  occlusion_eval  ->  aggregate

File YAML (vd configs/phase2/experiments/groupA_detrac_dev.yaml):
    name: groupA_detrac_dev
    dataset: detrac            split_name: DETRAC-all       part: train
    manifest: results/phase2/manifests/detrac_dev.json
    cache_id: yolov8n-coco_conf0.05_all
    classes: [2, 3, 5, 7]      max_det: 300      fps: 25     track_buffer_sec: 1.2    elapsed_mode: skip
    conditions: {A: {conf_min: 0.25}, B: {conf_min: 0.10}}
    trackers:
      - {name: cv,  yaml: bytetrack.yaml}
      - {name: pf100, yaml: configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml, seeds: [0, 1, 2]}
      - {name: ekf-b2.0, yaml: configs/bytetrack_ekf_ctrv.yaml, track_buffer_sec: 2.0}   # Moc 2: theo tracker
      - {name: ekf-th02, yaml: configs/bytetrack_ekf_ctrv.yaml, set: {ctrv_std_theta: 0.02}}
    baseline: cv
    assoc_log: false           # Moc 2: true -> ghi assoc_log/<video>_*.parquet cho moi lan chay (ByteTrack)
    pf_diag: false             # Moc 2: true -> pf_diag/<video>.csv cho tracker PF
    pairs: [[cv, ekf]]         # Moc 2: cac cap so sanh them (aggregate_phase2 -> paired_pairs.csv)
    occlusion: {part: train, context_s: 1.2, gaps: false, tag: null, n_boot: 5000}

Ten thu muc tracker: p2-<name>-<cond>-<tracker>[-s<seed>] trong data/processed/trackers/<split>/.
Ket qua: results/phase2/<name>/runs.csv, <cond>/{events,event_outcomes,paired,bootstrap}_*.csv,
         results/trackeval/<split>/p2-.../{per_video,combined}_metrics.csv, va bang tong hop
         (aggregate_phase2.py).

    python src/run_experiment.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml
    python src/run_experiment.py --exp ... --stage track      # chi tracking
    python src/run_experiment.py --exp ... --stage eval       # trackeval + occlusion
    python src/run_experiment.py --exp ... --dry-run          # in ke hoach
    python src/run_experiment.py --exp ... --force            # chay lai MOI buoc (tracking + danh gia + tong hop)
    python src/run_experiment.py --exp ... --force-eval       # chi buoc danh gia

TAI SU DUNG KET QUA (sua sau review doc lap, xem results/phase2/REVIEW_moc2_codex.md)
    Truoc day buoc danh gia bi BO QUA chi vi file ket qua da ton tai -> tracking moi co the bi cham bang
    metric CU. Nay moi buoc danh gia ghi mot DAU VAN TAY (src/run_fingerprint.py): sha256 cua tung file
    tracking, cau hinh, cache (meta + npz), danh sach sequence, phien ban ma nguon va evaluator. Ket qua
    chi duoc tai su dung khi dau van tay TRUNG KHIT; lech -> chay lai va in ro truong nao lech.
    Trang thai tracking duoc quan ly THEO TUNG VIDEO: `video_meta/<video>.json` (ghi NGUYEN TU sau khi
    video do chay xong, co `complete: true` + dau van tay). Mot video chi duoc tai su dung khi co bang
    chung hoan tat va dau van tay khop; thieu/hong metadata => chay lai. `run_meta.json` mang
    `status: running` tu truoc khi ghi de bat ky dau ra nao va chuyen sang `complete` khi xong, nen mot
    lan chay bi NGAT khong the duoc "xac nhan" boi metadata hoan tat cu (loi P1 cua review vong 2).
    Provenance (`created`, `git_commit`) duoc giu khi dau van tay khong doi; moi lan tiep tuc ghi them
    mot muc vao `resumed`.
    Moi loi cua buoc danh gia / tong hop deu tra ve exit code khac 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import run_trackeval  # noqa: E402
import run_fingerprint as fp  # noqa: E402
from time_units import track_buffer_frames  # noqa: E402

PY = sys.executable
ROOT = config.PROJECT_ROOT


def load_exp(path: str | Path) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.setdefault("conditions", {"A": {"conf_min": 0.25}})
    d.setdefault("classes", [2, 3, 5, 7])
    d.setdefault("max_det", 300)
    d.setdefault("elapsed_mode", "skip")
    d.setdefault("occlusion", {})
    d["_path"] = str(path)
    return d


def tracker_dir_name(exp: str, cond: str, tracker: str, seed: int | None) -> str:
    return f"p2-{exp}-{cond}-{tracker}" + (f"-s{seed}" if seed is not None else "")


def plan(exp: dict) -> list[dict]:
    """Danh sach (cond, tracker, seed, dir) can chay."""
    registry = run_tracking.tracker_registry()
    rows = []
    for cond, cc in exp["conditions"].items():
        for t in exp["trackers"]:
            cfg = run_tracking.load_tracker_cfg(t["yaml"], t.get("set"))
            stochastic = cfg.tracker_type in run_tracking.STOCHASTIC_TRACKERS
            seeds = t.get("seeds", [None]) if stochastic else [None]
            if stochastic and seeds == [None]:
                seeds = [int(getattr(cfg, "pf_seed", 0))]
            for s in seeds:
                rows.append({"cond": cond, "tracker": t["name"], "yaml": t["yaml"], "set": t.get("set") or {},
                             "seed": s, "conf_min": float(cc["conf_min"]),
                             "dir": tracker_dir_name(exp["name"], cond, t["name"], s),
                             "tracker_type": cfg.tracker_type,
                             "track_buffer_sec": t.get("track_buffer_sec", exp.get("track_buffer_sec"))})
    return rows


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Ghi JSON nguyen tu: ghi file tam roi doi ten (tranh file metadata bi cat doi khi bi ngat)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path):
    """None neu thieu HOAC hong (thieu/hong => KHONG duoc coi la tuong thich)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def video_meta_path(out_root: Path, video: str) -> Path:
    return out_root / "video_meta" / f"{video}.json"


def can_reuse_video(out_root: Path, video: str, fingerprint: str) -> tuple[bool, str]:
    """Chi tai su dung mot video khi CO BANG CHUNG HOAN TAT va dau van tay khop.

    Bang chung = `video_meta/<video>.json` (ghi NGUYEN TU sau khi video chay xong) voi
    `complete: true` va `input_fingerprint` trung. Thieu/hong metadata -> KHONG tai su dung
    (loi P1 vong 2: truoc day chi can ton tai `data/<video>.txt` la du).
    """
    if not (out_root / "data" / f"{video}.txt").exists():
        return False, "chua co file tracking"
    vm = _load_json(video_meta_path(out_root, video))
    if vm is None:
        return False, "thieu/hong video_meta"
    if not vm.get("complete"):
        return False, "video_meta chua danh dau hoan tat"
    if vm.get("input_fingerprint") != fingerprint:
        return False, "dau van tay khac"
    return True, ""


def mark_run_started(path: Path, meta: dict) -> dict:
    """Danh dau run DANG CHAY truoc khi ghi de bat ky dau ra nao.

    Giu `created`/`git_commit` GOC khi dau van tay khong doi (provenance cua phan thuc su tai su dung);
    neu dau van tay doi (hoac metadata cu thieu/hong) thi bat dau provenance moi.
    """
    old = _load_json(path)
    m = dict(meta)
    same = old is not None and old.get("input_fingerprint") == meta["input_fingerprint"]
    if same:
        m["created"] = old.get("created", m["created"])
        m["git_commit"] = old.get("git_commit", m["git_commit"])
        m["resumed"] = list(old.get("resumed", []))
    elif old is not None:
        m["previous"] = {"created": old.get("created"), "git_commit": old.get("git_commit"),
                         "input_fingerprint": old.get("input_fingerprint"), "status": old.get("status")}
    m["status"] = "running"
    m["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_json(path, m)
    return m


def mark_run_finished(path: Path, meta: dict, ran: int, reused: int, videos: list[str]) -> None:
    """Ket thuc mot lan chay. Neu tren dia da co metadata CUNG dau van tay (vd lan nay khong chay video nao
    nen `mark_run_started` khong duoc goi) thi GIU provenance goc va noi tiep danh sach `resumed`."""
    old = _load_json(path)
    m = dict(meta)
    if old is not None and old.get("input_fingerprint") == m.get("input_fingerprint"):
        m["created"] = old.get("created", m.get("created"))
        m["git_commit"] = old.get("git_commit", m.get("git_commit"))
        m["resumed"] = list(old.get("resumed", [])) if not m.get("resumed") else m["resumed"]
        if old.get("previous") and not m.get("previous"):
            m["previous"] = old["previous"]
    m["status"] = "complete"
    m["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    m["n_videos_ran"] = ran
    m["n_videos_reused"] = reused
    m["videos"] = list(videos)
    m.setdefault("resumed", []).append({"at": m["finished_at"], "git_commit": det_cache.git_commit(),
                                        "n_videos_ran": ran, "n_videos_reused": reused})
    _atomic_write_json(path, m)


def stage_track(exp: dict, rows: list[dict], videos: list[str], skip_existing: bool):
    fps = float(exp["fps"])
    dt = 1.0 / fps
    registry = run_tracking.tracker_registry()
    out_rows = []
    assoc_log = bool(exp.get("assoc_log", False))
    pf_diag = bool(exp.get("pf_diag", False))
    for r in rows:
        overrides = dict(r["set"])
        # track_buffer_sec theo TRACKER (Moc 2, C1) uu tien hon muc thi nghiem; doi sang buoc theo fps
        tb_sec = r.get("track_buffer_sec")
        if tb_sec is not None:
            overrides["track_buffer"] = track_buffer_frames(float(tb_sec), fps)
        cfg = run_tracking.load_tracker_cfg(r["yaml"], overrides)
        log_ok = assoc_log and cfg.tracker_type.startswith("bytetrack")     # SORT: khong co hook -> bo qua
        out_root = config.PROCESSED_DIR / "trackers" / exp["split_name"] / r["dir"]
        out_root.mkdir(parents=True, exist_ok=True)
        yaml_sha = hashlib.sha256(Path(run_tracking.resolve_tracker_yaml(r["yaml"])).read_bytes()).hexdigest()[:12]
        meta = {"experiment": exp["name"], "cond": r["cond"], "tracker": r["tracker"], "seed": r["seed"],
                "cache_id": exp["cache_id"], "tracker_yaml": r["yaml"], "tracker_yaml_sha256_12": yaml_sha,
                "tracker_cfg": run_tracking.cfg_to_dict(cfg), "overrides": overrides,
                "conf_min": r["conf_min"],
                "classes": exp["classes"], "max_det": exp["max_det"], "fps": fps, "dt_nominal_s": dt,
                "track_buffer_frames": int(cfg.track_buffer),
                "track_buffer_sec": round(int(cfg.track_buffer) / fps, 4),
                "elapsed_mode": exp["elapsed_mode"], "assoc_log": log_ok, "pf_diag": pf_diag,
                "git_commit": det_cache.git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        # DAU VAN TAY DAU VAO cua buoc tracking: cau hinh + cache + danh sach video + ma nguon.
        # Neu lech so voi lan chay truoc -> KHONG tai su dung file tracking cu (tranh "tiep tuc" mot run
        # voi cau hinh khac ma van giu ket qua cu).
        # `tracker_yaml_sha256_12` PHAI nam trong dau van tay: sua NOI DUNG file YAML tai CUNG duong
        # dan cung phai lam mat hieu luc tai su dung (review vong 5, P1-B).
        meta["input_fingerprint"] = fp.sha256_obj({
            "tracker_cfg": run_tracking.cfg_to_dict(cfg), "tracker_yaml_sha256_12": yaml_sha,
            "overrides": overrides, "conf_min": r["conf_min"],
            "classes": exp["classes"], "max_det": exp["max_det"], "elapsed_mode": exp["elapsed_mode"],
            "fps": fps, "seed": r["seed"], "assoc_log": log_ok, "pf_diag": pf_diag,
            "cache": fp.cache_fingerprint(exp["cache_id"], videos), "sequences": list(videos),
            "code": fp.code_fingerprint()})
        # Quyet dinh tai su dung THEO TUNG VIDEO, dua tren `video_meta/<video>.json` (ghi nguyen tu sau khi
        # video do chay xong). Thieu/hong metadata hoac dau van tay khac -> chay lai video do.
        fingerprint = meta["input_fingerprint"]
        plan_reuse, plan_run, reasons = [], [], {}
        for v in videos:
            ok, why = (can_reuse_video(out_root, v, fingerprint) if skip_existing else (False, "--no-skip/--force"))
            (plan_reuse if ok else plan_run).append(v)
            if not ok:
                reasons.setdefault(why, 0)
                reasons[why] += 1
        if plan_run and (out_root / "data").exists() and any((out_root / "data" / f"{v}.txt").exists() for v in plan_run):
            print(f"  [track] {r['dir']:<40} CHAY LAI {len(plan_run)} video (ghi de): "
                  + ", ".join(f"{k} x{n}" for k, n in reasons.items()))
        meta_path = out_root / "run_meta.json"
        if plan_run:
            meta = mark_run_started(meta_path, meta)      # danh dau DANG CHAY truoc khi ghi de bat ky dau ra nao
        t_all = 0.0
        summary = []
        for v in plan_run:
            vd = det_cache.open_cache(exp["cache_id"], v, r["conf_min"], exp["classes"], exp["max_det"])
            # dau ra cu cua video nay khong con dung -> xoa bang chung hoan tat truoc khi ghi de
            vmp = video_meta_path(out_root, v)
            if vmp.exists():
                vmp.unlink()
            t0 = time.time()
            df, df_t = run_tracking.run_video_to_dir(out_root, v, vd, cfg, dt, exp["elapsed_mode"], r["seed"], registry,
                                                     assoc_log=log_ok, pf_diag=pf_diag)
            sec = time.time() - t0
            t_all += sec
            row = {"video": v, "n_frames": len(vd), "n_boxes": len(df),
                   "n_tracks": int(df["id"].nunique()) if len(df) else 0,
                   "predict_ms_per_frame": round(float(df_t.t_predict_ms.mean()), 4),
                   "assoc_ms_per_frame": round(float(df_t.t_assoc_ms.mean()), 4),
                   "update_ms_per_frame": round(float(df_t.t_update_ms.mean()), 4), "seconds": round(sec, 2)}
            summary.append(row)
            # CONG BO ket qua cua video: ghi bang chung hoan tat (nguyen tu) SAU khi moi file da ghi xong
            _atomic_write_json(vmp, {"video": v, "complete": True, "input_fingerprint": fingerprint,
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     "git_commit": det_cache.git_commit(), "experiment": exp["name"],
                                     "cond": r["cond"], "tracker": r["tracker"], "seed": r["seed"],
                                     "n_frames": row["n_frames"], "n_boxes": row["n_boxes"],
                                     "n_tracks": row["n_tracks"]})
        if summary:
            old = out_root / "summary.csv"
            s = pd.DataFrame(summary)
            if plan_reuse and old.exists():
                s = pd.concat([pd.read_csv(old), s]).drop_duplicates("video", keep="last")
            s.to_csv(old, index=False)
        mark_run_finished(meta_path, meta, ran=len(summary), reused=len(plan_reuse), videos=videos)
        print(f"  [track] {r['dir']:<40} {len(summary):>3} video moi, {len(plan_reuse):>3} tai su dung  {t_all:6.1f}s")
        out_rows.append({**r, "seconds_track": round(t_all, 1)})
    return out_rows


def stage_eval(exp: dict, rows: list[dict], videos: list[str], force: bool = False) -> int:
    """Chay TrackEval + occlusion_eval. Tra ve so buoc THAT BAI (0 = tat ca thanh cong).

    Ket qua chi duoc tai su dung khi DAU VAN TAY trung khit (src/run_fingerprint.py) -> tracking moi khong
    the bi cham bang metric cu.
    """
    split = exp["split_name"]
    dataset_cls = "mot_crowd" if split.startswith("BDD-") else "mot"
    fps_exp = float(exp["fps"])
    n_fail = 0
    # --- TrackEval tung tracker ---
    for r in rows:
        out_dir = config.RESULTS_DIR / "trackeval" / split / r["dir"]
        overrides = dict(r["set"])
        tb_sec = r.get("track_buffer_sec")
        if tb_sec is not None:
            overrides["track_buffer"] = track_buffer_frames(float(tb_sec), fps_exp)
        cfg_r = run_tracking.load_tracker_cfg(r["yaml"], overrides)
        new_fp = fp.run_fingerprint(exp, r, videos, cfg_r, dataset_cls)
        have = ((out_dir / "combined_metrics.csv").exists() and (out_dir / "per_video_metrics.csv").exists()
                and (out_dir / "raw_fields.json").exists())
        diff = fp.compare(fp.load(out_dir / "eval_meta.json"), new_fp)
        if have and not diff and not force:
            print(f"  [eval ] {r['dir']:<40} TAI SU DUNG (dau van tay trung khit)")
            continue
        if have:
            print(f"  [eval ] {r['dir']:<40} CHAY LAI, ly do: {'--force' if force else ', '.join(diff)}")
        try:
            results, _ = run_trackeval.run(split, r["dir"], videos, dataset_cls)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  [eval ] {r['dir']:<40} THAT BAI: {type(e).__name__}: {e}")
            continue
        df = run_trackeval.flatten_results(results, r["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        df[df.video != "COMBINED_SEQ"].sort_values("video").to_csv(out_dir / "per_video_metrics.csv", index=False)
        df[df.video == "COMBINED_SEQ"].to_csv(out_dir / "combined_metrics.csv", index=False)
        # Moc 2: thong ke du per-sequence de bootstrap tai to hop dung cong thuc COMBINED (boot_stats.py)
        from boot_stats import save_raw_fields
        save_raw_fields(results, r["dir"], out_dir / "raw_fields.json")
        fp.save(out_dir / "eval_meta.json", new_fp, {"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        c = df[df.video == "COMBINED_SEQ"].iloc[0]
        print(f"  [eval ] {r['dir']:<40} HOTA {c.HOTA:.4f} IDF1 {c.IDF1:.4f} AssA {c.AssA:.4f} IDSW {int(c.IDSW)}")
    # --- occlusion_eval theo dieu kien (baseline dau tien) ---
    # Bo qua khi YAML KHONG khai bao `occlusion` (vd nguon nhan khong co thong tin che khuat):
    # chay occlusion_eval khi khong co du lieu chi tao ra loi kho hieu, khong tao ra thong tin.
    occ = exp.get("occlusion") or {}
    if not occ:
        print("  [occl ] bo qua: thi nghiem khong khai bao muc `occlusion` "
              "(nguon nhan khong co du lieu che khuat)")
        return n_fail
    for cond in exp["conditions"]:
        rr = [r for r in rows if r["cond"] == cond]
        base = [r["dir"] for r in rr if r["tracker"] == exp.get("baseline")]
        others = [r["dir"] for r in rr if r["tracker"] != exp.get("baseline")]
        if not base:
            print(f"  [occl ] {cond}: khong co baseline {exp.get('baseline')!r}, bo qua"); continue
        out = config.RESULTS_DIR / "phase2" / exp["name"] / cond
        out.mkdir(parents=True, exist_ok=True)
        new_occ_fp = fp.occl_fingerprint(exp, cond, [*base, *others], videos, occ)
        diff = fp.compare(fp.load(out / "occl_meta.json"), new_occ_fp)
        if (out / f"bootstrap_{split}.csv").exists() and not diff and not force:
            print(f"  [occl ] {cond}: TAI SU DUNG (dau van tay trung khit)"); continue
        if (out / f"bootstrap_{split}.csv").exists():
            print(f"  [occl ] {cond}: CHAY LAI, ly do: {'--force' if force else ', '.join(diff)}")
        cmd = [PY, "src/occlusion_eval.py", "--split-name", split, "--trackers", *base, *others,
               "--out-dir", str(out), "--n-boot", str(occ.get("n_boot", 5000)), "--videos", *videos]
        if occ.get("part"):
            cmd += ["--part", occ["part"]]
        if occ.get("tag"):
            cmd += ["--tag", occ["tag"]]
        if occ.get("fps"):
            cmd += ["--fps", str(occ["fps"])]
        if occ.get("context_s") is not None:
            cmd += ["--context-s", str(occ["context_s"])]
        if occ.get("gaps"):
            cmd += ["--gaps"]
        t0 = time.time()
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (out / "occlusion_eval.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
        print(f"  [occl ] {cond}: exit {res.returncode} ({time.time() - t0:.0f}s) -> {out}")
        if res.returncode != 0:
            n_fail += 1
            print(res.stderr[-1500:])
        else:
            fp.save(out / "occl_meta.json", new_occ_fp, {"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--stage", choices=["all", "track", "eval", "aggregate"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-skip", action="store_true", help="chay lai ca video da co (buoc tracking)")
    ap.add_argument("--force", action="store_true", help="chay lai MOI buoc (tracking + danh gia + tong hop)")
    ap.add_argument("--force-eval", action="store_true", help="buoc danh gia chay lai du dau van tay trung")
    a = ap.parse_args()
    exp = load_exp(a.exp)
    videos = json.loads((ROOT / exp["manifest"]).read_text(encoding="utf-8"))["videos"] if exp.get("manifest") \
        else exp["videos"]
    rows = plan(exp)
    print(f"[exp] {exp['name']}: split={exp['split_name']} cache={exp['cache_id']} {len(videos)} video, "
          f"{len(rows)} lan chay tracker ({len(exp['conditions'])} dieu kien)")
    out_dir = config.RESULTS_DIR / "phase2" / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.yaml").write_text(Path(a.exp).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "videos.json").write_text(json.dumps(videos, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)
    if a.dry_run:
        print(pd.DataFrame(rows)[["cond", "tracker", "seed", "conf_min", "dir"]].to_string(index=False)); return 0
    n_fail = 0
    if a.stage in ("all", "track"):
        stage_track(exp, rows, videos, skip_existing=not (a.no_skip or a.force))
    if a.stage in ("all", "eval"):
        n_fail += stage_eval(exp, rows, videos, force=a.force or a.force_eval)
    if a.stage in ("all", "aggregate"):
        res = subprocess.run([PY, "src/aggregate_phase2.py", "--exp", a.exp], cwd=ROOT)
        if res.returncode != 0:
            n_fail += 1
            print(f"[exp] aggregate THAT BAI (exit {res.returncode})")
    if n_fail:
        print(f"[exp] {n_fail} buoc THAT BAI -> exit 1")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
