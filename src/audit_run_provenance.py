"""
audit_run_provenance.py -- (review vong 2, muc 3) Kiem tra NGUON GOC dau ra tracking cua cac lan chay:
lan chay do co bat dau trong THU MUC MOI khong, co bi gian doan / ghi de / tai su dung khong, va bang chung
nao xac nhan cau hinh.

Voi moi thu muc `data/processed/trackers/<split>/<dir>`:
    n_videos            so file data/<video>.txt
    run_meta            co / thieu / hong
    status              'complete' | 'running' | '(khong co truong status)' - ban truoc khi sua khong ghi status
    created, git_commit provenance ghi trong run_meta
    n_resumed           so lan tiep tuc duoc ghi lai
    input_fingerprint   dau van tay dau vao (neu co)
    n_video_meta        so file video_meta/<video>.json (bang chung hoan tat theo tung video - chi co tu ban sua)
    first_txt, last_txt moc thoi gian file tracking som nhat / muon nhat (phat hien ghi de xen ke)
    txt_span_s          khoang cach giua hai moc do (giay)
    meta_after_last_txt run_meta duoc ghi SAU file tracking cuoi cung hay khong
    ket_luan            danh gia nguon goc

    python src/audit_run_provenance.py --pattern "p2-expB_detrac_*_fix1-*"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def load_json(p: Path):
    if not p.exists():
        return "missing"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "corrupt"


#: Nhom ket luan ve BACKEND TRACKER cua mot run.
BACKEND_NP = "NUMPY (xac nhan)"
BACKEND_UL = "ULTRALYTICS (xac nhan)"
BACKEND_UNKNOWN = "UNKNOWN/LEGACY (thieu bang chung)"


def backend_of(meta: dict) -> str:
    """BACKEND TRACKER cua mot run, CHI khi co bang chung tuong minh trong metadata.

    F3 (review vong 6): truoc day backend chi nam trong doi tuong da duoc hash thanh
    `input_fingerprint`, nen `grep '"backend": "np"'` tra 0 NGAY CA voi run NumPy that -> khong duoc
    dung phep grep do de ket luan. Tu vong nay, `run_meta.json` luu `tracker_backend` tuong minh.

    KHONG suy doan tu ten thu muc, tu viec THIEU mot truong, hay tu `cache_meta.backend`
    (do la backend DETECTOR cua cache, khac han backend TRACKER).
    """
    bk = meta.get("tracker_backend")
    if isinstance(bk, str) and bk:
        return {"np": BACKEND_NP, "ultralytics": BACKEND_UL}.get(bk, f"KHAC ({bk})")
    code = meta.get("code")
    if isinstance(code, dict):
        inner = code.get("backend")
        if isinstance(inner, dict) and isinstance(inner.get("backend"), str):
            return {"np": BACKEND_NP, "ultralytics": BACKEND_UL}.get(inner["backend"],
                                                                     f"KHAC ({inner['backend']})")
    return BACKEND_UNKNOWN


def audit_dir(d: Path) -> dict:
    txts = sorted((d / "data").glob("*.txt")) if (d / "data").is_dir() else []
    mtimes = [t.stat().st_mtime for t in txts]
    meta = load_json(d / "run_meta.json")
    vm = list((d / "video_meta").glob("*.json")) if (d / "video_meta").is_dir() else []
    row = {"dir": d.name, "split": d.parent.name, "n_videos": len(txts),
           "run_meta": "missing" if meta == "missing" else ("corrupt" if meta == "corrupt" else "ok"),
           "n_video_meta": len(vm)}
    if isinstance(meta, dict):
        row.update(backend=backend_of(meta), status=meta.get("status", "(khong co truong status)"),
                   created=meta.get("created"),
                   git_commit=meta.get("git_commit"), n_resumed=len(meta.get("resumed", [])),
                   input_fingerprint=(meta.get("input_fingerprint") or "")[:12],
                   experiment=meta.get("experiment"), cond=meta.get("cond"), tracker=meta.get("tracker"),
                   conf_min=meta.get("conf_min"), cache_id=meta.get("cache_id"))
        meta_m = (d / "run_meta.json").stat().st_mtime
    else:
        meta_m = None
    if mtimes:
        row["first_txt"] = dt.datetime.fromtimestamp(min(mtimes)).isoformat(timespec="seconds")
        row["last_txt"] = dt.datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")
        row["txt_span_s"] = round(max(mtimes) - min(mtimes), 1)
        row["meta_after_last_txt"] = (meta_m is not None and meta_m >= max(mtimes) - 1)
    return row


def verdict(r: dict) -> str:
    if r["run_meta"] != "ok":
        return "KHONG XAC MINH DUOC: thieu/hong run_meta"
    if r.get("status") == "running":
        return "NGHI VAN: run bi ngat (status=running)"
    if r.get("tracking_hash_match") is False:
        return "NGHI VAN: file tracking hien tai KHAC sha256 ghi trong eval_meta.json"
    legacy_runner = r.get("status") == "(khong co truong status)"
    if not r.get("meta_after_last_txt", True) and r.get("tracking_hash_match") is not True:
        if legacy_runner:
            # Ban runner CU ghi run_meta.json TRUOC vong lap -> thu tu nay la BINH THUONG voi ban do,
            # khong phai dau hieu ghi de. Nhung cung khong co fingerprint de xac minh -> danh dau LEGACY.
            return ("LEGACY: chay boi ban runner cu (metadata ghi truoc vong lap, chua co eval_meta/video_meta) "
                    "- khong xac minh duoc bang fingerprint")
        return "NGHI VAN: run_meta ghi TRUOC file tracking cuoi (co the da ghi de sau do)"
    n_res = r.get("n_resumed", 0)
    if r.get("n_video_meta", 0) == r["n_videos"] and r["n_videos"] > 0:
        base = "OK: moi video co bang chung hoan tat (video_meta)"
    elif r.get("tracking_hash_match") is True:
        base = "OK: provenance muc RUN + sha256 tung file khop eval_meta.json"
    elif r.get("status") == "(khong co truong status)":
        base = "LEGACY: ban runner cu, khong co fingerprint de doi chieu"
    else:
        base = "OK (ban truoc khi co video_meta): provenance o muc RUN"
    if n_res:
        base += f"; co {n_res} lan tiep tuc duoc ghi lai"
    return base


def verify_eval_fingerprint(d: Path, split: str) -> dict:
    """Doi chieu dau ra tracking HIEN TAI voi dau van tay ghi trong eval_meta.json cua lan danh gia.

    Day la bang chung MANH NHAT hien co ve nguon goc: `eval_meta.json` chua sha256 cua TUNG file
    `data/<video>.txt` tai thoi diem tinh metric; neu hash hien tai trung thi bang ket qua da bao cao
    ung voi DUNG cac file dang nam trong thu muc.
    """
    import run_fingerprint as fp
    em = load_json(config.RESULTS_DIR / "trackeval" / split / d.name / "eval_meta.json")
    if not isinstance(em, dict):
        return {"eval_meta": em if isinstance(em, str) else "missing", "tracking_hash_match": None}
    videos = list((em.get("tracking") or {}).get("per_video", {}).keys())
    now = fp.tracking_fingerprint(d, videos) if videos else None
    return {"eval_meta": "ok", "n_videos_in_eval": len(videos),
            "tracking_hash_match": (now["sha256"] == em["tracking"]["sha256"]) if now else None,
            "code_commit_at_eval": (em.get("code") or {}).get("git_commit"),
            "evaluated_at": em.get("evaluated_at")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-eval-fingerprint", action="store_true",
                    help="doi chieu file tracking hien tai voi sha256 ghi trong eval_meta.json")
    ap.add_argument("--pattern", default="p2-*")
    ap.add_argument("--splits", nargs="*", default=["DETRAC-all", "DETRAC-test", "BDDFX-val"])
    ap.add_argument("--out", default="results/phase2/run_provenance_audit.csv")
    a = ap.parse_args()
    rows = []
    for split in a.splits:
        base = config.PROCESSED_DIR / "trackers" / split
        if not base.is_dir():
            continue
        for d in sorted(base.glob(a.pattern)):
            if d.is_dir():
                r = audit_dir(d)
                if a.verify_eval_fingerprint:
                    r.update(verify_eval_fingerprint(d, split))
                r["ket_luan"] = verdict(r)
                rows.append(r)
    if not rows:
        print("khong tim thay thu muc nao"); return 1
    df = pd.DataFrame(rows)
    out = config.PROJECT_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    cols = [c for c in ("dir", "n_videos", "run_meta", "status", "n_video_meta", "n_resumed", "created",
                        "txt_span_s", "meta_after_last_txt", "tracking_hash_match", "evaluated_at",
                        "ket_luan") if c in df.columns]
    print(df[cols].to_string(index=False))
    print("\nPhan loai provenance:")
    print(df.ket_luan.str.split(":").str[0].value_counts().to_string())
    print("\nBACKEND TRACKER (chi tu bang chung TUONG MINH trong run_meta.json):")
    bk = df.get("backend")
    if bk is None:
        print("  (khong thu muc nao co metadata doc duoc)")
    else:
        for ten, n in bk.fillna(BACKEND_UNKNOWN).value_counts().items():
            print(f"  {ten:<32s} {n}")
        print(f"  --> {int((bk == BACKEND_UNKNOWN).sum())} run THIEU BANG CHUNG backend. KHONG suy ra "
              "chung dung Ultralytics, va cung khong ket luan 'chua co run NumPy'.")
    bad = df[df.ket_luan.str.startswith("NGHI VAN")]
    n_legacy = int(df.ket_luan.str.startswith("LEGACY").sum())
    print(f"\nSo thu muc NGHI VAN: {len(bad)}/{len(df)}; "
          f"LEGACY (chay boi runner cu, khong co fingerprint de doi chieu): {n_legacy}")
    if len(bad):
        print(bad[["dir", "ket_luan"]].to_string(index=False))
    print(f"\n-> {out}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
