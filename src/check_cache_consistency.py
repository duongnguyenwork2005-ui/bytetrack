"""
check_cache_consistency.py -- (sau review doc lap, loi P2) Doi chieu `meta.json` voi tung `<video>.json`
cua moi detection cache trong data/interim/det_cache/, de biet cache HIEN CO co bi tron cau hinh khong
truoc khi quyet dinh co phai tao lai hay khong.

Cot ket qua (results/phase2/cache_consistency.csv):
    cache_id, video, status ('ok' | 'legacy' | 'mismatch'), ly do, part, created, n_frames, n_boxes
`legacy` = cache sinh TRUOC khi co fingerprint (`spec_sha256` trong) -> khong the khang dinh bang hash;
khi do script in them cac dau hieu gian tiep (cung meta, cung `regions_hash` theo part, moc thoi gian).

    python src/check_cache_consistency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402


def main() -> int:
    rows = []
    for cdir in sorted(det_cache.CACHE_ROOT.glob("*")):
        if not cdir.is_dir() or not (cdir / "meta.json").exists():
            continue
        meta = det_cache.load_meta(cdir)
        for npz in sorted(cdir.glob("*.npz")):
            v = npz.stem
            st, why = det_cache.sequence_spec_status(cdir, v, meta)
            j = cdir / f"{v}.json"
            info = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
            rows.append({"cache_id": cdir.name, "video": v, "status": st, "reason": why,
                         "part": info.get("part"), "created": info.get("created"),
                         "n_frames": info.get("n_frames"), "n_boxes": info.get("n_boxes"),
                         "regions_hash": info.get("regions_hash"),
                         "meta_conf": meta.get("conf"), "meta_classes": meta.get("classes"),
                         "meta_max_det": meta.get("max_det"), "meta_weights_sha256": meta.get("weights_sha256"),
                         "meta_versions": json.dumps(meta.get("versions"), sort_keys=True),
                         "meta_spec_sha256": meta.get("spec_sha256"),
                         "n_version_history": len(meta.get("version_history", []))})
    if not rows:
        print("khong co cache nao"); return 1
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "phase2" / "cache_consistency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["cache_id", "status"]).size().rename("n_sequence").reset_index().to_string(index=False))
    print("\nMoi cache: mot cau hinh sinh du lieu duy nhat trong meta.json?")
    g = df.groupby("cache_id").agg(n=("video", "size"), conf=("meta_conf", "nunique"),
                                   classes=("meta_classes", "nunique"), max_det=("meta_max_det", "nunique"),
                                   weights=("meta_weights_sha256", "nunique"),
                                   versions=("meta_versions", "nunique"),
                                   n_version_history=("n_version_history", "max"),
                                   parts=("part", lambda s: ",".join(sorted({str(x) for x in s})))).reset_index()
    print(g.to_string(index=False))
    bad = df[df.status == "mismatch"]
    print(f"\nSequence KHONG khop cau hinh: {len(bad)}")
    if len(bad):
        print(bad[["cache_id", "video", "reason"]].to_string(index=False))
    print(f"\n-> {out}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
