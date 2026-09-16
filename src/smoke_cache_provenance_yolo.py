"""
smoke_cache_provenance_yolo.py -- Smoke YOLO THAT (~30 frame dev, cache-id MOI, thu muc cache TAM)
de kiem chung ban va cache review vong 3 tren moi truong THAT (Ultralytics + checkpoint that + anh
UA-DETRAC that). CAN GPU + anh UA-DETRAC; `src/test_cache_provenance.py` phu phan logic khong can
chung, nen suite tu dong khong goi file nay.

    python src/smoke_cache_provenance_yolo.py

Chay DUNG duong CLI `det_cache.main()`; chi can thiep hai cho, khong dung cham ket qua lich su:
  - CACHE_ROOT tro vao thu muc TAM
  - ds.frames() cat con 30 frame dau

Kiem:
  1. meta.json co weights_sha256 == sha256 cua CHINH file checkpoint YOLO da nap (model.ckpt_path)
  2. ghi cache thanh cong, doc lai strict -> trang thai 'ok'
  3. khai bao weights_sha256 SAI -> bi chan (khong ghi)
  4. thieu weights_sha256 (backend ultralytics) -> bi chan
  5. cache-id da co, doi spec_sha256 cach tinh cu -> bi chan append, meta.json GIU NGUYEN
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import det_cache as dc  # noqa: E402

N_FRAME = 30
VIDEO = json.loads((config.PROJECT_ROOT / "results/phase2/manifests/detrac_dev.json")
                   .read_text(encoding="utf-8"))["videos"][0]
TMP = Path(tempfile.mkdtemp(prefix="smoke_cache_"))
CACHE_ID = "smoke_review3_provenance"
ket_qua = []


def check(ten, ok, chi_tiet=""):
    ket_qua.append((ten, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {ten}")
    if chi_tiet:
        print(f"         {chi_tiet}")


def main() -> int:
    print(f"video dev: {VIDEO};  cache-id: {CACHE_ID};  thu muc tam: {TMP}")
    dc.CACHE_ROOT = TMP

    goc_get_dataset = dc.get_dataset

    def get_dataset_cat(*a, **k):
        ds = goc_get_dataset(*a, **k)
        goc_frames = ds.frames
        ds.frames = lambda v, part: goc_frames(v, part)[:N_FRAME]
        return ds
    dc.get_dataset = get_dataset_cat

    argv = sys.argv
    sys.argv = ["det_cache.py", "--dataset", "detrac", "--part", "train", "--videos", VIDEO,
                "--cache-id", CACHE_ID, "--source", "yolo", "--conf", "0.05", "--max-det", "3000"]
    try:
        rc = dc.main()
    finally:
        sys.argv = argv
    cdir = TMP / CACHE_ID
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))

    # --- 1. hash dung file checkpoint da nap ---
    from ultralytics import YOLO
    m = YOLO(config.YOLO_DEFAULT_MODEL)
    ckpt = Path(m.ckpt_path).resolve()
    that = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    check("weights_sha256 trong meta.json == sha256 cua checkpoint YOLO da nap (model.ckpt_path)",
          rc == 0 and meta["weights_sha256"] == that and Path(meta["weights"]).resolve() == ckpt,
          f"ckpt={ckpt}\n         meta={meta['weights_sha256'][:24]}...  thuc={that[:24]}...")

    # --- 2. doc lai strict ---
    npz = sorted(cdir.glob("*.npz"))
    trang_thai, ghi_chu = dc.sequence_spec_status(cdir, VIDEO, meta)
    goc_root = dc.CACHE_ROOT
    det = dc.open_cache(CACHE_ID, VIDEO, conf_min=0.25, classes=[2, 3, 5, 7], strict=True)
    n_frame = len(det) if hasattr(det, "__len__") else "?"
    check("ghi cache roi doc lai strict -> trang thai 'ok', doc duoc du frame",
          trang_thai == "ok" and len(npz) == 1 and n_frame == N_FRAME,
          f"trang_thai={trang_thai} ({ghi_chu}); npz={len(npz)}; n_frame doc lai={n_frame}")

    # --- 3. khai bao weights_sha256 SAI -> chan ---
    spec_sai = dc.DetectorSpec(backend="ultralytics", weights=str(ckpt), weights_sha256="b" * 64,
                               conf=0.05, classes=None, max_det=3000)
    err = ""
    try:
        spec_sai.fill_env()
    except RuntimeError as e:
        err = str(e)
    check("khai bao weights_sha256 KHAC file checkpoint thuc -> bi chan",
          "khong khop" in err.lower() or "khong khop" in err, err[:110] or "(khong loi!)")

    # --- 4. thieu weights_sha256 -> chan khi ghi ---
    spec_thieu = dc.DetectorSpec(backend="ultralytics", weights=str(ckpt), weights_sha256=None,
                                 conf=0.05, classes=None, max_det=3000)
    err = ""
    try:
        dc.write_meta(TMP / "cache_thieu_hash", spec_thieu)
    except RuntimeError as e:
        err = str(e)
    check("thieu weights_sha256 (backend ultralytics) -> tu choi ghi, khong tao thu muc",
          "weights_sha256" in err and not (TMP / "cache_thieu_hash").exists(), err[:110] or "(khong loi!)")

    # --- 5. fingerprint cu khong tuong thich -> chan append, meta GIU NGUYEN ---
    meta_truoc = (cdir / "meta.json").read_text(encoding="utf-8")
    m2 = json.loads(meta_truoc)
    m2["spec_sha256"] = "c" * 64                      # gia lap cach tinh fingerprint CU
    (cdir / "meta.json").write_text(json.dumps(m2, indent=2), encoding="utf-8")
    spec_lai = dc.DetectorSpec(**{k: v for k, v in meta.items()
                                  if k in dc.DetectorSpec.__dataclass_fields__})
    err = ""
    try:
        dc.write_meta(cdir, spec_lai)
    except RuntimeError as e:
        err = str(e)
    giu_nguyen = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))["spec_sha256"] == "c" * 64
    check("spec_sha256 cu khong tuong thich -> tu choi append, meta.json giu nguyen",
          "spec_sha256" in err and giu_nguyen, err[:130] or "(khong loi!)")
    (cdir / "meta.json").write_text(meta_truoc, encoding="utf-8")

    n_ok = sum(ok for _, ok in ket_qua)
    print(f"\nKET QUA SMOKE: {n_ok}/{len(ket_qua)} PASS  (video {VIDEO}, {N_FRAME} frame, GPU that)")
    return 0 if n_ok == len(ket_qua) else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        print(f"da xoa thu muc cache tam (khong dung cham {config.INTERIM_DIR / 'det_cache'})")
    raise SystemExit(code)
