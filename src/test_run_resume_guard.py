"""
test_run_resume_guard.py -- (review vòng 2, lỗi P1) Regression cho TRẠNG THÁI của bước tracking khi một lần
chạy bị **ngắt giữa chừng** hoặc khi **metadata thiếu/hỏng**. Chạy qua **đường chạy runner thật**
(`run_experiment.stage_track`) trên fixture BDD100K nhỏ, lỗi được giả lập có kiểm soát (ném exception từ
`run_tracking.run_video_to_dir` ở đúng video thứ hai).

Các kịch bản:
    1. A → B bị ngắt giữa chừng → quay về A: video đã bị ghi đè bằng cấu hình B **không** được tái sử dụng
       như thể nó thuộc A; `run_meta.json` của lần bị ngắt phải mang `status: running`.
    2. Có `data/<video>.txt` nhưng **thiếu** `video_meta/<video>.json` → chạy lại (không mặc nhiên tương thích).
    3. `video_meta/<video>.json` **hỏng** (JSON lỗi) → chạy lại.
    4. Resume **không đổi cấu hình** → không chạy video nào, giữ nguyên `created`/`git_commit` GỐC
       (so giá trị **lưu trước** resume với giá trị **sau** resume), ghi thêm mục `resumed`.
    5. Đổi cấu hình → chạy lại toàn bộ; `status` quay về `complete` sau khi xong.

    python src/test_run_resume_guard.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import run_experiment as rex  # noqa: E402
import run_tracking  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "resumeguard_bddfx"
SPLIT = "BDDFX-val"
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-4:]:
            print(f"         {line}")
    assert ok, name


def exp_dict(conf_min=0.25):
    return {"name": NAME, "dataset": "bdd100k", "split_name": SPLIT, "part": "val",
            "cache_id": "fixture_bddfx_gt", "classes": [1, 2, 3, 5, 7], "max_det": 300, "fps": 5,
            "track_buffer_sec": 1.2, "elapsed_mode": "steps", "assoc_log": False, "pf_diag": False,
            "conditions": {"c25": {"conf_min": conf_min}},
            "trackers": [{"name": "cv", "yaml": "bytetrack.yaml"}], "baseline": "cv", "occlusion": {}}


def rows_for(exp):
    return rex.plan(exp)


def out_root():
    return config.PROCESSED_DIR / "trackers" / SPLIT / f"p2-{NAME}-c25-cv"


def clean():
    d = out_root()
    if d.exists():
        shutil.rmtree(d)
    p = config.RESULTS_DIR / "phase2" / NAME
    if p.exists():
        shutil.rmtree(p)


def ensure_fixture(videos) -> bool:
    import det_cache
    cdir = det_cache.CACHE_ROOT / "fixture_bddfx_gt"
    return (config.PROCESSED_DIR / SPLIT).is_dir() and cdir.is_dir() and all((cdir / f"{v}.npz").exists() for v in videos)


class FailAt:
    """Bọc run_video_to_dir: chạy bình thường nhưng ném lỗi khi tới `video` chỉ định."""

    def __init__(self, video):
        self.video = video
        self.orig = run_tracking.run_video_to_dir

    def __enter__(self):
        orig, target = self.orig, self.video

        def wrapped(out_root_, v, *a, **kw):
            if v == target:
                raise RuntimeError(f"gia lap loi khi dang chay video {v}")
            return orig(out_root_, v, *a, **kw)
        run_tracking.run_video_to_dir = wrapped
        return self

    def __exit__(self, *exc):
        run_tracking.run_video_to_dir = self.orig
        return False


def main() -> int:
    videos = sorted(p.stem for p in (config.INTERIM_DIR / "det_cache" / "fixture_bddfx_gt").glob("*.npz")) \
        if (config.INTERIM_DIR / "det_cache" / "fixture_bddfx_gt").is_dir() else []
    if len(videos) < 2 or not ensure_fixture(videos):
        print("  [SKIP] chua co fixture BDDFX-val + cache fixture_bddfx_gt "
              "(chay src/test_smoke_phase2.py hoac src/test_run_experiment_guard.py truoc)")
        return 0
    v1, v2 = videos[0], videos[1]
    failed = 0
    try:
        clean()
        expA = exp_dict(0.25)
        rex.stage_track(expA, rows_for(expA), videos, skip_existing=True)
        metaA = json.loads((out_root() / "run_meta.json").read_text(encoding="utf-8"))
        _check("chay lan dau (cau hinh A): status=complete, moi video co video_meta",
               metaA["status"] == "complete"
               and all(rex.video_meta_path(out_root(), v).exists() for v in videos), str(metaA.get("status")))
        created_A, commit_A = metaA["created"], metaA["git_commit"]      # LUU TRUOC khi resume
        txt_A = {v: (out_root() / "data" / f"{v}.txt").read_text(encoding="utf-8") for v in videos}

        # --- 1. doi sang B, bi NGAT o video thu hai ---
        expB = exp_dict(0.40)
        crashed = False
        try:
            with FailAt(v2):
                rex.stage_track(expB, rows_for(expB), videos, skip_existing=True)
        except RuntimeError:
            crashed = True
        metaB = json.loads((out_root() / "run_meta.json").read_text(encoding="utf-8"))
        txt_B_v1 = (out_root() / "data" / f"{v1}.txt").read_text(encoding="utf-8")
        _check("bi ngat giua chung: run_meta co status='running' (khong phai 'complete')",
               crashed and metaB.get("status") == "running", f"crashed={crashed} status={metaB.get('status')}")
        _check(f"video dau ({v1}) da bi ghi de bang cau hinh B", txt_B_v1 != txt_A[v1])

        # --- quay ve A va resume ---
        ran = {"n": 0}
        orig = run_tracking.run_video_to_dir

        def counting(out_root_, v, *a, **kw):
            ran["n"] += 1
            return orig(out_root_, v, *a, **kw)
        run_tracking.run_video_to_dir = counting
        try:
            rex.stage_track(expA, rows_for(expA), videos, skip_existing=True)
        finally:
            run_tracking.run_video_to_dir = orig
        txt_after = {v: (out_root() / "data" / f"{v}.txt").read_text(encoding="utf-8") for v in videos}
        meta_after = json.loads((out_root() / "run_meta.json").read_text(encoding="utf-8"))
        _check(f"quay ve A: video bi ghi de ({v1}) duoc CHAY LAI (khong tai su dung dau ra cua B)",
               ran["n"] >= 1 and txt_after[v1] == txt_A[v1], f"so video da chay = {ran['n']}")
        _check("moi dau ra sau resume deu thuoc cau hinh A", all(txt_after[v] == txt_A[v] for v in videos))
        _check("run_meta tro lai status='complete'", meta_after["status"] == "complete")

        # --- 2. thieu video_meta ---
        rex.video_meta_path(out_root(), v2).unlink()
        ran["n"] = 0
        run_tracking.run_video_to_dir = counting
        try:
            rex.stage_track(expA, rows_for(expA), videos, skip_existing=True)
        finally:
            run_tracking.run_video_to_dir = orig
        _check(f"thieu video_meta cua {v2} -> chay lai dung 1 video", ran["n"] == 1, f"ran={ran['n']}")

        # --- 3. video_meta hong ---
        rex.video_meta_path(out_root(), v2).write_text("{khong phai JSON", encoding="utf-8")
        ran["n"] = 0
        run_tracking.run_video_to_dir = counting
        try:
            rex.stage_track(expA, rows_for(expA), videos, skip_existing=True)
        finally:
            run_tracking.run_video_to_dir = orig
        _check(f"video_meta hong ({v2}) -> chay lai dung 1 video", ran["n"] == 1, f"ran={ran['n']}")

        # --- 4. resume khong doi gi: giu provenance GOC ---
        meta_before_resume = json.loads((out_root() / "run_meta.json").read_text(encoding="utf-8"))
        n_resumed_before = len(meta_before_resume.get("resumed", []))
        ran["n"] = 0
        run_tracking.run_video_to_dir = counting
        try:
            rex.stage_track(expA, rows_for(expA), videos, skip_existing=True)
        finally:
            run_tracking.run_video_to_dir = orig
        meta_res = json.loads((out_root() / "run_meta.json").read_text(encoding="utf-8"))
        _check("resume khong doi gi: 0 video chay lai, created/git_commit GIU NGUYEN gia tri luu TRUOC resume, "
               "co them muc 'resumed'",
               ran["n"] == 0 and meta_res["created"] == meta_before_resume["created"]
               and meta_res["git_commit"] == meta_before_resume["git_commit"]
               and len(meta_res.get("resumed", [])) == n_resumed_before + 1,
               f"created {meta_before_resume['created']} -> {meta_res['created']}; "
               f"resumed {n_resumed_before} -> {len(meta_res.get('resumed', []))}")
        # Provenance cua lan chay A DAU TIEN (created_A/commit_A) da bi thay khi chuyen sang cau hinh B roi
        # quay lai - dung nhu ky vong: doi dau van tay => provenance moi.
        _check("provenance cua lan A dau tien da bi thay sau khi doi cau hinh roi quay lai",
               meta_res["created"] != created_A or meta_res["git_commit"] == commit_A)
        prov_truoc_doi_cfg = (meta_res["created"], meta_res["git_commit"])

        # --- 5. doi cau hinh -> chay lai toan bo ---
        ran["n"] = 0
        run_tracking.run_video_to_dir = counting
        try:
            rex.stage_track(expB, rows_for(expB), videos, skip_existing=True)
        finally:
            run_tracking.run_video_to_dir = orig
        meta_b2 = json.loads((out_root() / "run_meta.json").read_text(encoding="utf-8"))
        prev = meta_b2.get("previous", {})
        _check("doi cau hinh -> chay lai TOAN BO, status=complete, 'previous' giu provenance NGAY TRUOC do",
               ran["n"] == len(videos) and meta_b2["status"] == "complete"
               and (prev.get("created"), prev.get("git_commit")) == prov_truoc_doi_cfg
               and meta_b2["input_fingerprint"] != meta_res["input_fingerprint"],
               f"ran={ran['n']} created={meta_b2['created']} previous={prev.get('created')} "
               f"(mong doi {prov_truoc_doi_cfg[0]})")
    except AssertionError:
        failed += 1
    finally:
        clean()
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
