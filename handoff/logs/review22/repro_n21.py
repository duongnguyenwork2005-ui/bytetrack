# -*- coding: utf-8 -*-
"""Tai hien N21-F1/F2/F3 (CODEX_REVIEW_9f210e2) qua ENTRY POINT THAT `src/nano_live.py`, detector gia lap.
Chi QUAN SAT va in so lieu (khong assert) -> chay duoc ca truoc lan sau khi sua. Fixture: anh BMP + cache tong hop
(tai dung ham cua test_nano_stream). KHONG phai benchmark: toc do fixture khong mang y nghia hieu nang.

    python handoff/logs/review22/repro_n21.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import det_cache  # noqa: E402
import test_nano_stream as t  # noqa: E402

PY = sys.executable
ENV = {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8"}
CHAM = ("import sys, time, runpy; sys.path.insert(0, 'src'); import run_tracking as rt; _f = rt.load_tracker_cfg\n"
        "def _cham(*a, **k):\n    time.sleep(1.0)\n    return _f(*a, **k)\n"
        "rt.load_tracker_cfg = _cham; sys.argv = ['src/nano_live.py'] + sys.argv[1:]\n"
        "runpy.run_path('src/nano_live.py', run_name='__main__')\n")


def meta(out):
    p = out / "run_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def f1(tmp, anh, cache, stop_sau=None):
    ten = "F1_dung" if stop_sau else "F1"
    out = tmp / ten
    args = ["--source", str(anh), "--fps", "10", "--video-name", "seq", "--gia-lap-cache", str(cache),
            "--gia-lap-video", "seq", "--tracker", "cv", "--out", str(out), "--warmup-frames", "2", "--progress-every", "0",
            "--gia-lap-cham-ms", "20"]     # moi frame ~20 ms: lon hon do phan giai time.monotonic tren Windows (~15,6 ms)
    if stop_sau:
        (out).mkdir(parents=True)
        args += ["--stop-file", str(tmp / "STOP_F1")]
        (tmp / "STOP_F1").write_text("x")      # co san -> ma cu tu choi; ma moi: tuy hop dong
    r = subprocess.run([PY, "-c", CHAM, *args], cwd=ROOT, env=ENV, capture_output=True, text=True)
    m = meta(out)
    import pandas as pd
    pl = pd.read_csv(out / "pipeline" / "seq.csv") if (out / "pipeline" / "seq.csv").exists() else None
    th = m.get("thoi_gian", {})
    print(f"[F1{' dung' if stop_sau else ''}] exit {r.returncode} trang_thai {m.get('trang_thai')} frame {m.get('dem', {}).get('n_frame_da_xu_ly')}")
    if pl is not None and len(pl):
        bao_phu = float(pl["t_xong_cha"].max() - pl["t_bat_dau_con"].min())
        ten_w, ten_f = ("xu_ly_s", "fps_xu_ly") if "xu_ly_s" in th else ("wall_vong_lap_s", "fps_vong_lap")
        wv, fps = th.get(ten_w), th.get(ten_f)
        print(f"     {ten_w} bao cao = {wv:.6f} s | tu bat dau doc frame 1 den xong tracker frame cuoi = {bao_phu:.6f} s")
        print(f"     FPS bao cao = {fps:.2f} | FPS theo khoang bao phu = {len(pl) / bao_phu:.2f}")
        moc = th.get("moc_monotonic") or {}
        if moc:
            print(f"     cua so do (monotonic): gui BAT DAU {moc['gui_bat_dau']:.6f} -> ket thuc {moc['ket_thuc_xu_ly']:.6f} | "
                  f"frame 1 bat dau doc {pl['t_bat_dau_con'].min():.6f} (>= BAT DAU: {pl['t_bat_dau_con'].min() >= moc['gui_bat_dau']}), "
                  f"frame cuoi xong {pl['t_xong_cha'].max():.6f} (<= ket thuc: {pl['t_xong_cha'].max() <= moc['ket_thuc_xu_ly']}) | "
                  f"chuan_bi_cha_s {th.get('chuan_bi_cha_s'):.3f}")
    return r.returncode


def f2(tmp, anh, cache):
    root = tmp / "custom root" / "det cache"
    out = tmp / "F2"
    r = subprocess.run([PY, "src/nano_live.py", "--source", str(anh), "--fps", "10", "--video-name", "seq",
                        "--gia-lap-cache", str(cache), "--gia-lap-video", "seq", "--tracker", "cv", "--out", str(out),
                        "--save-cache", "review_custom", "--cache-root", str(root), "--progress-every", "0"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True)
    m = meta(out)
    print(f"[F2] live exit {r.returncode}; NPZ o root tuy chon: {(root / 'review_custom' / 'seq.npz').exists()}")
    argv = [PY if x == "python" else "REPRO-F2" if x == "<split>" else "rep-f2" if x == "<ten>" else x for x in m["replay_argv"]]
    print(f"     replay_argv: {m['replay_argv']}")
    rr = subprocess.run(argv, cwd=ROOT, env=ENV, capture_output=True, text=True)
    loi = [ln for ln in (rr.stdout + rr.stderr).splitlines() if "Error" in ln][-1:]
    print(f"     chay NGUYEN replay_argv -> exit {rr.returncode} {loi}")
    if rr.returncode == 0:
        rep = ROOT / "data/processed/trackers/REPRO-F2/rep-f2"
        p = subprocess.run([PY, "src/nano_parity.py", "--ref", str(rep), "--test", str(out)], cwd=ROOT, env=ENV,
                           capture_output=True, text=True)
        print(f"     nano_parity live vs replay: exit {p.returncode} {p.stdout.strip().splitlines()[-1:]}")
    shutil.rmtree(ROOT / "data/processed/trackers/REPRO-F2", ignore_errors=True)
    return rr.returncode


def f3(tmp, anh, cache):
    for ten, extra in (("F3_yaml_thieu", ["--tracker-yaml", str(tmp / "khong_co.yaml")]),
                       ("F3_yaml_hong", ["--tracker-yaml", str(tmp / "hong.yaml")]),
                       ("F3_set_sai", ["--tracker", "cv", "--set", "khong_co_dau_bang"])):
        (tmp / "hong.yaml").write_text("tracker_type: [khong: dong\n", encoding="utf-8")
        out = tmp / ten
        r = subprocess.run([PY, "src/nano_live.py", "--source", str(anh), "--fps", "10", "--video-name", "seq",
                            "--gia-lap-cache", str(cache), "--gia-lap-video", "seq", "--out", str(out), *extra,
                            "--progress-every", "0"], cwd=ROOT, env=ENV, capture_output=True, text=True)
        dong = (r.stdout + r.stderr).strip().splitlines()
        print(f"[{ten}] exit {r.returncode} | detector.pid co: {(out / 'detector.pid').exists()} | run_meta co: "
              f"{(out / 'run_meta.json').exists()} | dong cuoi: {dong[-1][:150] if dong else ''}")


def main():
    import time
    ci = time.get_clock_info("monotonic")
    print(f"# may: {sys.platform}, time.monotonic: {ci.implementation}, do phan giai {ci.resolution * 1e3:.3f} ms")
    tmp = Path(tempfile.mkdtemp(prefix="repro_n21_"))
    try:
        anh = t.tao_anh(tmp / "anh", n=8)
        cache = t.tao_cache_gia_lap(tmp / "cache", n=8)
        f1(tmp, anh, cache)
        f2(tmp, anh, cache)
        f3(tmp, anh, cache)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for p in (det_cache.CACHE_ROOT).glob("review_custom*"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
