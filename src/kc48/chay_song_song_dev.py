"""
chay_song_song_dev.py -- R48D: chay buoc `track` cua configs/phase2/experiments/kc48_pf_dev.yaml SONG SONG theo
(tracker, seed), moi lenh < 10 phut, tiep tuc duoc (giao thuc 04 muc 5.1, 9).

Moi don vi (tracker, seed) = mot tien trinh con goi DUNG `chay_thi_nghiem_kc48.chay_runner` (-> run_experiment.main,
goc config.PROCESSED_DIR doi sang results/phase2/kc48_pf_dev/_processed/) voi mot ban cau hinh LOC chi chua tracker do
(ghi o results/phase2/kc48_pf_dev/_exp_loc/, cung `name: kc48_pf_dev` -> cung ten thu muc run nhu cau hinh day du).
Ket qua tracking khong phu thuoc thu tu hay so tien trinh (rng PF = f(seed, video, track_id)); THOI GIAN trong
summary.csv bi anh huong boi tai song song -> chi dung de du toan, khong la chi so.
Don vi da xong (du video_meta cho 12 video) bi bo qua; runner tu kiem dau van tay tung video.
Buoc eval chay sau, tuan tu, bang wrapper voi cau hinh day du.

    python src/kc48/chay_song_song_dev.py --track [--so-tien-trinh 4] [--toi-da-giay 420]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import chay_thi_nghiem_kc48 as wr  # noqa: E402

EXP = "configs/phase2/experiments/kc48_pf_dev.yaml"


def don_vi(exp: dict) -> list[tuple[dict, int | None]]:
    out = []
    for t in exp["trackers"]:
        for s in t.get("seeds", [None]):
            out.append((t, s))
    return out


def ten_dir(exp: dict, t: dict, s) -> str:
    return f"p2-{exp['name']}-A-{t['name']}" + (f"-s{s}" if s is not None else "")


def da_xong(goc_moi: Path, exp: dict, t: dict, s, videos: list[str]) -> bool:
    import run_state
    out_root = goc_moi / "trackers" / exp["split_name"] / ten_dir(exp, t, s)
    return all(run_state.video_meta_path(out_root, v).exists() for v in videos)


def chay_mot(path_loc: str) -> int:
    os.chdir(ROOT)
    exp = wr.doc_exp(path_loc)
    out_dir = ROOT / "results" / "phase2" / exp["name"]
    return wr.chay_runner(path_loc, exp, "track", [], out_dir, out_dir / "_processed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--mot", default=None, help="(noi bo) chay mot cau hinh loc")
    ap.add_argument("--so-tien-trinh", type=int, default=4)
    ap.add_argument("--toi-da-giay", type=float, default=420.0)
    a = ap.parse_args()
    if a.mot:
        return chay_mot(a.mot)
    os.chdir(ROOT)
    exp = wr.doc_exp(EXP)
    videos = wr.videos_of(exp)
    out_dir = ROOT / "results" / "phase2" / exp["name"]
    goc_moi = out_dir / "_processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    wr.chuan_bi_gt(exp, videos, goc_moi, out_dir)                # mot lan, tuan tu (tranh tranh chap khi sao GT)
    raw = yaml.safe_load(Path(EXP).read_text(encoding="utf-8"))
    loc_dir = out_dir / "_exp_loc"
    log_dir = out_dir / "_log"
    loc_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    cho = [(t, s) for t, s in don_vi(exp) if not da_xong(goc_moi, exp, t, s, videos)]
    print(f"{len(videos)} video; {len(don_vi(exp))} don vi; con {len(cho)}", flush=True)
    t0 = time.perf_counter()
    dang: dict = {}
    while cho or dang:
        while cho and len(dang) < a.so_tien_trinh and time.perf_counter() - t0 <= a.toi_da_giay:
            t, s = cho.pop(0)
            d = dict(raw)
            tt = {k: v for k, v in t.items()}
            if s is not None:
                tt["seeds"] = [s]
            d["trackers"] = [tt]
            ten = ten_dir(exp, t, s)
            p = loc_dir / f"kc48_pf_dev__{ten}.yaml"
            p.write_text("# cau hinh LOC tu dong (chay_song_song_dev.py), khong sua tay\n" +
                         yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
            f = open(log_dir / f"{ten}.log", "w", encoding="utf-8")
            pr = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--mot", p.relative_to(ROOT).as_posix()],
                                  cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            dang[pr] = (ten, f, time.perf_counter())
        if not dang:
            break
        time.sleep(1.0)
        for pr in [q for q in dang if q.poll() is not None]:
            ten, f, ts = dang.pop(pr)
            f.close()
            dong = {"don_vi": ten, "exit": pr.returncode, "giay": round(time.perf_counter() - ts, 1),
                    "so_tien_trinh": a.so_tien_trinh, "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with open(out_dir / "nhat_ky_song_song.jsonl", "a", encoding="utf-8") as g:
                g.write(json.dumps(dong, ensure_ascii=False) + "\n")
            print(dong, flush=True)
            if pr.returncode != 0:
                print(f"LOI o {ten}: xem {log_dir / (ten + '.log')}")
    con = [ten_dir(exp, t, s) for t, s in don_vi(exp) if not da_xong(goc_moi, exp, t, s, videos)]
    print(f"{time.perf_counter() - t0:.0f} s; con {len(con)} don vi: {con}")
    return 3 if con else 0


if __name__ == "__main__":
    raise SystemExit(main())
