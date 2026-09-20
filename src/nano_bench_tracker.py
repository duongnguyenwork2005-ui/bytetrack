"""
nano_bench_tracker.py -- Do chi phi TRACKER tu detection cache tren Jetson Nano, theo giao thuc
da chot truoc o `scripts/nano/README.md` muc 4c.

CHAY TREN THIET BI. Khong chay detector: moi luot doc CUNG mot detection cache, nen hieu so giua
hai cau hinh la chi phi motion model + phan cap nhat cua bo loc, khong lan chi phi detector.

GIAO THUC (chot truoc khi do, khong doi giua chung):
  - Lich AB/BA CAN BANG: `--pairs N` cap, mot nua chay A->B, mot nua chay B->A. Thu tu co dinh
    lam nhiet do tich luy luon thiet cho cau hinh chay sau, nen phai can bang.
  - Moi LUOT co warm-up: bo `--warmup-frames` frame dau moi video, do phan con lai.
  - LUU MOI LUOT, ke ca luot bi throttling. Khong loai luot nao khoi du lieu tho.
  - Dai luong bao cao: TRUNG VI va IQR cua HIEU SO THEO CAP (B - A), khong phai hieu cua hai
    trung binh. Moi cap cho mot hieu so.
  - Loai (neu co) phai theo CAP: bo mot nua cap se lam lech hieu so ghep cap.
  - Ghi dieu kien chay moi luot: nvpmodel, jetson_clocks, nhiet do va tan so tu tegrastats.

DAI LUONG DO: `t_update_ms` -- TONG chi phi `tracker.update()`. `t_predict_ms` ghi kem de biet
chenh lech nam o buoc du doan hay buoc cap nhat. Ba cot KHONG phai ba pha doc lap
(xem `src/run_tracking.py`): `t_assoc_ms = t_update_ms - t_predict_ms` nen da GOM ca buoc cap nhat
cua bo loc va quan ly vong doi.

    python src/nano_bench_tracker.py --cache-id <id> --split-name <split> --videos v1 v2 \
        --a cv:bytetrack.yaml --b ekf:configs/bytetrack_ekf_ctrv.yaml \
        --conf-min 0.25 --fps 5 --track-buffer-sec 1.2 --pairs 6 \
        --out results/phase2/profile/nano_bench

KHONG suy ra so do neu chua chay tren thiet bi that: cong cu nay chi ghi cai no do duoc.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PY = sys.executable


# --------------------------------------------------------------------------- dieu kien thiet bi
def doc_lenh(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(khong chay duoc: {type(e).__name__})"


def nhiet_do_zone() -> dict:
    """Nhiet do tu sysfs (deg C). Rong neu khong phai Tegra."""
    out = {}
    for p in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        try:
            t = (p / "type").read_text().strip()
            v = int((p / "temp").read_text().strip()) / 1000.0
            out[t] = v
        except Exception:  # noqa: BLE001
            continue
    return out


def trang_thai_thiet_bi() -> dict:
    """Anh chup dieu kien chay. Tren may khong phai Jetson, cac truong se la '(khong chay duoc)'."""
    jc = "?"
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if p.exists():
        try:
            jc = p.read_text().strip()
        except Exception:  # noqa: BLE001
            pass
    return {"nvpmodel": doc_lenh(["nvpmodel", "-q"]), "cpu_governor": jc,
            "nhiet_do_C": nhiet_do_zone(), "thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S")}


class MauTegrastats(threading.Thread):
    """Lay mau `tegrastats` song song voi mot luot do. Khong co tegrastats -> khong co mau."""

    def __init__(self, chu_ky_ms: int = 500):
        super().__init__(daemon=True)
        self.chu_ky_ms, self.mau, self.p, self._dung = chu_ky_ms, [], None, False

    def run(self) -> None:
        try:
            self.p = subprocess.Popen(["tegrastats", "--interval", str(self.chu_ky_ms)],
                                      stdout=subprocess.PIPE, text=True)
        except Exception:  # noqa: BLE001
            return
        for line in self.p.stdout:                       # type: ignore[union-attr]
            if self._dung:
                break
            self.mau.append(line.strip())

    def dung(self) -> dict:
        self._dung = True
        if self.p:
            try:
                self.p.terminate()
            except Exception:  # noqa: BLE001
                pass
        return self.tom_tat()

    def tom_tat(self) -> dict:
        """Rut nhiet do GPU/CPU va tan so GPU tu cac dong tegrastats da lay."""
        if not self.mau:
            return {"n_mau": 0, "ghi_chu": "khong co tegrastats (khong phai Jetson hoac chua cai)"}
        temps, gr3d = [], []
        for ln in self.mau:
            for m in re.finditer(r"(\w+)@([\d.]+)C", ln):
                if m.group(1).lower() in ("gpu", "cpu", "tj", "thermal"):
                    temps.append(float(m.group(2)))
            m = re.search(r"GR3D_FREQ\s+(\d+)%(?:@(\d+))?", ln)
            if m and m.group(2):
                gr3d.append(int(m.group(2)))
        d: dict = {"n_mau": len(self.mau)}
        if temps:
            d.update(nhiet_max_C=max(temps), nhiet_tb_C=round(statistics.mean(temps), 1))
        if gr3d:
            d.update(gpu_freq_min_MHz=min(gr3d), gpu_freq_max_MHz=max(gr3d))
        return d


# --------------------------------------------------------------------------- mot luot do
def doc_timing(out_root: Path, videos: list[str], warmup: int) -> dict:
    """Gop `timing/<video>.csv`, BO `warmup` frame dau MOI video."""
    upd, pre, n = [], [], 0
    for v in videos:
        f = out_root / "timing" / f"{v}.csv"
        if not f.exists():
            raise FileNotFoundError(f"khong co {f}")
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows = rows[warmup:]
        for r in rows:
            upd.append(float(r["t_update_ms"]))
            pre.append(float(r["t_predict_ms"]))
        n += len(rows)
    if not n:
        raise ValueError(f"khong con frame nao sau khi bo {warmup} frame warm-up moi video")
    return {"n_frame_do": n,
            "update_ms_tb": statistics.mean(upd), "update_ms_trung_vi": statistics.median(upd),
            "predict_ms_tb": statistics.mean(pre), "predict_ms_trung_vi": statistics.median(pre)}


def chay_mot_luot(a, ten: str, yaml: str, videos: list[str], luot: int) -> dict:
    """Mot luot tracking tu cache + do thoi gian, kem anh chup dieu kien thiet bi."""
    tracker_name = f"{a.tag}-{ten}-r{luot}"
    cmd = [PY, "src/run_tracking.py", "--cache-id", a.cache_id, "--split-name", a.split_name,
           "--tracker-yaml", yaml, "--tracker-name", tracker_name,
           "--videos", *videos, "--conf-min", str(a.conf_min), "--max-det", str(a.max_det),
           "--fps", str(a.fps), "--elapsed-mode", a.elapsed_mode]
    if a.classes:
        cmd += ["--classes", *[str(c) for c in a.classes]]
    if a.track_buffer_sec is not None:
        cmd += ["--track-buffer-sec", str(a.track_buffer_sec)]

    truoc = trang_thai_thiet_bi()
    ts = MauTegrastats(a.tegrastats_ms)
    ts.start()
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    mau = ts.dung()
    sau = trang_thai_thiet_bi()

    d = {"luot": luot, "cau_hinh": ten, "tracker_name": tracker_name, "exit": r.returncode,
         "wall_s": round(wall, 3), "tegrastats": mau,
         "thiet_bi_truoc": truoc, "thiet_bi_sau": sau}
    if r.returncode != 0:
        d["loi"] = (r.stderr or r.stdout)[-600:]
        return d
    out_root = config.PROCESSED_DIR / "trackers" / a.split_name / tracker_name
    d.update(doc_timing(out_root, videos, a.warmup_frames))
    return d


# --------------------------------------------------------------------------- CLI
def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="Do tracker tu cache tren Nano (giao thuc muc 4c)")
    ap.add_argument("--cache-id", required=True)
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--a", required=True, help="cau hinh A dang ten:duong_dan_yaml (vd cv:bytetrack.yaml)")
    ap.add_argument("--b", required=True, help="cau hinh B dang ten:duong_dan_yaml")
    ap.add_argument("--conf-min", type=float, default=0.25)
    ap.add_argument("--classes", nargs="*", type=int, default=None)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--track-buffer-sec", type=float, default=1.2)
    ap.add_argument("--elapsed-mode", default="steps", choices=["skip", "steps"])
    ap.add_argument("--pairs", type=int, default=6,
                    help="so CAP (chia doi cho AB va BA). Giao thuc yeu cau it nhat 5; mac dinh 6 = 3+3")
    ap.add_argument("--warmup-frames", type=int, default=50,
                    help="bo bao nhieu frame dau MOI video truoc khi do")
    ap.add_argument("--nghi-giay", type=float, default=0.0,
                    help="cho giua cac luot de nhiet do on dinh (ghi lai trong ket qua)")
    ap.add_argument("--tegrastats-ms", type=int, default=500)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--out", default="results/phase2/profile/nano_bench")
    a = ap.parse_args()

    if a.pairs < 5:
        print(f"[LOI] giao thuc yeu cau it nhat 5 cap, nhan duoc --pairs {a.pairs}")
        return 2
    if a.pairs % 2:
        print(f"[LOI] --pairs phai CHAN de can bang AB/BA, nhan duoc {a.pairs}")
        return 2

    ten_a, yaml_a = a.a.split(":", 1)
    ten_b, yaml_b = a.b.split(":", 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("GIAO THUC DA CHOT TRUOC KHI DO (scripts/nano/README.md muc 4c)")
    print(f"  cau hinh A          : {ten_a}  ({yaml_a})")
    print(f"  cau hinh B          : {ten_b}  ({yaml_b})")
    print(f"  so cap              : {a.pairs}  ({a.pairs // 2} cap A->B, {a.pairs // 2} cap B->A)")
    print(f"  warm-up             : bo {a.warmup_frames} frame dau MOI video")
    print(f"  sequence            : {len(a.videos)} video -- {', '.join(a.videos[:4])}"
          + (" ..." if len(a.videos) > 4 else ""))
    print(f"  dai luong bao cao   : TRUNG VI + IQR cua hieu so THEO CAP ({ten_b} - {ten_a})")
    print(f"  luu moi luot        : co (khong loai luot nao khoi du lieu tho)")
    print("=" * 78)

    dev = trang_thai_thiet_bi()
    la_jetson = Path("/etc/nv_tegra_release").exists()
    print(f"thiet bi: {'Jetson (co /etc/nv_tegra_release)' if la_jetson else 'KHONG phai Jetson'}")
    if not la_jetson:
        print("  ! Dang chay tren may KHONG phai Jetson. Ket qua KHONG duoc bao cao la so Nano.")
    print()

    luots: list[dict] = []
    thu_tu = [(ten_a, yaml_a, ten_b, yaml_b)] * (a.pairs // 2) + \
             [(ten_b, yaml_b, ten_a, yaml_a)] * (a.pairs // 2)
    for i, (n1, y1, n2, y2) in enumerate(thu_tu, 1):
        print(f"[cap {i}/{a.pairs}] thu tu {n1} -> {n2}")
        for ten, yaml in ((n1, y1), (n2, y2)):
            d = chay_mot_luot(a, ten, yaml, a.videos, i)
            d["cap"] = i
            d["thu_tu_trong_cap"] = 1 if ten == n1 else 2
            luots.append(d)
            if d["exit"] != 0:
                print(f"    {ten:<6} THAT BAI exit={d['exit']}: {d.get('loi', '')[:150]}")
            else:
                print(f"    {ten:<6} update {d['update_ms_tb']:.3f} ms/frame "
                      f"(trung vi {d['update_ms_trung_vi']:.3f}), {d['n_frame_do']} frame do, "
                      f"wall {d['wall_s']:.1f}s, tegrastats {d['tegrastats'].get('n_mau', 0)} mau")
            if a.nghi_giay:
                time.sleep(a.nghi_giay)

    # --- hieu so THEO CAP ---
    hieu = []
    for i in range(1, a.pairs + 1):
        ca = [d for d in luots if d["cap"] == i and d["cau_hinh"] == ten_a and d["exit"] == 0]
        cb = [d for d in luots if d["cap"] == i and d["cau_hinh"] == ten_b and d["exit"] == 0]
        if len(ca) == 1 and len(cb) == 1:
            hieu.append({"cap": i, "thu_tu": "AB" if ca[0]["thu_tu_trong_cap"] == 1 else "BA",
                         "d_update_ms": cb[0]["update_ms_tb"] - ca[0]["update_ms_tb"],
                         "d_predict_ms": cb[0]["predict_ms_tb"] - ca[0]["predict_ms_tb"]})

    bc: dict = {"giao_thuc": {"pairs": a.pairs, "warmup_frames": a.warmup_frames,
                              "videos": a.videos, "A": ten_a, "B": ten_b,
                              "dai_luong": "trung vi + IQR cua hieu so theo cap (B - A)"},
                "la_jetson": la_jetson, "thiet_bi_ban_dau": dev,
                "luots": luots, "hieu_so_theo_cap": hieu}

    print()
    if len(hieu) < a.pairs:
        print(f"! chi {len(hieu)}/{a.pairs} cap hoan chinh (cap thieu mot nua KHONG duoc dung)")
    if hieu:
        ds = sorted(h["d_update_ms"] for h in hieu)
        # `statistics.quantiles(n=4)` can it nhat 2 diem; duoi 4 cap thi IQR khong co nghia
        # nen dung min/max. (Ternary gop mot dong truoc day bam sai ve -> q3 thanh tuple.)
        if len(ds) >= 4:
            qs = statistics.quantiles(ds, n=4)
            q1, q3 = qs[0], qs[2]
        else:
            q1, q3 = ds[0], ds[-1]
        print(f"HIEU SO {ten_b} - {ten_a} (update_ms/frame), {len(ds)} cap:")
        print(f"  trung vi = {statistics.median(ds):+.4f} ms")
        print(f"  IQR      = [{q1:+.4f}, {q3:+.4f}]")
        print(f"  min/max  = {ds[0]:+.4f} / {ds[-1]:+.4f}")
        bc["tong_hop"] = {"n_cap": len(ds), "trung_vi_d_update_ms": statistics.median(ds),
                          "iqr": [q1, q3], "min": ds[0], "max": ds[-1]}
    else:
        print("KHONG co cap hoan chinh nao -> khong bao cao so do.")

    f = out / f"{a.tag}_{ten_a}_vs_{ten_b}.json"
    f.write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {f}  (luu MOI luot, khong loai luot nao)")
    if not la_jetson:
        print("! Nhac lai: may nay KHONG phai Jetson -- khong duoc bao cao ket qua nay la so Nano.")
    return 0 if hieu else 1


if __name__ == "__main__":
    raise SystemExit(main())
