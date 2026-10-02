"""
test_pilot_chan_doan.py -- Kiem phan do pilot (vong 37) TRUOC khi chay pilot that: `train_detector.py --che-do pilot`
+ `pilot_chan_doan.py`. Fixture tong hop (khong dung du lieu nghien cuu); weights COCO that cua du an.

  P1  Cau hinh pilot: ke thua NGUYEN cau hinh goc (sha256 noi dung LF), CHI doi `epochs`; sai hash / them khoa / pilot
      khong doi gi -> tu choi; CLI: thieu --xac-nhan-pilot, cau hinh pilot cho smoke, --batch voi pilot, cờ A/B sai che do.
  P2  `on_fit_epoch_end` lan thu hai trong ultralytics `final_eval` KHONG ghi de val_s / metrics epoch cuoi (ma moi);
      cung chuoi su kien chay tren ma a22a6f5 -> tai hien loi ghi de (ky vong).
  P3  Chup / so sanh trang thai: BN buffer, che do train/eval, RNG python/numpy/torch, co cudnn -> phat hien; callback
      chi tieu thu RNG -> khoi phuc, khong loi; callback sua buffer -> LoiPilot.
  P4  Luot pilot THAT tren fixture (GPU neu co): diem do truoc hoc dung TRUOC cap nhat dau (ema.updates 0, optimizer rong,
      khong gradient, model == EMA); bang chung nap/remap khop dong 'Transferred N/M' cua ultralytics; moi phep do khong doi
      trang thai; LR / momentum / accumulate tung batch dung cong thuc warmup cua ma; chi so batch co cap nhat khop mo phong
      dieu kien `ni - last_opt_step >= accumulate`; 2 epoch + final_eval ghi rieng; train_s khong gom phep do; do checkpoint.
  P5  A/B tren CPU: hai luot KHONG do (A1, A2) va mot luot CO do (B), cung cau hinh -> neu A1 trung A2 tung bit (loss tung
      batch + trong so last.pt) thi B phai trung A1 (phep do khong lam doi qua trinh hoc); A1 != A2 -> SKIP (nen tang
      khong tat dinh, khong ket luan).

    python src/test_pilot_chan_doan.py [--bo-cpu-ab]
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import config  # noqa: E402

PY = sys.executable
KQ: list[tuple[str, bool]] = []
SKIP: list[tuple[str, str]] = []


def _check(ten: str, ok: bool, chi_tiet: str = "") -> bool:
    KQ.append((ten, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {ten}" + (f"  ({chi_tiet[:500]})" if chi_tiet else ""), flush=True)
    return bool(ok)


def _skip(ten: str, ly_do: str) -> None:
    SKIP.append((ten, ly_do))
    print(f"  [SKIP] {ten}  ({ly_do}) -- CHUA KIEM, khong tinh PASS", flush=True)


def _chay(nhom: str, fn) -> None:
    print(f"\n== {nhom}", flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        _check(f"{nhom}: loi khong mong doi {type(e).__name__}: {e}", False)
        traceback.print_exc()


def _yaml(p: Path, d: dict) -> Path:
    import yaml
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return p


def _sha_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cfg_goc_test(imgsz=160, workers=0) -> dict:
    import yaml
    c = yaml.safe_load((config.CONFIGS_DIR / "phase2" / "detector" / "train_bdd5.yaml").read_text(encoding="utf-8"))
    c["model"]["yolov8n"]["weights"] = str(config.MODELS_DIR / "yolov8n.pt")
    c["model"]["yolo11n"]["weights"] = str(config.MODELS_DIR / "candidates" / "yolo11n.pt")
    c["huan_luyen"]["imgsz"] = imgsz
    c["huan_luyen"]["workers"] = workers
    return c


def tao_cap_cfg(tmp: Path, ten: str, goc: dict, pilot: dict | None = None, them: dict | None = None) -> tuple[Path, Path]:
    pg = _yaml(tmp / f"{ten}_goc.yaml", goc)
    y = {"ke_thua": str(pg), "ke_thua_sha256": _sha_lf(pg), "pilot": pilot or {"epochs": 2}}
    y.update(them or {})
    return pg, _yaml(tmp / f"{ten}_pilot.yaml", y)


# =========================================================================== P1
def nhom_p1(tmp: Path):
    import train_detector as tr
    cfg = tr.doc_cfg(config.CONFIGS_DIR / "phase2" / "detector" / "train_bdd5_pilot.yaml")
    _check("P1a cau hinh pilot that: ke thua train_bdd5.yaml (sha256 LF a466a4fa...), pilot = {epochs: 2}",
           cfg["_ke_thua"]["sha256_noi_dung_lf"].startswith("a466a4fa") and cfg["pilot"] == {"epochs": 2}
           and cfg["huan_luyen"]["batch"] == 8 and cfg["smoke"] == {"epochs": 1, "patience": 1}, str(cfg["_ke_thua"]))
    goc = cfg_goc_test()

    def tu_choi(fn, tu=""):
        try:
            fn()
            return False, "CHAP NHAN"
        except tr.LoiCauHinh as e:
            return (tu in str(e)), str(e)[:200]
    _, pp = tao_cap_cfg(tmp, "p1b", goc)
    y = __import__("yaml").safe_load(pp.read_text(encoding="utf-8"))
    y["ke_thua_sha256"] = "0" * 64
    _check("P1b sha256 cau hinh goc sai -> tu choi", *tu_choi(lambda: tr.doc_cfg(_yaml(tmp / "p1b_sai.yaml", y)), "sha256"))
    _, pp = tao_cap_cfg(tmp, "p1c", goc, pilot={"epochs": 2, "lr0": 0.02})
    _check("P1c khoi pilot doi them lr0 -> tu choi (chi duoc doi epochs)", *tu_choi(lambda: tr.doc_cfg(pp), "chi duoc doi"))
    _, pp = tao_cap_cfg(tmp, "p1d", goc, them={"huan_luyen": {"batch": 16}})
    _check("P1d cau hinh pilot tu khai bao huan_luyen -> tu choi (moi khoa khac phai ke thua)",
           *tu_choi(lambda: tr.doc_cfg(pp), "chi duoc co"))
    _, pp = tao_cap_cfg(tmp, "p1e", goc)
    c = tr.doc_cfg(pp)
    khac = tr.kiem_pilot_chi_doi_epoch(c, tmp / "d.yaml", tmp / "runs", "x", "0")
    _check("P1e tham so hieu luc pilot so voi smoke: CHI khac epochs 1 -> 2", khac == {"epochs": [1, 2]}, str(khac))
    _, pp = tao_cap_cfg(tmp, "p1f", goc, pilot={"epochs": 1})
    c = tr.doc_cfg(pp)
    _check("P1f pilot khong doi gi so voi smoke (epochs 1) -> tu choi",
           *tu_choi(lambda: tr.kiem_pilot_chi_doi_epoch(c, tmp / "d.yaml", tmp / "runs", "x", "0"), "CHI khac"))
    base = [PY, str(SRC / "train_detector.py"), "--model", "yolov8n", "--project", str(tmp / "runs_p1"),
            "--ket-qua-dir", str(tmp / "kq_p1")]
    _, pp = tao_cap_cfg(tmp, "p1g", goc)
    r = subprocess.run(base + ["--config", str(pp), "--che-do", "pilot"], capture_output=True, text=True)
    _check("P1g --che-do pilot thieu --xac-nhan-pilot -> exit 1", r.returncode == 1 and "xac-nhan-pilot" in r.stderr,
           r.stderr[-200:])
    r = subprocess.run(base + ["--config", str(pp), "--che-do", "smoke"], capture_output=True, text=True)
    _check("P1h cau hinh pilot dung cho --che-do smoke -> exit 2", r.returncode == 2 and "cau hinh pilot" in r.stderr,
           r.stderr[-200:])
    r = subprocess.run(base + ["--config", str(pp), "--che-do", "pilot", "--xac-nhan-pilot", "--batch", "4"],
                       capture_output=True, text=True)
    _check("P1i --batch voi pilot -> exit 1 (pilot dung cau hinh)", r.returncode == 1, r.stderr[-200:])
    r = subprocess.run(base + ["--config", str(tmp / "p1g_goc.yaml"), "--che-do", "smoke", "--khong-do-chan-doan"],
                       capture_output=True, text=True)
    _check("P1j --khong-do-chan-doan ngoai pilot -> exit 1", r.returncode == 1, r.stderr[-200:])
    _check("P1k khong tao thu muc luot / ket qua o cac lan tu choi",
           not (tmp / "runs_p1").exists() and not (tmp / "kq_p1").exists())


# =========================================================================== P2
class _ModelGia:
    def __init__(self):
        self.cb: dict[str, list] = {}

    def add_callback(self, ev, fn):
        self.cb.setdefault(ev, []).append(fn)

    def chay(self, ev, x):
        for fn in self.cb.get(ev, []):
            fn(x)


class _LoaderGia:
    num_workers = 0
    dataset = [0] * 16

    def __len__(self):
        return 2


def _trainer_gia():
    pg = [{"lr": 0.01, "initial_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True, "param_group": g,
           "params": [0]} for g in ("weight", "bn", "bias")]
    return SimpleNamespace(amp=False, batch_size=8, accumulate=8, optimizer=SimpleNamespace(param_groups=pg),
                           scheduler=SimpleNamespace(), train_loader=_LoaderGia(),
                           test_loader=SimpleNamespace(dataset=[0] * 4), data={"nc": 5, "names": {0: "car"}},
                           save_dir="x", args=SimpleNamespace(a=1), tloss={"box": 1.0, "cls": 2.0}, epoch=0,
                           metrics={"m": 0.1}, fitness=0.1)


def _chuoi_su_kien(mod) -> dict:
    import time
    rec = {}
    m = _ModelGia()
    mod.gan_callback(m, rec)
    t = _trainer_gia()
    m.chay("on_train_start", t)
    for ep in range(2):
        t.epoch = ep
        for _ in range(2):
            m.chay("on_train_batch_end", t)
        m.chay("on_train_epoch_end", t)
        time.sleep(0.05)
        t.metrics, t.fitness = {"m": 0.1 * (ep + 1)}, 0.1 * (ep + 1)
        m.chay("on_fit_epoch_end", t)
    time.sleep(0.3)                                  # luu checkpoint + final_eval
    t.epoch, t.metrics = 2, {"m": 0.999}
    m.chay("on_fit_epoch_end", t)                    # final_eval: epoch tam +1
    return rec


def nhom_p2(tmp: Path):
    import train_detector as tr
    rec = _chuoi_su_kien(tr)
    e = rec.get("epoch", [])
    _check("P2a ma moi: 2 ban ghi epoch; lan goi final_eval KHONG ghi de val_s / metrics epoch 2; ghi rieng",
           len(e) == 2 and e[1]["metrics"] == {"m": 0.2} and e[1]["val_s"] < 0.25
           and rec.get("danh_gia_cuoi_ultralytics", {}).get("metrics") == {"m": 0.999}
           and not rec.get("loi_callback"),
           f"val_s {[x.get('val_s') for x in e]} metrics {[x.get('metrics') for x in e]} "
           f"cuoi {rec.get('danh_gia_cuoi_ultralytics')} loi_cb {rec.get('loi_callback')}")
    cu = tmp / "train_detector_a22a6f5.py"
    cu.write_bytes(subprocess.run(["git", "show", "a22a6f5:src/train_detector.py"], capture_output=True, check=True,
                                  cwd=SRC.parent).stdout)
    spec = importlib.util.spec_from_file_location("train_detector_a22a6f5", cu)
    mcu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcu)
    rc = _chuoi_su_kien(mcu)
    ec = rc.get("epoch", [])
    _check("P2b ma a22a6f5 (ky vong tai hien loi): final_eval GHI DE metrics + keo dai val_s cua epoch 2",
           len(ec) == 2 and ec[1]["metrics"] == {"m": 0.999} and ec[1]["val_s"] >= 0.3,
           f"val_s {[x.get('val_s') for x in ec]} metrics {[x.get('metrics') for x in ec]}")


# =========================================================================== P3
def nhom_p3(tmp: Path):
    import torch
    import torch.nn as nn

    import pilot_chan_doan as pcd
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    tr = SimpleNamespace(model=m, ema=SimpleNamespace(ema=copy.deepcopy(m), updates=0), optimizer=opt, scheduler=sch,
                         scaler=None, accumulate=8, epoch=0, best_fitness=None, stop=False)
    a = pcd.chup_trang_thai(tr)
    _check("P3a chup hai lan lien tiep, khong doi gi -> giong het", not pcd.so_sanh_trang_thai(a, pcd.chup_trang_thai(tr)))

    def phat_hien(sua, khoi_phuc, ten_muc):
        s0 = pcd.chup_trang_thai(tr)
        sua()
        k = pcd.so_sanh_trang_thai(s0, pcd.chup_trang_thai(tr))
        khoi_phuc()
        return any(ten_muc in x for x in k), str(k)[:200]
    bn = m[1]
    luu = bn.running_mean.clone()
    _check("P3b sua thong ke BatchNorm (running_mean) -> phat hien",
           *phat_hien(lambda: bn.running_mean.add_(1.0), lambda: bn.running_mean.copy_(luu), "model_tham_so_buffer"))
    _check("P3c doi che do train -> eval -> phat hien", *phat_hien(lambda: m.eval(), lambda: m.train(), "model_che_do_train"))
    st = random.getstate()
    _check("P3d tieu thu RNG python -> phat hien", *phat_hien(lambda: random.random(), lambda: random.setstate(st),
                                                             "rng_python"))
    snp = np.random.get_state()
    _check("P3e tieu thu RNG numpy -> phat hien", *phat_hien(lambda: np.random.rand(), lambda: np.random.set_state(snp),
                                                            "rng_numpy"))
    stt = torch.get_rng_state()
    _check("P3f tieu thu RNG torch -> phat hien", *phat_hien(lambda: torch.rand(1), lambda: torch.set_rng_state(stt),
                                                            "rng_torch"))
    b0 = torch.backends.cudnn.benchmark
    _check("P3g doi co cudnn.benchmark -> phat hien",
           *phat_hien(lambda: setattr(torch.backends.cudnn, "benchmark", not b0),
                      lambda: setattr(torch.backends.cudnn, "benchmark", b0), "co"))
    gia = SimpleNamespace()
    s0 = pcd.chup_trang_thai(tr)
    pcd.GhiPilot._kiem_ca_callback(gia, tr, lambda: (random.random(), np.random.rand(), torch.rand(3)), "chi rng")
    _check("P3h callback chi tieu thu RNG -> RNG duoc khoi phuc, trang thai giong truoc, khong LoiPilot",
           not pcd.so_sanh_trang_thai(s0, pcd.chup_trang_thai(tr)))
    try:
        pcd.GhiPilot._kiem_ca_callback(gia, tr, lambda: bn.running_var.mul_(2.0), "sua buffer")
        ok, ct = False, "khong loi"
    except pcd.LoiPilot as e:
        ok, ct = True, str(e)[:160]
    _check("P3i callback sua buffer BatchNorm -> LoiPilot (dung train)", ok, ct)


# =========================================================================== fixture + chay that
def tao_dataset_fixture(tmp: Path) -> Path:
    import bdd_yolo_export as bx
    import test_detector_5lop as t5
    t5.tao_moi_truong(tmp)
    cfg = t5.nap_cfg(tmp, t5.cfg_mau(tmp, "pf"))
    fs = bx.p_abs(cfg, cfg["file_split"])
    bx.tao_split(cfg, fs)
    bx.xuat_dataset(cfg, fs)
    return bx.p_abs(cfg, cfg["dau_ra"]) / "data.yaml"


def chay_pilot(tmp: Path, data: Path, ten: str, device: str, do: bool, goc: dict) -> tuple[int, dict, list[dict], str]:
    _, pp = tao_cap_cfg(tmp, f"cfg_{ten}", goc)
    lenh = [PY, str(SRC / "train_detector.py"), "--config", str(pp), "--model", "yolov8n", "--che-do", "pilot",
            "--xac-nhan-pilot", "--data", str(data), "--ten", ten, "--project", str(tmp / "runs"),
            "--ket-qua-dir", str(tmp / "kq"), "--device", device] + ([] if do else ["--khong-do-chan-doan"])
    r = subprocess.run(lenh, capture_output=True, text=True, encoding="utf-8", errors="replace")
    f = tmp / "kq" / f"{ten}.json"
    rec = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    fb = tmp / "kq" / f"{ten}_batch.csv"
    rows = list(csv.DictReader(open(fb, encoding="utf-8"))) if fb.is_file() else []
    return r.returncode, rec, rows, (r.stdout[-1500:] + r.stderr[-1500:])


def mo_phong_warmup(rows: list[dict], p: dict, nb: int, nw: int) -> tuple[list[str], list[int]]:
    """Doi chieu LR / momentum / accumulate tung batch voi cong thuc trong BaseTrainer._do_train; tra (loi, ni cap nhat
    theo mo phong dieu kien `ni - last_opt_step >= accumulate`)."""
    a = p["warmup_theo_ma"]["args"]
    E = int(a["epochs"])
    lf = lambda x: max(1 - x / E, 0) * (1.0 - a["lrf"]) + a["lrf"]  # noqa: E731
    loi, cap, last = [], [], -1
    acc = None
    for r in rows:
        ni, ep = int(r["ni"]), int(r["epoch"]) - 1
        if ni < nw:
            acc_md = max(1, int(np.interp(ni, [0, nw], [1, a["nbs"] / a["batch"]]).round()))
            for g in ("weight", "bn", "bias"):
                lr = float(np.interp(ni, [0, nw], [a["warmup_bias_lr"] if g == "bias" else 0.0, a["lr0"] * lf(ep)]))
                if abs(float(r[f"lr_{g}"]) - lr) > 1e-12:
                    loi.append(f"ni {ni} lr_{g} {r[f'lr_{g}']} != {lr}")
                mo = float(np.interp(ni, [0, nw], [a["warmup_momentum"], a["momentum"]]))
                if abs(float(r[f"momentum_{g}"]) - mo) > 1e-12:
                    loi.append(f"ni {ni} momentum_{g} {r[f'momentum_{g}']} != {mo}")
            acc = acc_md
        else:
            for g in ("weight", "bn", "bias"):
                lr = a["lr0"] * lf(ep)
                if abs(float(r[f"lr_{g}"]) - lr) > 1e-12:
                    loi.append(f"ni {ni} (sau warmup) lr_{g} {r[f'lr_{g}']} != {lr}")
        if acc is None:
            acc = max(round(a["nbs"] / a["batch"]), 1)
        if int(r["accumulate"]) != acc:
            loi.append(f"ni {ni} accumulate {r['accumulate']} != {acc}")
        if ni - last >= acc:
            cap.append(ni)
            last = ni
    return loi, cap


def nhom_p4(tmp: Path, data: Path):
    import torch
    dev = "0" if torch.cuda.is_available() else "cpu"
    code, rec, rows, out = chay_pilot(tmp, data, "p4_that", dev, True, cfg_goc_test())
    p = rec.get("pilot", {})
    _check(f"P4a luot pilot that tren fixture ({dev}) exit 0, tom tat DAT", code == 0 and p.get("tom_tat", {}).get("dat"),
           f"exit {code} {p.get('tom_tat')} {p.get('loi')} {out[-600:]}")
    if not p:
        return
    bc = p["truoc_hoc"]["bang_chung_khoi_tao"]
    o = bc["optimizer"]
    _check("P4b diem do truoc hoc: ema.updates 0, optimizer chua co trang thai, khong gradient, model == EMA, 5 lop dung "
           "thu tu", o["ema_updates"] == 0 and o["so_trang_thai"] == 0 and bc["so_tham_so_co_gradient"] == 0
           and bc["model_va_ema_giong_het"] and bc["ten_lop_model"] == {"0": "car", "1": "truck", "2": "bus",
                                                                       "3": "motorcycle", "4": "bicycle"}, str(o)[:300])
    n = bc["nap_trong_so"]
    tf = [x["dong"] for x in p["log_ultralytics"] if x["dong"].startswith("Transferred")]
    so = [int(s) for s in tf[0].split()[1].split("/")] if tf else None
    _check("P4c nap/remap doi chieu doc lap khop log ultralytics 'Transferred N/M': chep + remap = N, tong = M, khong "
           "tensor cung shape khac gia tri", so is not None and n["chep_nguyen"] + n["remap_hang_theo_ten"] == so[0]
           and n["tong_tensor_model"] == so[1] and not n["cung_shape_khac_gia_tri"]
           and n["chep_nguyen"] + n["remap_hang_theo_ten"] + n["khoi_tao_moi"] == n["tong_tensor_model"],
           f"log {tf} chep {n['chep_nguyen']} remap {n['remap_hang_theo_ten']} moi {n['khoi_tao_moi']}")
    do = p.get("do", {})
    can = ["truoc_hoc_model", "sau_epoch1_model", "sau_epoch1_ema", "sau_epoch2_model", "sau_epoch2_ema",
           "checkpoint_last", "checkpoint_best"]
    bb = [do[k]["bat_bien"]["khong_doi_sau_khoi_phuc"] for k in can[:5] if k in do]
    _check("P4d du 7 phep do; 5 phep do trong luc train + 3 callback: trang thai KHONG doi sau khoi phuc",
           all(k in do for k in can) and len(bb) == 5 and all(bb)
           and p["truoc_hoc"]["bat_bien_ca_callback"]["khong_doi_sau_khoi_phuc"]
           and all(e["bat_bien_ca_callback"]["khong_doi_sau_khoi_phuc"] for e in p["epoch"]),
           f"co {sorted(do)}")
    w = p["warmup_theo_ma"]
    nb, nw = w["so_batch_moi_epoch"], w["so_buoc_warmup_nw"]
    loi, cap = mo_phong_warmup(rows, p, nb, nw)
    cap_log = [ni for e in p["epoch"] for ni in e["ni_co_cap_nhat"]]
    _check("P4e 2 epoch x nb batch; nw theo ma = round(min(1, epochs-1) * nb); LR / momentum / accumulate tung batch dung "
           "cong thuc warmup; batch co cap nhat (ema.updates tang) khop mo phong",
           len(rows) == 2 * nb and nw == nb and not loi and cap == cap_log
           and int(rows[-1]["ema_updates"]) == len(cap_log),
           f"nb {nb} nw {nw} loi {loi[:3]} cap_md {cap[:12]} cap_log {cap_log[:12]}")
    e = rec.get("epoch", [])
    _check("P4f runner: 2 ban ghi epoch, final_eval ghi rieng (khong ghi de epoch 2)",
           len(e) == 2 and "danh_gia_cuoi_ultralytics" in rec and "final_eval_ultralytics" in p,
           f"{len(e)} epoch, cuoi {rec.get('danh_gia_cuoi_ultralytics', {}).get('thoi_gian_tu_het_train_epoch_cuoi_s')}")
    moc = {(m["ten"], m.get("epoch")): m["t"] for m in p["moc"]}
    t_sau_do1 = moc.get(("het_do_sau_epoch", 1))
    t_het2 = moc.get(("on_train_epoch_end", 2))
    lech = abs(e[1]["train_s"] - (t_het2 - t_sau_do1)) if len(e) == 2 and t_sau_do1 and t_het2 else None
    do1 = do["sau_epoch1_model"]["thoi_gian_tong_s"] + do["sau_epoch1_ema"]["thoi_gian_tong_s"]
    _check("P4g train_s epoch 2 tinh tu SAU phep do epoch 1 (khong gom thoi gian do)", lech is not None and lech < 0.2,
           f"lech {lech} s, phep do epoch 1 mat {do1:.1f} s")
    ep1 = p["epoch"][0]
    _check("P4h mosaic thuc te ghi theo epoch, khop du doan theo ma (epoch tat mosaic = epochs - close_mosaic = 0)",
           w["epoch_tat_mosaic_theo_ma"] == 0 and ep1.get("mosaic_thuc_te") is False
           and p["epoch"][1].get("mosaic_thuc_te") is False, str([x.get("mosaic_thuc_te") for x in p["epoch"]]))
    d0 = do["truoc_hoc_model"]
    tong = d0["detection"]["tong"]
    _check("P4i phep do dung quy tac danh gia chinh (conf 0,001, IoU 0,7, max_det 300, imgsz 640 cua evaluator) + co "
           "diem truoc nguong; so detection theo lop cong lai = tong",
           d0["quy_tac_du_doan"]["conf"] == 0.001 and d0["quy_tac_du_doan"]["iou_nms"] == 0.7
           and d0["quy_tac_du_doan"]["max_det"] == 300 and sum(d0["detection"]["theo_lop"].values()) == tong
           and d0["diem_truoc_nguong"]["max_moi_anh"]["n"] == d0["so_anh"], str(d0["diem_truoc_nguong"]["tien_xu_ly"]))
    ck = rec.get("checkpoint", {})
    _check("P4j checkpoint last/best nap lai dung 5 lop, train_args epochs = 2",
           {"last", "best"} <= set(ck) and all(c["dung_5_lop"] and not c["train_args_lech_cau_hinh"] for c in ck.values()))


def _trong_so_last(tmp: Path, ten: str) -> dict:
    import torch
    ck = torch.load(str(tmp / "runs" / ten / "weights" / "last.pt"), map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck["model"])
    return {k: v.detach().cpu().numpy().tobytes() for k, v in m.state_dict().items()}


def nhom_p5(tmp: Path, data: Path):
    goc = cfg_goc_test()
    kq = {}
    for ten, do in (("ab_a1", False), ("ab_a2", False), ("ab_b", True)):
        code, rec, rows, out = chay_pilot(tmp, data, ten, "cpu", do, goc)
        if code != 0:
            _check(f"P5 luot {ten} (CPU) exit 0", False, f"exit {code} {out[-500:]}")
            return
        cot_loss = [c for c in (rows[0] if rows else {}) if c == "loss_tong" or c.endswith("_loss")]
        if len(cot_loss) < 2:
            _check(f"P5 luot {ten}: batch CSV co cot loss (loss_tong + *_loss)", False, str(list(rows[0]) if rows else []))
            return
        kq[ten] = ([(r["ni"], *(r[c] for c in cot_loss), r["ema_updates"]) for r in rows], _trong_so_last(tmp, ten))
    a1, a2, b = kq["ab_a1"], kq["ab_a2"], kq["ab_b"]
    if a1 != a2:
        _skip("P5 A/B bat bien tren CPU", "hai luot KHONG do khac nhau tung bit -> nen tang khong tat dinh, khong ket luan")
        return
    _check("P5a hai luot KHONG do (A1, A2) trung tung bit: loss tung batch + trong so last.pt (nen tang tat dinh)", True)
    khac_t = [k for k in a1[1] if a1[1][k] != b[1].get(k)]
    _check("P5b luot CO do (B) trung A1 tung bit: loss tung batch va moi tensor last.pt -> phep do khong doi qua trinh hoc",
           a1[0] == b[0] and not khac_t, f"loss trung {a1[0] == b[0]}, tensor khac {khac_t[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bo-cpu-ab", action="store_true", help="bo nhom P5 (A/B CPU, ~3 luot train fixture)")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="test_pilot_"))
    try:
        _chay("P1. cau hinh pilot + CLI", lambda: nhom_p1(tmp))
        _chay("P2. final_eval khong ghi de epoch cuoi (ma moi) / tai hien loi (ma a22a6f5)", lambda: nhom_p2(tmp))
        _chay("P3. chup / so sanh / khoi phuc trang thai", lambda: nhom_p3(tmp))
        data = None

        def fx():
            nonlocal data
            data = tao_dataset_fixture(tmp / "fx")
        _chay("Fixture dataset (bo xuat that, hop dong v2)", fx)
        if data is not None:
            _chay("P4. luot pilot that tren fixture", lambda: nhom_p4(tmp, data))
            if a.bo_cpu_ab:
                _skip("P5 A/B CPU", "--bo-cpu-ab")
            else:
                _chay("P5. A/B bat bien tren CPU (do / khong do)", lambda: nhom_p5(tmp, data))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = sum(ok for _, ok in KQ)
    print(f"\nKET QUA: {n}/{len(KQ)} PASS" + (f", {len(SKIP)} SKIP (CHUA KIEM): {[s for s, _ in SKIP]}" if SKIP else ""))
    return 0 if KQ and n == len(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
