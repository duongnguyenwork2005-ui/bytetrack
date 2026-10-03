#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 39 -- KIEM THAT resume tren GPU voi dataset SMOKE da khoa (334/84 anh), ca YOLOv8n va YOLO11n. Day la kiem phan
mem cho tinh nang resume va bo giam sat; KHONG phai pilot chon tham so, KHONG phai training dai.

Chay tu goc repo:  python handoff/logs/review39/kiem_resume_gpu_smoke.py --ra handoff/logs/review39/06_kiem_resume_gpu.json
Moi model (tuan tu), MOI tien trinh deu chay qua `src/chay_giam_sat.py chay` (bam PID do cong cu tao):
  C  luot lien tuc 3 epoch     --che-do day-du --xac-nhan-train-dai --data smoke --config cau_hinh_kiem_resume_3epoch.yaml
  I  cung cau hinh; KILL tien trinh runner (PID tu trang_thai.json cua cong cu) + con ngay khi nhat ky ghi epoch 1 da luu
  R  `--tiep-tuc` I toi du 3 epoch
Thu muc: models/finetune_phase2/kiem_resume_v39/ (luot), results/phase2/kiem_resume_v39/ (ket qua runner),
handoff/logs/review39/05_gpu/<model>_<C|I|R>/ (log + RAM cua cong cu giam sat). Khong ghi de luot / ket qua cu.
So sanh theo ke hoach khoa truoc (01_ke_hoach_kiem_resume.md), cung ham voi src/test_tiep_tuc_huan_luyen.py.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))
import test_tiep_tuc_huan_luyen as tt  # noqa: E402

PY = sys.executable
L = Path(__file__).resolve().parent
CFG = L / "cau_hinh_kiem_resume_3epoch.yaml"
DATA = GOC / "data" / "yolo_bdd5_v1_smoke" / "data.yaml"
PROJECT = GOC / "models" / "finetune_phase2" / "kiem_resume_v39"
KQ = GOC / "results" / "phase2" / "kiem_resume_v39"


def lenh_runner(model, ten, *them):
    return [PY, "-u", str(GOC / "src" / "train_detector.py"), "--config", str(CFG), "--model", model, "--che-do",
            "day-du", "--xac-nhan-train-dai", "--data", str(DATA), "--ten", ten, "--project", str(PROJECT),
            "--ket-qua-dir", str(KQ), "--device", "0", *them]


def giam_sat(thu_muc: Path, lenh: list[str]) -> list[str]:
    return [PY, str(GOC / "src" / "chay_giam_sat.py"), "chay", "--thu-muc", str(thu_muc), "--khong-in", "--", *lenh]


def chay(thu_muc: Path, lenh: list[str]) -> int:
    return subprocess.run(giam_sat(thu_muc, lenh), cwd=GOC, timeout=3600).returncode


def chay_va_ngat(thu_muc: Path, lenh: list[str], run_dir: Path, timeout=1800) -> dict:
    import psutil
    p = subprocess.Popen(giam_sat(thu_muc, lenh), cwd=GOC)
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            if p.poll() is not None:
                return {"bi_kill": False, "exit_cong_cu": p.returncode, "ly_do": "tu thoat truoc khi ngat"}
            if any(s.get("loai") == "epoch_da_luu" and s.get("epoch") == 1 for s in tt.doc_nk(run_dir)):
                pid = json.loads((thu_muc / "trang_thai.json").read_text(encoding="utf-8"))["pid"]
                pr = psutil.Process(pid)
                con = pr.children(recursive=True)
                for c in con:
                    c.kill()
                pr.kill()
                rc = p.wait(timeout=300)
                return {"bi_kill": True, "pid_runner_bi_kill": pid, "so_con_bi_kill": len(con), "exit_cong_cu": rc,
                        "sau_s": round(time.time() - t0, 1)}
            time.sleep(0.2)
        return {"bi_kill": False, "ly_do": "het thoi gian"}
    finally:
        if p.poll() is None:
            p.kill()


def chep_bang_chung(model: str) -> dict:
    """Chep file NHO cua luot (nhat ky, theo batch, results, args) + JSON runner vao 05_gpu/<model>_luot/ de reviewer
    tinh lai; KHONG chep checkpoint (.pt)."""
    dich = L / "05_gpu" / f"{model}_luot"
    dich.mkdir(exist_ok=True)
    da_chep = {}
    for t in ("C", "I"):
        for f in ("nhat_ky_luot.jsonl", "theo_batch.csv", "results.csv", "args.yaml"):
            p = PROJECT / f"{model}_{t}" / f
            if p.is_file():
                shutil.copyfile(p, dich / f"{t}__{f}")
                da_chep[f"{t}__{f}"] = tt.sha(p)
    for p in sorted(KQ.glob(f"{model}_[CI]*.json")):
        shutil.copyfile(p, dich / f"ket_qua_runner__{p.name}")
        da_chep[f"ket_qua_runner__{p.name}"] = tt.sha(p)
    return da_chep


def danh_gia(out: dict) -> dict:
    """Tieu chi DAT/KHONG DAT da khoa truoc trong 01_ke_hoach_kiem_resume.md muc G; K2/K6 chi bao."""
    k, gs = out["kiem"], out["kiem"]["giam_sat"]
    d = {
        "C_exit_0": out["C"]["exit"] == 0,
        "I_bi_kill_sau_epoch_1": out["I"].get("bi_kill") is True and out["I"]["nhat_ky"] == [("bat_dau", None),
                                                                                             ("epoch_da_luu", 1)],
        "R_exit_0": out["R"]["exit"] == 0,
        "K3_runner_tu_kiem_dat": len(k["K3_kiem_khoi_phuc"]) >= 13 and all(k["K3_kiem_khoi_phuc"].values()),
        "K3_nguon_e001": k["K3_nguon"]["chon_sha"] == k["K3_nguon"]["e001_sha"] and k["K3_nguon"]["start_epoch"] == 1
        and k["K3_nguon"]["tong_epoch"] == 3,
        "K4_lich_trung_C": not k["K4_lr_momentum_ni_batch_trung_C"]["lech"] and k["K4_lr_momentum_ni_batch_trung_C"][
            "so_dong_R"] == k["K4_lr_momentum_ni_batch_trung_C"]["so_dong_C"] > 0
        and k["K4_lr_momentum_ni_batch_trung_C"]["mosaic_dat"],
        "K5_mo_phong_khop": not k["K5_mo_phong"]["lech_C"] and not k["K5_mo_phong"]["lech_R"],
        "K7_nhat_ky_dung_thu_tu": k["K7_nhat_ky"]["dung_thu_tu"],
        "K8_hoan_tat_5_lop": k["K8_ket_thuc"]["trang_thai"] == "hoan_tat" and k["K8_ket_thuc"]["exit_code"] == 0
        and bool(k["K8_ket_thuc"]["checkpoint_5_lop"]) and all(k["K8_ket_thuc"]["checkpoint_5_lop"].values()),
    }
    for t, ma in (("C", 0), ("I", 15), ("R", 0)):
        g = gs[t]
        d[f"giam_sat_{t}_bam_dung_pid_runner"] = g["pid_runner_tu_ghi_nhat_ky"] == [g["pid_cong_cu_bam"]] \
            and g["pid_trong_csv"] == [g["pid_cong_cu_bam"]]
        d[f"giam_sat_{t}_exit_{ma}"] = g["exit_code"] == ma
        d[f"giam_sat_{t}_co_mau_khong_ghi_0"] = g["so_mau"] > 0 and (g["rss_chinh_max_mib"] or 0) > 0 \
            and g["so_mau_khong_doc_duoc"] == 0
        d[f"giam_sat_{t}_tinh_lai_khop"] = g["tinh_lai_khop"]
    return {"tieu_chi": d, "dat": all(d.values())}


def mot_model(model: str) -> dict:
    out = {"model": model}
    run_c, run_i = PROJECT / f"{model}_C", PROJECT / f"{model}_I"
    for ten in ("C", "I", "R", "luot"):
        if (L / "05_gpu" / f"{model}_{ten}").exists():
            raise SystemExit(f"{L / '05_gpu' / f'{model}_{ten}'} da ton tai -- khong ghi de")
    t0 = time.time()
    out["C"] = {"exit": chay(L / "05_gpu" / f"{model}_C", lenh_runner(model, f"{model}_C")), "s": round(time.time() - t0)}
    print(f"[{model}] C exit {out['C']['exit']} ({out['C']['s']} s)", flush=True)
    t0 = time.time()
    out["I"] = chay_va_ngat(L / "05_gpu" / f"{model}_I", lenh_runner(model, f"{model}_I"), run_i)
    out["I"]["s"] = round(time.time() - t0)
    nk_truoc = tt.doc_nk(run_i)
    out["I"]["nhat_ky"] = [(s["loai"], s.get("epoch")) for s in nk_truoc]
    print(f"[{model}] I {out['I']}", flush=True)
    t0 = time.time()
    out["R"] = {"exit": chay(L / "05_gpu" / f"{model}_R", lenh_runner(model, f"{model}_I", "--tiep-tuc")),
                "s": round(time.time() - t0)}
    print(f"[{model}] R exit {out['R']['exit']} ({out['R']['s']} s)", flush=True)
    out["bang_chung_da_chep"] = chep_bang_chung(model)
    try:
        out["kiem"] = so_sanh(model, run_c, run_i)
        out["danh_gia"] = danh_gia(out)
    except Exception as e:  # noqa: BLE001 -- giu loi that, khong tu bo qua
        import traceback
        out["loi_so_sanh"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        out["danh_gia"] = {"dat": False, "ly_do": "so sanh loi -- xem loi_so_sanh"}
    return out


def so_sanh(model: str, run_c: Path, run_i: Path) -> dict:
    import yaml
    c = yaml.safe_load(CFG.read_text(encoding="utf-8"))["huan_luyen"]
    E, nbs, batch = int(c["epochs"]), int(c["nbs"]), int(c["batch"])
    bc, bi = tt.doc_batch(run_c), tt.doc_batch(run_i)
    nb = sum(1 for x in bc if x["epoch"] == "1")
    nw = round(min(float(c["warmup_epochs"]), max(E - 1, 0)) * nb)
    nk = tt.doc_nk(run_i)
    nk_i1 = next(s for s in nk if s["loai"] == "epoch_da_luu" and s["epoch"] == 1)
    bd = next(s for s in nk if s["loai"] == "bat_dau" and s.get("che_do_luot") == "tiep_tuc")
    rr = json.loads((KQ / f"{model}_I__tiep_tuc_1.json").read_text(encoding="utf-8"))
    loai_bo = ("t", "lan")
    i_e1 = [x for x in bi if x["lan"] == "0" and x["epoch"] == "1"]
    k2 = [x["ni"] for x, y in zip(i_e1, bc[:nb]) if any(x[k] != y[k] for k in x if k not in loai_bo)]
    ck_c1 = tt.tensor_ck(run_c / "khoi_phuc" / "last_e001.pt")
    ck_i1 = tt.tensor_ck(run_i / "khoi_phuc" / "last_e001.pt")
    r_rows = [x for x in bi if x["lan"] == "1"]
    c_sau = [x for x in bc if int(x["epoch"]) >= 2]
    cot4 = ("epoch", "i", "ni", "lr_weight", "lr_bn", "lr_bias", "momentum_weight", "momentum_bn", "momentum_bias",
            "batch_size")
    k4 = [(x["ni"], k) for x, y in zip(r_rows, c_sau) for k in cot4 if x[k] != y[k]]
    mos = {t: {s["epoch"]: s.get("mosaic_thuc_te") for s in tt.doc_nk(r) if s["loai"] == "epoch_da_luu"}
           for t, r in (("C", run_c), ("I_R", run_i))}
    mos_p = {t: {s["epoch"]: s.get("mosaic_p") for s in tt.doc_nk(r) if s["loai"] == "epoch_da_luu"}
             for t, r in (("C", run_c), ("I_R", run_i))}
    mos_mong = {e: e - 1 < E - int(c["close_mosaic"]) for e in range(1, E + 1)}
    mp_c, buoc_c = tt.mo_phong(E, nb, nw, nbs, batch)
    mp_r, buoc_r = tt.mo_phong(E, nb, nw, nbs, batch, tu_epoch=1, updates_dau=nk_i1["ema_updates"])
    k5c = [x["ni"] for x in bc if (int(x["accumulate"]), int(x["ema_updates"])) != mp_c[int(x["ni"])]]
    k5r = [x["ni"] for x in r_rows if (int(x["accumulate"]), int(x["ema_updates"])) != mp_r[int(x["ni"])]]
    ck_rE = tt.tensor_ck(run_i / "khoi_phuc" / f"last_e{E:03d}.pt")
    ck_cE = tt.tensor_ck(run_c / "khoi_phuc" / f"last_e{E:03d}.pt")
    fit = {t: next(s["fitness"] for s in tt.doc_nk(r) if s["loai"] == "epoch_da_luu" and s["epoch"] == E)
           for t, r in (("C", run_c), ("R", run_i))}
    loai = [(s["loai"], s.get("lan"), s.get("epoch")) for s in nk]
    mong = [("bat_dau", 0, None), ("epoch_da_luu", 0, 1), ("bat_dau", 1, None)] + \
           [("epoch_da_luu", 1, e) for e in range(2, E + 1)] + [("ket_thuc", 1, None)]
    # bo giam sat: PID cong cu bam == PID runner tu ghi trong nhat ky (doc lap)
    gs = {}
    for ten, r in (("C", run_c), ("I", run_i), ("R", run_i)):
        d = L / "05_gpu" / f"{model}_{ten}"
        tom = json.loads((d / "tom_tat.json").read_text(encoding="utf-8"))
        nk_r = tt.doc_nk(r)
        pid_nk = [s["pid"] for s in nk_r if s["loai"] == "bat_dau" and s.get("lan") == (1 if ten == "R" else 0)]
        lai = subprocess.run([PY, str(GOC / "src" / "chay_giam_sat.py"), "tinh-lai", "--thu-muc", str(d)],
                             capture_output=True, text=True, cwd=GOC)
        gs[ten] = {"pid_cong_cu_bam": tom["pid"], "pid_runner_tu_ghi_nhat_ky": pid_nk, "exit_code": tom["exit_code"],
                   "exit_code_hex": tom["exit_code_hex"], "so_mau": tom["ram"]["so_mau"],
                   "so_mau_khong_doc_duoc": tom["ram"]["so_mau_khong_doc_duoc"],
                   "khoang_do_s": tom["ram"]["khoang_do_s"], "thoi_gian_lenh_s": tom["thoi_gian_s"],
                   "pid_trong_csv": tom["ram"]["pid_trong_csv"], "rss_tong_max_mib": (tom["ram"]["rss_tong_mib"] or {})
                   .get("max"), "rss_chinh_max_mib": (tom["ram"]["rss_chinh_mib"] or {}).get("max"),
                   "so_con_max": (tom["ram"]["so_con"] or {}).get("max"),
                   "ram_he_thong_con_trong_min_mib": (tom["ram"]["ram_he_thong_con_trong_mib"] or {}).get("min"),
                   "tinh_lai_khop": lai.returncode == 0, "ly_do_dung": tom["ly_do_dung_lay_mau"]}
    return {
        "K2_I_truoc_ngat_trung_C_epoch1": {"so_dong": len(i_e1), "dong_khac": len(k2), "ni_khac_dau": k2[:5],
                                           "checkpoint_e001": tt.ck_giong_het(ck_i1, ck_c1)[1],
                                           "ghi_chu": "GPU: tat dinh KHONG duoc gia dinh -- chi bao"},
        "K3_kiem_khoi_phuc": {k: v["dat"] for k, v in (rr.get("kiem_khoi_phuc") or {}).items()},
        "K3_nguon": {"chon_sha": bd["tiep_tuc"]["chon"]["sha256"], "e001_sha": nk_i1["last_sha256"],
                     "start_epoch": bd["start_epoch"], "tong_epoch": bd["tong_epoch"]},
        "K4_lr_momentum_ni_batch_trung_C": {"so_dong_R": len(r_rows), "so_dong_C": len(c_sau), "lech": k4[:6],
                                            "mosaic_theo_epoch": mos, "mosaic_p": mos_p, "mosaic_du_doan": mos_mong,
                                            "mosaic_dat": mos["C"] == mos["I_R"] == mos_mong},
        "K5_mo_phong": {"nb": nb, "nw": nw, "lech_C": k5c[:5], "lech_R": k5r[:5],
                        "buoc_C_sau_epoch1": [n for n in buoc_c if n >= nb], "buoc_R": buoc_r,
                        "accumulate_cuoi_warmup": mp_c[nw - 1][0] if nw else None, "accumulate_sau_resume": round(nbs / batch)},
        "K6_khac_biet": {"loss_khac": sum(1 for x, y in zip(r_rows, c_sau) if x["loss_tong"] != y["loss_tong"]),
                         "thu_tu_khac": sum(1 for x, y in zip(r_rows, c_sau) if x["thu_tu_batch"] != y["thu_tu_batch"]),
                         "lech_ema_max_cuoi": tt.lech_ema_max(ck_rE, ck_cE), "fitness_cuoi": fit},
        "K7_nhat_ky": {"dung_thu_tu": loai == mong, "su_kien": loai},
        "K8_ket_thuc": {"trang_thai": nk[-1].get("trang_thai"), "exit_code": nk[-1].get("exit_code"),
                        "checkpoint_5_lop": {k: v["dung_5_lop"] for k, v in rr.get("checkpoint", {}).items()}},
        "giam_sat": gs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--model", nargs="+", default=["yolov8n", "yolo11n"])
    a = ap.parse_args()
    if Path(a.ra).exists():
        raise SystemExit(f"{a.ra} da ton tai -- khong ghi de")
    (L / "05_gpu").mkdir(exist_ok=True)
    kq = {"mo_ta": __doc__.split("\n")[1], "cau_hinh": str(CFG.relative_to(GOC)).replace("\\", "/"),
          "cau_hinh_sha256": tt.sha(CFG), "data": str(DATA.relative_to(GOC)).replace("\\", "/"), "model": []}
    for m in a.model:
        kq["model"].append(mot_model(m))
        Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8",
                              newline="\n")
        print(json.dumps(kq["model"][-1], ensure_ascii=False, default=str)[:3000], flush=True)
    dat = {x["model"]: x["danh_gia"]["dat"] for x in kq["model"]}
    print(f"[ra] {a.ra}")
    print(f"[KET LUAN] {dat} -> {'DAT' if all(dat.values()) else 'KHONG DAT'}")
    return 0 if all(dat.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
