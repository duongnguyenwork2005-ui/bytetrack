"""
kiem_pf_diag_moc2.py -- (R48, buoc P9a8ee8bf) Kiem CHI DOC: log chan doan PF moc 2 va src/pf_diag_report.py co tron
dong tai khoi tao (nhanh suy bien cua PFCTRV.update_particles) vao ESS/N va ti le resample hay khong; tinh lai thong ke
ESS moc 2 voi mau so (update hop le) va so tai khoi tao TACH RIENG.

Quy uoc ghi cua nhanh suy bien (src/pf_ctrv.py, khong doi tu 6e28ffa): kind = "update", ess_ratio = 0.0,
resampled = True, n_unique = N, degenerate = True. pf_diag_report.py loc `kind == "update"` roi lay trung binh truc tiep
nen VE MA, dong suy bien bi tron vao ess_ratio_mean / p10 / frac_ess_lt_0.2 / frac_resampled / n_unique_ratio.
Script nay dem dong suy bien tren CSV tho va tinh lai:
    - "cu"  : dung cong thuc cua pf_diag_report (moi dong update) -> doi chieu voi summary.csv / by_n_lost.csv da ghi;
    - "moi" : chi tren update hop le (degenerate = False); tai khoi tao dem rieng (n_tai_khoi_tao).
Bat bien kiem: degenerate <=> ess_ratio == 0 <=> maha2_best > pf_degenerate_maha2; moi ESS/N hop le >= 1/N.

Chi doc. Kiem ke MOI thu muc pf_diag duoi data/processed va results (sua F1: ban dau chi quet DETRAC-all nen bo sot
6 run test). Hai nhom duoc phan tich, bang tach rieng:
    - dev : data/processed/trackers/DETRAC-all/p2-expA_detrac_dev-*-pf*-s*/pf_diag  (18 run x 12 video), doi chieu voi
            results/phase2/pf_diag/expA_detrac_dev/*.csv;
    - test: data/processed/trackers/DETRAC-test/p2-expA_detrac_test-*-pf*-s*/pf_diag (6 run x 40 video), KHONG co bao
            cao cu (pf_diag_report chua tung chay tren test) -> khong doi chieu; tap test chi bao cao, khong chon gi.
Chup sha256 moi CSV pf_diag kiem ke duoc + run_meta.json + ket qua cu TRUOC va SAU khi tinh; bang dev (theo_run.csv,
theo_tracker.csv, theo_n_lost.csv) giu ten cu, sha256 truoc/sau khi ghi lai duoc so (bang dev khong doi).
Khong import pandas (tranh loi Smart App Control voi pyarrow); quantile = numpy 'linear' (= mac dinh pandas).

    python src/kc48/kiem_pf_diag_moc2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRK = REPO / "data" / "processed" / "trackers" / "DETRAC-all"
TRK_TEST = REPO / "data" / "processed" / "trackers" / "DETRAC-test"
CU = REPO / "results" / "phase2" / "pf_diag" / "expA_detrac_dev"
OUT = REPO / "results" / "phase2" / "kc48_pf_code" / "ess_moc2_tach_tai_khoi_tao"
FILE_CU = [CU / "summary.csv", CU / "by_n_lost.csv", CU / "summary_by_tracker.csv",
           REPO / "handoff" / "logs" / "pf_diag_report.log", REPO / "src" / "pf_diag_report.py"]
RE_RUN = re.compile(r"^p2-expA_detrac_dev-(c\d+)-(pf(\d+))-s(\d+)$")
RE_RUN_TEST = re.compile(r"^p2-expA_detrac_test-(c\d+)-(pf(\d+))-s(\d+)$")
# (ten nhom, thu muc split, regex, so run ky vong, so video moi run ky vong, hau to file ket qua)
NHOM = [("dev", TRK, RE_RUN, 18, 12, ""), ("test", TRK_TEST, RE_RUN_TEST, 6, 40, "_test")]
GOC_KIEM_KE = [REPO / "data" / "processed", REPO / "results"]
FILE_BANG = ["theo_run", "theo_tracker", "theo_n_lost"]
# cung bin voi pf_diag_report: pd.cut(n_lost, [-1, 0, 2, 5, 10, 30, 10000]) (dong ben phai)
BIN_EDGES = [-1, 0, 2, 5, 10, 30, 10_000]
BIN_LABELS = ["0", "1-2", "3-5", "6-10", "11-30", ">30"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def chup_hash(files: list[Path]) -> dict:
    return {rel(p): sha256(p) for p in files}


def doc_csv(p: Path) -> dict:
    with open(p, newline="", encoding="utf-8") as f:
        return tu_dong(list(csv.DictReader(f)))


def tu_dong(rows: list[dict]) -> dict:
    """Dong CSV (chuoi, nhu pandas.to_csv ghi) -> mang theo cot."""
    col = lambda k, fn: np.array([fn(r[k]) for r in rows])  # noqa: E731
    b = lambda s: s == "True"  # noqa: E731
    return {"kind": col("kind", str), "ess": col("ess_ratio", float), "rs": col("resampled", b),
            "nu": col("n_unique", float), "deg": col("degenerate", b), "maha": col("maha2_best", float),
            "nl": col("n_lost", float)}


def gop(ds: list[dict]) -> dict:
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def q10(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.10)) if len(x) else float("nan")


def tk(ess, rs, nu, N) -> dict:
    """Thong ke cua pf_diag_report tren tap dong cho truoc."""
    n = len(ess)
    if n == 0:
        return {k: float("nan") for k in ("ess_ratio_mean", "ess_ratio_p10", "frac_resampled", "frac_ess_lt_0.2",
                                           "n_unique_ratio_mean", "n_unique_ratio_p10")}
    return {"ess_ratio_mean": float(np.mean(ess)), "ess_ratio_p10": q10(ess), "frac_resampled": float(np.mean(rs)),
            "frac_ess_lt_0.2": float(np.mean(ess < 0.2)), "n_unique_ratio_mean": float(np.mean(nu / N)),
            "n_unique_ratio_p10": q10(nu / N)}


def mat_na_bin(nl: np.ndarray, lab: str) -> np.ndarray:
    """Mat na cua bin n_lost `lab`, khoang (lo, hi] nhu pd.cut mac dinh."""
    i = BIN_LABELS.index(lab)
    return (nl > BIN_EDGES[i]) & (nl <= BIN_EDGES[i + 1])


def doc_bang(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ghi_csv(p: Path, rows: list[dict]) -> None:
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def kiem_ke_pf_diag() -> tuple[list[Path], list[Path]]:
    """Moi thu muc ten pf_diag duoi GOC_KIEM_KE (bat ke split / ten run) -> (thu muc run cha, thu muc khac).
    Ngoai le duy nhat: results/phase2/pf_diag la thu muc DAU RA cua pf_diag_report (bao cao cu, FILE_CU), khong phai
    log cua mot run -> tach ra, ghi rieng."""
    runs, khac = [], []
    for goc in GOC_KIEM_KE:
        for d, dirs, _ in os.walk(goc):
            if "pf_diag" in dirs:
                (khac if Path(d) / "pf_diag" == CU.parent else runs).append(Path(d))
    return sorted(runs), sorted(khac)


def phan_tich_nhom(runs: list[Path], rx: re.Pattern, n_video_ky_vong: int, sum_cu: dict | None,
                   bl_cu: dict | None) -> tuple[list, list, list, float]:
    """Thong ke tung run + theo bin n_lost. sum_cu/bl_cu = None: khong co bao cao cu de doi chieu."""
    theo_run, theo_bin, loi_bat_bien = [], [], []
    lech_cu_max = 0.0
    for d in runs:
        cond, tracker, N, seed = rx.match(d.name).group(1), rx.match(d.name).group(2), \
            int(rx.match(d.name).group(3)), rx.match(d.name).group(4)
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8")) if (d / "run_meta.json").exists() else {}
        cfg = meta.get("tracker_cfg", {})
        thr = float(cfg.get("pf_degenerate_maha2", 1e4))
        if int(cfg.get("pf_num_particles", N)) != N:
            loi_bat_bien.append(f"{d.name}: pf_num_particles {cfg.get('pf_num_particles')} != {N}")
        files = sorted((d / "pf_diag").glob("*.csv"))
        if len(files) != n_video_ky_vong:
            loi_bat_bien.append(f"{d.name}: {len(files)} CSV != {n_video_ky_vong}")
        a = gop([doc_csv(f) for f in files])
        upd = a["kind"] == "update"
        deg = upd & a["deg"]
        hop = upd & ~a["deg"]
        kinds = sorted(set(a["kind"].tolist()))
        # bat bien
        if set(kinds) - {"update", "init_motion"}:
            loi_bat_bien.append(f"{d.name}: kind la {kinds}")
        if np.any(deg != (upd & (a["ess"] == 0.0))):
            loi_bat_bien.append(f"{d.name}: degenerate khong trung ess_ratio == 0")
        if np.any(deg != (upd & (a["maha"] > thr))):
            loi_bat_bien.append(f"{d.name}: degenerate khong trung maha2_best > {thr}")
        if np.any(a["ess"][hop] < 1.0 / N - 1e-12):
            loi_bat_bien.append(f"{d.name}: co ESS/N hop le < 1/N")
        if np.any(a["nu"][upd] > N):
            loi_bat_bien.append(f"{d.name}: n_unique > N")

        cu = tk(a["ess"][upd], a["rs"][upd], a["nu"][upd], N)
        moi = tk(a["ess"][hop], a["rs"][hop], a["nu"][hop], N)
        rec = {"run": d.name, "cond": cond, "tracker": tracker, "seed": int(seed), "N": N, "n_video": len(files),
               "pf_degenerate_maha2": thr, "n_dong": int(len(a["kind"])), "n_init_motion": int(np.sum(~upd)),
               "n_update_tat_ca": int(np.sum(upd)), "n_tai_khoi_tao": int(np.sum(deg)),
               "n_update_hop_le": int(np.sum(hop)),
               "ti_le_tai_khoi_tao": float(np.sum(deg) / max(np.sum(upd), 1)),
               "maha2_best_max_hop_le": float(np.max(a["maha"][hop])) if np.any(hop) else float("nan")}
        for k, v in moi.items():
            rec[f"{k}_hop_le"] = v
        for k, v in cu.items():
            rec[f"{k}_cong_thuc_cu"] = v
        if sum_cu is not None:
            ref = sum_cu.get((cond, tracker, seed))
            if ref is None:
                loi_bat_bien.append(f"{d.name}: khong co dong trong summary.csv cu")
            else:
                if int(ref["n_update"]) != rec["n_update_tat_ca"]:
                    loi_bat_bien.append(f"{d.name}: n_update {rec['n_update_tat_ca']} != cu {ref['n_update']}")
                lech = max(abs(cu[k] - float(ref[k])) for k in cu)
                rec["lech_max_vs_summary_cu"] = lech
                lech_cu_max = max(lech_cu_max, lech)
                rec["frac_degenerate_cu"] = float(ref["frac_degenerate"])
        theo_run.append(rec)

        # theo bin n_lost
        for lab in BIN_LABELS:
            m_bin = upd & mat_na_bin(a["nl"], lab)
            mb_hop = m_bin & ~a["deg"]
            if not np.any(m_bin):
                continue
            e_cu, e_moi = a["ess"][m_bin], a["ess"][mb_hop]
            row = {"cond": cond, "tracker": tracker, "seed": int(seed), "n_lost_bin": lab,
                   "n_update_tat_ca": int(np.sum(m_bin)), "n_tai_khoi_tao": int(np.sum(m_bin & a["deg"])),
                   "n_update_hop_le": int(np.sum(mb_hop)),
                   "ess_ratio_hop_le": float(np.mean(e_moi)) if len(e_moi) else float("nan"),
                   "frac_resampled_hop_le": float(np.mean(a["rs"][mb_hop])) if np.any(mb_hop) else float("nan"),
                   "n_unique_ratio_hop_le": float(np.mean(a["nu"][mb_hop] / N)) if np.any(mb_hop) else float("nan"),
                   "ess_ratio_cong_thuc_cu": float(np.mean(e_cu))}
            if bl_cu is not None:
                r0 = bl_cu.get((cond, tracker, seed, lab))
                if r0 is None:
                    loi_bat_bien.append(f"{d.name} bin {lab}: khong co trong by_n_lost.csv cu")
                else:
                    if int(r0["n"]) != row["n_update_tat_ca"]:
                        loi_bat_bien.append(f"{d.name} bin {lab}: n {row['n_update_tat_ca']} != cu {r0['n']}")
                    row["lech_ess_vs_by_n_lost_cu"] = abs(row["ess_ratio_cong_thuc_cu"] - float(r0["ess_ratio"]))
                    lech_cu_max = max(lech_cu_max, row["lech_ess_vs_by_n_lost_cu"])
            theo_bin.append(row)
    return theo_run, theo_bin, loi_bat_bien, lech_cu_max


def gop_tracker(theo_run: list[dict]) -> list[dict]:
    """Gop theo (cond, tracker): mau so va tai khoi tao la TONG qua seed; ty le la trung binh giua seed."""
    theo_tracker = []
    for key in sorted({(r["cond"], r["tracker"]) for r in theo_run}, key=lambda k: (k[0], int(k[1][2:]))):
        rs = [r for r in theo_run if (r["cond"], r["tracker"]) == key]
        g = {"cond": key[0], "tracker": key[1], "N": rs[0]["N"], "n_seed": len(rs),
             "n_update_tat_ca_tong": sum(r["n_update_tat_ca"] for r in rs),
             "n_tai_khoi_tao_tong": sum(r["n_tai_khoi_tao"] for r in rs),
             "n_update_hop_le_tong": sum(r["n_update_hop_le"] for r in rs)}
        for k in ("ess_ratio_mean", "ess_ratio_p10", "frac_resampled", "frac_ess_lt_0.2", "n_unique_ratio_mean"):
            v = np.array([r[f"{k}_hop_le"] for r in rs])
            g[f"{k}_hop_le"] = float(np.mean(v))
            g[f"{k}_hop_le_sd_seed"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        theo_tracker.append(g)
    return theo_tracker


def main() -> int:
    # kiem ke toan bo truoc, phan nhom sau: thu muc pf_diag nao khong thuoc nhom nao -> bao loi (khong bo sot nua)
    tat_ca_run, thu_muc_bao_cao = kiem_ke_pf_diag()
    kiem_ke = [{"run": rel(d), "n_csv": len(list((d / "pf_diag").glob("*.csv"))),
                "co_run_meta": (d / "run_meta.json").exists()} for d in tat_ca_run]
    csv_kiem_ke = [f for d in tat_ca_run for f in sorted((d / "pf_diag").glob("*.csv"))]
    meta_kiem_ke = [d / "run_meta.json" for d in tat_ca_run if (d / "run_meta.json").exists()]
    hash_truoc = chup_hash(csv_kiem_ke + meta_kiem_ke + FILE_CU)
    bang_out = [OUT / f"{b}{nh[5]}.csv" for nh in NHOM for b in FILE_BANG]
    hash_bang_truoc = chup_hash([p for p in bang_out if p.exists()])

    loi, nhom_kq, da_phan_nhom = [], {}, set()
    lech_cu_max = 0.0
    for ten, goc, rx, n_run_kv, n_vid_kv, hau_to in NHOM:
        runs = [d for d in tat_ca_run if d.parent == goc and rx.match(d.name)]
        da_phan_nhom.update(runs)
        if len(runs) != n_run_kv:
            loi.append(f"nhom {ten}: {len(runs)} run != {n_run_kv}")
        if ten == "dev":
            sum_cu = {(r["cond"], r["tracker"], r["seed"]): r for r in doc_bang(CU / "summary.csv")}
            bl_cu = {(r["cond"], r["tracker"], r["seed"], r["n_lost_bin"]): r for r in doc_bang(CU / "by_n_lost.csv")}
        else:
            sum_cu = bl_cu = None
        theo_run, theo_bin, loi_nhom, lech = phan_tich_nhom(runs, rx, n_vid_kv, sum_cu, bl_cu)
        loi += [f"[{ten}] {e}" for e in loi_nhom]
        if ten == "dev":
            lech_cu_max = lech
        theo_tracker = gop_tracker(theo_run)
        OUT.mkdir(parents=True, exist_ok=True)
        ghi_csv(OUT / f"theo_run{hau_to}.csv", theo_run)
        ghi_csv(OUT / f"theo_tracker{hau_to}.csv", theo_tracker)
        ghi_csv(OUT / f"theo_n_lost{hau_to}.csv", theo_bin)
        nhom_kq[ten] = {
            "so_run": len(runs), "so_file_csv": sum(r["n_video"] for r in theo_run),
            "n_update_tat_ca": sum(r["n_update_tat_ca"] for r in theo_run),
            "n_tai_khoi_tao": sum(r["n_tai_khoi_tao"] for r in theo_run),
            "n_update_hop_le": sum(r["n_update_hop_le"] for r in theo_run),
            "maha2_best_max_hop_le": max(r["maha2_best_max_hop_le"] for r in theo_run),
            "co_bao_cao_cu_de_doi_chieu": sum_cu is not None,
            "bang": [f"{b}{hau_to}.csv" for b in FILE_BANG], "theo_tracker": theo_tracker}
    khong_phan_nhom = [rel(d) for d in tat_ca_run if d not in da_phan_nhom]
    if khong_phan_nhom:
        loi.append(f"thu muc pf_diag khong thuoc nhom nao: {khong_phan_nhom}")

    hash_sau = chup_hash(csv_kiem_ke + meta_kiem_ke + FILE_CU)
    doi = sorted(k for k in hash_truoc if hash_truoc[k] != hash_sau.get(k))
    hash_bang_sau = chup_hash(bang_out)
    bang_dev_doi = sorted(rel(OUT / f"{b}.csv") for b in FILE_BANG
                          if rel(OUT / f"{b}.csv") in hash_bang_truoc
                          and hash_bang_truoc[rel(OUT / f"{b}.csv")] != hash_bang_sau[rel(OUT / f"{b}.csv")])
    tong_tkt = sum(v["n_tai_khoi_tao"] for v in nhom_kq.values())
    tong_upd = sum(v["n_update_tat_ca"] for v in nhom_kq.values())
    ket = {
        "ma_nguon_tron_ve_cong_thuc": True,
        "giai_thich": "pf_diag_report.py loc kind=='update' (gom ca dong suy bien ess_ratio=0, resampled=True, "
                      "n_unique=N) roi lay mean/quantile -> tron neu co dong suy bien.",
        "kiem_ke": {"goc": [rel(g) for g in GOC_KIEM_KE], "so_run_co_pf_diag": len(tat_ca_run),
                    "so_csv": len(csv_kiem_ke), "so_run_meta": len(meta_kiem_ke), "run": kiem_ke,
                    "khong_thuoc_nhom": khong_phan_nhom,
                    "thu_muc_pf_diag_la_bao_cao_cu_khong_phai_run": [rel(d / "pf_diag") for d in thu_muc_bao_cao]},
        "nhom": nhom_kq,
        "n_update_tat_ca": tong_upd, "n_tai_khoi_tao": tong_tkt,
        "so_lieu_moc2_bi_tron_thuc_te": bool(tong_tkt > 0),
        "lech_max_cong_thuc_cu_vs_file_cu_dev": lech_cu_max,
        "loi_bat_bien": loi,
        "hash_truoc": hash_truoc, "file_doi_sau_khi_chay": doi,
        "hash_bang_kq_truoc_khi_ghi": hash_bang_truoc, "hash_bang_kq_sau_khi_ghi": hash_bang_sau,
        "bang_dev_doi_so_voi_lan_ghi_truoc": bang_dev_doi,
    }
    (OUT / "ket_qua.json").write_text(json.dumps(ket, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"kiem ke: {len(tat_ca_run)} run co pf_diag, {len(csv_kiem_ke)} CSV, {len(meta_kiem_ke)} run_meta; "
          f"khong thuoc nhom: {len(khong_phan_nhom)}; thu muc bao cao cu (khong phai run): "
          f"{[rel(d / 'pf_diag') for d in thu_muc_bao_cao]}")
    for ten, v in nhom_kq.items():
        print(f"[{ten}] run {v['so_run']}, file {v['so_file_csv']}, update {v['n_update_tat_ca']}, "
              f"tai khoi tao {v['n_tai_khoi_tao']}, hop le {v['n_update_hop_le']}, "
              f"maha2 hop le max {v['maha2_best_max_hop_le']:.1f}")
        for g in v["theo_tracker"]:
            print(f"  {g['cond']} {g['tracker']}: upd {g['n_update_tat_ca_tong']}, tkt {g['n_tai_khoi_tao_tong']}, "
                  f"hop le {g['n_update_hop_le_tong']}, ESS/N {g['ess_ratio_mean_hop_le']:.4f}, "
                  f"resample {g['frac_resampled_hop_le']:.4f}")
    print(f"lech max cong thuc cu vs file cu (dev): {lech_cu_max:.3e}; loi bat bien: {len(loi)}; "
          f"file dau vao/cu doi: {len(doi)}; bang dev doi so voi lan ghi truoc: {len(bang_dev_doi)} "
          f"(co hash truoc: {sum(rel(OUT / f'{b}.csv') in hash_bang_truoc for b in FILE_BANG)}/3)")
    for e in loi[:20]:
        print("  BAT BIEN:", e)
    print(f"-> {OUT}")
    ok = (len(tat_ca_run) == 24 and len(csv_kiem_ke) == 456 and not loi and not doi and not bang_dev_doi
          and lech_cu_max < 1e-9)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
