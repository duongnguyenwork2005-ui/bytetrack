"""Read-only audit of round-38 receipts; no dataset bytes, training or inference.

Usage: python check_round38.py --tree TREE --prior PRIOR --out results.json
PRIOR contains split.json and estimate.json copied from accepted reviews.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jfile(p):
    return json.loads(p.read_text(encoding="utf-8"))


def receipt(p):
    s = p.read_text(encoding="utf-8")
    return json.JSONDecoder().raw_decode(s[s.index("{"):])[0]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    log = a.tree / "handoff/logs/review38"
    old = a.tree / "handoff/review_evidence/CODEX_0c0a07a"
    report = {"scope": "receipt/hash/arithmetic consistency; not an independent validation of dataset bytes", "checks": []}

    def check(name, ok):
        report["checks"].append({"name": name, "pass": bool(ok)})

    # The prior review's exact file bytes are preserved, not merely its summary.
    lines = (old / "SHA256SUMS.txt").read_text().splitlines()
    for ln in lines:
        h, f = ln.split(maxsplit=1)
        check("prior evidence " + f, sha((old / f.lstrip("*")).read_bytes()) == h)
    check("91 prior evidence files", len(lines) == 91)
    report["prior_evidence_files"] = len(lines)

    split_p = a.prior / "split.json"
    sp = jfile(split_p)
    check("prior split exact bytes", sha(split_p.read_bytes()) == "b8a2848e3da053a2b1b79e7fa6cfad812540315af6e48b12b4174a40efc06cdb")
    t, d = sp["train_fit"], sp["dev"]
    check("1261/139 locked, unique and disjoint videos", len(t) == len(set(t)) == 1261 and len(d) == len(set(d)) == 139 and not set(t) & set(d))
    list_sha = sha(json.dumps({"train_fit": t, "dev": d}, sort_keys=True).encode())
    check("video list hash", list_sha == sp["sha256_danh_sach"])
    before = (log / "02_kiem_truoc_xuat.log").read_text()
    for h in (sha(split_p.read_bytes()), list_sha, sp["nguon_nhan"]["sha256"], sp["cau_hinh"]["sha256"]):
        check("pre-export receipt vs prior lock " + h[:12], h in before)

    source = receipt(log / "01_kiem_nguon.log")
    ex = receipt(log / "03_xuat_day_du.log")
    contract = receipt(log / "04_kiem_hop_dong.log")
    independent = jfile(log / "05_kiem_doc_lap.json")
    check("source receipt 1400 videos / 278079 frames", source["so_video_nhan"] == source["so_video_anh"] == source["so_video_dung_duoc"] == 1400 and source["so_frame_nhan"] == 278079 and source["so_anh_thieu"] == 0)
    check("source frame count vs prior split", sum(x["so_frame"] for x in sp["thong_ke_nguon_theo_split"].values()) == source["so_frame_nhan"])
    check("frame partition arithmetic", sum(contract["so_anh"].values()) + ex["dem_frame"]["khong_theo_buoc"] + ex["dem_frame"]["ngoai_khoang_co_nhan"] == ex["dem_frame"]["frame_nguon"] == 278079)
    check("export time from timestamp", "23:35:25" in (log / "03_xuat_day_du.log").read_text() and "23:48:22" in (log / "03_xuat_day_du.log").read_text() and "wall_s=777" in (log / "03_xuat_day_du.meta.txt").read_text())
    ds_hash = {"sha256_data_yaml": "1280e57bcc5c6927a3a1143f59b3ffaf2a05fc9d16b63507ab4f81be2eaeecd1", "sha256_tom_tat": "9eae5eb3ee0a70a13dee0dca7afd682044b480d55690257f2ac4cbe574749010", "manifest_sha256": "1ee6174f8be328f4f62ea10e3a9bb0502fd18c4e55f9b9aa0d2512cd565f6069", "sha256_video_json": "05140872cc35b4399700960139ce287810640985e2ee20f9700e848826a09750"}
    receipts = {"export": ex["kiem_sau_xuat"], "recheck": contract}
    dry = {}
    for model in ("yolov8n", "yolo11n"):
        dry[model] = receipt(log / ("06_dry_run_" + model + ".log"))
        receipts[model] = dry[model]["dataset"]
        text = (log / ("06_dry_run_" + model + ".log")).read_text()
        check(model + " dry run declares no training / no output", "KHONG train, KHONG tao" in text and "# exit=0" in text)
    for key, r in receipts.items():
        check(key + " v2 / 0 error", r["phien_ban_manifest"] == 2 and r["loi"] == 0 and r["loai"] == "day_du" and r["tap_con"] is None)
        for k, h in ds_hash.items():
            check(key + " consistent dataset hash " + k, r[k] == h)
        check(key + " exact locked split hash", r["split_da_khoa"]["sha256"] == sha(split_p.read_bytes()) and r["split_da_khoa"]["sha256_danh_sach"] == list_sha)
        check(key + " image counts", r["so_anh"] == {"train": 50487, "dev": 5559})
        check(key + " file hash counts", r["so_file_da_bam"] == {"anh": 56046, "nhan": 56046, "ignore": 18600})
        check(key + " no image cache", r["cache_anh"]["so_npy_canh_anh"] == {"train": 0, "dev": 0})
        check(key + " train/dev paths", all(r["duong_dan_resolve"][k].replace("\\", "/").endswith("/images/" + s) for k, s in (("train", "train"), ("val", "dev"))))

    check("manifest receipt internally reconciled", ex["manifest_anh"]["so_dong"] == 56046 and ex["manifest_anh"]["sha256"] == contract["manifest_sha256"])
    box_total = 0
    for s in ("train", "dev"):
        exported, checked = ex["thong_ke"][s], independent["thong_ke"][s]
        for k in ("so_video", "so_anh", "am_tinh", "box", "anh_co_lop", "video_co_lop"):
            check(s + " statistic agrees " + k, exported[k] == checked[k])
        check(s + " ignore image counts", exported["anh_to_xam"] == checked["anh_co_vung_ignore"] == contract["so_anh_co_file_ignore"][s])
        check(s + " negative images", exported["am_tinh"] == contract["so_nhan_rong_hop_le"][s])
        prior_s = sp["thong_ke_nguon_theo_split"]["train_fit" if s == "train" else "dev"]
        for c in ("car", "truck", "bus", "motorcycle", "bicycle"):
            check(s + " selected support bounds " + c, 0 < checked["video_co_lop"][c] <= prior_s["video_co_lop"][c] and checked["video_co_lop"][c] <= checked["anh_co_lop"][c] <= checked["box"][c] <= prior_s["box"][c] and checked["anh_co_lop"][c] <= checked["so_anh"])
        box_total += sum(checked["box"].values())
    check("475234 boxes arithmetic", box_total == independent["nhan"]["so_dong_box"] == 475234)
    check("independent checker receipt has no errors", independent["ket_luan"] == "DAT" and independent["loi"] == [] and independent["nhan"]["dong_loi"] == independent["nhan"]["anh_so_box_lech_manifest"] == 0)
    check("frame set receipt 56046", all(independent["khop_tap_frame"][k] == 56046 for k in ("mong_doi", "manifest", "dong_manifest")) and independent["khop_tap_frame"]["thieu_trong_manifest"] == independent["khop_tap_frame"]["thua_trong_manifest"] == 0)
    check("class name mapping receipt", list(independent["data_yaml"]["names"].values()) == ["car", "truck", "bus", "motorcycle", "bicycle"])
    report["dataset_receipt"] = {"hashes": ds_hash, "stats": independent["thong_ke"], "box_total": box_total, "file_count_receipt": 130699, "hashed_content_files": 56046 * 2 + 18600, "bytes_receipt": 8034288965}

    with (log / "03_ram_xuat_mau_tho.csv").open(newline="") as f:
        ram = list(csv.DictReader(f))
    available = [int(r["ram_he_thong_con_trong_mib"]) for r in ram]
    rsum = jfile(log / "03_ram_xuat.json")
    check("RAM system samples", len(ram) == rsum["so_mau"] == 781)
    check("RAM system min/median", min(available) == rsum["ram_he_thong_con_trong_min_mib"] == 449 and statistics.median(available) == rsum["ram_he_thong_con_trong_trung_vi_mib"] == 727)
    check("RAM timestamps strictly increasing", all(float(x["t_unix"]) < float(y["t_unix"]) for x, y in zip(ram, ram[1:])))
    report["ram_system"] = {"samples": len(ram), "min_mib": min(available), "median_mib": statistics.median(available), "max_mib": max(available), "process_rss_usable": False}

    ts = dry["yolov8n"]["tham_so_khai_bao"]
    for m, rec in dry.items():
        p = rec["tham_so_khai_bao"]
        check(m + " proposed config", (p["epochs"], p["batch"], p["imgsz"], p["optimizer"], p["warmup_epochs"], p["cache"], p["amp"], p["seed"]) == (15, 8, 640, "SGD", 1.0, False, False, 0))
        check(m + " static arithmetic", rec["suy_ra"]["batch_moi_epoch"] == math.ceil(50487 / 8) == 6311 and rec["suy_ra"]["tong_batch"] == 15 * 6311 == 94665 and rec["suy_ra"]["buoc_toi_uu_moi_epoch"] == math.ceil(6311 / 8) == 789)
        check(m + " same non-model training arguments", all(v == ts[k] for k, v in p.items() if k not in ("name", "model")))
    # Exact existing helper evaluated without importing torch or training.
    runner = old / "tree/src/train_detector.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "suy_ra")
    ns = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(runner), "exec"), ns)
    pilot = jfile(old / "tree/handoff/logs/review37/06_ket_qua_runner_yolov8n.json")
    rough = ns["suy_ra"](pilot["tham_so_khai_bao"], 334)
    actual = [x["so_lan_cap_nhat"] for x in pilot["pilot"]["epoch"]]
    check("counterexample to exact optimizer steps", rough["buoc_toi_uu_moi_epoch"] == 6 and actual == [12, 5])
    report["nonblocking_note_optimizer_steps"] = {"full_dataset_static_estimate": 789, "pilot_static_estimate_per_epoch": rough["buoc_toi_uu_moi_epoch"], "pilot_observed_per_epoch": actual, "interpretation": "static estimate excludes dynamic warmup accumulation and update boundary effects; not a full-dataset measured count"}
    expected = jfile(a.prior / "estimate.json")
    candidates = [x for x in expected["thoi_gian_theo_phuong_an"] if x.get("buoc_frame") == 5 and x.get("epoch") == 15]
    check("estimate unchanged scenario available", len(candidates) == 1)
    if candidates:
        case = candidates[0]
        check("56.64/73.95 hour scenarios", case["tong_hai_model_h_trung_tam"] == 56.64 and case["tong_hai_model_h_cao"] == 73.95)
    report["pass"] = sum(x["pass"] for x in report["checks"])
    report["fail"] = sum(not x["pass"] for x in report["checks"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False, indent=2))
    for c in report["checks"]:
        if not c["pass"]:
            print("FAIL", c["name"])
    return 0 if report["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
