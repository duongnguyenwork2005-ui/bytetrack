#!/usr/bin/env python3
"""Check delivered split, smoke receipts and estimate arithmetic; no training.

Inputs are a reconstructed delta tree and the accepted native-label manifest.
This does not inspect the Windows image dataset or load any checkpoint.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--accepted-label-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logs = a.tree / "handoff/logs/review35"
    p = a.tree / "configs/phase2/detector/splits/bdd5_train_v1.json"
    raw = p.read_bytes()
    split = load(p)
    train, dev = split["train_fit"], split["dev"]
    assert len(train) == len(set(train)) == 1261
    assert len(dev) == len(set(dev)) == 139
    assert not set(train) & set(dev)
    assert train == sorted(train) and dev == sorted(dev)
    rows = list(csv.DictReader(a.accepted_label_manifest.open(encoding="utf-8"), delimiter="\t"))
    assert sha(a.accepted_label_manifest.read_bytes()) == "a2e416f26d2ba44a50c7f3ec8578deec39f364cd2381c36dc12fe45a905fcf39"
    native_train = {Path(r["duong_dan"]).stem: r for r in rows if "/train/" in r["duong_dan"]}
    native_val = {Path(r["duong_dan"]).stem for r in rows if "/val/" in r["duong_dan"]}
    assert set(train) | set(dev) == set(native_train)
    assert not (set(train) | set(dev)) & native_val
    fingerprint = sha("".join(f"{name}.json\t{native_train[name]['sha256']}\n" for name in sorted(native_train)).encode())
    assert fingerprint == split["nguon_nhan"]["sha256"]
    list_hash = sha(json.dumps({"train_fit": train, "dev": dev}, sort_keys=True).encode())
    assert list_hash == split["sha256_danh_sach"]
    assert sha(raw) == "b8a2848e3da053a2b1b79e7fa6cfad812540315af6e48b12b4174a40efc06cdb"
    assert sha(raw.replace(b"\r\n", b"\n")) == "bc9c85d1e71891ec3247b602c9bb5e54b2c047f9e7fe9405a8baa8c6b8ce429a"
    strata = split["chia"]["tang"]
    assert sum(s["so_video"] for s in strata.values()) == 1400
    assert sum(s["so_dev"] for s in strata.values()) == 139
    assert all(s["so_dev"] == math.floor(s["so_video"] * .1 + .5) for s in strata.values())
    selection = load(logs / "06_kiem_chon_tap_con.json")
    rng = np.random.default_rng(0)
    selected = {}
    for key, full, count in [("train", train, 16), ("dev", dev, 4)]:
        chosen = sorted(full[int(i)] for i in rng.permutation(len(full))[:count])
        assert chosen == selection[key]["video"]
        selected[key] = chosen
    split_result = {"train_videos":1261, "dev_videos":139, "common":0,
                    "native_manifest_video_matches":1400, "overlap_native_val_names":0,
                    "split_raw_sha256":sha(raw), "split_list_sha256":list_hash,
                    "source_label_fingerprint":fingerprint, "smoke_rng_selection_matches":True,
                    "crlf_bytes":raw.count(b"\r\n"), "strata_sizes_rounding_match":True,
                    "stratum_membership_recomputed_from_raw_labels":False}
    (a.out / "split_verification.json").write_text(json.dumps(split_result, indent=2) + "\n")

    records = {}
    for model in ("yolov8n", "yolo11n"):
        r = load(logs / f"12_ket_qua_runner_{model}_lan2.json")
        assert r["che_do"] == "smoke" and r["loi"] is None and r["loss_huu_han"]
        assert len(r["epoch"]) == 1
        args = r["hieu_luc"]["args"]
        assert args["epochs"] == 1 and args["batch"] == 8 and args["imgsz"] == 640
        assert args["cache"] is False and args["resume"] is False
        assert args["optimizer"] == "SGD" and args["seed"] == 0
        ds = r["dataset"]
        assert ds["split_da_khoa"]["sha256"] == sha(raw)
        assert ds["split_da_khoa"]["sha256_danh_sach"] == list_hash
        assert ds["so_anh"] == {"train":334, "dev":84}
        for part, count in [("train",334),("dev",84)]:
            loaded = r["dau_vao_da_nap"][part]
            assert loaded["so_anh_nap"] == loaded["so_anh_da_xac_minh"] == count
            assert all(loaded[k] == 0 for k in ("thieu","thua","nhan_khac","npy_canh_anh"))
        for ckpt in r["checkpoint"].values():
            assert ckpt["nap_lai"] and ckpt["dung_5_lop"] and not ckpt["train_args_lech_cau_hinh"]
        meta = (logs / f"12_smoke_{model}_lan2.meta.txt").read_text()
        assert "exit_runner=0" in meta and "exit_moc_gio=0" in meta
        records[model] = r
    for field in ("manifest_sha256", "sha256_data_yaml", "sha256_video_json", "sha256_tom_tat"):
        assert records["yolov8n"]["dataset"][field] == records["yolo11n"]["dataset"][field]

    # Compute time totals independently of the provided estimate script.
    estimate = load(logs / "15_du_toan_training.json")
    count_data = load(logs / "14_dem_anh_day_du.json")
    diffs = []
    for option in estimate["thoi_gian_theo_phuong_an"]:
        b = str(option["buoc_frame"])
        E = option["epoch"]
        c = count_data["theo_buoc"][b]
        nt, nv = c["train"]["so_anh"], c["dev"]["so_anh"]
        source_bytes = c["train"]["byte_anh_nguon"] + c["dev"]["byte_anh_nguon"]
        ignored = c["train"]["anh_to_xam_xap_xi"] + c["dev"]["anh_to_xam_xap_xi"]
        exported = source_bytes * (1 + .551 * ignored/(nt+nv)) + (nt+nv) * (4096*1.4+600)
        sums = [0., 0.]
        for name, r in records.items():
            epoch = r["epoch"][0]
            batch_s = epoch["on_dinh"]["s_moi_batch_trung_vi"]
            init = estimate["do_o_smoke"][name]["khoi_tao_s"] + exported/42e6 + (nt+nv)/1100
            residual = round(epoch["train_s"]-epoch["so_batch"]*batch_s,1)
            train_s = math.ceil(nt/8)*batch_s
            speed = estimate["do_o_smoke"][name]["ms_anh_val_speed"]
            validation = epoch["val_s"] + (nv-84)*sum(speed.values())/1000
            validation_high = validation + (nv-84)*.005
            central = init + residual + E*(train_s+validation)+validation
            high = (init+residual+E*(train_s+validation_high)+validation_high)*1.3
            reported = option["theo_model"][name]
            assert round(central/3600,2) == reported["tong_h_trung_tam"]
            assert round(high/3600,2) == reported["tong_h_cao"]
            sums[0] += central; sums[1] += high
        assert round(sums[0]/3600,2) == option["tong_hai_model_h_trung_tam"]
        assert round(sums[1]/3600,2) == option["tong_hai_model_h_cao"]
        diffs.append({"stride":int(b), "epochs":E, "hours":round(sums[0]/3600,2),
                      "high_scenario_hours":round(sums[1]/3600,2)})
    smoke_result = {"models_checked":list(records), "epochs_per_model":1, "images_train":334,
                    "images_dev":84, "matching_dataset_fingerprints":True,
                    "gpu_runs_repeated_by_reviewer":False, "checkpoint_bytes_loaded_by_reviewer":False,
                    "independent_estimate_arithmetic":diffs}
    (a.out / "smoke_estimate_verification.json").write_text(json.dumps(smoke_result, indent=2) + "\n")
    print(json.dumps({"split":split_result,"smoke_and_estimate":smoke_result}, indent=2))


if __name__ == "__main__":
    main()
