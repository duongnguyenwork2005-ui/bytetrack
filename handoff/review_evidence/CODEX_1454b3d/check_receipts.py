"""Independently reconcile supplied bytes/CSV/JSON. Does not rerun training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
root = a.tree / "handoff/logs/review39"
checks, ram_table, training = [], [], []


def check(name, ok):
    checks.append({"name": name, "pass": bool(ok)})


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


receipt = read(root / "06_kiem_resume_gpu.json")
check("GPU test configuration hash", sha(root / "cau_hinh_kiem_resume_3epoch.yaml") == receipt["cau_hinh_sha256"])
for model in receipt["model"]:
    name = model["model"]
    md = root / "05_gpu" / (name + "_luot")
    for f, h in model["bang_chung_da_chep"].items():
        check(name + ": copied evidence hash " + f, sha(md / f) == h)
    journals = {k: [json.loads(s) for s in (md / (k + "__nhat_ky_luot.jsonl")).read_text().splitlines() if s.strip()]
                for k in ("C", "I")}
    for run in ("C", "I", "R"):
        d = root / "05_gpu" / (name + "_" + run)
        rows = list(csv.DictReader((d / "ram_mau_tho.csv").open(encoding="utf-8", newline="")))
        summary, status = read(d / "tom_tat.json"), read(d / "trang_thai.json")
        j = journals["C" if run == "C" else "I"]
        pid = [e["pid"] for e in j if e["loai"] == "bat_dau" and e["lan"] == (1 if run == "R" else 0)]
        label = name + "_" + run
        check(label + ": PID CSV == monitor == runner journal", {int(r["pid_chinh"]) for r in rows} == set(pid) == {summary["pid"]})
        check(label + ": status PID/create time", status["pid"] == summary["pid"] and status["create_time"] == summary["create_time"])
        expected_exit = 15 if run == "I" else 0
        check(label + ": actual exit recorded in status and summary", status["exit_code"] == summary["exit_code"] == expected_exit)
        check(label + ": final log exit", (d / "log.txt").read_text(encoding="utf-8").rstrip().endswith("# exit=" + str(expected_exit)))
        ts = [float(r["t_unix"]) for r in rows]
        check(label + ": sample count and increasing time", len(rows) == summary["ram"]["so_mau"] and all(y > x for x, y in zip(ts, ts[1:])))
        check(label + ": sampled duration", round(ts[-1] - ts[0], 1) == summary["ram"]["khoang_do_s"])
        table = {"run": label, "pid": pid[0], "samples": len(rows), "exit": expected_exit}
        for col in ("rss_chinh_mib", "rss_tong_mib", "so_con", "ram_he_thong_con_trong_mib"):
            values = sorted(float(r[col]) for r in rows if r[col] != "")
            recomputed = {"n": len(values), "min": min(values), "max": max(values), "trung_vi": values[len(values)//2]}
            check(label + ": summary " + col, recomputed == summary["ram"][col])
            if col == "ram_he_thong_con_trong_mib":
                table["available_min_mib"] = min(values)
            elif col.startswith("rss"):
                table[col + "_max"] = max(values)
        check(label + ": all RAM observations positive/finite", all(r[c] and math.isfinite(float(r[c])) and float(r[c]) > 0
              for r in rows for c in ("rss_chinh_mib", "rss_tong_mib", "ram_he_thong_con_trong_mib")))
        check(label + ": RSS total sum within CSV rounding", all(abs(float(r["rss_tong_mib"]) - float(r["rss_chinh_mib"]) - float(r["rss_con_mib"])) < .151 for r in rows))
        ram_table.append(table)
    batches = {k: list(csv.DictReader((md / (k + "__theo_batch.csv")).open(newline="", encoding="utf-8"))) for k in ("C", "I")}
    c, ir = batches["C"], batches["I"]
    check(name + ": 126 batches each, unique ni=0..125", all(len(v) == 126 and [int(r["ni"]) for r in v] == list(range(126)) for v in (c, ir)))
    check(name + ": resume lane after epoch 1", all(int(r["lan"]) == (0 if int(r["epoch"]) == 1 else 1) for r in ir))
    check(name + ": finite losses", all(math.isfinite(float(r["loss_tong"])) for r in c + ir))
    columns = ("ni", "epoch", "i", "lr_weight", "lr_bn", "lr_bias", "momentum_weight", "momentum_bn", "momentum_bias", "batch_size")
    check(name + ": post-resume LR/momentum/batch match continuous", all(all(x[k] == y[k] for k in columns) for x,y in zip(c[42:], ir[42:])))
    rj = read(md / ("ket_qua_runner__" + name + "_I__tiep_tuc_1.json"))
    resumed = next(e for e in journals["I"] if e["loai"] == "bat_dau" and e["lan"] == 1)
    flags = {k: v["dat"] for k,v in resumed["kiem_khoi_phuc"].items()}
    check(name + ": 13 declared restoration checks agree across receipts", len(flags) == 13 and all(flags.values()) and flags == model["kiem"]["K3_kiem_khoi_phuc"] and resumed["kiem_khoi_phuc"] == rj["kiem_khoi_phuc"])
    check(name + ": selected source is recorded epoch 1 checkpoint", rj["tiep_tuc"]["chon"]["sha256"] == journals["I"][1]["last_sha256"])
    check(name + ": completed epochs exactly 1,2,3", all([e["epoch"] for e in j if e["loai"] == "epoch_da_luu"] == [1,2,3] for j in journals.values()))
    mosaic = {k:{e["epoch"]: e["mosaic_p"] for e in j if e["loai"] == "epoch_da_luu"} for k,j in journals.items()}
    check(name + ": logged mosaic p 1,0,0", all(v == {1:1.0,2:0.0,3:0.0} for v in mosaic.values()))
    loss_diff = sum(x["loss_tong"] != y["loss_tong"] for x,y in zip(c[42:], ir[42:]))
    order_diff = sum(x["thu_tu_batch"] != y["thu_tu_batch"] for x,y in zip(c[42:], ir[42:]))
    check(name + ": 84 post-resume loss/order differences", loss_diff == order_diff == 84)
    training.append({"model":name,"rows_continuous":len(c),"rows_interrupted_resumed":len(ir),"post_resume_loss_diff":loss_diff,
                     "post_resume_order_diff":order_diff,"restoration_flags_from_receipts":flags,"mosaic_p":mosaic})
r = {"scope":"independent byte/hash and CSV/JSON reconciliation; NO checkpoint tensors, GPU training, dataset or Windows signal rerun",
     "median_convention":"summary uses sorted[n//2], including even n; checked the declared implementation, not statistical inference",
     "ram":ram_table,"training":training,"checks":checks,"pass":sum(x["pass"] for x in checks),"fail":sum(not x["pass"] for x in checks)}
a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"pass":r["pass"],"fail":r["fail"],"failed":[x for x in checks if not x["pass"]],"ram":ram_table},ensure_ascii=False,indent=2))
raise SystemExit(int(r["fail"]>0))
