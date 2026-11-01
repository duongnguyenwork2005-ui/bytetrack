#!/usr/bin/env python3
"""R63 - doc log trtexec (TensorRT 8.2) -> JSON. Chay duoc tren PC va Python 3.6 cua Nano (khong f-string
phuc tap, khong dataclass, chi thu vien chuan).

Cach dung: python3 phan_tich_trtexec.py <log> [<log> ...] [--json tom_tat.json]
Moi log: thong ke dong "=== Performance summary ===" (Throughput, Latency, End-to-End Host Latency, Enqueue Time,
H2D Latency, GPU Compute Time, D2H Latency: min/max/mean/median/percentile), tong walltime, top lop cua
--dumpProfile neu co, va median/p90 tinh lai tu file --exportTimes (<log bo .log>.times.json) neu ton tai.
"""
import argparse
import json
import os
import re
import sys

RE_TIEN_TO = re.compile(r"^(?:\[[^\]]*\]\s*)*")            # bo "[10/30/2026-10:00:00] [I] "
RE_THONG_KE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 \-]*?):\s+(min = .*)$")
RE_CAP = re.compile(r"(min|max|mean|median|percentile\((\d+(?:\.\d+)?)%\))\s*=\s*([-\d.eE+]+)\s*ms")
RE_DON = re.compile(r"^(Throughput|Total Host Walltime|Total GPU Compute Time):\s+([-\d.eE+]+)\s*(\S+)")
RE_LOP = re.compile(r"^(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def bo_tien_to(dong):
    return RE_TIEN_TO.sub("", dong.rstrip("\r\n")).strip()


def doc_log(duong):
    kq = {"log": os.path.basename(duong), "lenh": None, "thong_ke_ms": {}, "don": {}, "lop_top": [], "exit": None,
          "loi": []}
    trong_tom_tat = trong_profile = False
    lop = []
    with open(duong, errors="replace") as f:
        for dong_goc in f:
            if dong_goc.startswith("== ") and kq["lenh"] is None:
                kq["lenh"] = dong_goc[3:].strip()
            m = re.match(r"^\[exit=(-?\d+)\]", dong_goc.strip())
            if m:
                kq["exit"] = int(m.group(1))
            d = bo_tien_to(dong_goc)
            if "Performance summary" in d:
                trong_tom_tat, trong_profile = True, False
                continue
            if d.startswith("=== Profile"):
                trong_profile, trong_tom_tat = True, False
                continue
            if d.startswith("===") and trong_tom_tat:
                trong_tom_tat = False
            if re.search(r"\b(Error|FAILED|failed)\b", d) and "PASSED" not in d:
                kq["loi"].append(d[:200])
            if trong_tom_tat:
                m = RE_THONG_KE.match(d)
                if m:
                    ten = m.group(1).strip()
                    tk = {}
                    for k, p, v in RE_CAP.findall(m.group(2)):
                        tk["p%s" % p if p else k] = float(v)
                    kq["thong_ke_ms"][ten] = tk
                    continue
                m = RE_DON.match(d)
                if m:
                    kq["don"][m.group(1)] = {"gia_tri": float(m.group(2)), "don_vi": m.group(3)}
            elif trong_profile:
                m = RE_LOP.match(d)
                if m and not d.startswith("Layer"):
                    lop.append({"lop": m.group(1).strip(), "tong_ms": float(m.group(2)), "tb_ms": float(m.group(3)),
                                "phan_tram": float(m.group(4))})
    # dong "Total" cua profile khong phai mot lop
    lop = [x for x in lop if x["lop"].lower() != "total"]
    kq["lop_top"] = sorted(lop, key=lambda x: -x["tb_ms"])[:15]
    kq["so_lop_profile"] = len(lop)
    times = re.sub(r"\.log$", "", duong) + ".times.json"
    if os.path.exists(times):
        kq["export_times"] = tom_tat_times(times)
    return kq


def phan_vi(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (k - i)


def tom_tat_times(duong):
    with open(duong) as f:
        ds = json.load(f)
    if not isinstance(ds, list) or not ds:
        return {"n": 0}
    out = {"n": len(ds)}
    for k in sorted(ds[0]):
        if not k.endswith("Ms") or k.startswith(("start", "end")) and k != "endToEndMs":
            continue
        xs = [float(d[k]) for d in ds if k in d]
        out[k] = {"median": phan_vi(xs, 0.5), "p90": phan_vi(xs, 0.9), "mean": sum(xs) / len(xs)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    tat_ca = [doc_log(p) for p in a.logs]
    for kq in tat_ca:
        g = kq["thong_ke_ms"].get("GPU Compute Time", {})
        l = kq["thong_ke_ms"].get("Latency", kq["thong_ke_ms"].get("Host Latency", {}))
        print("%-34s exit=%s gpu_median=%s latency_median=%s thong_ke=%d lop=%d" % (
            kq["log"], kq["exit"], g.get("median"), l.get("median"), len(kq["thong_ke_ms"]), kq["so_lop_profile"]))
    # Luot hop le chi khi: co dong [exit=0], khong co dong loi, co GPU Compute Time (GPT review R63 Q7).
    hong = []
    for k in tat_ca:
        ly_do = []
        if k["exit"] is None:
            ly_do.append("thieu exit")
        elif k["exit"] != 0:
            ly_do.append("exit=%d" % k["exit"])
        if k["loi"]:
            ly_do.append("co %d dong loi" % len(k["loi"]))
        if "GPU Compute Time" not in k["thong_ke_ms"]:
            ly_do.append("thieu GPU Compute Time")
        k["hop_le"] = not ly_do
        k["ly_do_khong_hop_le"] = ly_do
        if ly_do:
            hong.append("%s (%s)" % (k["log"], ", ".join(ly_do)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(tat_ca, f, indent=2, ensure_ascii=False)
    if hong:
        print("[LOI] luot khong hop le: %s" % "; ".join(hong), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
