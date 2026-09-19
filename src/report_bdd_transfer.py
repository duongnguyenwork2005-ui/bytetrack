"""
report_bdd_transfer.py -- Bao cao thi nghiem CHUYEN CAU HINH UA-DETRAC -> BDD100K val.

Chi DOC ket qua da co (`results/phase2/<exp>/<cond>/`) va trinh bay dung pham vi; KHONG chay lai
thi nghiem, KHONG chon lai tham so.

Sinh:
    <out>/bang_chinh.md          HOTA/DetA/AssA/IDF1/FP/FN/IDSW theo TUNG dieu kien confidence
    <out>/cap_so_sanh.md         chenh lech ghep cap theo video (bootstrap 5000, seed 0)
    <out>/pf_seed.md             bien thien giua 3 seed cua PF va cach tong hop
    <out>/lop_gt_det.md          lop duoc danh gia, anh xa COCO-BDD, so luong GT/detection theo lop
    <out>/toc_do.md              thoi gian tracker TU CACHE (rieng khoi thoi gian co detector)
    <out>/theo_video.csv         bang tung video

    python src/report_bdd_transfer.py --exp configs/phase2/experiments/bdd_transfer_cocovid_val.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from mot_datasets.bdd100k_coco import BDD100KCocoDataset, parse_coco  # noqa: E402
from mot_datasets.bdd100k_mot import COCO_TO_BDD, VEHICLE_CATEGORIES  # noqa: E402
from run_experiment import load_exp  # noqa: E402

M = ["HOTA", "DetA", "AssA", "IDF1", "FP", "FN", "IDSW"]


def fmt(v, n=4):
    return "—" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.{n}f}")


def bang_chinh(exp: dict, out: Path) -> None:
    lines = ["# Bảng chính — chuyển cấu hình UA-DETRAC → BDD100K val", "",
             f"Thí nghiệm `{exp['name']}`, split `{exp['split_name']}`, "
             f"nguồn nhãn **COCO-video chuyển đổi** (không phải `box_track_20` native).", "",
             "> Đây là **kết quả thực nghiệm** trên toàn bộ val. Phân tầng che khuất **không** được báo "
             "cáo: nguồn nhãn thiếu `occluded`.", ""]
    for cond in exp["conditions"]:
        f = config.RESULTS_DIR / "phase2" / exp["name"] / cond / "summary.csv"
        if not f.exists():
            lines += [f"## Điều kiện `{cond}` — CHƯA CÓ KẾT QUẢ ({f})", ""]
            continue
        d = pd.read_csv(f)
        conf = exp["conditions"][cond]["conf_min"]
        lines += [f"## Điều kiện `{cond}` (conf > {conf})", "",
                  "| tracker | " + " | ".join(M) + " |", "|:--|" + "--:|" * len(M)]
        for _, r in d.iterrows():
            lines.append(f"| {r['tracker']} | " + " | ".join(fmt(r.get(m)) for m in M) + " |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cap_so_sanh(exp: dict, out: Path) -> None:
    lines = ["# Cặp so sánh — chênh lệch ghép cặp theo video", "",
             "Bootstrap **5000 lần, seed 0**, ghép cặp **theo video**; COMBINED tái tổ hợp nhất quán với "
             "bảng chính (`src/boot_stats.py`). Các cặp đã **khóa trước** trong YAML.", ""]
    for cond in exp["conditions"]:
        f = config.RESULTS_DIR / "phase2" / exp["name"] / cond / "paired_pairs.csv"
        if not f.exists():
            lines += [f"## `{cond}` — CHƯA CÓ", ""]
            continue
        d = pd.read_csv(f)
        d = d[d.method == "combined_resample_videos"]
        lines += [f"## `{cond}`", "", "| cặp | metric | Δ (COMBINED) | CI 95 % | CI chứa 0 |",
                  "|:--|:--|--:|:--|:--|"]
        for _, r in d.iterrows():
            if r.metric not in ("HOTA", "IDF1", "AssA", "IDSW"):
                continue
            lines.append(f"| {r.tracker} − {r.baseline} | {r.metric} | {fmt(r.diff_combined)} | "
                         f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}] | "
                         f"{'**có**' if r.ci_contains_zero else 'không'} |")
        lines.append("")
    lines += ["", "**Đọc đúng mức bằng chứng**: CI chứa 0 ⇒ *không đủ bằng chứng về khác biệt*, "
                  "**không** phải bằng chứng về sự tương đương.", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pf_seed(exp: dict, out: Path) -> None:
    lines = ["# PF — biến thiên giữa ba seed và cách tổng hợp", "",
             "Cách tổng hợp **đã khóa trước** (`results/phase2/lock/lock_bdd_transfer.json`):", "",
             "- **Bảng chính**: trung bình **metric** theo seed.",
             "- **Bootstrap**: trung bình **thống kê đủ** theo seed rồi tái tổ hợp.",
             "- Hai đại lượng này **khác nhau** với metric phi tuyến (xem `PHASE2_REPORT.md` mục 15.2).", ""]
    for cond in exp["conditions"]:
        f = config.RESULTS_DIR / "phase2" / exp["name"] / cond / "summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        pf = d[d.tracker.astype(str).str.startswith("pf")]
        if not len(pf):
            continue
        lines += [f"## `{cond}`", "", "| tracker | n_seeds | HOTA | SD HOTA | SD IDF1 | SD AssA |",
                  "|:--|--:|--:|--:|--:|--:|"]
        for _, r in pf.iterrows():
            lines.append(f"| {r['tracker']} | {fmt(r.get('n_seeds'), 0)} | {fmt(r.get('HOTA'))} | "
                         f"{fmt(r.get('HOTA_sd_seed'), 5)} | {fmt(r.get('IDF1_sd_seed'), 5)} | "
                         f"{fmt(r.get('AssA_sd_seed'), 5)} |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def lop_gt_det(exp: dict, labels: str, out: Path) -> None:
    ds = BDD100KCocoDataset(labels)
    df = parse_coco(ds._load(), part=exp["part"])
    gt = df.category.value_counts()
    muc_tieu = set(VEHICLE_CATEGORIES)
    lines = ["# Lớp được đánh giá, ánh xạ COCO–BDD, số lượng GT/detection", "",
             "## Lớp được đánh giá", "",
             f"Mục tiêu đánh giá là **`vehicle`** = {', '.join(sorted(muc_tieu))} → gộp thành **một lớp** "
             "(class 1) trong GT MOTChallenge, liên tục với UA-DETRAC.", "",
             "- `train` (tàu hỏa) → **vùng bỏ qua**.",
             "- `pedestrian`, `rider` → **không** đánh giá (detector COCO không có lớp `rider`); "
             "chỉ thống kê.", "",
             "> Vì GT gộp 5 lớp xe thành một lớp, **metric riêng từng lớp chưa được báo cáo**: đường tách "
             "lớp (`split_tracker_output_by_class`) và chính sách ignore theo lớp **chưa được kiểm chứng** "
             "trên nguồn này.", "",
             "## Ánh xạ COCO → BDD (detector yolov8n pretrained COCO)", "",
             "| COCO id | lớp BDD | thuộc mục tiêu `vehicle` |", "|--:|:--|:--|"]
    for cid, name in sorted(COCO_TO_BDD.items()):
        lines.append(f"| {cid} | {name} | {'**có**' if name in muc_tieu else 'không'} |")
    lines += ["", "`person` (COCO id 0) **không** được ánh xạ: COCO gộp `pedestrian` và `rider`.", "",
              "## Số lượng GT theo lớp (toàn bộ val)", "",
              "| lớp | số box GT | tỉ lệ | vai trò |", "|:--|--:|--:|:--|"]
    tong = int(gt.sum())
    for k, v in gt.items():
        vai = ("mục tiêu" if k in muc_tieu else ("vùng bỏ qua" if k == "train" else "chỉ thống kê"))
        lines.append(f"| {k} | {int(v):,} | {100 * v / tong:.2f} % | {vai} |")
    mc = int(gt.get("motorcycle", 0))
    lines += ["", f"> **`motorcycle`: {mc:,} box ({100 * mc / tong:.2f} %)** — rất hiếm trong BDD val. "
                  "Với đề tài giao thông hỗn hợp (xe máy chiếm đa số), **BDD val không phải là tập đại "
                  "diện**; mọi kết luận về xe máy từ nguồn này đều dựa trên mẫu rất nhỏ.", "",
              "## Số detection theo lớp (từ detection cache, sau khi lọc)", ""]
    dem = _det_counts(exp)
    ten = {v: k for k, v in COCO_TO_BDD.items()}
    lines += ["Số detection đưa vào tracker, đếm từ **chính detection cache** sau khi lọc theo "
              "`classes` và ngưỡng confidence của từng điều kiện (trước association):", "",
              "| lớp | COCO id | " + " | ".join(f"det `{c}`" for c in exp["conditions"])
              + " | GT | det/GT `" + list(exp["conditions"])[0] + "` |",
              "|:--|--:|" + "--:|" * (len(exp["conditions"]) + 2)]
    c0 = list(exp["conditions"])[0]
    for cid in sorted(exp["classes"]):
        nb = COCO_TO_BDD.get(cid, str(cid))
        g = int(gt.get(nb, 0))
        o = [f"| {nb} | {cid} "]
        for c in exp["conditions"]:
            o.append(f"| {dem.get(c, {}).get(cid, 0):,} ")
        r = dem.get(c0, {}).get(cid, 0) / g if g else float("nan")
        o.append(f"| {g:,} | {fmt(r, 2)} |")
        lines.append("".join(o))
    lines += ["",
              "> `det/GT` **không** là độ chính xác: detection chưa được ghép với GT và hai bên dùng "
              "**định nghĩa lớp khác nhau** (COCO vs BDD). Nó chỉ cho thấy detector **đề xuất nhiều hay "
              "ít hơn** GT ở mỗi lớp.", "",
              ]
    g_mo = int(gt.get("motorcycle", 0))
    ty = {c: (dem.get(c, {}).get(3, 0) / g_mo if g_mo else float("nan")) for c in exp["conditions"]}
    lines += ["> **Xe máy** — số đề xuất so với GT đổi chiều theo ngưỡng: "
              + "; ".join(f"`{c}` {dem.get(c, {}).get(3, 0):,}/{g_mo:,} = {ty[c]:.2f}"
                          for c in exp["conditions"])
              + ". Vì GT gộp 5 lớp xe thành một, **nhầm lớp giữa các lớp xe không bị phạt**, nên "
              "những con số này **không** nói được detector phát hiện xe máy tốt hay kém; muốn biết "
              "phải ghép theo lớp, mà đường đó chưa được kiểm chứng trên nguồn này.", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _det_counts(exp: dict) -> dict:
    """Dem detection THEO LOP COCO tu cache, ung voi nguong cua TUNG dieu kien.

    Doc dung cache ma tracker da dung; loc y het duong chay that (`classes` + `conf_min`).
    Cot cua `data`: [x1, y1, x2, y2, conf, cls].
    """
    import numpy as np
    d = config.INTERIM_DIR / "det_cache" / exp["cache_id"]
    giu = set(exp["classes"])
    out = {c: collections.Counter() for c in exp["conditions"]}
    for f in sorted(d.glob("*.npz")):
        with np.load(f) as z:
            a = z["data"]
        if not len(a):
            continue
        cls, conf = a[:, 5].astype(int), a[:, 4]
        m0 = np.isin(cls, list(giu))
        for c, cc in exp["conditions"].items():
            m = m0 & (conf > float(cc["conf_min"]))
            for k, v in zip(*np.unique(cls[m], return_counts=True)):
                out[c][int(k)] += int(v)
    return out


def toc_do(exp: dict, out: Path) -> None:
    lines = ["# Tốc độ — tách **tracker từ cache** khỏi **toàn pipeline**", "",
             "**Chỉ là số đo mô tả của chính các lượt đã chạy** (máy còn chạy việc khác, không lặp có "
             "kiểm soát) ⇒ **không** đủ để xếp hạng hiệu năng.", "",
             "Ba cột **không** phải ba pha độc lập: `update` = **tổng** lời gọi `tracker.update()`; "
             "`predict` = chỉ `multi_predict`; phần còn lại = `update − predict`, **bao gồm** "
             "association, bước cập nhật của bộ lọc và quản lý vòng đời.", "",
             "## 1. Tracker chạy TỪ CACHE (không gồm detector)", "",
             "Đo từ cache **loại** chi phí **detector**. Mục 2 ghép thêm chi phí detector đã ghi "
             "lúc dựng cache để có con số **toàn pipeline**.", ""]
    for cond in exp["conditions"]:
        f = config.RESULTS_DIR / "phase2" / exp["name"] / cond / "summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        lines += [f"### `{cond}`", "", "| tracker | update ms/frame | trong đó predict ms | còn lại ms |",
                  "|:--|--:|--:|--:|"]
        for _, r in d.iterrows():
            u, p = r.get("update_ms_per_frame"), r.get("predict_ms_per_frame")
            lines.append(f"| {r['tracker']} | {fmt(u, 3)} | {fmt(p, 3)} | {fmt(u - p, 3)} |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detector_seconds(exp: dict) -> tuple[float, float, int]:
    """Tong thoi gian detector + so frame, doc tu sidecar cache (truong `seconds`).

    `seconds` do luc DUNG CACHE: gom doc/giai ma JPEG + suy luan + NMS cho ca 80 lop COCO
    (`max_det` 3000), KHONG phai chi rieng suy luan.
    """
    d = config.INTERIM_DIR / "det_cache" / exp["cache_id"]
    tong_s, tong_f = 0.0, 0
    for f in sorted(d.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        tong_s += float(m.get("seconds") or 0.0)
        tong_f += int(m.get("n_frames") or 0)
    return tong_s, (1000.0 * tong_s / tong_f if tong_f else float("nan")), tong_f


def toan_pipeline(exp: dict, out: Path) -> None:
    """Muc 2 cua bao cao toc do: detector + tracker, noi ro day la TONG HAI PHA DO RIENG."""
    tong_s, det_ms, n_frame = _detector_seconds(exp)
    lines = ["", "## 2. Toàn pipeline (detector + tracker)", "",
             "⚠️ Đây là **tổng của hai pha đo RIÊNG** (detector đo lúc dựng cache, tracker đo lúc chạy "
             "ma trận), **không phải** một phép đo end-to-end. Chạy thật một lượt liền mạch có thể khác "
             "(chồng lấn I/O, giữ GPU nóng, không ghi/đọc cache).", "",
             f"- Detector (`yolov8n`, conf 0,05, **80 lớp COCO**, `max_det` 3000): "
             f"**{det_ms:.1f} ms/frame** — {tong_s / 60:.1f} phút cho {n_frame:,} frame.",
             "- Số này **gom cả đọc + giải mã JPEG**, không phải riêng suy luận.",
             "- Detector ở đây **dư thừa có chủ ý** cho mục đích cache (giữ 80 lớp để mọi điều kiện dùng chung "
             "một cache); một triển khai thật chỉ cần 5 lớp và `max_det` 300 sẽ **nhanh hơn**. "
             "Vì vậy con số này là **cận trên**, không dùng để suy ra FPS triển khai.", ""]
    for cond in exp["conditions"]:
        f = config.RESULTS_DIR / "phase2" / exp["name"] / cond / "summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        lines += [f"### `{cond}`", "",
                  "| tracker | detector ms | tracker ms | tổng ms/frame | FPS suy ra | tracker chiếm |",
                  "|:--|--:|--:|--:|--:|--:|"]
        for _, r in d.iterrows():
            u = r.get("update_ms_per_frame")
            if pd.isna(u):
                continue
            tot = det_ms + u
            lines.append(f"| {r['tracker']} | {det_ms:.1f} | {u:.3f} | {tot:.1f} | {1000 / tot:.1f} | "
                         f"{100 * u / tot:.1f} % |")
        lines.append("")
    lines += ["> **Đọc đúng mức**: với cấu hình cache này, chi phí **detector áp đảo**; chênh lệch giữa "
              "các bộ lọc gần như không đổi FPS toàn pipeline trên máy này. Điều này **không** suy ra được cho "
              "Jetson Nano: tỷ lệ CPU/GPU khác hẳn, chưa đo.", ""]
    with out.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--labels", default="data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exp = load_exp(a.exp)
    out = Path(a.out) if a.out else (config.RESULTS_DIR / "phase2" / exp["name"] / "bao_cao")
    out.mkdir(parents=True, exist_ok=True)

    bang_chinh(exp, out / "bang_chinh.md")
    cap_so_sanh(exp, out / "cap_so_sanh.md")
    pf_seed(exp, out / "pf_seed.md")
    lop_gt_det(exp, a.labels, out / "lop_gt_det.md")
    toc_do(exp, out / "toc_do.md")
    toan_pipeline(exp, out / "toc_do.md")

    per = []
    for cond in exp["conditions"]:
        f = config.RESULTS_DIR / "phase2" / exp["name"] / cond / "per_video.csv"
        if f.exists():
            d = pd.read_csv(f)
            d.insert(0, "cond", cond)
            per.append(d)
    if per:
        pd.concat(per).to_csv(out / "theo_video.csv", index=False)
    print(f"-> {out}")
    for p in sorted(out.iterdir()):
        print(f"   {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
