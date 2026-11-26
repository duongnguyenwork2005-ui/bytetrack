"""Test R53B `chuan_bi_danh_gia.py` trên fixture tổng hợp (không dữ liệu thật, không nạp model, không GPU).

Mỗi lần chạy ghi vào thư mục MỚI results/phase2/r53_detector/tests_eval/lan_<thoi diem>_pid<pid>/, không xóa, không
ghi đè. Fixture UA-DETRAC dùng lại ảnh/XML/box của test adapter cũ; fingerprint dựng bằng hàm băm của mốc A vì script
mốc A từ chối ghi vào results/; fixture BDD tự dựng (COCO-video nhỏ, đủ năm lớp). Dự đoán giả lập thay `du_doan_ultralytics` ở module adapter (ranh giới nạp
model); mọi bước khác (cổng adapter, scorer, D3, NPZ, khóa) chạy thật qua `main(argv, ch)`.

  A  ánh xạ COCO-80 / 5 lớp theo tên; loại checkpoint sai -> smoke exit 3.
  M  chọn mẫu: ảnh đầu có GT từng lớp, loại trùng, bổ sung tới 64, giữ thứ tự; ít ảnh -> không vượt; thiếu lớp -> lỗi.
  T  tiền kiểm: lệch hồ sơ -> exit 2, không ghi; đạt -> exit 0; chạy lại -> exit 1, file cũ không đổi.
  K  khóa: thiếu tài liệu / thiếu lệnh / còn dấu hiệu chưa xong -> exit 2, không tạo khóa; đạt -> exit 0; chạy lại -> 1.
  S  smoke trước khóa -> exit 2; sáu smoke giả lập -> exit 0, cùng mẫu, tham số dự đoán đúng khóa, NPZ/hash/nguồn,
     kết quả = scorer gọi trực tiếp; D3 (truck/motorcycle/bicycle = N/A); BDD đủ năm lớp; chạy lại -> exit 1.
  H  cổng hash: file khóa bị sửa, checkpoint đổi, SHA256SUMS đổi -> exit 2, KHÔNG gọi dự đoán.

  python src/r53_detector/test_chuan_bi_danh_gia.py
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _luong in (sys.stdout, sys.stderr):
    try:
        _luong.reconfigure(encoding="utf-8")
    except Exception:
        pass

import contextlib  # noqa: E402
import csv  # noqa: E402
import copy  # noqa: E402
import datetime as _dt  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import chuan_bi_danh_gia as cb  # noqa: E402
import test_danh_gia_detector_detrac_test as tdt  # noqa: E402  (chỉ dùng hằng XML / VIDEOS / W / H của fixture)
from detector_classes import LOP_5, mo_ta_anh_xa  # noqa: E402

KQ = []
COCO80 = {i: f"lop_coco_{i}" for i in range(80)}
COCO80.update({1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})


def kiem(ten, ok, chi_tiet=""):
    KQ.append((ten, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {ten}" + (f"  ({chi_tiet})" if chi_tiet else ""))


def chay(argv, ch):
    e = io.StringIO()
    with contextlib.redirect_stderr(e), contextlib.redirect_stdout(io.StringIO()):
        ma = cb.main(argv, ch)
    return ma, e.getvalue()


class Gia:
    """Thay du_doan_ultralytics: ghi lại lời gọi, trả dự đoán định sẵn (conf biểu diễn đúng ở float32)."""

    def __init__(self, ep_names=None):
        self.goi, self.ep_names = [], ep_names

    def __call__(self, weights, anh, qt, batch=16, device="0"):
        self.goi.append({"weights": weights, "anh": [p.name for p in anh], "qt": dict(qt), "batch": batch, "device": device})
        ra = [{"xyxy": np.array([[20, 20, 120, 120], [200, 200, 320, 300], [5, 5, 30, 30]], float),
               "conf": np.array([0.875, 0.5, 0.375]), "cls": np.array([0, 2, 1])} for _ in anh]
        names = self.ep_names or (COCO80 if "coco" in Path(weights).name else list(LOP_5))
        return ra, mo_ta_anh_xa(names)


@contextlib.contextmanager
def gia_lap(g):
    goc = (cb.dg.du_doan_ultralytics, cb.bv.du_doan_ultralytics)
    cb.dg.du_doan_ultralytics = cb.bv.du_doan_ultralytics = g
    try:
        yield g
    finally:
        cb.dg.du_doan_ultralytics, cb.bv.du_doan_ultralytics = goc


def tao_bdd(goc: Path) -> dict:
    """2 video x 3 ảnh 64x48, đủ năm lớp GT, một vùng bỏ qua, một ảnh không annotation."""
    from PIL import Image
    rng = np.random.default_rng(1)
    videos = [{"id": 1, "name": "v1"}, {"id": 2, "name": "v2"}]
    cats = [{"id": 1, "name": "pedestrian"}, {"id": 3, "name": "car"}, {"id": 4, "name": "truck"}, {"id": 5, "name": "bus"},
            {"id": 7, "name": "motorcycle"}, {"id": 8, "name": "bicycle"}]
    images, ann, iid = [], [], 0
    for v in videos:
        (goc / "anh" / v["name"]).mkdir(parents=True)
        for k in range(3):
            iid += 1
            r = f"{v['name']}/{v['name']}-{k + 1:07d}.jpg"
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(goc / "anh" / r, quality=90)
            images.append({"id": iid, "video_id": v["id"], "frame_id": k, "file_name": r, "width": 64, "height": 48})
    for i, (img, cat, box, crowd, ign) in enumerate([(1, 3, [2, 2, 20, 10], 0, 0), (1, 4, [25, 5, 15, 15], 0, 0),
                                                     (2, 5, [1, 1, 12, 12], 0, 0), (2, 3, [40, 25, 10, 10], 0, 1),
                                                     (4, 7, [5, 5, 10, 10], 0, 0), (5, 8, [30, 20, 10, 10], 0, 0),
                                                     (6, 1, [3, 3, 5, 9], 0, 0)], 1):
        ann.append({"id": i, "image_id": img, "category_id": cat, "instance_id": i, "bbox": box, "iscrowd": crowd,
                    "ignore": ign})
    nhan = goc / "nhan.json"
    nhan.write_text(json.dumps({"videos": videos, "images": images, "annotations": ann, "categories": cats}), encoding="utf-8")
    mf = goc / "manifest.json"
    mf.write_text(json.dumps({"videos": ["v1", "v2"], "nguon_nhan_sha256": cb.sha(nhan)}), encoding="utf-8")
    bang = goc / "hash.csv"
    with open(bang, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel", "size", "mtime_ns", "sha256"])
        for im in images:
            p = goc / "anh" / im["file_name"]
            w.writerow([im["file_name"], p.stat().st_size, p.stat().st_mtime_ns, cb.sha(p)])
    return {"nhan": nhan, "anh": goc / "anh", "manifest": mf, "hash_anh": bang}


def tao_detrac(goc: Path) -> dict:
    """Ảnh + XML như `tao_fixture` của test adapter cũ (2 video, 5 ảnh, 1 ảnh D4). Script mốc A từ chối --out trong
    results/ (cây chỉ đọc), nên fingerprint dựng bằng chính hàm băm / đọc bảng của mốc A (fpt), cùng định dạng adapter đọc."""
    from PIL import Image
    fpt = cb.fpt
    anh, xml, fp = goc / "anh", goc / "xml", goc / "fp"
    xml.mkdir(parents=True)
    fp.mkdir()
    for v, (n, frames, ign) in tdt.XML.items():
        (anh / v).mkdir(parents=True)
        for i in range(1, n + 1):
            Image.new("RGB", (tdt.W, tdt.H), (i * 40 % 256, 80, 120)).save(anh / v / f"img{i:05d}.jpg", quality=60)
        dong = [f'<sequence name="{v}">', "<ignored_region>"]
        dong += [f'<box left="{x}" top="{y}" width="{w}" height="{h}"/>' for x, y, w, h in ign]
        dong.append("</ignored_region>")
        for num, tg in frames.items():
            dong.append(f'<frame density="{len(tg)}" num="{num}"><target_list>')
            dong += [f'<target id="{k}"><box left="{x}" top="{y}" width="{w}" height="{h}"/>'
                     f'<attribute vehicle_type="{loai}"/></target>' for k, (loai, x, y, w, h) in enumerate(tg, 1)]
            dong.append("</target_list></frame>")
        dong.append("</sequence>")
        (xml / f"{v}.xml").write_text("\n".join(dong) + "\n", encoding="utf-8")
    mf = goc / "manifest.json"
    mf.write_text(json.dumps({"name": "fixture", "videos": tdt.VIDEOS, "n_videos": len(tdt.VIDEOS)}) + "\n", encoding="utf-8")
    rows, d4, ev_rel, ev_hash, xmls = [], [], [], [], []
    for v in tdt.VIDEOS:
        xmls.append({"video": v, "sha256": cb.sha(xml / f"{v}.xml")})
        _, frames = cb.dg.doc_gt_xml(xml / f"{v}.xml")
        for num, p in fpt.liet_ke_anh(anh / v):
            info, r = fpt.bam_anh(p, Image), f"{v}/{p.name}"
            rows.append({k: info[k] if k != "rel" else r for k in fpt.COT_BANG})
            if num in frames:
                ev_rel.append(r)
                ev_hash.append((r, info["sha256"]))
            else:
                d4.append({"rel": r, "video": v, "frame": num})
    (fp / "detrac_test_sha256.csv").write_bytes(fpt.csv_bytes(rows, fpt.COT_BANG))
    (fp / "detrac_test_loai_D4.csv").write_bytes(fpt.csv_bytes(d4, fpt.COT_D4))
    (fp / "detrac_test_fingerprint.json").write_text(json.dumps({
        "van_tay": {"manifest": {"sha256": cb.sha(mf), "videos": tdt.VIDEOS}, "xml": xmls,
                    "tap_danh_gia_sau_D4": {"so_anh": len(ev_rel), "danh_sach_anh_co_thu_tu_sha256": fpt.sha256_thu_tu(ev_rel),
                                            "noi_dung_anh_sha256_tong_hop": fpt.sha256_ds(ev_hash)}},
        "dau_ra": {"bang": {"sha256": cb.sha(fp / "detrac_test_sha256.csv")},
                   "d4": {"sha256": cb.sha(fp / "detrac_test_loai_D4.csv")}}}), encoding="utf-8")
    return {"anh": anh, "xml": xml, "manifest": mf, "fp": fp}


def tao_ch(goc: Path) -> dict:
    fx = tao_detrac(goc / "fx_detrac")
    models = {}
    for ten, (_, _, lo) in cb.MODELS.items():
        p = goc / "models" / f"{ten}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"checkpoint gia lap {ten}".encode())
        models[ten] = (p, cb.sha(p), lo)
    return {"repo": cb.REPO, "chuan_bi": goc / "chuan_bi", "smoke": goc / "smoke", "doc02": goc / "02_ke_hoach.md",
            "du_lieu": {"detrac_test": {"xml": fx["xml"], "anh": fx["anh"], "manifest": fx["manifest"],
                                        "fingerprint": fx["fp"] / "detrac_test_fingerprint.json",
                                        "hash_anh": fx["fp"] / "detrac_test_sha256.csv",
                                        "loai_d4": fx["fp"] / "detrac_test_loai_D4.csv"},
                        "bdd_val": tao_bdd(goc / "fx_bdd")},
            "ky_vong": {"detrac_test": {"so_video": 2, "dem.anh_tren_dia": 5, "dem.anh_loai_D4": 1, "dem.anh_danh_gia": 4},
                        "bdd_val": {"so_video": 2, "dem.anh": 6}},
            "models": models, "ma": [cb.REPO / m for m in cb.MA], "gb_toi_thieu": 1}


def bam_cay(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): cb.sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


# ----------------------------------------------------------------------------------------------- nhóm
def nhom_a():
    m = mo_ta_anh_xa(COCO80)
    kiem("A1 COCO-80 -> 5 lop theo ten (car 2, truck 7, bus 5, motorcycle 3, bicycle 1)",
         m["loai_checkpoint"] == "coco80" and m["model_id_sang_lop5"] == {"1": 4, "2": 0, "3": 3, "5": 2, "7": 1}, str(m))
    m5 = mo_ta_anh_xa(list(LOP_5))
    kiem("A2 checkpoint 5 lop -> dong nhat 0..4", m5["loai_checkpoint"] == "lop5" and
         m5["model_id_sang_lop5"] == {str(i): i for i in range(5)})
    kiem("A3 khoa ky vong loai checkpoint: coco cho baseline, lop5 cho mixed5",
         {t: v[2] for t, v in cb.MODELS.items()} == {"yolov8n_coco": "coco80", "yolov8n_mixed5": "lop5",
                                                     "yolo11n_mixed5": "lop5"})


def gt_gia(n, lop_tai):
    gt = [{"cls": np.array([], int)} for _ in range(n)]
    for i, cs in lop_tai.items():
        gt[i] = {"cls": np.array(cs, int)}
    return gt


def nhom_m():
    gt = gt_gia(200, {0: [0, 1], 30: [3], 120: [2], 150: [4], 151: [0, 2, 4]})
    chi_so, ly_do = cb.chon_mau(gt, LOP_5)
    ky = sorted({0, 30, 120, 150} | set(range(1, 30)) | set(range(31, 62)))
    kiem("M1 anh dau co GT tung lop + bo sung anh dau toi 64, loai trung, giu thu tu",
         chi_so == ky and len(chi_so) == 64 and ly_do == {"car": 0, "truck": 0, "bus": 120, "motorcycle": 30, "bicycle": 150},
         f"{len(chi_so)} anh, ly do {ly_do}")
    kiem("M2 gioi han 64: hang so va mau DETRAC car/bus", cb.SO_ANH_TOI_DA == 64 and
         len(cb.chon_mau(gt_gia(500, {0: [0], 400: [2]}), ("car", "bus"))[0]) == 64)
    kiem("M3 it anh -> khong vuot so anh co", cb.chon_mau(gt_gia(10, {0: [0], 9: [2]}), ("car", "bus"))[0] == list(range(10)))
    try:
        cb.chon_mau(gt_gia(50, {0: [0, 1, 2, 3]}), LOP_5)
        kiem("M4 thieu lop bicycle -> loi", False)
    except cb.LoiChuanBi as e:
        kiem("M4 thieu lop bicycle -> loi", "bicycle" in str(e), str(e))


def nhom_tks(goc: Path):
    ch = tao_ch(goc)
    # T: lệch hồ sơ -> 2, không ghi
    ch_sai = copy.deepcopy(ch)
    ch_sai["ky_vong"]["detrac_test"]["dem.anh_danh_gia"] = 5
    ma, e = chay(["--buoc", "tien-kiem", "--tap", "detrac_test"], ch_sai)
    kiem("T1 so anh khac ho so -> exit 2, khong ghi", ma == 2 and not ch["chuan_bi"].exists() and "LOI HO SO" in e, e[-200:])
    # S0: smoke trước khóa -> 2, không gọi dự đoán
    ch_tk = dict(ch, smoke=goc / "smoke_truoc_khoa")
    with gia_lap(Gia()) as g:
        ma, e = chay(["--buoc", "smoke", "--tap", "detrac_test", "--model", "yolov8n_coco"], ch_tk)
    kiem("S0 smoke truoc khi khoa -> exit 2, khong du doan", ma == 2 and not g.goi and "LOI KHOA" in e, e[-160:])
    for tap in cb.TAP:
        ma, e = chay(["--buoc", "tien-kiem", "--tap", tap], ch)
        kiem(f"T2 tien kiem {tap} -> exit 0", ma == 0, e[-300:])
    mau = {tap: cb.doc_json(ch["chuan_bi"] / f"mau_smoke_{tap}.json") for tap in cb.TAP}
    kiem("T3 mau DETRAC: 4 anh sau D4, co car + bus; mau BDD: du 5 lop",
         mau["detrac_test"]["so_anh"] == 4 and all(mau["detrac_test"]["gt_theo_lop_trong_mau"][c] for c in ("car", "bus"))
         and all(mau["bdd_val"]["gt_theo_lop_trong_mau"][c] for c in LOP_5), json.dumps(mau["bdd_val"]["ly_do_theo_lop"]))
    truoc = bam_cay(ch["chuan_bi"])
    ma, e = chay(["--buoc", "tien-kiem", "--tap", "bdd_val"], ch)
    kiem("T4 tien kiem lai -> exit 1, file cu khong doi", ma == 1 and bam_cay(ch["chuan_bi"]) == truoc)
    # K: tài liệu 02
    ma, e = chay(["--buoc", "khoa"], ch)
    kiem("K1 thieu tai lieu 02 -> exit 2, khong tao khoa", ma == 2 and not (ch["chuan_bi"] / "SHA256SUMS").exists(), e[-150:])
    can = cb.chuoi_bat_buoc_doc02(cb.hash_ma(ch))
    ch["doc02"].write_text("\n".join(can[1:]) + "\n", encoding="utf-8")
    ma, e = chay(["--buoc", "khoa"], ch)
    kiem("K2 tai lieu 02 thieu mot lenh day du -> exit 2", ma == 2 and "thieu" in e and
         not (ch["chuan_bi"] / "khoa_danh_gia.json").exists(), e[-150:])
    ch["doc02"].write_text("\n".join(can) + "\nTODO\n", encoding="utf-8")
    ma, e = chay(["--buoc", "khoa"], ch)
    kiem("K3 tai lieu 02 con TODO -> exit 2", ma == 2 and "chua xong" in e, e[-150:])
    ch["doc02"].write_text("# ke hoach fixture\n" + "\n".join(can) + "\n", encoding="utf-8")
    ma, e = chay(["--buoc", "khoa"], ch)
    kiem("K4 khoa -> exit 0, co SHA256SUMS + thoi diem", ma == 0 and (ch["chuan_bi"] / "SHA256SUMS").exists() and
         (ch["chuan_bi"] / "thoi_diem_khoa.json").exists(), e[-300:])
    truoc = bam_cay(ch["chuan_bi"])
    ma, e = chay(["--buoc", "khoa"], ch)
    kiem("K5 khoa lai -> exit 1, khong doi", ma == 1 and bam_cay(ch["chuan_bi"]) == truoc)
    khoa = cb.doc_json(ch["chuan_bi"] / "khoa_danh_gia.json")
    kiem("K6 khoa chua 6 lenh day du (batch 16, device 0, conf 0.25, gioi han 0, full/<tap>/<model>) va quy tac",
         len(khoa["lenh_day_du"]) == 6 and all("--batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out "
                                                f"results/phase2/r53_detector/full/{x['tap']}/{x['model']}" in x["lenh"]
                                                for x in khoa["lenh_day_du"])
         and khoa["quy_tac"]["du_doan"]["imgsz"] == 640 and khoa["quy_tac"]["du_doan"]["conf"] == 0.001
         and khoa["quy_tac"]["du_doan"]["iou_nms"] == 0.7 and khoa["quy_tac"]["du_doan"]["max_det"] == 300
         and khoa["quy_tac"]["du_doan"]["half"] is False and khoa["quy_tac"]["van_hanh"] == {"conf": 0.25, "iou": 0.5})
    # S: sáu smoke
    with gia_lap(Gia()) as g:
        ra = {(tap, ten): chay(["--buoc", "smoke", "--tap", tap, "--model", ten], ch) for tap in cb.TAP for ten in cb.MODELS}
    kiem("S1 sau smoke gia lap -> exit 0", all(v[0] == 0 for v in ra.values()),
         str({k: (v[0], v[1][-200:]) for k, v in ra.items() if v[0]}))
    kiem("S2 tham so du doan dung khoa (imgsz 640, conf 0.001, NMS 0.7, max_det 300, FP32, batch 16, device 0)",
         len(g.goi) == 6 and all(x["qt"]["imgsz"] == 640 and x["qt"]["conf"] == 0.001 and x["qt"]["iou_nms"] == 0.7 and
                                 x["qt"]["max_det"] == 300 and x["qt"]["half"] is False and x["qt"]["agnostic_nms"] is False
                                 and x["batch"] == 16 and x["device"] == "0" for x in g.goi))
    for tap in cb.TAP:
        ds = [cb.doc_json(ch["smoke"] / tap / ten / "danh_sach_anh.json")["rel"] for ten in cb.MODELS]
        kiem(f"S3 {tap}: cung mau giua 3 model, = mau da khoa, <= 64",
             ds[0] == ds[1] == ds[2] == khoa["mau_smoke"][tap]["rel"] and len(ds[0]) <= 64)
    ok_npz, ok_cham = True, True
    for tap in cb.TAP:
        for ten in cb.MODELS:
            d = ch["smoke"] / tap / ten
            ng = cb.doc_json(d / f"nguon_du_doan_{ten}.json")
            kq = cb.doc_json(d / "ket_qua.json")
            ok_npz &= ng["npz_sha256"] == cb.sha(d / f"du_doan_{ten}.npz") == kq["mo_hinh"][ten]["npz_sha256"]
            ok_npz &= cb.doc_json(d / "exit.json")["exit"] == 0 and (d / "log.txt").exists()
            ok_cham &= all(kq["kiem"].values()) and kq["canh_bao"] == cb.CANH_BAO_SMOKE
    kiem("S4 NPZ + sha256 khop nguon_du_doan va ket_qua; exit.json 0; log.txt", ok_npz)
    kiem("S5 moi kiem sau du doan dat (cham lai tu NPZ trung, loai checkpoint dung)", ok_cham)
    # D3 + scorer gọi trực tiếp
    d = ch["smoke"] / "detrac_test" / "yolov8n_mixed5"
    kq = cb.doc_json(d / "ket_qua.json")
    d3 = kq["mo_hinh"]["yolov8n_mixed5"]["ket_qua_D3"]
    _, anh, rel_ds, gt = cb.doc_gt("detrac_test", ch)
    g2 = Gia()
    chi = [rel_ds.index(r) for r in khoa["mau_smoke"]["detrac_test"]["rel"]]
    dd, _ = g2("x.pt", [anh[i] for i in chi], {})
    r = cb.ev.danh_gia([gt[i] for i in chi], dd, 0.25)
    kiem("S6 D3: truck/motorcycle/bicycle = N/A; mAP (car, bus) = trung binh scorer goi truc tiep",
         all(d3["theo_lop"][c] == "N/A" for c in ("truck", "motorcycle", "bicycle"))
         and abs(d3["mAP (car, bus)"]["mAP50"] - (r["theo_lop"]["car"]["AP50"] + r["theo_lop"]["bus"]["AP50"]) / 2) < 1e-12
         and d3["so_lieu_tho_khong_phai_metric"]["theo_lop"]["truck"]["n_det_conf_thap"] == r["theo_lop"]["truck"]["n_det_conf_thap"])
    with open(d / "theo_lop.csv", newline="", encoding="utf-8") as f:
        rows = {x["lop"]: x for x in csv.DictReader(f)}
    kiem("S7 D3 trong CSV: 3 lop ngoai tap ghi N/A", all(rows[c]["AP50"] == "N/A" for c in ("truck", "motorcycle", "bicycle")))
    kb = cb.doc_json(ch["smoke"] / "bdd_val" / "yolo11n_mixed5" / "ket_qua.json")["mo_hinh"]["yolo11n_mixed5"]
    kiem("S8 BDD: bao du nam lop, moi lop co GT", all(kb["theo_lop"][c]["n_gt"] > 0 for c in LOP_5), str(
        {c: kb["theo_lop"][c]["n_gt"] for c in LOP_5}))
    truoc = bam_cay(ch["smoke"])
    with gia_lap(Gia()) as g:
        ma, e = chay(["--buoc", "smoke", "--tap", "bdd_val", "--model", "yolov8n_coco"], ch)
    kiem("S9 smoke lai cung thu muc -> exit 1, khong du doan, khong doi", ma == 1 and not g.goi and bam_cay(ch["smoke"]) == truoc)
    # A4: loại checkpoint sai -> 3
    with gia_lap(Gia(ep_names=COCO80)):
        ma, e = chay(["--buoc", "smoke", "--tap", "detrac_test", "--model", "yolo11n_mixed5", "--lan", "2"], ch)
    kiem("A4 mixed5 nhung checkpoint COCO-80 -> exit 3", ma == 3 and "LOI ANH XA" in e, e[-160:])
    # H: cổng hash
    for nhan, p, lan in (("mau smoke", ch["chuan_bi"] / "mau_smoke_detrac_test.json", 3),
                         ("checkpoint", ch["models"]["yolov8n_coco"][0], 4),
                         ("SHA256SUMS", ch["chuan_bi"] / "SHA256SUMS", 5),
                         ("tai lieu 02", ch["doc02"], 6)):
        goc_b = p.read_bytes()
        p.write_bytes(goc_b + b"\n")
        try:
            with gia_lap(Gia()) as g:
                ma, e = chay(["--buoc", "smoke", "--tap", "detrac_test", "--model", "yolov8n_coco", "--lan", str(lan)], ch)
            ma_kk, _ = chay(["--buoc", "kiem-khoa"], ch)
        finally:
            p.write_bytes(goc_b)
        kiem(f"H {nhan} bi sua sau khoa -> smoke exit 2, khong du doan; kiem-khoa exit 2", ma == 2 and not g.goi
             and ma_kk == 2, e[-160:])
    for tap, p, lan in (("detrac_test", ch["du_lieu"]["detrac_test"]["anh"] / "MVI_00002" / "img00002.jpg", 8),
                        ("bdd_val", ch["du_lieu"]["bdd_val"]["anh"] / "v2" / "v2-0000003.jpg", 8)):
        goc_b = p.read_bytes()
        p.write_bytes(goc_b + b"\0")
        try:
            with gia_lap(Gia()) as g:
                ma, e = chay(["--buoc", "smoke", "--tap", tap, "--model", "yolov8n_mixed5", "--lan", str(lan)], ch)
        finally:
            p.write_bytes(goc_b)
        kiem(f"H {tap}: noi dung anh doi sau khoa -> cong adapter exit 2, khong du doan",
             ma == 2 and not g.goi and "LOI DATASET" in e, e[-160:])
    with gia_lap(Gia()) as g:
        ma, e = chay(["--buoc", "smoke", "--tap", "detrac_test", "--model", "yolov8n_coco", "--lan", "7"], ch)
    kiem("H khoi phuc dung byte -> smoke lai vao thu muc moi exit 0; kiem-khoa exit 0", ma == 0 and len(g.goi) == 1
         and chay(["--buoc", "kiem-khoa"], ch)[0] == 0, e[-160:])
    ma, _ = chay(["--buoc", "khoa", "--tap", "bdd_val"], ch)
    kiem("P tham so sai (--tap voi khoa) -> exit 2", ma == 2)


def main() -> int:
    goc = cb.R53 / "tests_eval" / f"lan_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}"
    goc.mkdir(parents=True)
    print(f"thu muc test: {goc}")
    for f in (nhom_a, nhom_m, lambda: nhom_tks(goc / "fixture")):
        try:
            f()
        except Exception:
            traceback.print_exc()
            KQ.append((getattr(f, "__name__", "nhom"), False))
    dat = sum(ok for _, ok in KQ)
    tom = {"thu_muc": str(goc), "dat": dat, "tong": len(KQ), "ket_qua": [{"ten": t, "dat": ok} for t, ok in KQ],
           "ma_sha256": {m: cb.sha(cb.REPO / m) for m in cb.MA}, "luc": _dt.datetime.now().isoformat(timespec="seconds")}
    (goc / "ket_qua_test.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{dat}/{len(KQ)} dat")
    return 0 if dat == len(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
