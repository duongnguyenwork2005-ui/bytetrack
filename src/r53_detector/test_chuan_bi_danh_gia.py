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
  Phụ lục 1 (sửa review R53B F1–F3):
  E  đối số bàn giao = lệnh đầy đủ đã khóa; 6 dòng lệnh khởi chạy độc lập đặt đủ biến môi trường; ký tự cmd bị từ chối.
  C  CLI CHÍNH hai adapter gốc (`main` + parser của adapter, dự đoán giả lập): tham số bàn giao -> exit 0, đủ đầu ra;
     sai sha256 -> 2; thư mục đã có -> 1; ảnh đổi -> 2; không gọi dự đoán khi bị chặn.
  R  phục hồi khi NPZ đã lưu nhưng chấm lỗi: UA-DETRAC `--cham-lai` trùng kết quả, NPZ bị sửa -> 2; BDD: NPZ còn, không
     có ket_qua; chấm lại bằng hàm cũ trên NPZ trùng lượt bình thường; adapter BDD không có `--cham-lai`.
  F  khởi chạy độc lập: môi trường sạch (bỏ các biến cần đặt) -> cmd.exe đặt biến -> chay_giam_sat -> tiến trình adapter
     giả lập; kiểm bytecode tắt, cấu hình ultralytics trong vùng đầu ra, offline, không .pyc mới; chấm lỗi -> exit 1.
  L  phụ lục khóa: không đổi gì / thiếu bản gốc / sửa phần đã khóa / thiếu chuỗi / đổi file ngoài tập -> 2; đạt -> 0;
     tạo lại -> 1; kiem-khoa theo chuỗi; file đổi sau phụ lục, bản sao bị sửa, thiếu thời điểm -> 2; smoke vẫn qua cổng.
  Phụ lục 2 (bước hẹp Pdaa0fd9d, F2):
  P  phu-luc-2: không đổi gì / thiếu bản phụ lục 1 trong snapshot / sửa phần trước mốc / thiếu chuỗi -> 2; đạt -> 0, bản
     trước = hash phụ lục 1; tạo lại -> 1; kiem-khoa chuỗi 3 mức; sửa mã / 02 / bản sao PL1, PL2 / SUMS PL1 -> 2.
  X  cham-lai-bdd trên thư mục full giả lập (lượt adapter BDD gốc qua main(), dự đoán giả lập): chưa có PL2 -> 2; phục
     hồi sau lỗi chấm -> 0, trùng từng trường lượt thường; không suy luận, đầu vào không đổi; đích đã có -> 1; NPZ sửa /
     đảo thứ tự / thiếu ảnh -> 2; nguồn (lệnh giám sát) sai -> 2; thiếu bằng chứng hash -> 2; khóa hỏng -> 2.
  Phụ lục 3 (review R53B vòng 4, F2 [P2]):
  X12 adapter cùng tên (bản sao nguyên byte) ở checkout khác, tuyệt đối hoặc tương đối với cwd khác -> 2; trang_thai.json
     thiếu / thiếu cwd / lệnh khác tom_tat -> 2; đường dẫn tuyệt đối đúng adapter repo -> 0.
  Q  phu-luc-3: không đổi gì / thiếu bản phụ lục 2 / sửa phần trước mốc / thiếu chuỗi -> 2; đạt -> 0, bản trước = hash
     phụ lục 2; tạo lại -> 1; kiem-khoa chuỗi 4 mức; sửa mã / 02 / bản sao PL3 / SUMS PL2 -> 2; cham-lai-bdd vẫn đạt.

  python src/r53_detector/test_chuan_bi_danh_gia.py
  (nội bộ, chỉ cho nhóm F: --con-adapter <cau_hinh.json trong tests_eval/>)
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
import subprocess  # noqa: E402
import time  # noqa: E402
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


# ----------------------------------------------------------------------------------------------- phụ lục 1: E, C, R, F, L
def nhom_e():
    ok = True
    for x in cb.lenh_day_du():
        ok &= " ".join(cb.doi_so_day_du(x["tap"], x["model"])) in x["lenh"]
    kiem("E1 doi so ban giao (doi_so_day_du) = 6 lenh day du da khoa", ok)
    ch = cb.cau_hinh_mac_dinh()
    env, ds = cb.moi_truong_doc_lap(ch), cb.lenh_doc_lap(ch)
    ok = len(ds) == 6 and len({x["giam_sat"] for x in ds}) == 6
    for x in ds:
        ok &= x["dong_lenh"].startswith("cmd.exe /d /c set ") and all(f'set "{k}={v}"' in x["dong_lenh"] for k, v in env.items())
        ok &= f"-B -u {cb.TAP[x['tap']]['adapter'].replace('/', chr(92))} " in x["dong_lenh"] and " --khong-in -- " in x["dong_lenh"]
        ok &= f'(if not exist "{env["YOLO_CONFIG_DIR"]}\\" mkdir "{env["YOLO_CONFIG_DIR"]}") && "' in x["dong_lenh"]
        ok &= x["dong_lenh"].endswith(" ".join(cb.doi_so_day_du(x["tap"], x["model"])))
        ok &= x["giam_sat"].startswith("results\\phase2\\r53_detector\\full\\_giam_sat\\")
    kiem("E2 6 dong lenh doc lap: cmd dat du bien, -B cho ca giam sat lan adapter, giam sat moi, doi so = lenh khoa",
         ok and env["YOLO_CONFIG_DIR"] == str(cb.R53 / "full" / "_cau_hinh_thu_vien") and env["YOLO_OFFLINE"] == "1"
         and env["PYTHONDONTWRITEBYTECODE"] == "1", ds[0]["dong_lenh"][:160])
    try:
        cb.dong_lenh_doc_lap(env, "gs", ["a.py", "x&y"])
        kiem("E3 ky tu dieu khien cmd trong doi so -> tu choi", False)
    except cb.LoiChuanBi:
        kiem("E3 ky tu dieu khien cmd trong doi so -> tu choi", True)


def du_lieu_argv(tap, ch) -> list:
    d = ch["du_lieu"][tap]
    if tap == "detrac_test":
        return ["--xml", str(d["xml"]), "--anh", str(d["anh"]), "--manifest", str(d["manifest"]),
                "--fingerprint", str(d["fingerprint"]), "--hash-anh", str(d["hash_anh"]), "--loai-d4", str(d["loai_d4"]),
                "--kich-thuoc", f"{tdt.W}x{tdt.H}"]
    return ["--nhan-cocovid", str(d["nhan"]), "--anh-goc", str(d["anh"]), "--manifest-video", str(d["manifest"]),
            "--hash-anh", str(d["hash_anh"])]


def argv_ban_giao(tap, ten, ch, out: Path, sha_kb=None) -> list:
    """Đối số bàn giao đã khóa; chỉ thay giá trị checkpoint / sha256 / --out bằng fixture, rồi trỏ dữ liệu vào fixture."""
    a = list(cb.doi_so_day_du(tap, ten))
    p, s, _ = ch["models"][ten]
    a[a.index("--model") + 1] = f"{ten}={p}"
    a[a.index("--sha256") + 1] = f"{ten}={sha_kb or s}"
    a[a.index("--out") + 1] = str(out)
    return a + du_lieu_argv(tap, ch)


def chay_adapter(mod, argv, goc: Path, nhan: str):
    """main() CỦA CHÍNH adapter (parser + đường chạy gốc); BDD đọc sys.argv. Ngoại lệ không bắt -> ma = 'ngoai_le'
    (tiến trình thật: traceback, exit 1). Log + exit lưu dưới log_cli/."""
    o, goc_argv = io.StringIO(), sys.argv
    with contextlib.redirect_stderr(o), contextlib.redirect_stdout(o):
        try:
            if mod is cb.bv:
                sys.argv = [mod.__file__, *argv]
                ma = mod.main()
            else:
                ma = mod.main(argv)
        except SystemExit as e:
            ma = e.code
        except Exception:
            traceback.print_exc(file=o)
            ma = "ngoai_le"
        finally:
            sys.argv = goc_argv
    d = goc / "log_cli"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{nhan}.txt").write_text(f"# argv: {json.dumps(argv, ensure_ascii=False)}\n{o.getvalue()}# exit={ma}\n",
                                   encoding="utf-8")
    return ma, o.getvalue()


@contextlib.contextmanager
def cham_hong(mod):
    """Bước chấm lỗi SAU khi adapter đã lưu NPZ (thay `danh_gia` trong module adapter)."""
    goc = mod.danh_gia

    def hong(*a, **k):
        raise RuntimeError("loi cham gia lap sau khi da luu NPZ")
    mod.danh_gia = hong
    try:
        yield
    finally:
        mod.danh_gia = goc


def so_json(a, b) -> bool:
    return json.dumps(cb.ev._json_safe(a), sort_keys=True) == json.dumps(cb.ev._json_safe(b), sort_keys=True)


def nhom_c_r(goc: Path) -> dict:
    ch = tao_ch(goc)
    out = goc / "cli"
    td, tb = "yolov8n_mixed5", "yolov8n_coco"
    # C1/C2: tham số bàn giao qua CLI chính của adapter
    with gia_lap(Gia()) as g:
        ma, log = chay_adapter(cb.dg, argv_ban_giao("detrac_test", td, ch, out / "detrac_test" / td), goc, "C1_detrac")
    d = out / "detrac_test" / td
    tep = ["ket_qua.json", "danh_sach_anh.json", f"nguon_du_doan_{td}.json", f"du_doan_{td}.npz", "theo_lop.csv",
           "tong_hop.csv", "so_lieu_tho_khong_phai_metric.csv"]
    ok = ma == 0 and all((d / t).exists() for t in tep) and len(g.goi) == 1
    if ok:
        kq, ng = cb.doc_json(d / "ket_qua.json"), cb.doc_json(d / f"nguon_du_doan_{td}.json")
        x = g.goi[0]
        ok = (x["batch"] == 16 and x["device"] == "0" and len(x["anh"]) == 4 and x["qt"]["imgsz"] == 640
              and x["qt"]["conf"] == 0.001 and kq["dataset"]["gioi_han_anh"] == 0 and kq["quy_tac"]["van_hanh"]["conf"] == 0.25
              and kq["mo_hinh"][td]["so_anh_da_du_doan"] == 4 and ng["npz_sha256"] == cb.sha(d / f"du_doan_{td}.npz")
              and kq["mo_hinh"][td]["sha256"] == ch["models"][td][1]
              and all(kq["mo_hinh"][td]["ket_qua_D3"]["theo_lop"][c] == "N/A" for c in ("truck", "motorcycle", "bicycle")))
    kiem("C1 UA-DETRAC: CLI adapter goc + tham so ban giao -> exit 0, 4/4 anh, batch 16, device 0, conf 0.25, du dau ra, D3",
         ok, log[-300:])
    with gia_lap(Gia()) as g:
        ma, log = chay_adapter(cb.bv, argv_ban_giao("bdd_val", tb, ch, out / "bdd_val" / tb), goc, "C2_bdd")
    d = out / "bdd_val" / tb
    ok = ma == 0 and (d / "ket_qua.json").exists() and (d / "theo_lop.csv").exists() and (d / f"du_doan_{tb}.npz").exists()
    kq_b = cb.doc_json(d / "ket_qua.json") if ok else {}
    ok = ok and len(g.goi) == 1 and g.goi[0]["batch"] == 16 and g.goi[0]["device"] == "0" and len(g.goi[0]["anh"]) == 6 \
        and kq_b["mo_hinh"][tb]["so_anh_da_du_doan"] == 6 and kq_b["mo_hinh"][tb]["anh_xa_lop"]["loai_checkpoint"] == "coco80" \
        and all(kq_b["mo_hinh"][tb]["theo_lop"][c]["n_gt"] > 0 for c in LOP_5)
    kiem("C2 BDD val: CLI adapter goc + tham so ban giao -> exit 0, 6/6 anh, coco80, du nam lop co GT", ok, log[-300:])
    # C3–C5: cổng của chính adapter
    for tap, mod, ten in (("detrac_test", cb.dg, td), ("bdd_val", cb.bv, tb)):
        o3 = out / tap / f"{ten}__sha_sai"
        with gia_lap(Gia()) as g:
            ma, log = chay_adapter(mod, argv_ban_giao(tap, ten, ch, o3, sha_kb="0" * 64), goc, f"C3_{tap}")
        kiem(f"C3 {tap}: sha256 checkpoint sai -> exit 2, khong du doan, khong tao thu muc",
             ma == 2 and not g.goi and not o3.exists() and "LOI CHECKPOINT" in log, log[-160:])
        truoc = bam_cay(out / tap / ten)
        with gia_lap(Gia()) as g:
            ma, log = chay_adapter(mod, argv_ban_giao(tap, ten, ch, out / tap / ten), goc, f"C4_{tap}")
        kiem(f"C4 {tap}: --out da ton tai -> exit 1, khong du doan, khong doi", ma == 1 and not g.goi
             and bam_cay(out / tap / ten) == truoc, log[-160:])
        p = (ch["du_lieu"][tap]["anh"] / ("MVI_00002/img00002.jpg" if tap == "detrac_test" else "v2/v2-0000003.jpg"))
        goc_b = p.read_bytes()
        p.write_bytes(goc_b + b"\0")
        o5 = out / tap / f"{ten}__anh_doi"
        try:
            with gia_lap(Gia()) as g:
                ma, log = chay_adapter(mod, argv_ban_giao(tap, ten, ch, o5), goc, f"C5_{tap}")
        finally:
            p.write_bytes(goc_b)
        kiem(f"C5 {tap}: noi dung anh doi -> exit 2, khong du doan, khong tao thu muc",
             ma == 2 and not g.goi and not o5.exists() and "LOI DATASET" in log, log[-160:])
    # R1–R3: UA-DETRAC chấm lỗi sau NPZ -> --cham-lai
    r1 = out / "detrac_test" / f"{td}__cham_loi"
    with gia_lap(Gia()), cham_hong(cb.dg):
        ma, log = chay_adapter(cb.dg, argv_ban_giao("detrac_test", td, ch, r1), goc, "R1_detrac_cham_loi")
    npz = r1 / f"du_doan_{td}.npz"
    kiem("R1 UA-DETRAC cham loi: ngoai le (exit 1), NPZ + nguon_du_doan (sha256 khop) da ghi, khong co ket_qua.json",
         ma == "ngoai_le" and npz.exists() and not (r1 / "ket_qua.json").exists()
         and cb.doc_json(r1 / f"nguon_du_doan_{td}.json")["npz_sha256"] == cb.sha(npz), log[-200:])
    r2 = out / "detrac_test" / f"{td}__cham_loi__cham_lai"
    with gia_lap(Gia()) as g:
        ma, log = chay_adapter(cb.dg, ["--cham-lai", str(r1), "--out", str(r2)] + du_lieu_argv("detrac_test", ch), goc,
                               "R2_detrac_cham_lai")
    ok = ma == 0 and not g.goi and (r2 / "ket_qua.json").exists()
    if ok:
        k1 = cb.doc_json(out / "detrac_test" / td / "ket_qua.json")["mo_hinh"][td]["ket_qua_D3"]
        ok = so_json(cb.doc_json(r2 / "ket_qua.json")["mo_hinh"][td]["ket_qua_D3"], k1)
    kiem("R2 UA-DETRAC --cham-lai tu thu muc loi -> exit 0, khong du doan, D3 trung luot binh thuong", ok, log[-200:])
    goc_b = npz.read_bytes()
    npz.write_bytes(goc_b + b"\0")
    try:
        with gia_lap(Gia()) as g:
            ma, log = chay_adapter(cb.dg, ["--cham-lai", str(r1), "--out", str(r2) + "_npz_doi"] + du_lieu_argv("detrac_test", ch),
                                   goc, "R3_detrac_npz_doi")
    finally:
        npz.write_bytes(goc_b)
    kiem("R3 UA-DETRAC: NPZ doi sau khi luu -> --cham-lai exit 2", ma == 2 and not g.goi and "da doi" in log, log[-160:])
    # R4–R6: BDD chấm lỗi sau NPZ
    r4 = out / "bdd_val" / f"{tb}__cham_loi"
    with gia_lap(Gia()), cham_hong(cb.bv):
        ma, log = chay_adapter(cb.bv, argv_ban_giao("bdd_val", tb, ch, r4), goc, "R4_bdd_cham_loi")
    npz = r4 / f"du_doan_{tb}.npz"
    sha_ngay = cb.sha(npz) if npz.exists() else None
    kiem("R4 BDD cham loi: ngoai le (exit 1), NPZ con, khong co ket_qua.json / theo_lop.csv; log co dong [du lieu]",
         ma == "ngoai_le" and npz.exists() and not (r4 / "ket_qua.json").exists() and not (r4 / "theo_lop.csv").exists()
         and "[du lieu]" in log, log[-200:])
    danh_tinh, anh, gt = cb.bv.kiem_va_doc_gt(ch["du_lieu"]["bdd_val"]["nhan"], ch["du_lieu"]["bdd_val"]["anh"],
                                              ch["du_lieu"]["bdd_val"]["manifest"], ch["du_lieu"]["bdd_val"]["hash_anh"])
    ten_npz, dd = cb.dg.doc_du_doan(npz, len(anh))
    r = cb.ev.danh_gia(gt, dd, cb.bv.quy_tac_hieu_luc(0.25)["van_hanh"]["conf"])
    kiem("R5 BDD: ten anh NPZ = thu tu adapter; cham lai tu NPZ bang ham cu = ket qua luot binh thuong; NPZ khong doi",
         [str(x) for x in ten_npz] == [p.name for p in anh] and all(so_json(r[k], kq_b["mo_hinh"][tb][k]) for k in r)
         and cb.sha(npz) == sha_ngay and danh_tinh["so_video"] == 2)
    ma, log = chay_adapter(cb.bv, ["--cham-lai", str(r4), "--out", str(r4) + "_x"], goc, "R6_bdd_khong_co_cham_lai")
    kiem("R6 adapter BDD khong co --cham-lai (parser tu choi, exit 2)", ma == 2 and not Path(str(r4) + "_x").exists())
    return ch


def chay_doc_lap(ch, goc: Path, nhan: str, cau_hinh: dict):
    """Như WMI: môi trường KHÔNG có các biến cần đặt; dòng lệnh từ cb.dong_lenh_doc_lap (cùng hàm sinh 6 lệnh bàn giao)."""
    d = goc / "doc_lap"
    d.mkdir(parents=True, exist_ok=True)
    env_can = cb.moi_truong_doc_lap(ch)
    p_cfg = d / f"cau_hinh_{nhan}.json"
    cb.ghi_json_moi(p_cfg, dict(cau_hinh, env_kiem=sorted(env_can), bao_cao=str(d / f"bao_cao_{nhan}.json")))
    gs = d / f"giam_sat_{nhan}"
    dong = cb.dong_lenh_doc_lap(env_can, str(gs), [str(Path(__file__).resolve()), "--con-adapter", str(p_cfg)],
                                py=sys.executable)
    env = {k: v for k, v in os.environ.items() if k not in env_can}
    t0 = time.time()
    p = subprocess.run(dong, cwd=cb.REPO, env=env, capture_output=True, timeout=600)
    (d / f"{nhan}_ket_qua.txt").write_text(f"# dong lenh: {dong}\n# returncode={p.returncode}\n# stdout:\n"
                                           f"{p.stdout.decode('utf-8', 'replace')}\n# stderr:\n"
                                           f"{p.stderr.decode('utf-8', 'replace')}\n", encoding="utf-8")
    bc = cb.doc_json(d / f"bao_cao_{nhan}.json") if (d / f"bao_cao_{nhan}.json").exists() else {}
    log = (gs / "log.txt").read_text(encoding="utf-8") if (gs / "log.txt").exists() else ""
    pyc_moi = [str(x) for x in (cb.REPO / "src").rglob("*.pyc") if x.stat().st_mtime >= t0 - 1]
    return p.returncode, bc, log, pyc_moi, env_can


def nhom_f(ch, goc: Path):
    ch = dict(ch, full=goc / "full_gia")
    out = goc / "cli_doc_lap"
    td = "yolov8n_mixed5"
    ma, bc, log, pyc, env = chay_doc_lap(ch, goc, "F1_detrac", {"tap": "detrac_test", "loi_cham": False,
                                                              "argv": argv_ban_giao("detrac_test", td, ch, out / td)})
    cfg = Path(env["YOLO_CONFIG_DIR"])
    sf = bc.get("ultralytics_settings_file")
    kiem("F1 khoi chay doc lap UA-DETRAC: exit 0 qua cmd + chay_giam_sat, log '# exit=0', du dau ra",
         ma == 0 and "# exit=0" in log and (out / td / "ket_qua.json").exists(), log[-200:])
    kiem("F2 tien trinh adapter: bytecode tat (co + bien), UTF-8, bien dat dung; ultralytics doc cau hinh trong vung dau ra",
         bc.get("dont_write_bytecode") == 1 and bc.get("utf8_mode") == 1 and bc.get("env") == env
         and Path(bc.get("ultralytics_user_config_dir", "")).resolve().is_relative_to(cfg.resolve()) and sf is not None
         and Path(sf).resolve().is_relative_to(cfg.resolve())
         and Path(sf).exists(), json.dumps(bc, ensure_ascii=False)[:300])
    kiem("F3 khong co .pyc moi trong src/ sau lan chay doc lap", not pyc, str(pyc[:3]))
    tb = "yolov8n_coco"
    ma, bc, log, pyc, _ = chay_doc_lap(ch, goc, "F4_bdd_cham_loi", {"tap": "bdd_val", "loi_cham": True,
                                                                  "argv": argv_ban_giao("bdd_val", tb, ch, out / f"bdd_{tb}")})
    d = out / f"bdd_{tb}"
    kiem("F4 BDD cham loi khi chay doc lap: exit 1 that qua giam sat, log giu traceback + [du lieu], NPZ con, khong ket_qua",
         ma == 1 and "# exit=1" in log and "RuntimeError" in log and "[du lieu]" in log
         and (d / f"du_doan_{tb}.npz").exists() and not (d / "ket_qua.json").exists() and not pyc, log[-200:])


def nhom_l(goc: Path):
    ch = tao_ch(goc)
    ma_gia = goc / "fx_ma" / "ma_gia.py"
    ma_gia.parent.mkdir(parents=True)
    ma_gia.write_bytes(b"x = 1\n")
    ch.update(ma=ch["ma"] + [ma_gia], ban_goc=goc / "ban_goc", yeu_cau_test=False, doi_sau_khoa=[ch["doc02"], ma_gia],
              full=goc / "full")
    for tap in cb.TAP:
        chay(["--buoc", "tien-kiem", "--tap", tap], ch)
    ch["doc02"].write_text("# fixture\n" + "\n".join(cb.chuoi_bat_buoc_doc02(cb.hash_ma(ch))) + "\n", encoding="utf-8")
    ma, e = chay(["--buoc", "khoa"], ch)
    kiem("L0 khoa fixture -> exit 0", ma == 0, e[-200:])
    pl_dir = ch["chuan_bi"] / cb.PHU_LUC
    ma, e = chay(["--buoc", "phu-luc"], ch)
    kiem("L1 khong file nao doi -> phu-luc exit 2, khong tao", ma == 2 and not pl_dir.exists() and "khong co file" in e, e[-150:])
    ma_gia.write_bytes(b"x = 2\n")
    ma, e = chay(["--buoc", "phu-luc"], ch)
    ma_kk, _ = chay(["--buoc", "kiem-khoa"], ch)
    kiem("L2 ma doi nhung khong co ban goc -> phu-luc 2; chua co phu luc -> kiem-khoa 2",
         ma == 2 and ma_kk == 2 and not pl_dir.exists() and "ban goc" in e, e[-150:])
    bg = ch["ban_goc"] / cb.rel(ma_gia, ch)
    bg.parent.mkdir(parents=True)
    bg.write_bytes(b"x = 1\n")
    ma, e = chay(["--buoc", "phu-luc"], ch)
    kiem("L3 tai lieu 02 chua co phu luc -> exit 2", ma == 2 and not pl_dir.exists() and "tai lieu 02" in e, e[-150:])
    doc_goc = ch["doc02"].read_bytes()
    phu = (cb.MOC_PHU_LUC + "\n" + "\n".join(cb.chuoi_bat_buoc_phu_luc(ch)) + "\n").encode("utf-8")
    ch["doc02"].write_bytes(doc_goc.replace(b"# fixture", b"# fixturE") + phu)
    ma, e = chay(["--buoc", "phu-luc"], ch)
    kiem("L4 sua phan tai lieu 02 da khoa -> exit 2", ma == 2 and not pl_dir.exists() and "noi them" in e, e[-150:])
    ch["doc02"].write_bytes(doc_goc + (cb.MOC_PHU_LUC + "\n").encode("utf-8"))
    ma, e = chay(["--buoc", "phu-luc"], ch)
    kiem("L5 phu luc thieu lenh / bien moi truong -> exit 2", ma == 2 and not pl_dir.exists() and "thieu" in e, e[-150:])
    ch["doc02"].write_bytes(doc_goc + phu)
    p_mau = ch["chuan_bi"] / "mau_smoke_bdd_val.json"
    b = p_mau.read_bytes()
    p_mau.write_bytes(b + b"\n")
    try:
        ma, e = chay(["--buoc", "phu-luc"], ch)
    finally:
        p_mau.write_bytes(b)
    kiem("L6 file ngoai tap duoc phep doi bi sua -> exit 2", ma == 2 and not pl_dir.exists() and "khong thuoc" in e, e[-150:])
    ma, e = chay(["--buoc", "phu-luc"], ch)
    pl = cb.doc_json(pl_dir / "khoa_phu_luc_1.json") if ma == 0 else {"thay_doi": []}
    ma_kk, e_kk = chay(["--buoc", "kiem-khoa"], ch)
    kiem("L7 phu-luc -> exit 0: 2 file doi, ban sao goc dung hash, tien to 02 = ban khoa; kiem-khoa theo chuoi -> 0",
         ma == 0 and ma_kk == 0 and len(pl["thay_doi"]) == 2
         and {x["cach_giu_ban_goc"] for x in pl["thay_doi"]} == {"tien_to", "ban_sao"}
         and all(cb.sha(cb.tu_rel(x["ban_sao"], ch)) == x["sha256_khoa_goc"] for x in pl["thay_doi"] if "ban_sao" in x),
         (e + e_kk)[-200:])
    truoc = bam_cay(ch["chuan_bi"])
    ma, _ = chay(["--buoc", "phu-luc"], ch)
    kiem("L8 tao lai phu luc -> exit 1, khong doi", ma == 1 and bam_cay(ch["chuan_bi"]) == truoc)
    for nhan, p in (("ma doi sau phu luc", ma_gia), ("tai lieu 02 doi sau phu luc", ch["doc02"]),
                    ("ban sao goc bi sua", cb.tu_rel(next(x["ban_sao"] for x in pl["thay_doi"] if "ban_sao" in x), ch)),
                    ("SHA256SUMS goc", ch["chuan_bi"] / "SHA256SUMS")):
        b = p.read_bytes()
        p.write_bytes(b + b"#")
        try:
            ma_kk, e_kk = chay(["--buoc", "kiem-khoa"], ch)
            with gia_lap(Gia()) as g:
                ma, _ = chay(["--buoc", "smoke", "--tap", "detrac_test", "--model", "yolov8n_coco", "--lan", "9"], ch)
        finally:
            p.write_bytes(b)
        kiem(f"L9 {nhan} -> kiem-khoa 2, smoke 2 khong du doan", ma_kk == 2 and ma == 2 and not g.goi, e_kk[-150:])
        sm = ch["smoke"] / "detrac_test" / "yolov8n_coco__lan9"
        if sm.exists():
            sm.rename(sm.with_name(f"yolov8n_coco__lan9_{len(KQ)}"))
    p_tg = pl_dir / "thoi_diem_phu_luc_1.json"
    b = p_tg.read_bytes()
    p_tg.unlink()
    try:
        ma_kk, e_kk = chay(["--buoc", "kiem-khoa"], ch)
    finally:
        p_tg.write_bytes(b)
    kiem("L10 phu luc do dang (thieu thoi_diem_phu_luc_1.json) -> kiem-khoa 2", ma_kk == 2, e_kk[-150:])
    with gia_lap(Gia()) as g:
        ma, e = chay(["--buoc", "smoke", "--tap", "detrac_test", "--model", "yolov8n_coco"], ch)
    kiem("L11 sau phu luc: smoke qua cong khoa theo chuoi -> exit 0", ma == 0 and len(g.goi) == 1, e[-150:])
    tham_chieu = dung_full_bdd(ch, goc)
    ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", "yolov8n_coco", "--lan", "9"], ch, goc, "X0_chua_phu_luc_2")
    kiem("X0 cham-lai-bdd khi chua co phu luc 2 -> exit 2, khong ket_qua", ma == 2 and "phu luc 2" in e and not
         (ch["full"] / "bdd_val" / "yolov8n_coco__cham_lai_lan9" / "ket_qua.json").exists(), e[-150:])
    nhom_p2(ch, goc, ma_gia)
    nhom_x(ch, goc, tham_chieu)
    nhom_q(ch, goc, ma_gia)


# ----------------------------------------------------------------------------------------------- phụ lục 2 + chấm lại
def chay_log(argv, ch, goc: Path, nhan: str):
    """Như `chay` nhưng lưu argv / stderr / exit dưới log_cli/ (cả ca bị từ chối trước khi có thư mục đích)."""
    ma, e = chay(argv, ch)
    d = goc / "log_cli"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{nhan}.txt").write_text(f"# argv: {json.dumps(argv, ensure_ascii=False)}\n{e}# exit={ma}\n", encoding="utf-8")
    return ma, e


def ghi_bang_chung(ch, ten: str, npz: Path) -> Path:
    """Giống đầu ra 2 dòng PowerShell `lenh_ghi_bang_chung_npz` (BOM, CRLF, Format-List)."""
    p = ch["full"] / "_hau_kiem" / f"bdd_val__{ten}__npz.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    t = (f"\r\n\r\nFullName      : {npz.resolve()}\r\nLength        : {npz.stat().st_size}\r\nLastWriteTime : 20/10/2026\r\n"
         f"\r\n\r\n\r\n\r\nAlgorithm : SHA256\r\nHash      : {cb.sha(npz).upper()}\r\nPath      : {npz.resolve()}\r\n\r\n\r\n")
    p.write_bytes(b"\xef\xbb\xbf" + t.encode("utf-8"))
    return p


def ghi_tom_tat(ch, ten: str, argv: list, exit_code: int, adapter="src\\danh_gia_detector_bdd_val.py", cwd=None,
                lenh_tt=None) -> Path:
    """tom_tat.json + trang_thai.json tổng hợp theo dạng của chay_giam_sat (lệnh = python -B -u <adapter BDD> <đối số bàn
    giao fixture>; tom_tat.json không có cwd, trang_thai.json cuối có lenh + cwd). cwd="" -> trang_thai không có cwd."""
    d = ch["full"] / "_giam_sat" / f"bdd_val__{ten}"
    d.mkdir(parents=True, exist_ok=True)
    lenh = [sys.executable, "-B", "-u", str(adapter), *argv]
    (d / "tom_tat.json").write_text(json.dumps({"lenh": lenh, "exit_code": exit_code}, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    tt = {"trang_thai": "ket_thuc", "lenh": lenh_tt or lenh, "exit_code": exit_code}
    if cwd != "":
        tt["cwd"] = str(cwd or cb.REPO)
    (d / "trang_thai.json").write_text(json.dumps(tt, ensure_ascii=False) + "\n", encoding="utf-8")
    return d / "tom_tat.json"


class GiaGt(Gia):
    """Dự đoán giả lập bám GT (lệch 1 px, conf 0,875 / 0,625 xen kẽ, thêm một FP 0,375) để AP fixture khác 0; tọa độ
    nguyên và conf biểu diễn đúng ở float32 -> chấm từ NPZ (float32) và chấm trong bộ nhớ phải trùng tuyệt đối."""

    def __init__(self, gt_theo_ten: dict):
        super().__init__()
        self.gt = gt_theo_ten

    def __call__(self, weights, anh, qt, batch=16, device="0"):
        _, mt = super().__call__(weights, anh, qt, batch, device)
        ra = []
        for p in anh:
            g = self.gt[p.name]
            n = len(g["cls"])
            ra.append({"xyxy": np.vstack([g["xyxy"] + 1.0, [[30, 30, 60, 46]]]).astype(float),
                       "conf": np.array([0.875 if k % 2 == 0 else 0.625 for k in range(n)] + [0.375]),
                       "cls": np.concatenate([g["cls"], [0]]).astype(int)})
        return ra, mt


def dung_full_bdd(ch, goc: Path) -> Path:
    """Thư mục full giả lập: yolov8n_coco chấm lỗi sau NPZ (exit 1, không ket_qua); yolov8n_mixed5 và yolo11n_mixed5 chạy
    thường. Lượt thường tham chiếu của yolov8n_coco nằm NGOÀI full/. Mọi lượt qua main() của adapter BDD gốc."""
    d = ch["du_lieu"]["bdd_val"]
    _, _, gt = cb.bv.kiem_va_doc_gt(d["nhan"], d["anh"], d["manifest"], d["hash_anh"])
    gt_ten = {g["anh"]: g for g in gt}
    ref = goc / "luot_thuong" / "yolov8n_coco"
    with gia_lap(GiaGt(gt_ten)):
        chay_adapter(cb.bv, argv_ban_giao("bdd_val", "yolov8n_coco", ch, ref), goc, "X_luot_thuong_coco")
    for ten, hong in (("yolov8n_coco", True), ("yolov8n_mixed5", False), ("yolo11n_mixed5", False)):
        d = ch["full"] / "bdd_val" / ten
        argv = argv_ban_giao("bdd_val", ten, ch, d)
        with gia_lap(GiaGt(gt_ten)), (cham_hong(cb.bv) if hong else contextlib.nullcontext()):
            ma, _ = chay_adapter(cb.bv, argv, goc, f"X_full_{ten}")
        ghi_tom_tat(ch, ten, argv, 1 if ma == "ngoai_le" else ma)
        ghi_bang_chung(ch, ten, d / f"du_doan_{ten}.npz")
    return ref


def nhom_p2(ch, goc: Path, ma_gia: Path):
    pl2 = ch["chuan_bi"] / "phu_luc_2"
    ch.update(ban_goc_2=[goc / "ban_goc_2"])
    ma, e = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P1")
    kiem("P1 phu-luc-2 khi khong file nao doi sau phu luc 1 -> exit 2, khong tao", ma == 2 and not pl2.exists(), e[-150:])
    ban_pl1 = ma_gia.read_bytes()
    ma_gia.write_bytes(b"x = 3\n")
    ma, e = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P2")
    kiem("P2 ma doi nhung snapshot khong co ban phu luc 1 -> exit 2", ma == 2 and not pl2.exists() and "snapshot" in e,
         e[-150:])
    bg = goc / "ban_goc_2" / cb.rel(ma_gia, ch)
    bg.parent.mkdir(parents=True)
    bg.write_bytes(ban_pl1)
    doc_pl1 = ch["doc02"].read_bytes()
    phu2 = (cb.PHU_LUC_DS[2][1] + "\n" + "\n".join(cb.chuoi_bat_buoc_phu_luc_2(ch)) + "\n").encode("utf-8")
    ch["doc02"].write_bytes(doc_pl1.replace(b"# fixture", b"# fixturE") + phu2)
    ma, e = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P3a")
    kiem("P3a sua phan tai lieu 02 cua khoa goc -> exit 2 (ban goc phu luc 1 khong khop)", ma == 2 and not pl2.exists()
         and "ban goc phu luc 1" in e, e[-150:])
    ch["doc02"].write_bytes(doc_pl1 + b"#" + phu2)
    ma, e = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P3")
    kiem("P3 sua phan phu luc 1 cua tai lieu 02 (truoc moc phu luc 2) -> exit 2", ma == 2 and not pl2.exists()
         and "noi them" in e, e[-150:])
    ch["doc02"].write_bytes(doc_pl1 + (cb.PHU_LUC_DS[2][1] + "\n").encode("utf-8"))
    ma, e = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P4")
    kiem("P4 phu luc 2 thieu ba lenh cham lai / bang chung NPZ / hash ma -> exit 2", ma == 2 and not pl2.exists()
         and "thieu" in e, e[-150:])
    ch["doc02"].write_bytes(doc_pl1 + phu2)
    ma, e = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P5")
    ma_kk, e_kk = chay_log(["--buoc", "kiem-khoa"], ch, goc, "P5_kiem_khoa")
    pl = cb.doc_json(pl2 / "khoa_phu_luc_2.json") if ma == 0 else {"thay_doi": [], "khoa_truoc": {}}
    p1 = cb.doc_json(ch["chuan_bi"] / "phu_luc_1" / "thoi_diem_phu_luc_1.json")
    kiem("P5 phu-luc-2 -> exit 0: tai lieu 02 + ma doi, ban truoc = hash phu luc 1, tro phu luc 1; kiem-khoa chuoi 2 muc -> 0",
         ma == 0 and ma_kk == 0 and len(pl["thay_doi"]) == 2 and pl["khoa_truoc"].get("sha256sums_sha256") ==
         p1["sha256sums_sha256"] and all(x["muc_truoc"] == "phu_luc_1" for x in pl["thay_doi"])
         and all(cb.sha(cb.tu_rel(x["ban_sao"], ch)) == x["sha256_khoa_truoc"] == cb.sha_byte(ban_pl1)
                 for x in pl["thay_doi"] if "ban_sao" in x)
         and ma_kk == 0 and cb.kiem_khoa(ch)[1]["_cap_phu_luc"] == ["phu_luc_1", "phu_luc_2"], (e + e_kk)[-200:])
    truoc = bam_cay(ch["chuan_bi"])
    ma, _ = chay_log(["--buoc", "phu-luc-2"], ch, goc, "P6")
    kiem("P6 tao lai phu luc 2 -> exit 1, khong doi", ma == 1 and bam_cay(ch["chuan_bi"]) == truoc)
    ma, _ = chay_log(["--buoc", "phu-luc"], ch, goc, "P6b")
    kiem("P6b phu-luc (muc 1) sau phu luc 2 -> exit 1, khong doi", ma == 1 and bam_cay(ch["chuan_bi"]) == truoc)
    sao1 = next(cb.tu_rel(x["ban_sao"], ch) for x in cb.doc_json(ch["chuan_bi"] / "phu_luc_1" / "khoa_phu_luc_1.json")
                ["thay_doi"] if "ban_sao" in x)
    sao2 = next(cb.tu_rel(x["ban_sao"], ch) for x in pl["thay_doi"] if "ban_sao" in x)
    for nhan, p in (("ma doi sau phu luc 2", ma_gia), ("tai lieu 02 doi sau phu luc 2", ch["doc02"]),
                    ("ban sao phu luc 1", sao1), ("ban sao phu luc 2", sao2),
                    ("SHA256SUMS phu luc 1", ch["chuan_bi"] / "phu_luc_1" / "SHA256SUMS")):
        b = p.read_bytes()
        p.write_bytes(b + b"#")
        try:
            ma_kk, e_kk = chay(["--buoc", "kiem-khoa"], ch)
        finally:
            p.write_bytes(b)
        kiem(f"P7 {nhan} -> kiem-khoa 2", ma_kk == 2, e_kk[-150:])
    with gia_lap(Gia()) as g:
        ma, e = chay(["--buoc", "smoke", "--tap", "bdd_val", "--model", "yolov8n_coco", "--lan", "2"], ch)
    kiem("P8 sau phu luc 2: smoke qua cong khoa 3 muc -> exit 0", ma == 0 and len(g.goi) == 1, e[-150:])


def nhom_x(ch, goc: Path, ref: Path):
    full = ch["full"] / "bdd_val"
    tb, tm, tk = "yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"
    vao = [full / tb, ch["full"] / "_hau_kiem", ch["full"] / "_giam_sat", ch["du_lieu"]["bdd_val"]["anh"].parent]
    truoc = {str(d): bam_cay(d) for d in vao}
    with gia_lap(Gia()) as g:
        ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb], ch, goc, "X1")
    o1 = full / f"{tb}__cham_lai"
    ok = ma == 0 and not g.goi and (o1 / "ket_qua.json").exists() and (o1 / "log.txt").exists() \
        and cb.doc_json(o1 / "exit.json")["exit"] == 0
    kq = cb.doc_json(o1 / "ket_qua.json") if ok else {}
    kr = cb.doc_json(ref / "ket_qua.json")
    if ok:
        m, mr = kq["mo_hinh"][tb], kr["mo_hinh"][tb]
        bo = set(cb.TRUONG_KHONG_TU_NPZ)
        ok = (all(so_json(kq[k], kr[k]) for k in ("quy_tac", "quy_tac_gt", "dataset"))
              and set(mr) - bo <= set(m) and all(so_json(m[k], mr[k]) for k in set(mr) - bo)
              and (o1 / "theo_lop.csv").read_bytes() == (ref / "theo_lop.csv").read_bytes()
              and kq["cham_lai"]["exit_goc"] == 1 and "doi_chieu_ket_qua_goc" not in kq["cham_lai"]
              and mr["macro"]["mAP50"] > 0 and all(mr["theo_lop"][c]["AP50"] > 0 for c in LOP_5))
    kiem("X1 phuc hoi sau loi cham (exit goc 1, khong ket_qua): cham-lai-bdd exit 0; moi truong ket_qua + theo_lop.csv "
         "trung luot thuong (scorer cu, AP fixture > 0 ca 5 lop), tru anh_xa_lop / thoi_gian_s khong co trong NPZ", ok,
         (e + (json.dumps(kr["mo_hinh"][tb]["macro"]) if kr else ""))[-200:])
    sau = {str(d): bam_cay(d) for d in vao}
    kiem("X2 khong suy luan (du doan gia lap khong duoc goi, goi_suy_luan 0, ham bi chan); dau vao + du lieu khong doi",
         ok and not g.goi and kq["cham_lai"]["goi_suy_luan"] == 0 and kq["cham_lai"]["dau_vao_khong_doi"] and truoc == sau
         and kq["cham_lai"]["npz_sha256"] == cb.sha(full / tb / f"du_doan_{tb}.npz"))
    t1 = bam_cay(o1)
    ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb], ch, goc, "X3")
    kiem("X3 thu muc dich da co -> exit 1, khong doi", ma == 1 and bam_cay(o1) == t1, e[-120:])
    with gia_lap(Gia()) as g:
        ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tm], ch, goc, "X4")
    o4 = full / f"{tm}__cham_lai"
    k4 = cb.doc_json(o4 / "ket_qua.json") if (o4 / "ket_qua.json").exists() else {}
    kiem("X4 thu muc chay thuong (co ket_qua.json): cham lai exit 0, doi chieu ket qua goc khong truong nao khac",
         ma == 0 and not g.goi and k4.get("cham_lai", {}).get("doi_chieu_ket_qua_goc", {}).get("truong_khac") == [],
         e[-160:])
    npz = full / tb / f"du_doan_{tb}.npz"
    b = npz.read_bytes()
    npz.write_bytes(b + b"\0")
    try:
        ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", "2"], ch, goc, "X5")
    finally:
        npz.write_bytes(b)
    kiem("X5 NPZ bi sua sau khi ghi bang chung -> exit 2, khong ket_qua", ma == 2 and "NPZ da doi" in e
         and not (full / f"{tb}__cham_lai_lan2" / "ket_qua.json").exists(), e[-150:])
    d = ch["du_lieu"]["bdd_val"]
    _, anh, _ = cb.bv.kiem_va_doc_gt(d["nhan"], d["anh"], d["manifest"], d["hash_anh"])
    npz_k = full / tk / f"du_doan_{tk}.npz"
    b_k = npz_k.read_bytes()
    _, dd = cb.dg.doc_du_doan(npz_k, len(anh))
    for lan, nhan, a2, d2 in ((2, "thu tu anh dao", anh[::-1], dd[::-1]), (3, "thieu anh cuoi", anh[:-1], dd[:-1])):
        cb.ev.luu_du_doan(npz_k, a2, d2)
        ghi_bang_chung(ch, tk, npz_k)                   # hash khớp -> chỉ cổng danh sách / thứ tự chặn
        try:
            ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tk, "--lan", str(lan)], ch, goc, f"X6_{lan}")
        finally:
            npz_k.write_bytes(b_k)
            ghi_bang_chung(ch, tk, npz_k)
        kiem(f"X6 NPZ {nhan} (hash bang chung khop) -> exit 2, khong ket_qua", ma == 2 and "thu tu" in e
             and not (full / f"{tk}__cham_lai_lan{lan}" / "ket_qua.json").exists(), e[-150:])
    ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tk, "--lan", "4"], ch, goc, "X7")
    kiem("X7 khoi phuc NPZ dung -> chay lai vao thu muc moi __lan4 exit 0", ma == 0
         and (full / f"{tk}__cham_lai_lan4" / "ket_qua.json").exists(), e[-150:])
    p_tom = ch["full"] / "_giam_sat" / f"bdd_val__{tb}" / "tom_tat.json"
    p_tt = p_tom.with_name("trang_thai.json")
    b_tom, b_tt = p_tom.read_bytes(), p_tt.read_bytes()
    argv = argv_ban_giao("bdd_val", tb, ch, full / tb)
    for lan, nhan, a2 in ((3, "--gioi-han-anh 5", [x if i != argv.index("--gioi-han-anh") + 1 else "5"
                                                   for i, x in enumerate(argv)]),
                          (4, "sha256 khac khoa", argv_ban_giao("bdd_val", tb, ch, full / tb, sha_kb="0" * 64)),
                          (5, "--out thu muc khac", argv_ban_giao("bdd_val", tb, ch, full / tm))):
        ghi_tom_tat(ch, tb, a2, 1)
        try:
            ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", str(lan)], ch, goc, f"X8_{lan}")
        finally:
            p_tom.write_bytes(b_tom)
            p_tt.write_bytes(b_tt)
        kiem(f"X8 nguon: lenh giam sat {nhan} -> exit 2", ma == 2 and "lenh giam sat" in e, e[-150:])
    p_bc = ch["full"] / "_hau_kiem" / f"bdd_val__{tb}__npz.txt"
    b_bc = p_bc.read_bytes()
    p_bc.unlink()
    try:
        ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", "6"], ch, goc, "X9")
    finally:
        p_bc.write_bytes(b_bc)
    kiem("X9 thieu bang chung hash NPZ -> exit 2", ma == 2 and "thieu" in e, e[-150:])
    ma_gia = next(p for p in ch["ma"] if p.name == "ma_gia.py")
    b = ma_gia.read_bytes()
    ma_gia.write_bytes(b + b"#")
    try:
        with gia_lap(Gia()) as g:
            ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", "7"], ch, goc, "X10")
    finally:
        ma_gia.write_bytes(b)
    kiem("X10 khoa hong (ma doi sau phu luc 2) -> exit 2, khong doc NPZ / khong ket_qua", ma == 2 and not g.goi and
         not (full / f"{tb}__cham_lai_lan7" / "ket_qua.json").exists(), e[-150:])
    ma1, _ = chay(["--buoc", "cham-lai-bdd", "--tap", "bdd_val", "--model", tb], ch)
    ma2, _ = chay(["--buoc", "cham-lai-bdd"], ch)
    kiem("X11 tham so: --tap voi cham-lai-bdd / thieu --model -> exit 2", ma1 == 2 and ma2 == 2)
    nhom_x12(ch, goc, tb, argv, p_tom, p_tt, b_tom, b_tt)


def nhom_x12(ch, goc: Path, tb: str, argv: list, p_tom: Path, p_tt: Path, b_tom: bytes, b_tt: bytes):
    """Phụ lục 3 (review R53B vòng 4, F2 [P2]): adapter trong lệnh giám sát phải là file adapter BDD của repo sau resolve
    (tương đối -> theo cwd trong trang_thai.json), không chỉ trùng tên. Bản sao NGUYÊN BYTE adapter ở checkout khác."""
    full = ch["full"] / "bdd_val"
    khac = goc / "checkout_khac"
    p_khac = khac / "src" / "danh_gia_detector_bdd_val.py"
    p_khac.parent.mkdir(parents=True)
    p_khac.write_bytes((cb.REPO / cb.TAP["bdd_val"]["adapter"]).read_bytes())
    p_repo = (cb.REPO / cb.TAP["bdd_val"]["adapter"]).resolve()
    ca = ((21, "script cung ten, duong dan tuyet doi o checkout khac", dict(adapter=p_khac), 2, "khong phai adapter BDD"),
          (22, "duong dan tuong doi nhung cwd giam sat la checkout khac", dict(cwd=khac), 2, "khong phai adapter BDD"),
          (23, "trang_thai.json khong co cwd", dict(cwd=""), 2, "cwd"),
          (24, "trang_thai.json ghi lenh khac tom_tat.json", dict(lenh_tt=[sys.executable, "-B", "-u", str(p_khac), *argv]),
           2, "lenh khac nhau"),
          (25, "duong dan tuyet doi dung adapter trong repo", dict(adapter=p_repo), 0, "[DAT]"))
    for lan, nhan, kw, ky_vong, dau in ca:
        ghi_tom_tat(ch, tb, argv, 1, **kw)
        try:
            with gia_lap(Gia()) as g:
                ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", str(lan)], ch, goc, f"X12_{lan}")
        finally:
            p_tom.write_bytes(b_tom)
            p_tt.write_bytes(b_tt)
        o = full / f"{tb}__cham_lai_lan{lan}"
        co_kq = (o / "ket_qua.json").exists()
        ok = ma == ky_vong and not g.goi and (co_kq if ky_vong == 0 else not co_kq) and (dau in e if ky_vong else True)
        if ky_vong == 0 and co_kq:
            kq = cb.doc_json(o / "ket_qua.json")["cham_lai"]
            ok = ok and kq["adapter_goc"] == cb.TAP["bdd_val"]["adapter"] and kq["cwd_giam_sat"] == str(cb.REPO)
        kiem(f"X12 nguon: {nhan} -> exit {ky_vong}" + (", khong ket_qua" if ky_vong else ", co ket_qua, adapter/cwd ghi lai"),
             ok, e[-180:])
    p_tt.unlink()
    try:
        ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", "26"], ch, goc, "X12_26")
    finally:
        p_tt.write_bytes(b_tt)
    kiem("X12 nguon: thieu trang_thai.json -> exit 2, khong ket_qua", ma == 2 and "trang_thai.json" in e
         and not (full / f"{tb}__cham_lai_lan26" / "ket_qua.json").exists(), e[-150:])


def nhom_q(ch, goc: Path, ma_gia: Path):
    """Phụ lục 3 nối tiếp phụ lục 2 (cùng cơ chế khóa, mức 3)."""
    pl3 = ch["chuan_bi"] / "phu_luc_3"
    ch.update(ban_goc_3=[goc / "ban_goc_3"])
    ma, e = chay_log(["--buoc", "phu-luc-3"], ch, goc, "Q1")
    kiem("Q1 phu-luc-3 khi khong file nao doi sau phu luc 2 -> exit 2, khong tao", ma == 2 and not pl3.exists(), e[-150:])
    ban_pl2 = ma_gia.read_bytes()
    ma_gia.write_bytes(b"x = 4\n")
    ma, e = chay_log(["--buoc", "phu-luc-3"], ch, goc, "Q2")
    kiem("Q2 ma doi nhung snapshot khong co ban phu luc 2 -> exit 2", ma == 2 and not pl3.exists() and "snapshot" in e,
         e[-150:])
    bg = goc / "ban_goc_3" / cb.rel(ma_gia, ch)
    bg.parent.mkdir(parents=True)
    bg.write_bytes(ban_pl2)
    doc_pl2 = ch["doc02"].read_bytes()
    phu3 = (cb.PHU_LUC_DS[3][1] + "\n" + "\n".join(cb.chuoi_bat_buoc_phu_luc_3(ch)) + "\n").encode("utf-8")
    ch["doc02"].write_bytes(doc_pl2 + b"#" + phu3)
    ma, e = chay_log(["--buoc", "phu-luc-3"], ch, goc, "Q3")
    kiem("Q3 sua phan phu luc 2 cua tai lieu 02 (truoc moc phu luc 3) -> exit 2", ma == 2 and not pl3.exists()
         and "noi them" in e, e[-150:])
    ch["doc02"].write_bytes(doc_pl2 + (cb.PHU_LUC_DS[3][1] + "\n").encode("utf-8"))
    ma, e = chay_log(["--buoc", "phu-luc-3"], ch, goc, "Q4")
    kiem("Q4 phu luc 3 thieu lenh cham lai / hash ma -> exit 2", ma == 2 and not pl3.exists() and "thieu" in e, e[-150:])
    ch["doc02"].write_bytes(doc_pl2 + phu3)
    ma, e = chay_log(["--buoc", "phu-luc-3"], ch, goc, "Q5")
    ma_kk, e_kk = chay_log(["--buoc", "kiem-khoa"], ch, goc, "Q5_kiem_khoa")
    pl = cb.doc_json(pl3 / "khoa_phu_luc_3.json") if ma == 0 else {"thay_doi": [], "khoa_truoc": {}}
    p2 = cb.doc_json(ch["chuan_bi"] / "phu_luc_2" / "thoi_diem_phu_luc_2.json")
    kiem("Q5 phu-luc-3 -> exit 0: tai lieu 02 + ma doi, ban truoc = hash phu luc 2, tro phu luc 2; kiem-khoa chuoi 3 muc -> 0",
         ma == 0 and ma_kk == 0 and len(pl["thay_doi"]) == 2 and pl["muc"] == 3
         and pl["khoa_truoc"].get("sha256sums_sha256") == p2["sha256sums_sha256"]
         and all(x["muc_truoc"] == "phu_luc_2" for x in pl["thay_doi"])
         and all(cb.sha(cb.tu_rel(x["ban_sao"], ch)) == x["sha256_khoa_truoc"] == cb.sha_byte(ban_pl2)
                 for x in pl["thay_doi"] if "ban_sao" in x)
         and cb.kiem_khoa(ch)[1]["_cap_phu_luc"] == ["phu_luc_1", "phu_luc_2", "phu_luc_3"], (e + e_kk)[-200:])
    truoc = bam_cay(ch["chuan_bi"])
    ma3, _ = chay_log(["--buoc", "phu-luc-3"], ch, goc, "Q6")
    ma2, _ = chay_log(["--buoc", "phu-luc-2"], ch, goc, "Q6b")
    kiem("Q6 tao lai phu luc 3 / phu luc 2 -> exit 1, khong doi", ma3 == 1 and ma2 == 1 and bam_cay(ch["chuan_bi"]) == truoc)
    sao3 = next(cb.tu_rel(x["ban_sao"], ch) for x in pl["thay_doi"] if "ban_sao" in x)
    for nhan, p in (("ma doi sau phu luc 3", ma_gia), ("tai lieu 02 doi sau phu luc 3", ch["doc02"]),
                    ("ban sao phu luc 3", sao3), ("SHA256SUMS phu luc 2", ch["chuan_bi"] / "phu_luc_2" / "SHA256SUMS")):
        b = p.read_bytes()
        p.write_bytes(b + b"#")
        try:
            ma_kk, e_kk = chay(["--buoc", "kiem-khoa"], ch)
        finally:
            p.write_bytes(b)
        kiem(f"Q7 {nhan} -> kiem-khoa 2", ma_kk == 2, e_kk[-150:])
    tb = "yolov8n_coco"
    with gia_lap(Gia()) as g:
        ma, e = chay_log(["--buoc", "cham-lai-bdd", "--model", tb, "--lan", "14"], ch, goc, "Q8")
    o = ch["full"] / "bdd_val" / f"{tb}__cham_lai_lan14"
    kq = cb.doc_json(o / "ket_qua.json") if (o / "ket_qua.json").exists() else {}
    kr = cb.doc_json(o.with_name(f"{tb}__cham_lai") / "ket_qua.json")
    kiem("Q8 sau phu luc 3: cham-lai-bdd exit 0, chuoi khoa 4 muc, cung so lieu lan dau, khong suy luan",
         ma == 0 and not g.goi and kq.get("cham_lai", {}).get("khoa", {}).get("chuoi") ==
         ["khoa_goc", "phu_luc_1", "phu_luc_2", "phu_luc_3"]
         and so_json(kq["mo_hinh"][tb]["theo_lop"], kr["mo_hinh"][tb]["theo_lop"]), e[-150:])


def con_adapter(p_cfg: Path) -> int:
    """Tiến trình con của nhóm F: dự đoán giả lập rồi gọi main() của adapter gốc. Chỉ nhận cấu hình trong tests_eval/."""
    p_cfg = p_cfg.resolve()
    if not p_cfg.is_relative_to((cb.R53 / "tests_eval").resolve()):
        print(f"[LOI] {p_cfg} ngoai tests_eval/", file=sys.stderr)
        return 2
    c = cb.doc_json(p_cfg)
    mod = cb.dg if c["tap"] == "detrac_test" else cb.bv
    bc = {"pid": os.getpid(), "dont_write_bytecode": sys.flags.dont_write_bytecode, "utf8_mode": sys.flags.utf8_mode,
          "env": {k: os.environ.get(k) for k in c["env_kiem"]}}
    try:
        import ultralytics.utils as uu
        bc["ultralytics_user_config_dir"] = str(uu.USER_CONFIG_DIR)
        sf = getattr(uu, "SETTINGS_FILE", None) or getattr(getattr(uu, "SETTINGS", None), "file", None)
        bc["ultralytics_settings_file"] = str(sf) if sf else None
    except Exception as e:  # noqa: BLE001 -- ghi lại để test báo FAIL
        bc["loi_ultralytics"] = repr(e)
    cb.ghi_json_moi(Path(c["bao_cao"]), bc)
    mod.du_doan_ultralytics = Gia()
    if c["loi_cham"]:
        def hong(*a, **k):
            raise RuntimeError("loi cham gia lap sau khi da luu NPZ")
        mod.danh_gia = hong
    if mod is cb.bv:
        sys.argv = [mod.__file__, *c["argv"]]
        return mod.main()
    return mod.main(c["argv"])


def main() -> int:
    goc = cb.R53 / "tests_eval" / f"lan_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}"
    goc.mkdir(parents=True)
    print(f"thu muc test: {goc}")
    def nhom_crf():
        ch = nhom_c_r(goc / "fixture_cli")
        nhom_f(ch, goc / "fixture_cli")
    for f in (nhom_a, nhom_m, lambda: nhom_tks(goc / "fixture"), nhom_e, nhom_crf, lambda: nhom_l(goc / "fixture_phu_luc")):
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
    if len(sys.argv) == 3 and sys.argv[1] == "--con-adapter":
        raise SystemExit(con_adapter(Path(sys.argv[2])))
    raise SystemExit(main())
