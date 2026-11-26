"""R53B — tiền kiểm, khóa giao thức và smoke cho đánh giá ba detector trên UA-DETRAC test và BDD val.

Không sửa và không thay hàm cũ: kiểm dữ liệu + đọc GT (`kiem_va_doc_gt` của hai adapter), dự đoán
(`du_doan_ultralytics`, gọi qua module adapter), lưu NPZ (`luu_du_doan`), scorer (`eval_detector_5lop.danh_gia`),
ngưỡng hiệu lực (`quy_tac_hieu_luc`), chuẩn hóa D3 (`chuan_hoa_d3`), đọc lại NPZ (`doc_du_doan`), ghi nguồn/kết quả
UA-DETRAC (`ghi_nguon`, `ghi_ket_qua`). File này chỉ thêm: kiểm số lượng/danh tính so với hồ sơ, chọn mẫu smoke,
khóa SHA256 và vỏ chạy smoke. Không nhận lệnh hay đường dẫn tùy ý; không chạy đánh giá đầy đủ.

  python src/r53_detector/chuan_bi_danh_gia.py --buoc tien-kiem --tap detrac_test
  python src/r53_detector/chuan_bi_danh_gia.py --buoc tien-kiem --tap bdd_val
  python src/r53_detector/chuan_bi_danh_gia.py --buoc lenh-day-du        # chỉ in lệnh bàn giao + hash + căn cứ dự toán
  python src/r53_detector/chuan_bi_danh_gia.py --buoc khoa
  python src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa         # chỉ đọc: băm lại mọi file trong SHA256SUMS
  python src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap <detrac_test|bdd_val> --model <ten> [--lan N]

Mẫu smoke (tối đa 64 ảnh/tập): theo thứ tự ảnh của adapter, lấy ảnh đầu tiên có GT cho từng lớp cần kiểm (car, bus
trên UA-DETRAC; năm lớp trên BDD val), loại trùng, rồi bổ sung các ảnh đầu chưa chọn tới 64; giữ thứ tự adapter.
Exit: 0 đạt; 1 từ chối ghi đè; 2 cổng khóa / dữ liệu / checkpoint / tham số; 3 kiểm sau dự đoán hoặc chấm lỗi.
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

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import importlib.metadata as _md  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import danh_gia_detector_bdd_val as bv  # noqa: E402
import danh_gia_detector_detrac_test as dg  # noqa: E402
import eval_detector_5lop as ev  # noqa: E402
import fingerprint_detrac_test as fpt  # noqa: E402
from detector_classes import ID_LOP_5, LOP_5  # noqa: E402

PYTHON = "C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe"
R53 = REPO / "results" / "phase2" / "r53_detector"
SO_ANH_TOI_DA = 64
BATCH, DEVICE, CONF_VH = 16, "0", 0.25
GB_TOI_THIEU = 15
# Checkpoint/hash: baseline theo kế hoạch R43/R47; hai best.pt mixed5 theo bàn giao R53A (ban_giao/checkpoint_sha256.csv).
MODELS = {
    "yolov8n_coco": ("models/yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36", "coco80"),
    "yolov8n_mixed5": ("models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt",
                       "542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694", "lop5"),
    "yolo11n_mixed5": ("models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt",
                       "9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb", "lop5"),
}
TAP = {"detrac_test": {"adapter": "src/danh_gia_detector_detrac_test.py", "lop_kiem": ("car", "bus")},
       "bdd_val": {"adapter": "src/danh_gia_detector_bdd_val.py", "lop_kiem": tuple(LOP_5)}}
# Trường danh tính lấy từ kết quả kiem_va_doc_gt; khóa lưu các trường này, smoke so lại từng trường.
KHOA_DANH_TINH = {
    "detrac_test": ["so_video", "dem.anh_tren_dia", "dem.anh_loai_D4", "dem.anh_danh_gia", "dem.gt_theo_lop",
                    "manifest_sha256", "fingerprint_sha256", "bang_hash_sha256", "loai_D4_sha256",
                    "danh_sach_anh_co_thu_tu_sha256", "noi_dung_anh_sha256_tong_hop"],
    "bdd_val": ["so_video", "dem.anh", "dem.gt_theo_lop", "nhan_sha256", "manifest_sha256", "danh_sach_anh_sha256",
                "noi_dung_anh_sha256_tong_hop", "doi_chieu_hash_anh.file_sha256"],
}
# Hồ sơ: UA-DETRAC theo R47 (03_ke_hoach_khoa..., kết quả 20261010_*); BDD val theo R43 (ket_qua.json r43_*).
KY_VONG_DU_LIEU = {
    "detrac_test": {"so_video": 40, "dem.anh_tren_dia": 56340, "dem.anh_loai_D4": 173, "dem.anh_danh_gia": 56167,
                    "manifest_sha256": "ff2b4b9c658aced54567247c16b5a0fc43ae1e9836af70e7ea656fce3077f47c",
                    "fingerprint_sha256": "dd0f2a0e9dcaf11f26fe397ff4c4eb7ef8a8ecfd248d4682aa652b3af156343a",
                    "bang_hash_sha256": "3cea34881d90031a5ed2161a774154e87691228da9b25b1ac0b07dbe8717b166",
                    "loai_D4_sha256": "4d7312530505651b92da414b2dcf82c4ec74486dc408fa37dc9f927d85723551",
                    "danh_sach_anh_co_thu_tu_sha256": "46204978758016cb16627b2035d34a40c9b31c5eb477db6dd442afab3eba92d7",
                    "noi_dung_anh_sha256_tong_hop": "e4fff621ae775ad7d571d67e23c5f75db100434b4905c1e405fea24dec696cb6"},
    "bdd_val": {"so_video": 200, "dem.anh": 39973,
                "nhan_sha256": "074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d",
                "manifest_sha256": "2071c15bd300e5dfff156fd708393b7018b074777ec4ef83da236409eb34c3ba",
                "danh_sach_anh_sha256": "4bdf4889984ad9b9f37602f711fd066622f5473364e033fda0b886e7829666d5",
                "noi_dung_anh_sha256_tong_hop": "1e209af9a7077c173d46006b25c3eb6f7f66b9aed484b9d2a8dd281f61317555",
                "doi_chieu_hash_anh.file_sha256": "60fea2a9ce623450c1526953b7e427016ed917cfd702a7c6275525fca3582d20"},
}
MA = ["src/danh_gia_detector_detrac_test.py", "src/danh_gia_detector_bdd_val.py", "src/eval_detector_5lop.py",
      "src/detector_classes.py", "src/fingerprint_detrac_test.py", "src/bdd_yolo_export.py", "src/config.py",
      "src/r53_detector/chuan_bi_danh_gia.py", "src/r53_detector/test_chuan_bi_danh_gia.py"]
# Hash đã ghi trong hồ sơ (R47 03 mục 2 + PL1 v3) -- chỉ để đối chiếu và báo, khóa dùng hash thực tế.
HO_SO_MA = {"src/danh_gia_detector_detrac_test.py": "bb4b497b4f89a4464777144a35d66cbd109d03d3c485c4176cde0b67e358ccb4",
            "src/danh_gia_detector_bdd_val.py": "de2c06f4db2eb99c15b44d873cc63b6b6b446bc632000701671dedd65677020a",
            "src/eval_detector_5lop.py": "30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f",
            "src/fingerprint_detrac_test.py": "37b62da99b77880a27aa234a887158e51498b6822921b430c1366d271f621255",
            "src/bdd_yolo_export.py": "1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493",
            "src/detector_classes.py": "7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e"}
# Căn cứ dự toán lượt dài: các lượt đầy đủ đã chạy cùng adapter, cùng kiến trúc, cùng máy.
CAN_CU_DU_TOAN = {"detrac_test": ["results/phase2/detector_eval/detrac_test/20261010_yolov8n_coco",
                                  "results/phase2/detector_eval/detrac_test/20261010_yolov8n_ft",
                                  "results/phase2/detector_eval/detrac_test/20261010_yolo11n_ft"],
                  "bdd_val": ["results/phase2/detector_eval/bdd_val_cocovid/r43_yolov8n_coco",
                              "results/phase2/detector_eval/bdd_val_cocovid/r43_yolov8n_ft",
                              "results/phase2/detector_eval/bdd_val_cocovid/r43_yolo11n_ft"]}
DAU_HIEU_CHUA_XONG = ("TODO", "TBD", "PLACEHOLDER", "<hex>", "<sha256>", "<SHA256>", "{{", "}}")
CANH_BAO_SMOKE = ("SMOKE KY THUAT tren toi da 64 anh: chi kiem duong chay; KHONG dung AP smoke de xep hang detector "
                  "hay chinh tham so")
LOI_DU_LIEU = (dg.LoiDuLieu, bv.LoiDuLieu, FileNotFoundError, ValueError, KeyError)


class LoiChuanBi(Exception):
    """Cổng khóa / mẫu / tài liệu không đạt."""


# ----------------------------------------------------------------------------------------------- tiện ích
def cau_hinh_mac_dinh() -> dict:
    return {"repo": REPO, "chuan_bi": R53 / "chuan_bi", "smoke": R53 / "smoke",
            "doc02": REPO / "handoff" / "logs" / "review53" / "02_ke_hoach_khoa_danh_gia.md",
            "du_lieu": {"detrac_test": {k: dg.p_abs(v) for k, v in dg.MAC_DINH.items() if k != "kich_thuoc"},
                        "bdd_val": {k: bv.p_abs(v) for k, v in bv.MAC_DINH.items()}},
            "ky_vong": KY_VONG_DU_LIEU, "models": {t: (REPO / p, s, lo) for t, (p, s, lo) in MODELS.items()},
            "ma": [REPO / m for m in MA], "gb_toi_thieu": GB_TOI_THIEU}


def bay_gio() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def sha(p) -> str:
    return ev.sha256_file(Path(p))


def rel(p, ch) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(Path(ch["repo"]).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def tu_rel(s: str, ch) -> Path:
    p = Path(s)
    return p if p.is_absolute() else Path(ch["repo"]) / p


def lay(d: dict, khoa: str):
    for k in khoa.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def ghi_json_moi(p: Path, obj) -> None:
    """Ghi file MỚI (mode 'x'): file đã có -> FileExistsError, không ghi đè."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(ev._json_safe(obj), ensure_ascii=False, indent=1, default=str) + "\n")


def doc_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def o_dia_trong_gb(p: Path) -> float:
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2 ** 30


def hash_ma(ch) -> dict:
    ra = {}
    for p in ch["ma"]:
        r = rel(p, ch)
        s = sha(p)
        ra[r] = {"sha256": s, **({"ho_so": HO_SO_MA[r], "khop_ho_so": HO_SO_MA[r] == s} if r in HO_SO_MA else {})}
    return ra


def moi_truong() -> dict:
    goi = {}
    for ten in ("ultralytics", "torch", "torchvision", "numpy", "pillow", "opencv-python"):
        try:
            goi[ten] = _md.version(ten)
        except _md.PackageNotFoundError:
            goi[ten] = None
    return {"python": sys.version.split()[0], "python_exe": sys.executable, "platform": platform.platform(), "goi": goi}


# ----------------------------------------------------------------------------------------------- dữ liệu + mẫu
def doc_gt(tap: str, ch) -> tuple[dict, list, list, list]:
    """Cổng adapter (kiểm đủ nội dung ảnh) -> (danh tính, ảnh, rel theo thứ tự adapter, gt)."""
    d = ch["du_lieu"][tap]
    if tap == "detrac_test":
        W, H = (int(x) for x in dg.MAC_DINH["kich_thuoc"].split("x"))
        return dg.kiem_va_doc_gt(d["xml"], d["anh"], d["manifest"], d["fingerprint"], d["hash_anh"], d["loai_d4"], (W, H))
    danh_tinh, anh, gt = bv.kiem_va_doc_gt(d["nhan"], d["anh"], d["manifest"], d["hash_anh"])
    return danh_tinh, anh, [p.relative_to(d["anh"]).as_posix() for p in anh], gt


def bang_hash(tap: str, ch) -> dict:
    p = ch["du_lieu"][tap]["hash_anh"]
    if tap == "detrac_test":
        return {r["rel"]: r["sha256"] for r in fpt.doc_bang_hash(p)}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["rel"]: r["sha256"] for r in csv.DictReader(f)}


def danh_tinh_khoa(tap: str, danh_tinh: dict) -> dict:
    return {k: lay(danh_tinh, k) for k in KHOA_DANH_TINH[tap]}


def kiem_ky_vong(danh_tinh: dict, ky_vong: dict) -> list:
    return [{"truong": k, "ky_vong": v, "thuc_te": lay(danh_tinh, k)} for k, v in ky_vong.items() if lay(danh_tinh, k) != v]


def chon_mau(gt: list, lop_kiem, toi_da: int = SO_ANH_TOI_DA) -> tuple[list, dict]:
    """Chỉ số ảnh mẫu (thứ tự adapter) + {lớp: chỉ số ảnh đầu tiên có GT lớp đó}."""
    chon, ly_do = set(), {}
    for lop in lop_kiem:
        c = ID_LOP_5[lop]
        i = next((i for i, g in enumerate(gt) if (g["cls"] == c).any()), None)
        if i is None:
            raise LoiChuanBi(f"khong co anh nao co GT lop {lop}")
        ly_do[lop] = i
        chon.add(i)
    if len(chon) > toi_da:
        raise LoiChuanBi(f"so anh bat buoc theo lop ({len(chon)}) vuot {toi_da}")
    for i in range(len(gt)):
        if len(chon) >= toi_da:
            break
        chon.add(i)
    return sorted(chon), ly_do


def mo_ta_mau(tap: str, ch, rel_ds: list, anh: list, gt: list, chi_so: list, ly_do: dict) -> dict:
    """Danh sách mẫu + sha256 từng ảnh băm lại, đối chiếu bảng hash của tập."""
    bang = bang_hash(tap, ch)
    rels = [rel_ds[i] for i in chi_so]
    sha_anh = [sha(anh[i]) for i in chi_so]
    lech = [r for r, s in zip(rels, sha_anh) if bang.get(r) != s]
    if lech:
        raise LoiChuanBi(f"{len(lech)} anh mau khac bang hash (vd {lech[:3]})")
    gt_lop = {lop: int(sum(int((gt[i]["cls"] == ID_LOP_5[lop]).sum()) for i in chi_so)) for lop in LOP_5}
    thieu = [lop for lop in TAP[tap]["lop_kiem"] if not gt_lop[lop]]
    if thieu or len(rels) > SO_ANH_TOI_DA or len(set(rels)) != len(rels):
        raise LoiChuanBi(f"mau khong hop le: thieu lop {thieu}, {len(rels)} anh")
    return {"tap": tap, "so_anh": len(rels), "toi_da": SO_ANH_TOI_DA, "lop_kiem": list(TAP[tap]["lop_kiem"]),
            "quy_tac_chon": ("theo thu tu anh cua adapter: anh dau tien co GT cho tung lop can kiem, loai trung, roi bo "
                             "sung cac anh dau chua chon toi 64; giu thu tu adapter"),
            "ly_do_theo_lop": {lop: {"chi_so": i, "rel": rel_ds[i]} for lop, i in ly_do.items()},
            "chi_so": chi_so, "rel": rels, "sha256_anh": sha_anh,
            "danh_sach_anh_co_thu_tu_sha256": fpt.sha256_thu_tu(rels), "gt_theo_lop_trong_mau": gt_lop}


# ----------------------------------------------------------------------------------------------- lệnh bàn giao
def lenh_day_du() -> list:
    ra = []
    for tap, t in TAP.items():
        for ten, (p, s, _) in MODELS.items():
            out = f"results/phase2/r53_detector/full/{tap}/{ten}"
            ra.append({"tap": tap, "model": ten, "out": out,
                       "lenh": (f"{PYTHON} -u {t['adapter']} --model {ten}={p} --sha256 {ten}={s} --batch {BATCH} "
                                f"--device {DEVICE} --conf-van-hanh {CONF_VH} --gioi-han-anh 0 --out {out}")})
    return ra


def lenh_smoke() -> list:
    return [f"{PYTHON} src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap {tap} --model {ten}"
            for tap in TAP for ten in MODELS]


def can_cu_du_toan(ch) -> dict:
    ra = {}
    for tap, dirs in CAN_CU_DU_TOAN.items():
        ra[tap] = []
        for d in dirs:
            p = tu_rel(d, ch)
            try:
                kq = doc_json(p / "ket_qua.json")
            except (FileNotFoundError, ValueError):
                ra[tap].append({"thu_muc": d, "loi": "khong doc duoc ket_qua.json"})
                continue
            for ten, m in kq["mo_hinh"].items():
                npz = p / f"du_doan_{ten}.npz"
                ra[tap].append({"thu_muc": d, "mo_hinh": ten, "so_anh": m.get("so_anh_da_du_doan"),
                                "thoi_gian_du_doan_va_cham_s": m.get("thoi_gian_s"),
                                "npz_bytes": npz.stat().st_size if npz.exists() else None})
    return ra


def chuoi_bat_buoc_doc02(ma: dict) -> list:
    """Doc 02 phải chứa nguyên văn: 6 lệnh đầy đủ, 6 lệnh smoke, sha256 3 checkpoint, sha256 mọi file mã."""
    return ([x["lenh"] for x in lenh_day_du()] + lenh_smoke() + [s for _, s, _ in MODELS.values()]
            + [v["sha256"] for v in ma.values()])


# ----------------------------------------------------------------------------------------------- bước
def buoc_tien_kiem(tap: str, ch, argv) -> int:
    out_tk, out_mau = ch["chuan_bi"] / f"tien_kiem_{tap}.json", ch["chuan_bi"] / f"mau_smoke_{tap}.json"
    for p in (out_tk, out_mau):
        if p.exists():
            print(f"[LOI] {p} da ton tai -- khong ghi de", file=sys.stderr)
            return 1
    t0 = _dt.datetime.now()
    try:
        danh_tinh, anh, rel_ds, gt = doc_gt(tap, ch)
    except LOI_DU_LIEU as e:
        print(f"[LOI DATASET] {tap}: {e}", file=sys.stderr)
        return 2
    t_kiem = (_dt.datetime.now() - t0).total_seconds()
    lech = kiem_ky_vong(danh_tinh, ch["ky_vong"][tap])
    if lech:
        print(f"[LOI HO SO] {tap}: danh tinh / so luong khac ho so: {json.dumps(lech, ensure_ascii=False)}", file=sys.stderr)
        return 2
    try:
        chi_so, ly_do = chon_mau(gt, TAP[tap]["lop_kiem"])
        mau = mo_ta_mau(tap, ch, rel_ds, anh, gt, chi_so, ly_do)
    except LoiChuanBi as e:
        print(f"[LOI MAU] {tap}: {e}", file=sys.stderr)
        return 2
    mau["tao_luc"] = bay_gio()
    tk = {"buoc": "tien-kiem", "tap": tap, "tao_luc": bay_gio(), "lenh": [sys.executable, *argv],
          "cong_adapter": f"{TAP[tap]['adapter']}::kiem_va_doc_gt (kiem_noi_dung=True)", "thoi_gian_kiem_s": round(t_kiem, 1),
          "ky_vong_ho_so": ch["ky_vong"][tap], "lech_ho_so": [], "dat": True,
          "danh_tinh_khoa": danh_tinh_khoa(tap, danh_tinh),
          "danh_tinh_day_du": {k: v for k, v in danh_tinh.items() if k != "loai_D4"},
          **({"loai_D4": danh_tinh["loai_D4"]} if "loai_D4" in danh_tinh else {}),
          "so_anh_danh_gia": len(anh), "ma": hash_ma(ch), "moi_truong": moi_truong(),
          "o_dia_trong_gb": round(o_dia_trong_gb(ch["chuan_bi"]), 1)}
    ghi_json_moi(out_tk, tk)
    ghi_json_moi(out_mau, mau)
    print(f"[DAT] tien-kiem {tap}: {danh_tinh['so_video']} video, {len(anh)} anh danh gia, kiem {t_kiem:.0f} s; "
          f"mau {mau['so_anh']} anh, ly do {json.dumps(mau['ly_do_theo_lop'], ensure_ascii=False)}")
    return 0


def buoc_lenh_day_du(ch) -> int:
    ra = {"lenh_day_du": lenh_day_du(), "lenh_smoke": lenh_smoke(), "ma": hash_ma(ch),
          "checkpoint": {t: {"duong_dan": rel(p, ch), "sha256_khai_bao": s, "khop": sha(p) == s}
                         for t, (p, s, _) in ch["models"].items()},
          "can_cu_du_toan": can_cu_du_toan(ch), "o_dia_trong_gb": round(o_dia_trong_gb(ch["chuan_bi"]), 1)}
    print(json.dumps(ra, ensure_ascii=False, indent=1))
    return 0 if all(v["khop"] for v in ra["checkpoint"].values()) else 2


def file_du_lieu(tap: str, ch) -> dict:
    return {k: {"duong_dan": rel(p, ch), "sha256": sha(p)} for k, p in ch["du_lieu"][tap].items() if Path(p).is_file()}


def buoc_khoa(ch) -> int:
    cb = ch["chuan_bi"]
    p_khoa, p_sums, p_tg = cb / "khoa_danh_gia.json", cb / "SHA256SUMS", cb / "thoi_diem_khoa.json"
    for p in (p_khoa, p_sums, p_tg):
        if p.exists():
            print(f"[LOI] {p} da ton tai -- khoa da tao, khong ghi de", file=sys.stderr)
            return 1
    try:
        for tap in TAP:
            d = ch["smoke"] / tap
            if d.exists() and any(d.iterdir()):
                raise LoiChuanBi(f"{d} da co ket qua smoke -- khoa phai truoc moi smoke")
        tk, mau = {}, {}
        for tap in TAP:
            p1, p2 = cb / f"tien_kiem_{tap}.json", cb / f"mau_smoke_{tap}.json"
            if not (p1.exists() and p2.exists()):
                raise LoiChuanBi(f"thieu tien kiem / mau cua {tap}")
            tk[tap], mau[tap] = doc_json(p1), doc_json(p2)
            if not tk[tap].get("dat") or tk[tap].get("lech_ho_so"):
                raise LoiChuanBi(f"tien kiem {tap} khong dat")
        for ten, (p, s, _) in ch["models"].items():
            if sha(p) != s:
                raise LoiChuanBi(f"checkpoint {ten} sha256 khac {s[:16]}")
        ma = hash_ma(ch)
        if not ch["doc02"].exists():
            raise LoiChuanBi(f"thieu tai lieu {ch['doc02']}")
        van_ban = ch["doc02"].read_text(encoding="utf-8")
        thieu = [x for x in chuoi_bat_buoc_doc02(ma) if x not in van_ban]
        if thieu:
            raise LoiChuanBi(f"tai lieu 02 thieu {len(thieu)} chuoi bat buoc, vd {thieu[:2]}")
        cho_trong = [x for x in DAU_HIEU_CHUA_XONG if x in van_ban]
        if cho_trong:
            raise LoiChuanBi(f"tai lieu 02 con dau hieu chua xong: {cho_trong}")
        trong = o_dia_trong_gb(cb)
        if trong < ch["gb_toi_thieu"]:
            raise LoiChuanBi(f"o dia con {trong:.1f} GB < {ch['gb_toi_thieu']} GB")
    except LoiChuanBi as e:
        print(f"[LOI KHOA] {e}", file=sys.stderr)
        return 2
    qt = bv.quy_tac_hieu_luc(CONF_VH)
    khoa = {
        "phien_ban": 1, "buoc": "R53B_LOCK_SMOKE", "tao_luc": bay_gio(),
        "ghi_chu_khoa": ("khoa bang noi dung + SHA256 (SHA256SUMS) kem thoi diem (thoi_diem_khoa.json); KHONG commit theo "
                         "yeu cau hien tai; thay doi sau smoke chi qua phu luc va review lai truoc luot dai"),
        "mo_hinh": {ten: {"checkpoint": rel(p, ch), "sha256": s, "loai_checkpoint_ky_vong": lo,
                          "nguon": ("baseline COCO, ke hoach R43/R47" if lo == "coco80" else
                                    "best.pt chon tren dev noi bo mixed5, ban giao R53A (ban_giao/checkpoint_sha256.csv)")}
                    for ten, (p, s, lo) in ch["models"].items()},
        "du_lieu": {tap: {"adapter": TAP[tap]["adapter"], "file": file_du_lieu(tap, ch),
                          "danh_tinh_khoa": tk[tap]["danh_tinh_khoa"], "ky_vong_ho_so": tk[tap]["ky_vong_ho_so"],
                          "tien_kiem": {"file": rel(cb / f"tien_kiem_{tap}.json", ch),
                                        "sha256": sha(cb / f"tien_kiem_{tap}.json")}} for tap in TAP},
        "ma": ma, "moi_truong": moi_truong(),
        "quy_tac": {"du_doan": qt["du_doan"], "batch": BATCH, "device": DEVICE, "ghep": qt["ghep"], "bo_qua": qt["bo_qua"],
                    "ap": qt["ap"], "van_hanh": qt["van_hanh"], "gop_lop": qt["gop_lop"],
                    "gt_detrac_test": dg.QUY_TAC_GT, "gt_bdd_val": bv.QUY_TAC_GT,
                    "bao_cao": {"detrac_test": "D3: AP car, bus va 'mAP (car, bus)'; truck, motorcycle, bicycle = N/A",
                                "bdd_val": "du nam lop car, truck, bus, motorcycle, bicycle; motorcycle/bicycle chi tu BDD val"},
                    "khong_lam": ["khong chon lai checkpoint, khong tuning bang BDD val / UA-DETRAC test / smoke",
                                  "khong dung AP smoke de xep hang", "khong bootstrap, khong ket luan nhan qua ve tron du lieu"]},
        "mau_smoke": {tap: {"file": rel(cb / f"mau_smoke_{tap}.json", ch), "sha256": sha(cb / f"mau_smoke_{tap}.json"),
                            **{k: mau[tap][k] for k in ("so_anh", "danh_sach_anh_co_thu_tu_sha256", "rel", "sha256_anh",
                                                         "ly_do_theo_lop", "gt_theo_lop_trong_mau")}} for tap in TAP},
        "lenh_smoke": lenh_smoke(), "lenh_day_du": lenh_day_du(),
        "dau_ra": {"chuan_bi": rel(cb, ch), "smoke": rel(ch["smoke"], ch) + "/<tap>/<model>/",
                   "day_du": "results/phase2/r53_detector/full/<tap>/<model>/ (thu muc moi, adapter tu choi neu da co)"},
        "tieu_chi_nghiem_thu": {
            "smoke": ["exit 0", "khoa kiem dat truoc khi nap model", "sha256 checkpoint dung khoa",
                      "danh tinh du lieu = khoa", "dung mau da khoa (<= 64 anh, cung mau giua cac model)",
                      "loai checkpoint dung (coco80 / lop5)", "NPZ + sha256 + nguon du doan", "cham lai tu NPZ trung",
                      "UA-DETRAC: chuan hoa D3 dat; BDD: du nam lop co GT trong mau"],
            "luot_day_du": ["exit 0 cho tung lenh", "so_anh_da_du_doan = 56167 (UA-DETRAC) / 39973 (BDD val)",
                            "du_doan_<ten>.npz + sha256; UA-DETRAC co nguon_du_doan_<ten>.json",
                            "ket_qua.json / CSV hop le; sha256 checkpoint va danh tinh du lieu = khoa",
                            "AP tung lop va so sanh mo ta voi baseline yolov8n_coco",
                            "cham lai tu NPZ bang scorer cu de doi chieu; loi chi o buoc cham -> cham lai, khong suy luan lai"]},
        "tai_lieu_ke_hoach": {"file": rel(ch["doc02"], ch), "sha256": sha(ch["doc02"])},
    }
    ghi_json_moi(p_khoa, khoa)
    ds = [p_khoa, ch["doc02"]] + [cb / f"{a}_{tap}.json" for tap in TAP for a in ("tien_kiem", "mau_smoke")]
    ds += list(ch["ma"]) + [p for p, _, _ in ch["models"].values()]
    ds += [p for tap in TAP for p in ch["du_lieu"][tap].values() if Path(p).is_file()]
    dong = "".join(f"{sha(p)}  {rel(p, ch)}\n" for p in ds)
    with open(p_sums, "x", encoding="utf-8", newline="\n") as f:
        f.write(dong)
    ghi_json_moi(p_tg, {"thoi_diem_khoa": bay_gio(), "sha256sums": rel(p_sums, ch), "sha256sums_sha256": sha(p_sums),
                        "so_file": len(ds), "khoa_danh_gia_sha256": sha(p_khoa),
                        "ghi_chu": "khoa truoc moi smoke; khong commit theo yeu cau hien tai; nguoi dung quyet commit"})
    print(f"[DAT] khoa: {len(ds)} file trong SHA256SUMS ({sha(p_sums)[:16]}), khoa_danh_gia {sha(p_khoa)[:16]}")
    return 0


def kiem_khoa(ch) -> tuple[dict, dict]:
    cb = ch["chuan_bi"]
    try:
        tg = doc_json(cb / "thoi_diem_khoa.json")
        p_sums = tu_rel(tg["sha256sums"], ch)
        if sha(p_sums) != tg["sha256sums_sha256"]:
            raise LoiChuanBi("SHA256SUMS da doi sau khi khoa")
        dong = p_sums.read_text(encoding="utf-8").splitlines()
        if len(dong) != tg["so_file"]:
            raise LoiChuanBi(f"SHA256SUMS co {len(dong)} dong, khoa ghi {tg['so_file']}")
        for d in dong:
            s, sep, r = d.partition("  ")
            if not sep or len(s) != 64:
                raise LoiChuanBi(f"dong SHA256SUMS sai dinh dang: {d[:80]}")
            if sha(tu_rel(r, ch)) != s:
                raise LoiChuanBi(f"{r}: sha256 khac khoa")
        khoa = doc_json(cb / "khoa_danh_gia.json")
        if sha(cb / "khoa_danh_gia.json") != tg["khoa_danh_gia_sha256"]:
            raise LoiChuanBi("khoa_danh_gia.json khac thoi_diem_khoa")
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise LoiChuanBi(f"khong doc duoc khoa: {e}") from None
    return khoa, tg


class Tee:
    def __init__(self, goc, f):
        self.goc, self.f = goc, f
        self.encoding = "utf-8"

    def write(self, s):
        try:
            self.goc.write(s)
        except Exception:
            pass
        self.f.write(s)
        self.f.flush()
        return len(s)

    def flush(self):
        try:
            self.goc.flush()
        except Exception:
            pass
        self.f.flush()

    def isatty(self):
        return False

    def __getattr__(self, ten):            # thuộc tính khác (fileno, reconfigure, ...) chuyển cho luồng gốc
        return getattr(self.goc, ten)


def buoc_smoke(tap: str, ten: str, lan: int, ch, argv) -> int:
    out = ch["smoke"] / tap / (ten if lan == 1 else f"{ten}__lan{lan}")
    if out.exists():
        print(f"[LOI] {out} da ton tai -- khong ghi de; chay lai dung --lan moi", file=sys.stderr)
        return 1
    out.mkdir(parents=True)
    gi = ch["smoke"] / ".gitignore"
    if not gi.exists():
        gi.write_text("# R53B: NPZ smoke khong vao Git\n*.npz\n", encoding="utf-8")
    t0 = bay_gio()
    goc_out, goc_err = sys.stdout, sys.stderr
    ma = 3
    with open(out / "log.txt", "w", encoding="utf-8", newline="\n") as f:
        sys.stdout, sys.stderr = Tee(goc_out, f), Tee(goc_err, f)
        try:
            print(f"# {t0} lenh: {' '.join([sys.executable, *argv])}")
            ma = _smoke(tap, ten, out, ch, argv)
        except Exception:
            traceback.print_exc()
            ma = 3
        finally:
            print(f"# {bay_gio()} exit={ma}")
            sys.stdout, sys.stderr = goc_out, goc_err
    ghi_json_moi(out / "exit.json", {"exit": ma, "bat_dau": t0, "ket_thuc": bay_gio(), "lenh": [sys.executable, *argv]})
    return ma


def _smoke(tap: str, ten: str, out: Path, ch, argv) -> int:
    try:
        khoa, tg = kiem_khoa(ch)
        if ten not in khoa["mo_hinh"] or tap not in khoa["mau_smoke"]:
            raise LoiChuanBi(f"{tap}/{ten} khong co trong khoa")
    except LoiChuanBi as e:
        print(f"[LOI KHOA] {e}", file=sys.stderr)
        return 2
    print(f"[khoa] dat: {tg['so_file']} file khop SHA256SUMS, khoa luc {tg['thoi_diem_khoa']}", flush=True)
    m = khoa["mo_hinh"][ten]
    w = tu_rel(m["checkpoint"], ch)
    s = sha(w)
    if s != m["sha256"]:
        print(f"[LOI CHECKPOINT] {ten}: sha256 {s[:16]} khac khoa {m['sha256'][:16]}", file=sys.stderr)
        return 2
    trong = o_dia_trong_gb(out)
    if trong < ch["gb_toi_thieu"]:
        print(f"[LOI] o dia con {trong:.1f} GB < {ch['gb_toi_thieu']} GB", file=sys.stderr)
        return 2
    t1 = _dt.datetime.now()
    try:
        danh_tinh, anh, rel_ds, gt = doc_gt(tap, ch)
    except LOI_DU_LIEU as e:
        print(f"[LOI DATASET] {e}", file=sys.stderr)
        return 2
    t_kiem = (_dt.datetime.now() - t1).total_seconds()
    dtk = danh_tinh_khoa(tap, danh_tinh)
    khac = sorted(k for k, v in dtk.items() if json.dumps(v, sort_keys=True) !=
                  json.dumps(khoa["du_lieu"][tap]["danh_tinh_khoa"].get(k), sort_keys=True))
    if khac:
        print(f"[LOI DATASET] danh tinh du lieu khac khoa: {khac}", file=sys.stderr)
        return 2
    mk = khoa["mau_smoke"][tap]
    try:
        chi_so, _ = chon_mau(gt, TAP[tap]["lop_kiem"])
    except LoiChuanBi as e:
        print(f"[LOI MAU] {e}", file=sys.stderr)
        return 2
    rels = [rel_ds[i] for i in chi_so]
    sha_anh = [sha(anh[i]) for i in chi_so]
    if (rels != mk["rel"] or fpt.sha256_thu_tu(rels) != mk["danh_sach_anh_co_thu_tu_sha256"] or sha_anh != mk["sha256_anh"]
            or len(rels) > SO_ANH_TOI_DA):
        print("[LOI MAU] mau tinh lai / noi dung anh mau khac khoa", file=sys.stderr)
        return 2
    print(f"[du lieu] {tap}: kiem adapter {t_kiem:.0f} s, danh tinh = khoa; mau {len(rels)} anh = khoa", flush=True)
    anh_m, gt_m = [anh[i] for i in chi_so], [gt[i] for i in chi_so]
    ghi_json_moi(out / "danh_sach_anh.json", {"rel": rels, "sha256": fpt.sha256_thu_tu(rels), "sha256_anh": sha_anh,
                                               "cach_tinh": fpt.CACH_TINH["danh_sach_anh_co_thu_tu_sha256"],
                                               "ghi_luc": bay_gio(), "ghi_chu": "ghi TRUOC suy luan"})
    qt = bv.quy_tac_hieu_luc(CONF_VH)
    if json.dumps(qt["du_doan"], sort_keys=True) != json.dumps(khoa["quy_tac"]["du_doan"], sort_keys=True) or \
            qt["van_hanh"] != khoa["quy_tac"]["van_hanh"]:
        print("[LOI KHOA] quy tac du doan / van hanh hien tai khac khoa", file=sys.stderr)
        return 2
    cfg = ch["smoke"] / "_cau_hinh_thu_vien"            # cấu hình ultralytics/matplotlib trong vùng ghi, không vào hồ sơ người dùng
    cfg.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"], os.environ["MPLCONFIGDIR"], os.environ["YOLO_OFFLINE"] = str(cfg), str(cfg), "1"
    mod = dg if tap == "detrac_test" else bv
    t2 = _dt.datetime.now()
    dd, mt = mod.du_doan_ultralytics(str(w), anh_m, qt["du_doan"], BATCH, DEVICE)
    t_dd = (_dt.datetime.now() - t2).total_seconds()
    if len(dd) != len(anh_m):
        print(f"[LOI] {len(dd)} ket qua du doan cho {len(anh_m)} anh", file=sys.stderr)
        return 3
    npz = out / f"du_doan_{ten}.npz"
    ev.luu_du_doan(npz, anh_m, dd)
    if tap == "detrac_test":
        dg.ghi_nguon(out, ten, m["checkpoint"], s, mt, qt, danh_tinh, len(anh_m))
    else:
        ghi_json_moi(out / f"nguon_du_doan_{ten}.json", {
            "phien_ban": 1, "mo_hinh": ten, "checkpoint": m["checkpoint"], "sha256": s, "anh_xa_lop": mt,
            "quy_tac_du_doan": qt["du_doan"], "npz": npz.name, "npz_sha256": sha(npz), "so_anh": len(anh_m),
            "du_lieu": dtk, "adapter_sha256": sha(Path(bv.__file__)), "tao_luc": bay_gio()})
    kiem = {"khoa": True, "checkpoint_sha256": True, "danh_tinh_du_lieu": True, "mau_bang_khoa": True,
            "so_anh_du_doan": len(dd) == len(anh_m),
            "loai_checkpoint": (mt or {}).get("loai_checkpoint") == m["loai_checkpoint_ky_vong"]}
    if not kiem["loai_checkpoint"]:
        print(f"[LOI ANH XA] {ten}: loai checkpoint {(mt or {}).get('loai_checkpoint')} khac "
              f"{m['loai_checkpoint_ky_vong']}", file=sys.stderr)
        return 3
    r = ev.danh_gia(gt_m, dd, qt["van_hanh"]["conf"])
    if r.get("van_hanh") != qt["van_hanh"]:
        print(f"[LOI] nguong van hanh cua scorer {r.get('van_hanh')} khac {qt['van_hanh']}", file=sys.stderr)
        return 3
    ten_npz, dd2 = dg.doc_du_doan(npz, len(anh_m))
    r2 = ev.danh_gia(gt_m, dd2, qt["van_hanh"]["conf"])
    kiem["npz_ten_anh"] = [str(x) for x in ten_npz] == [p.name for p in anh_m]
    kiem["cham_lai_tu_npz_trung"] = json.dumps(ev._json_safe(r), sort_keys=True) == json.dumps(ev._json_safe(r2), sort_keys=True)
    ket = None
    if tap == "detrac_test":
        try:
            ket = dg.chuan_hoa_d3(r)
            kiem["chuan_hoa_D3"] = True
        except dg.LoiDuLieu as e:
            print(f"[LOI D3] {e}", file=sys.stderr)
            kiem["chuan_hoa_D3"] = False
    else:
        kiem["du_nam_lop_co_gt"] = all(r["theo_lop"][lop]["n_gt"] > 0 for lop in LOP_5)
    gpu = {}
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            gpu = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except Exception as e:  # noqa: BLE001 -- chỉ là metadata
            gpu = {"loi_doc_gpu": repr(e)}
    kq = {"canh_bao": CANH_BAO_SMOKE, "tap": tap, "mo_hinh_ten": ten, "tao_luc": bay_gio(), "lenh": [sys.executable, *argv],
          "khoa": {"thoi_diem_khoa": tg["thoi_diem_khoa"], "sha256sums_sha256": tg["sha256sums_sha256"]},
          "quy_tac": qt, "batch": BATCH, "device": DEVICE,
          "dataset": {"ten": tap, "n_anh": len(anh_m), "mau_smoke": True, "danh_tinh_khoa": dtk},
          "mo_hinh": {ten: {"checkpoint": m["checkpoint"], "sha256": s, "anh_xa_lop": mt, "so_anh_da_du_doan": len(dd),
                            "npz": npz.name, "npz_sha256": sha(npz), "thoi_gian_du_doan_s": round(t_dd, 1),
                            "chan_doan_tho": bv.chan_doan(gt_m, dd, qt["van_hanh"]["conf"]),
                            **({"ket_qua_D3": ket} if tap == "detrac_test" else r)}},
          "thoi_gian_kiem_du_lieu_s": round(t_kiem, 1), "kiem": kiem, "moi_truong": {**moi_truong(), **gpu},
          "cau_hinh_thu_vien": {"YOLO_CONFIG_DIR": rel(cfg, ch), "YOLO_OFFLINE": "1"}}
    if tap == "detrac_test" and ket is not None:
        dg.ghi_ket_qua(out, kq, {ten: ket})
    else:
        ghi_json_moi(out / "ket_qua.json", kq)
        with open(out / "theo_lop.csv", "x", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=["mo_hinh", "lop", "n_gt", "AP50", "AP50_95", "precision_vh", "recall_vh"])
            cw.writeheader()
            cw.writerows(ev._json_safe([{"mo_hinh": ten, "lop": lop, "n_gt": v["n_gt"], "AP50": v["AP50"],
                                         "AP50_95": v["AP50_95"], "precision_vh": v["van_hanh"]["precision"],
                                         "recall_vh": v["van_hanh"]["recall"]} for lop, v in r["theo_lop"].items()]))
    hong = [k for k, v in kiem.items() if not v]
    if hong:
        print(f"[LOI] kiem sau du doan khong dat: {hong}", file=sys.stderr)
        return 3
    print(f"[DAT] smoke {tap}/{ten}: {len(dd)} anh, du doan {t_dd:.1f} s, NPZ {sha(npz)[:16]}, kiem {sorted(kiem)}", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- main
def main(argv=None, ch=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="R53B: tien kiem, khoa, smoke danh gia ba detector (khong chay danh gia day du)")
    ap.add_argument("--buoc", required=True, choices=["tien-kiem", "lenh-day-du", "khoa", "kiem-khoa", "smoke"])
    ap.add_argument("--tap", choices=list(TAP))
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--lan", type=int, default=1, help="chi cho smoke: lan chay lai vao thu muc moi <model>__lan<N>")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    ch = ch or cau_hinh_mac_dinh()
    can_tap = a.buoc in ("tien-kiem", "smoke")
    if bool(a.tap) != can_tap or bool(a.model) != (a.buoc == "smoke") or a.lan < 1 or (a.lan != 1 and a.buoc != "smoke"):
        print(f"[LOI] --buoc {a.buoc}: --tap / --model / --lan khong hop le", file=sys.stderr)
        return 2
    if a.buoc == "tien-kiem":
        return buoc_tien_kiem(a.tap, ch, argv)
    if a.buoc == "lenh-day-du":
        return buoc_lenh_day_du(ch)
    if a.buoc == "khoa":
        return buoc_khoa(ch)
    if a.buoc == "kiem-khoa":
        try:
            _, tg = kiem_khoa(ch)
        except LoiChuanBi as e:
            print(f"[LOI KHOA] {e}", file=sys.stderr)
            return 2
        print(f"[DAT] kiem-khoa: {tg['so_file']} file khop SHA256SUMS ({tg['sha256sums_sha256']}), khoa luc "
              f"{tg['thoi_diem_khoa']}")
        return 0
    return buoc_smoke(a.tap, a.model, a.lan, ch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
