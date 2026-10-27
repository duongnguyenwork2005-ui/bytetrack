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
  python src/r53_detector/chuan_bi_danh_gia.py --buoc phu-luc           # phụ lục 1 sau khóa (review R53B F1–F3)
  python src/r53_detector/chuan_bi_danh_gia.py --buoc phu-luc-2         # phụ lục 2 nối tiếp phụ lục 1 (F2, Pdaa0fd9d)
  python src/r53_detector/chuan_bi_danh_gia.py --buoc phu-luc-3         # phụ lục 3 nối tiếp phụ lục 2 (F2 [P2], vòng 5)
  python src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model <ten> [--lan N]   # sau lượt đầy đủ BDD

Phụ lục 2: `cham-lai-bdd` chỉ đọc full/bdd_val/<ten>/du_doan_<ten>.npz, kiểm hash NPZ với bằng chứng
full/_hau_kiem/bdd_val__<ten>__npz.txt, lệnh gốc trong full/_giam_sat/bdd_val__<ten>/tom_tat.json (adapter so theo đường
dẫn sau resolve với cwd trong trang_thai.json cùng thư mục -- phụ lục 3), cổng dữ liệu đầy đủ
(`bv.kiem_va_doc_gt`), tên + thứ tự ảnh NPZ (`dg.doc_du_doan`), rồi chấm bằng `ev.danh_gia` với `bv.quy_tac_hieu_luc(0.25)`;
ghi ket_qua.json/theo_lop.csv vào full/bdd_val/<ten>__cham_lai/ (mới; đã có -> exit 1). Không suy luận, không so detector.

Phụ lục 1 (sau khóa, không ghi đè khóa gốc): chỉ ba file được đổi (tài liệu 02 — chỉ nối thêm sau mốc "## Phụ lục 1",
file này và file test). `phu-luc` kiểm mọi file khác của SHA256SUMS gốc vẫn đúng hash, giữ bản gốc (tiền tố byte của
tài liệu 02; bản sao hai file mã từ snapshot review, đã kiểm hash gốc), đòi một lần test đạt trên đúng mã hiện tại, rồi
ghi chuan_bi/phu_luc_1/ (khoa_phu_luc_1.json, SHA256SUMS, thoi_diem_phu_luc_1.json). `kiem-khoa` kiểm cả chuỗi:
SHA256SUMS gốc + phụ lục; file đổi phải đúng hash phụ lục và bản gốc phải còn khớp hash khóa gốc.

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
import hashlib  # noqa: E402
import importlib.metadata as _md  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
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
# Phụ lục 1 (sau khóa): mốc trong tài liệu 02, bản gốc mã lấy từ snapshot review của orchestrator (chỉ đọc).
PYTHON_WIN = PYTHON.replace("/", "\\")
PHU_LUC = "phu_luc_1"
MOC_PHU_LUC = "\n## Phụ lục 1"
BAN_GOC_SNAPSHOT = Path("D:/LABOTARY/automation/runs_local/bytetrack_r53b_lock_smoke_20261020_0356/review_points/"
                        "0002_review/files")
LY_DO_PHU_LUC = {
    "nguon": "review Codex R53B vong 1 (bytetrack_r53b_lock_smoke_20261020_0356/reports/review_001.json): revise F1-F3",
    "F1": "fixture CLI cua chinh hai adapter goc (parser, tham so ban giao, cong hash, tu choi ghi de)",
    "F2": "quy trinh BDD cu the khi NPZ da luu nhung cham loi; kiem tren fixture",
    "F3": "moi truong dat ngay trong tien trinh khoi chay doc lap (bytecode, cache/cau hinh thu vien, offline)",
    "khong_doi": "quy tac, ba checkpoint, du lieu, mau smoke, sau lenh day du va bang chung smoke giu nguyen",
}
# Phụ lục 2 (sau review R53B vòng 2-3, F2): bước chấm lại BDD từ NPZ, người dùng duyệt riêng (Pdaa0fd9d).
PHU_LUC_DS = {1: ("phu_luc_1", "\n## Phụ lục 1"), 2: ("phu_luc_2", "\n## Phụ lục 2"), 3: ("phu_luc_3", "\n## Phụ lục 3")}
BAN_GOC_SNAPSHOT_2 = [Path("D:/LABOTARY/automation/runs_local/bytetrack_r53b_lock_smoke_20261020_0356/review_points/") / d
                      / "files" for d in ("0006_review", "0004_review")]
# Phụ lục 3 (sau review R53B vòng 4, F2 [P2]): cổng nguồn so đường dẫn adapter sau resolve; bản phụ lục 2 của mã lấy
# từ snapshot review vòng 4 của orchestrator (chỉ đọc, kiểm hash).
BAN_GOC_SNAPSHOT_3 = [Path("D:/LABOTARY/automation/runs_local/bytetrack_r53b_lock_smoke_20261020_0356/review_points/"
                           "0009_review/files")]
LY_DO_PHU_LUC_2 = {
    "nguon": ("review Codex R53B vong 2 (review_002.json) va vong 3: blocked, F2 con mo; nguoi dung duyet buoc hep "
              "Pdaa0fd9d (cham lai BDD tu NPZ, chi doc); P1517bbbd van bi tu choi"),
    "F2": "--buoc cham-lai-bdd: cham lai mot thu muc full BDD tu NPZ bang ham cu, khong suy luan, khong bang so sanh",
    "khong_doi": ("quy tac, ba checkpoint, du lieu, mau smoke, sau lenh day du, khoa goc, phu luc 1 va bang chung smoke "
                  "giu nguyen; khong kiem DETRAC trong buoc nay"),
}
LY_DO_PHU_LUC_3 = {
    "nguon": ("review Codex R53B vong 4 (review_004.json): blocked, F2 [P2] -- cong nguon chi so ten file adapter; vong "
              "sua 5/5 trong buoc Pdaa0fd9d"),
    "F2_P2": ("cham-lai-bdd: duong dan adapter trong lenh giam sat, resolve theo cwd cua bo giam sat (trang_thai.json), "
              "phai la file adapter BDD trong repo; script cung ten o duong dan khac -> exit 2"),
    "khong_doi": ("quy tac, ba checkpoint, du lieu, mau smoke, sau lenh day du, ba lenh cham lai, khoa goc, phu luc 1, "
                  "phu luc 2 va bang chung smoke giu nguyen; khong kiem DETRAC, khong bang so sanh"),
}
LY_DO_PHU_LUC_DS = {2: ("R53B_LOCK_SMOKE (buoc hep Pdaa0fd9d, F2)", LY_DO_PHU_LUC_2),
                    3: ("R53B_LOCK_SMOKE (buoc hep Pdaa0fd9d, sua F2 [P2] sau review vong 4)", LY_DO_PHU_LUC_3)}
HAM_CHAM_LAI = ["danh_gia_detector_bdd_val.kiem_va_doc_gt", "danh_gia_detector_detrac_test.doc_du_doan",
                "danh_gia_detector_bdd_val.quy_tac_hieu_luc(0.25)", "eval_detector_5lop.danh_gia",
                "danh_gia_detector_bdd_val.chan_doan"]
DOI_SO_DU_LIEU_BDD = {"--nhan-cocovid": "nhan", "--anh-goc": "anh", "--manifest-video": "manifest", "--hash-anh": "hash_anh"}


class LoiChuanBi(Exception):
    """Cổng khóa / mẫu / tài liệu không đạt."""


# ----------------------------------------------------------------------------------------------- tiện ích
def cau_hinh_mac_dinh() -> dict:
    return {"repo": REPO, "chuan_bi": R53 / "chuan_bi", "smoke": R53 / "smoke",
            "doc02": REPO / "handoff" / "logs" / "review53" / "02_ke_hoach_khoa_danh_gia.md",
            "du_lieu": {"detrac_test": {k: dg.p_abs(v) for k, v in dg.MAC_DINH.items() if k != "kich_thuoc"},
                        "bdd_val": {k: bv.p_abs(v) for k, v in bv.MAC_DINH.items()}},
            "ky_vong": KY_VONG_DU_LIEU, "models": {t: (REPO / p, s, lo) for t, (p, s, lo) in MODELS.items()},
            "ma": [REPO / m for m in MA], "gb_toi_thieu": GB_TOI_THIEU,
            "full": R53 / "full", "tests_eval": R53 / "tests_eval", "ban_goc": BAN_GOC_SNAPSHOT, "yeu_cau_test": True,
            "ban_goc_2": BAN_GOC_SNAPSHOT_2, "ban_goc_3": BAN_GOC_SNAPSHOT_3,
            "doi_sau_khoa": [REPO / "handoff" / "logs" / "review53" / "02_ke_hoach_khoa_danh_gia.md",
                             REPO / "src" / "r53_detector" / "chuan_bi_danh_gia.py",
                             REPO / "src" / "r53_detector" / "test_chuan_bi_danh_gia.py"]}


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


def moi_truong_doc_lap(ch) -> dict:
    """Biến môi trường đặt NGAY trong tiến trình khởi chạy độc lập (WMI nhận môi trường chuẩn, không kế thừa phiên):
    tắt bytecode, UTF-8, không đệm, cấu hình ultralytics/matplotlib trong vùng đầu ra, ultralytics offline."""
    cfg = str(Path(ch["full"]) / "_cau_hinh_thu_vien")
    return {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
            "YOLO_CONFIG_DIR": cfg, "MPLCONFIGDIR": cfg, "YOLO_OFFLINE": "1"}


def doi_so_day_du(tap: str, ten: str) -> list:
    """Đối số adapter của lệnh đầy đủ -- cùng nội dung với `lenh_day_du` (test kiểm chuỗi nối lại nằm trong lệnh đó)."""
    p, s, _ = MODELS[ten]
    return ["--model", f"{ten}={p}", "--sha256", f"{ten}={s}", "--batch", str(BATCH), "--device", DEVICE,
            "--conf-van-hanh", str(CONF_VH), "--gioi-han-anh", "0", "--out", f"results/phase2/r53_detector/full/{tap}/{ten}"]


def dong_lenh_doc_lap(env: dict, giam_sat: str, con: list, py: str = PYTHON_WIN) -> str:
    """Dòng lệnh cho WMI Win32_Process.Create: cmd.exe đặt biến rồi chạy bộ giám sát, bộ giám sát chạy `con` bằng cùng
    python với -B. Thư mục cấu hình thư viện được tạo TRƯỚC python: ultralytics gặp YOLO_CONFIG_DIR chưa tồn tại thì
    lặng lẽ dùng <cwd>\\Ultralytics (test F2 vòng sửa đã gặp). Không nhận ký tự điều khiển của cmd trong đối số."""
    for x in [*env.values(), giam_sat, *con, py]:
        if any(c in str(x) for c in '&|<>^%"()\r\n'):
            raise LoiChuanBi(f"doi so co ky tu dieu khien cmd: {x!r}")
    dat = " && ".join(f'set "{k}={v}"' for k, v in env.items())
    for k in dict.fromkeys(k for k in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR") if k in env):
        if not (k == "MPLCONFIGDIR" and env.get("YOLO_CONFIG_DIR") == env[k]):
            dat += f' && (if not exist "{env[k]}\\" mkdir "{env[k]}")'
    return (f'cmd.exe /d /c {dat} && "{py}" -B -u src\\chay_giam_sat.py chay --thu-muc {giam_sat} --khong-in -- '
            f'"{py}" -B -u ' + " ".join(str(x) for x in con))


def lenh_doc_lap(ch) -> list:
    env = moi_truong_doc_lap(ch)
    ra = []
    for tap, t in TAP.items():
        for ten in MODELS:
            gs = rel(Path(ch["full"]) / "_giam_sat" / f"{tap}__{ten}", ch).replace("/", "\\")
            ra.append({"tap": tap, "model": ten, "giam_sat": gs,
                       "dong_lenh": dong_lenh_doc_lap(env, gs, [t["adapter"].replace("/", "\\"), *doi_so_day_du(tap, ten)])})
    return ra


def chuoi_bat_buoc_phu_luc(ch) -> list:
    """Phần phụ lục của tài liệu 02 phải chứa nguyên văn: 6 dòng lệnh khởi chạy độc lập và từng biến môi trường."""
    return [x["dong_lenh"] for x in lenh_doc_lap(ch)] + [f'set "{k}={v}"' for k, v in moi_truong_doc_lap(ch).items()]


def lenh_cham_lai_bdd() -> list:
    """Ba lệnh chấm lại BDD (phụ lục 2) -- chỉ bàn giao, chạy sau lượt đầy đủ; mỗi lệnh một thư mục full/bdd_val/<model>."""
    return [{"model": ten, "lenh": f"{PYTHON} src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model {ten}"}
            for ten in MODELS]


def lenh_ghi_bang_chung_npz(ten: str) -> list:
    """Hai dòng PowerShell ghi bằng chứng NPZ (kích thước + SHA256) ngay khi lệnh BDD kết thúc; -NoClobber: không ghi đè."""
    npz = f"results\\phase2\\r53_detector\\full\\bdd_val\\{ten}\\du_doan_{ten}.npz"
    bc = f"results\\phase2\\r53_detector\\full\\_hau_kiem\\bdd_val__{ten}__npz.txt"
    return [f"Get-Item {npz} | Select-Object FullName, Length, LastWriteTime | Format-List | Out-File {bc} -Encoding utf8 "
            f"-Width 500 -NoClobber",
            f"Get-FileHash -Algorithm SHA256 {npz} | Format-List | Out-File {bc} -Append -Encoding utf8 -Width 500"]


def chuoi_bat_buoc_phu_luc_2(ch) -> list:
    """Phụ lục 2 của tài liệu 02 phải chứa nguyên văn: 3 lệnh chấm lại, 6 dòng ghi bằng chứng NPZ, sha256 các file mã đổi."""
    return ([x["lenh"] for x in lenh_cham_lai_bdd()] + [d for ten in MODELS for d in lenh_ghi_bang_chung_npz(ten)]
            + [sha(p) for p in ch["doi_sau_khoa"] if Path(p).resolve() != Path(ch["doc02"]).resolve()])


def chuoi_bat_buoc_phu_luc_3(ch) -> list:
    """Phụ lục 3 của tài liệu 02 phải chứa nguyên văn: 3 lệnh chấm lại (không đổi), sha256 các file mã đổi."""
    return ([x["lenh"] for x in lenh_cham_lai_bdd()]
            + [sha(p) for p in ch["doi_sau_khoa"] if Path(p).resolve() != Path(ch["doc02"]).resolve()])


CHUOI_BAT_BUOC_PHU_LUC = {2: chuoi_bat_buoc_phu_luc_2, 3: chuoi_bat_buoc_phu_luc_3}


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
    ra["moi_truong_doc_lap"] = moi_truong_doc_lap(ch)
    print(json.dumps(ra, ensure_ascii=False, indent=1))
    print("# dong lenh khoi chay doc lap (CommandLine cho WMI Win32_Process.Create, CurrentDirectory = repo):")
    for x in lenh_doc_lap(ch):
        print(x["dong_lenh"])
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
        dong = doc_sums(p_sums, tg["so_file"])
        pl, cap = kiem_chuoi_phu_luc(ch, tg)
        for r, s in dong.items():
            if r in pl:                                   # đổi sau khóa: hash mới + bản gốc đã kiểm trong kiem_phu_luc
                if hash_truoc(pl[r]) != s:
                    raise LoiChuanBi(f"{r}: phu luc ghi hash goc khac SHA256SUMS goc")
                continue
            if sha(tu_rel(r, ch)) != s:
                raise LoiChuanBi(f"{r}: sha256 khac khoa")
        khoa = doc_json(cb / "khoa_danh_gia.json")
        if sha(cb / "khoa_danh_gia.json") != tg["khoa_danh_gia_sha256"]:
            raise LoiChuanBi("khoa_danh_gia.json khac thoi_diem_khoa")
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise LoiChuanBi(f"khong doc duoc khoa: {e}") from None
    return khoa, dict(tg, _phu_luc=sorted(pl), _cap_phu_luc=cap)


def doc_sums(p: Path, so_file: int) -> dict:
    dong = p.read_text(encoding="utf-8").splitlines()
    if len(dong) != so_file:
        raise LoiChuanBi(f"{p.name} co {len(dong)} dong, khoa ghi {so_file}")
    ra = {}
    for d in dong:
        s, sep, r = d.partition("  ")
        if not sep or len(s) != 64 or r in ra:
            raise LoiChuanBi(f"dong {p.name} sai dinh dang / trung: {d[:80]}")
        ra[r] = s
    return ra


def sha_byte(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_truoc(x: dict) -> str:
    """Hash của file ở khóa liền trước mức phụ lục đã đổi nó (PL1 ghi 'sha256_khoa_goc', PL2 ghi 'sha256_khoa_truoc')."""
    return x.get("sha256_khoa_truoc", x.get("sha256_khoa_goc"))


def doc_cap_phu_luc(ch, k: int, tg: dict, truoc: dict | None) -> dict | None:
    """Đọc + kiểm toàn vẹn MỘT mức phụ lục (SHA256SUMS, khoa_*.json, con trỏ về khóa gốc / mức trước). Không kiểm file hiện
    tại. Không có thư mục -> None; dở dang (thiếu thoi_diem_*.json) -> lỗi."""
    ten = PHU_LUC_DS[k][0]
    d = ch["chuan_bi"] / ten
    if not d.exists():
        return None
    tp = doc_json(d / f"thoi_diem_{ten}.json")
    p_sums = d / "SHA256SUMS"
    if sha(p_sums) != tp["sha256sums_sha256"]:
        raise LoiChuanBi(f"SHA256SUMS {ten} da doi")
    sums = doc_sums(p_sums, tp["so_file"])
    pl = doc_json(d / f"khoa_{ten}.json")
    if sha(d / f"khoa_{ten}.json") != tp["khoa_phu_luc_sha256"]:
        raise LoiChuanBi(f"khoa_{ten}.json khac thoi_diem_{ten}")
    if (pl["khoa_goc"]["sha256sums_sha256"] != tg["sha256sums_sha256"]
            or pl["khoa_goc"]["thoi_diem_khoa"] != tg["thoi_diem_khoa"]):
        raise LoiChuanBi(f"{ten} khong tro dung khoa goc")
    if truoc is not None:
        kt = pl.get("khoa_truoc") or {}
        if kt.get("phu_luc") != truoc["ten"] or kt.get("sha256sums_sha256") != truoc["tp"]["sha256sums_sha256"]:
            raise LoiChuanBi(f"{ten} khong tro dung {truoc['ten']}")
    return {"k": k, "ten": ten, "dir": d, "tp": tp, "sums": sums, "pl": pl}


def doc_cac_cap(ch, tg: dict) -> list:
    """Các mức phụ lục liên tiếp từ 1; mức sau tồn tại mà thiếu mức trước -> lỗi."""
    cap, truoc = [], None
    for k in sorted(PHU_LUC_DS):
        L = doc_cap_phu_luc(ch, k, tg, truoc)
        if L is None:
            sau = [PHU_LUC_DS[j][0] for j in PHU_LUC_DS if j > k and (ch["chuan_bi"] / PHU_LUC_DS[j][0]).exists()]
            if sau:
                raise LoiChuanBi(f"{sau} ton tai nhung thieu {PHU_LUC_DS[k][0]}")
            break
        cap.append(L)
        truoc = L
    return cap


def kiem_chuoi_phu_luc(ch, tg: dict) -> tuple[dict, list]:
    """Kiểm cả chuỗi phụ lục trên file hiện tại. Trả ({rel: mục thay đổi ĐẦU TIÊN, hash_truoc = hash khóa gốc}, tên các mức).
    Mỗi mức: dòng SHA256SUMS của nó phải khớp file hiện tại, trừ file bị mức SAU đổi (khi đó mức sau phải ghi đúng hash
    này là hash trước và giữ bản đó); bản trước của từng file đổi (tiền tố tài liệu 02 / bản sao mã) phải còn khớp."""
    cap = doc_cac_cap(ch, tg)
    for i, L in enumerate(cap):
        doi_sau = {}
        for L2 in cap[i + 1:]:
            for x in L2["pl"]["thay_doi"]:
                doi_sau.setdefault(x["file"], x)
        for r, s in L["sums"].items():
            if r in doi_sau:
                if hash_truoc(doi_sau[r]) != s:
                    raise LoiChuanBi(f"{r}: muc sau {L['ten']} ghi hash truoc khac SHA256SUMS {L['ten']}")
            elif sha(tu_rel(r, ch)) != s:
                raise LoiChuanBi(f"{r}: sha256 khac {L['ten']}")
        for x in L["pl"]["thay_doi"]:
            p = tu_rel(x["file"], ch)
            if x["file"] in doi_sau:
                if hash_truoc(doi_sau[x["file"]]) != x["sha256_moi"]:
                    raise LoiChuanBi(f"{x['file']}: muc sau khong noi tiep {L['ten']}")
            elif sha(p) != x["sha256_moi"]:
                raise LoiChuanBi(f"{x['file']}: doi sau {L['ten']}")
            goc = (p.read_bytes()[:x["so_byte_tien_to"]] if x["cach_giu_ban_goc"] == "tien_to"
                   else tu_rel(x["ban_sao"], ch).read_bytes())
            if sha_byte(goc) != hash_truoc(x):
                raise LoiChuanBi(f"{x['file']}: ban truoc {L['ten']} giu lai khong con khop hash")
    ra = {}
    for L in cap:
        for x in L["pl"]["thay_doi"]:
            ra.setdefault(x["file"], x)
    return ra, [L["ten"] for L in cap]


def kiem_phu_luc(ch, tg: dict) -> dict:
    """Không có thư mục phụ lục -> {} (khóa gốc nguyên vẹn). Có -> kiểm cả chuỗi, trả {rel: mục thay đổi}."""
    return kiem_chuoi_phu_luc(ch, tg)[0]


def ket_qua_test_dat(ch) -> Path:
    """Lần test mới nhất trong tests_eval/ đạt toàn bộ và ghi đúng hash mã hiện tại."""
    hien = {m: sha(REPO / m) for m in MA}
    for p in sorted(Path(ch["tests_eval"]).glob("lan_*/ket_qua_test.json"), reverse=True):
        t = doc_json(p)
        if t.get("ma_sha256") == hien:
            if t.get("tong") and t.get("dat") == t.get("tong"):
                return p
            raise LoiChuanBi(f"lan test moi nhat tren ma hien tai khong dat: {rel(p, ch)} {t.get('dat')}/{t.get('tong')}")
    raise LoiChuanBi("chua co lan test nao tren dung ma hien tai")


def buoc_phu_luc(ch) -> int:
    cb, d = ch["chuan_bi"], ch["chuan_bi"] / PHU_LUC
    if d.exists():
        print(f"[LOI] {d} da ton tai -- phu luc 1 da tao, khong ghi de", file=sys.stderr)
        return 1
    try:
        try:
            tg = doc_json(cb / "thoi_diem_khoa.json")
            p_sums = tu_rel(tg["sha256sums"], ch)
            if sha(p_sums) != tg["sha256sums_sha256"] or sha(cb / "khoa_danh_gia.json") != tg["khoa_danh_gia_sha256"]:
                raise LoiChuanBi("khoa goc (SHA256SUMS / khoa_danh_gia.json) da doi")
            dong = doc_sums(p_sums, tg["so_file"])
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise LoiChuanBi(f"khong doc duoc khoa goc: {e}") from None
        duoc_doi = {rel(p, ch): Path(p) for p in ch["doi_sau_khoa"]}
        if set(duoc_doi) - set(dong):
            raise LoiChuanBi(f"file duoc phep doi khong co trong khoa goc: {sorted(set(duoc_doi) - set(dong))}")
        thay_doi, sao = [], []
        for r, s in dong.items():
            p = tu_rel(r, ch)
            moi = sha(p)
            if moi == s:
                continue
            if r not in duoc_doi:
                raise LoiChuanBi(f"{r}: doi sau khoa nhung khong thuoc tap duoc phep doi")
            x = {"file": r, "sha256_khoa_goc": s, "sha256_moi": moi}
            if p.resolve() == Path(ch["doc02"]).resolve():
                b = p.read_bytes()
                i = b.find(MOC_PHU_LUC.encode("utf-8"))
                if i < 0 or sha_byte(b[:i]) != s:
                    raise LoiChuanBi("tai lieu 02: phan truoc moc phu luc khong con trung ban da khoa (chi duoc noi them)")
                phan = b[i:].decode("utf-8")
                thieu = [c for c in chuoi_bat_buoc_phu_luc(ch) if c not in phan]
                cho_trong = [c for c in DAU_HIEU_CHUA_XONG if c in phan]
                if thieu or cho_trong:
                    raise LoiChuanBi(f"phu luc tai lieu 02 thieu {len(thieu)} chuoi bat buoc (vd {thieu[:1]}) / "
                                     f"con dau hieu chua xong {cho_trong}")
                x.update(cach_giu_ban_goc="tien_to", so_byte_tien_to=i, moc=MOC_PHU_LUC.strip())
            else:
                g = Path(ch["ban_goc"]) / r
                if not g.is_file() or sha(g) != s:
                    raise LoiChuanBi(f"{r}: khong co ban goc dung hash khoa goc tai {g}")
                x.update(cach_giu_ban_goc="ban_sao", ban_sao=rel(d / "ban_khoa_goc" / r, ch), nguon_ban_sao=str(g))
                sao.append((g, d / "ban_khoa_goc" / r, s))
            thay_doi.append(x)
        if not thay_doi:
            raise LoiChuanBi("khong co file nao doi sau khoa -- khong can phu luc")
        if rel(ch["doc02"], ch) not in {x["file"] for x in thay_doi}:
            raise LoiChuanBi("phu luc phai duoc ghi trong tai lieu 02 (sau moc '## Phụ lục 1')")
        p_test = ket_qua_test_dat(ch) if ch["yeu_cau_test"] else None
    except LoiChuanBi as e:
        print(f"[LOI PHU LUC] {e}", file=sys.stderr)
        return 2
    d.mkdir(parents=True)
    for g, dich, s in sao:
        dich.parent.mkdir(parents=True, exist_ok=True)
        with open(g, "rb") as fi, open(dich, "xb") as fo:
            shutil.copyfileobj(fi, fo)
        if sha(dich) != s:
            print(f"[LOI PHU LUC] ban sao {dich} khac hash goc", file=sys.stderr)
            return 2
    pl = {"phien_ban": 1, "buoc": "R53B_LOCK_SMOKE (sua sau review)", "tao_luc": bay_gio(), "ly_do": LY_DO_PHU_LUC,
          "khoa_goc": {"thoi_diem_khoa": tg["thoi_diem_khoa"], "sha256sums": tg["sha256sums"],
                       "sha256sums_sha256": tg["sha256sums_sha256"], "khoa_danh_gia_sha256": tg["khoa_danh_gia_sha256"]},
          "thay_doi": thay_doi, "file_khoa_goc_khong_doi": len(dong) - len(thay_doi), "ma": hash_ma(ch),
          "moi_truong_doc_lap": moi_truong_doc_lap(ch), "lenh_doc_lap": lenh_doc_lap(ch),
          "test": ({"file": rel(p_test, ch), "sha256": sha(p_test), "dat": doc_json(p_test)["dat"],
                    "tong": doc_json(p_test)["tong"]} if p_test else None),
          "ghi_chu": ("khoa goc va bang chung smoke giu nguyen; khong commit theo yeu cau hien tai; quy tac, checkpoint, "
                      "du lieu, mau smoke va sau lenh day du khong doi")}
    p_pl = d / "khoa_phu_luc_1.json"
    ghi_json_moi(p_pl, pl)
    ds = [p_pl, p_sums, cb / "thoi_diem_khoa.json", cb / "khoa_danh_gia.json"]
    ds += [tu_rel(x["file"], ch) for x in thay_doi] + [dich for _, dich, _ in sao] + ([p_test] if p_test else [])
    with open(d / "SHA256SUMS", "x", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{sha(p)}  {rel(p, ch)}\n" for p in ds))
    ghi_json_moi(d / "thoi_diem_phu_luc_1.json", {
        "thoi_diem_phu_luc": bay_gio(), "sha256sums_sha256": sha(d / "SHA256SUMS"), "so_file": len(ds),
        "khoa_phu_luc_sha256": sha(p_pl), "khoa_goc_sha256sums_sha256": tg["sha256sums_sha256"],
        "ghi_chu": "sau smoke; khong commit; nguoi dung quyet commit"})
    print(f"[DAT] phu luc 1: {len(thay_doi)} file doi sau khoa ({', '.join(x['file'] for x in thay_doi)}); "
          f"SHA256SUMS phu luc {sha(d / 'SHA256SUMS')[:16]}")
    return 0


def buoc_phu_luc_2(ch) -> int:
    return buoc_phu_luc_noi(ch, 2)


def buoc_phu_luc_3(ch) -> int:
    return buoc_phu_luc_noi(ch, 3)


def buoc_phu_luc_noi(ch, k: int) -> int:
    """Phụ lục k (>= 2) nối tiếp các mức 1..k-1 (không ghi đè khóa gốc / mức trước). Hash 'trước' của mỗi file = mức gần
    nhất ghi nó (SHA256SUMS mức k-1, ..., nếu không thì khóa gốc). Tài liệu 02 chỉ nối thêm sau '## Phụ lục k'; bản trước
    của mã lấy từ snapshot review (ch['ban_goc_k'], kiểm đúng hash) rồi chép vào phu_luc_k/ban_khoa_truoc/."""
    cb = ch["chuan_bi"]
    ten, moc = PHU_LUC_DS[k]
    d = cb / ten
    if d.exists():
        print(f"[LOI] {d} da ton tai -- phu luc {k} da tao, khong ghi de", file=sys.stderr)
        return 1
    can_truoc = [PHU_LUC_DS[j][0] for j in range(1, k)]
    try:
        try:
            tg = doc_json(cb / "thoi_diem_khoa.json")
            p_sums = tu_rel(tg["sha256sums"], ch)
            if sha(p_sums) != tg["sha256sums_sha256"] or sha(cb / "khoa_danh_gia.json") != tg["khoa_danh_gia_sha256"]:
                raise LoiChuanBi("khoa goc (SHA256SUMS / khoa_danh_gia.json) da doi")
            dong = doc_sums(p_sums, tg["so_file"])
            cap = doc_cac_cap(ch, tg)
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise LoiChuanBi(f"khong doc duoc khoa goc / phu luc truoc: {e}") from None
        if [L["ten"] for L in cap] != can_truoc:
            raise LoiChuanBi(f"phu luc {k} can dung {k - 1} muc truoc ({', '.join(can_truoc)}), "
                             f"dang co {[L['ten'] for L in cap]}")
        for L in cap:                                  # bản trước giữ lại ở mọi mức vẫn đúng hash
            for x in L["pl"]["thay_doi"]:
                p = tu_rel(x["file"], ch)
                goc = (p.read_bytes()[:x["so_byte_tien_to"]] if x["cach_giu_ban_goc"] == "tien_to"
                       else tu_rel(x["ban_sao"], ch).read_bytes())
                if sha_byte(goc) != hash_truoc(x):
                    raise LoiChuanBi(f"{x['file']}: ban goc {L['ten'].replace('_', ' ')} giu lai khong con khop hash")
        L1 = cap[-1]                                   # mức liền trước
        ky_vong = {r: (s, "khoa_goc") for r, s in dong.items()}
        for L in cap:
            ky_vong.update({r: (s, L["ten"]) for r, s in L["sums"].items()})
        duoc_doi = {rel(p, ch): Path(p) for p in ch["doi_sau_khoa"]}
        thay_doi, sao = [], []
        for r, (s, muc) in ky_vong.items():
            p = tu_rel(r, ch)
            moi = sha(p)
            if moi == s:
                continue
            if r not in duoc_doi:
                raise LoiChuanBi(f"{r}: doi sau {muc} nhung khong thuoc tap duoc phep doi")
            x = {"file": r, "sha256_khoa_truoc": s, "muc_truoc": muc, "sha256_moi": moi}
            if p.resolve() == Path(ch["doc02"]).resolve():
                b = p.read_bytes()
                i = b.find(moc.encode("utf-8"))
                if i < 0 or sha_byte(b[:i]) != s:
                    raise LoiChuanBi(f"tai lieu 02: phan truoc moc phu luc {k} khong con trung ban {muc} (chi duoc noi them)")
                phan = b[i:].decode("utf-8")
                thieu = [c for c in CHUOI_BAT_BUOC_PHU_LUC[k](ch) if c not in phan]
                cho_trong = [c for c in DAU_HIEU_CHUA_XONG if c in phan]
                if thieu or cho_trong:
                    raise LoiChuanBi(f"phu luc {k} tai lieu 02 thieu {len(thieu)} chuoi bat buoc (vd {thieu[:1]}) / "
                                     f"con dau hieu chua xong {cho_trong}")
                x.update(cach_giu_ban_goc="tien_to", so_byte_tien_to=i, moc=moc.strip())
            else:
                g = next((Path(b) / r for b in ch[f"ban_goc_{k}"] if (Path(b) / r).is_file() and sha(Path(b) / r) == s),
                         None)
                if g is None:
                    raise LoiChuanBi(f"{r}: khong co ban {muc} dung hash {s[:16]} trong snapshot review")
                x.update(cach_giu_ban_goc="ban_sao", ban_sao=rel(d / "ban_khoa_truoc" / r, ch), nguon_ban_sao=str(g))
                sao.append((g, d / "ban_khoa_truoc" / r, s))
            thay_doi.append(x)
        if rel(ch["doc02"], ch) not in {x["file"] for x in thay_doi}:
            raise LoiChuanBi(f"phu luc {k} phai duoc ghi trong tai lieu 02 (sau moc '{moc.strip()}')")
        p_test = ket_qua_test_dat(ch) if ch["yeu_cau_test"] else None
    except LoiChuanBi as e:
        print(f"[LOI PHU LUC {k}] {e}", file=sys.stderr)
        return 2
    d.mkdir(parents=True)
    for g, dich, s in sao:
        dich.parent.mkdir(parents=True, exist_ok=True)
        with open(g, "rb") as fi, open(dich, "xb") as fo:
            shutil.copyfileobj(fi, fo)
        if sha(dich) != s:
            print(f"[LOI PHU LUC {k}] ban sao {dich} khac hash muc truoc", file=sys.stderr)
            return 2
    pl = {"phien_ban": 1, "muc": k, "buoc": LY_DO_PHU_LUC_DS[k][0], "tao_luc": bay_gio(),
          "ly_do": LY_DO_PHU_LUC_DS[k][1],
          "khoa_goc": {"thoi_diem_khoa": tg["thoi_diem_khoa"], "sha256sums": tg["sha256sums"],
                       "sha256sums_sha256": tg["sha256sums_sha256"], "khoa_danh_gia_sha256": tg["khoa_danh_gia_sha256"]},
          "khoa_truoc": {"phu_luc": L1["ten"], "sha256sums": rel(L1["dir"] / "SHA256SUMS", ch),
                         "sha256sums_sha256": L1["tp"]["sha256sums_sha256"],
                         "thoi_diem_phu_luc": L1["tp"]["thoi_diem_phu_luc"]},
          "thay_doi": thay_doi, "ma": hash_ma(ch), "ham_cham_lai": HAM_CHAM_LAI, "lenh_cham_lai_bdd": lenh_cham_lai_bdd(),
          "lenh_ghi_bang_chung_npz": {t: lenh_ghi_bang_chung_npz(t) for t in MODELS},
          "test": ({"file": rel(p_test, ch), "sha256": sha(p_test), "dat": doc_json(p_test)["dat"],
                    "tong": doc_json(p_test)["tong"]} if p_test else None),
          "ghi_chu": (f"khoa goc, {', '.join(x.replace('_', ' ') for x in can_truoc)} va bang chung smoke giu nguyen; "
                      "khong commit theo yeu cau hien tai; quy tac, checkpoint, du lieu, mau smoke va sau lenh day du "
                      "khong doi; ba lenh cham lai chi ban giao")}
    p_pl = d / f"khoa_{ten}.json"
    ghi_json_moi(p_pl, pl)
    ds = [p_pl, p_sums, cb / "thoi_diem_khoa.json", cb / "khoa_danh_gia.json"]
    for L in cap:
        ds += [L["dir"] / "SHA256SUMS", L["dir"] / f"thoi_diem_{L['ten']}.json", L["dir"] / f"khoa_{L['ten']}.json"]
    ds += [tu_rel(x["file"], ch) for x in thay_doi] + [dich for _, dich, _ in sao] + ([p_test] if p_test else [])
    with open(d / "SHA256SUMS", "x", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{sha(p)}  {rel(p, ch)}\n" for p in ds))
    ghi_json_moi(d / f"thoi_diem_{ten}.json", {
        "thoi_diem_phu_luc": bay_gio(), "sha256sums_sha256": sha(d / "SHA256SUMS"), "so_file": len(ds),
        "khoa_phu_luc_sha256": sha(p_pl), "khoa_goc_sha256sums_sha256": tg["sha256sums_sha256"],
        "khoa_truoc_sha256sums_sha256": L1["tp"]["sha256sums_sha256"],
        "ghi_chu": f"sau smoke va {', '.join(x.replace('_', ' ') for x in can_truoc)}; khong commit; nguoi dung quyet commit"})
    print(f"[DAT] phu luc {k}: {len(thay_doi)} file doi sau {L1['ten'].replace('_', ' ')} "
          f"({', '.join(x['file'] for x in thay_doi)}); SHA256SUMS phu luc {k} {sha(d / 'SHA256SUMS')[:16]}")
    return 0


# ----------------------------------------------------------------------------------------------- chấm lại BDD (PL2)
def bang_chung_npz(p: Path, npz: Path) -> str:
    """SHA256 trong file bằng chứng PowerShell (Format-List của Get-FileHash, lệnh `lenh_ghi_bang_chung_npz`); 'Path'
    phải trỏ đúng NPZ. Chịu BOM (PowerShell 5 -Encoding utf8)."""
    t = p.read_text(encoding="utf-8-sig").replace("﻿", "")
    h = re.findall(r"^\s*Hash\s*:\s*([0-9A-Fa-f]{64})\s*$", t, re.M)
    duong = re.findall(r"^\s*Path\s*:\s*(.+?)\s*$", t, re.M)
    if len(h) != 1 or len(duong) != 1:
        raise LoiChuanBi(f"{p.name}: can dung mot dong Hash va mot dong Path (co {len(h)} / {len(duong)})")
    if Path(duong[0]).resolve() != npz.resolve():
        raise LoiChuanBi(f"{p.name}: Path {duong[0]} khong phai {npz}")
    return h[0].lower()


def cwd_giam_sat(tom: dict, tt: dict) -> Path:
    """Thư mục làm việc của bộ giám sát (= của tiến trình adapter) lấy từ trang_thai.json; lệnh phải trùng tom_tat.json.
    Không suy ra cwd từ repo hiện tại: đường dẫn tương đối chỉ có nghĩa theo cwd thật của lượt chạy."""
    if [str(x) for x in tt.get("lenh") or []] != [str(x) for x in tom["lenh"]]:
        raise LoiChuanBi("lenh giam sat: trang_thai.json va tom_tat.json ghi lenh khac nhau")
    cwd = tt.get("cwd")
    if not cwd or not Path(cwd).is_absolute():
        raise LoiChuanBi(f"lenh giam sat: trang_thai.json khong co cwd tuyet doi ({cwd!r})")
    return Path(cwd)


def kiem_nguon_bdd(tom: dict, ten: str, vao: Path, khoa: dict, ch, cwd: Path) -> dict:
    """Lệnh trong tom_tat.json của bộ giám sát = lệnh bàn giao BDD của `ten` (adapter gốc, checkpoint/hash đã khóa,
    batch 16, device 0, conf 0,25, giới hạn 0, --out = thư mục đang chấm, dữ liệu = cổng đầy đủ). Adapter so theo đường
    dẫn sau resolve (tương đối -> theo `cwd` của bộ giám sát) với file adapter BDD trong repo, không chỉ theo tên file:
    file cùng tên ở bản checkout / thư mục khác -> lỗi. Nội dung adapter trong repo do cổng khóa kiểm (SHA256SUMS)."""
    lenh = [str(x) for x in tom["lenh"]]
    tuyet = (lambda s: (Path(s) if Path(s).is_absolute() else Path(cwd) / s).resolve())
    p_adapter = (Path(ch["repo"]) / TAP["bdd_val"]["adapter"]).resolve()
    i = next((k for k, x in enumerate(lenh) if Path(x.replace("\\", "/")).name == p_adapter.name), None)
    if i is None:
        raise LoiChuanBi("lenh giam sat khong chay adapter BDD goc")
    if tuyet(lenh[i]) != p_adapter:
        raise LoiChuanBi(f"lenh giam sat chay {tuyet(lenh[i])} (cwd {cwd}), khong phai adapter BDD trong repo {p_adapter}")
    con = lenh[i + 1:]
    if len(con) % 2 or any(not k.startswith("--") for k in con[::2]) or len(set(con[::2])) != len(con[::2]):
        raise LoiChuanBi(f"doi so lenh giam sat khong theo cap --khoa gia_tri: {con[:6]}")
    a = dict(zip(con[::2], con[1::2]))
    can = dict(zip(doi_so_day_du("bdd_val", ten)[::2], doi_so_day_du("bdd_val", ten)[1::2]))
    la = set(a) - set(can) - set(DOI_SO_DU_LIEU_BDD)
    if la or set(can) - set(a):
        raise LoiChuanBi(f"lenh giam sat: doi so la {sorted(la)} / thieu {sorted(set(can) - set(a))}")
    m = khoa["mo_hinh"][ten]
    tm, _, w = a["--model"].partition("=")
    ts, _, s = a["--sha256"].partition("=")
    sai = [k for k in ("--batch", "--device", "--conf-van-hanh", "--gioi-han-anh") if a[k] != can[k]]
    if tm != ten or tuyet(w) != Path(ch["models"][ten][0]).resolve():
        sai.append("--model")
    if ts != ten or s != m["sha256"] or s != ch["models"][ten][1]:
        sai.append("--sha256")
    if tuyet(a["--out"]) != vao.resolve():
        sai.append("--out")
    for k, khoa_dl in DOI_SO_DU_LIEU_BDD.items():
        p = tuyet(a[k]) if k in a else bv.p_abs(bv.MAC_DINH[khoa_dl]).resolve()
        if p != Path(ch["du_lieu"]["bdd_val"][khoa_dl]).resolve():
            sai.append(k)
    if sai:
        raise LoiChuanBi(f"lenh giam sat khac lenh ban giao o {sai}")
    return {"lenh": lenh, "checkpoint_doi_so": w, "exit_code": tom.get("exit_code"), "cwd": str(cwd),
            "adapter": rel(p_adapter, ch)}


def bam_dau_vao(ds) -> dict:
    ra = {}
    for g in ds:
        g = Path(g)
        for p in sorted(g.rglob("*") if g.is_dir() else [g]):
            if p.is_file():
                ra[p.as_posix()] = sha(p)
    return ra


TRUONG_KHONG_TU_NPZ = ("anh_xa_lop", "thoi_gian_s")   # adapter ghi lúc suy luận; NPZ không chứa -> không đối chiếu


def doi_chieu_goc(kq: dict, goc: dict, ten: str) -> list:
    """Trường khác nhau giữa kết quả chấm lại và ket_qua.json gốc của cùng thư mục (cùng model; không so detector khác)."""
    khac = [k for k in ("quy_tac", "quy_tac_gt", "dataset") if json.dumps(ev._json_safe(kq[k]), sort_keys=True)
            != json.dumps(ev._json_safe(goc.get(k)), sort_keys=True)]
    m, g = kq["mo_hinh"][ten], (goc.get("mo_hinh") or {}).get(ten, {})
    for k in sorted(set(m) | set(g)):
        if k in TRUONG_KHONG_TU_NPZ or k in ("npz", "npz_sha256"):
            continue
        if json.dumps(ev._json_safe(m.get(k)), sort_keys=True) != json.dumps(ev._json_safe(g.get(k)), sort_keys=True):
            khac.append(f"mo_hinh.{k}")
    return khac


def buoc_cham_lai_bdd(ten: str, lan: int, ch, argv) -> int:
    vao = Path(ch["full"]) / "bdd_val" / ten
    out = vao.with_name(f"{ten}__cham_lai" if lan == 1 else f"{ten}__cham_lai_lan{lan}")
    if out.exists():
        print(f"[LOI] {out} da ton tai -- khong ghi de; chay lai dung --lan moi", file=sys.stderr)
        return 1
    out.mkdir(parents=True)
    t0 = bay_gio()
    goc_out, goc_err = sys.stdout, sys.stderr
    ma = 3
    with open(out / "log.txt", "w", encoding="utf-8", newline="\n") as f:
        sys.stdout, sys.stderr = Tee(goc_out, f), Tee(goc_err, f)
        try:
            print(f"# {t0} lenh: {' '.join([sys.executable, *argv])}")
            ma = _cham_lai_bdd(ten, vao, out, ch, argv)
        except Exception:
            traceback.print_exc()
            ma = 3
        finally:
            print(f"# {bay_gio()} exit={ma}")
            sys.stdout, sys.stderr = goc_out, goc_err
    ghi_json_moi(out / "exit.json", {"exit": ma, "bat_dau": t0, "ket_thuc": bay_gio(), "lenh": [sys.executable, *argv]})
    return ma


def _cham_lai_bdd(ten: str, vao: Path, out: Path, ch, argv) -> int:
    """Chỉ đọc: cổng khóa -> nguồn + hash NPZ -> cổng dữ liệu đầy đủ của adapter -> NPZ (đúng danh sách + thứ tự ảnh)
    -> scorer cũ. Không nạp model; hàm dự đoán của hai adapter bị chặn trong lúc chạy."""
    npz = vao / f"du_doan_{ten}.npz"
    p_bc = Path(ch["full"]) / "_hau_kiem" / f"bdd_val__{ten}__npz.txt"
    p_tom = Path(ch["full"]) / "_giam_sat" / f"bdd_val__{ten}" / "tom_tat.json"
    p_goc = vao / "ket_qua.json"
    try:
        khoa, tg = kiem_khoa(ch)
        if ten not in khoa["mo_hinh"] or PHU_LUC_DS[2][0] not in tg["_cap_phu_luc"]:
            raise LoiChuanBi(f"{ten} khong co trong khoa / chua co phu luc 2 (buoc cham lai chua duoc khoa)")
        print(f"[khoa] dat theo chuoi {['khoa_goc', *tg['_cap_phu_luc']]}", flush=True)
        for p in (vao, npz, p_bc, p_tom):
            if not p.exists():
                raise LoiChuanBi(f"thieu {p}")
        truoc = bam_dau_vao([vao, p_bc, p_tom.parent])
        s_npz = sha(npz)
        s_bc = bang_chung_npz(p_bc, npz)
        if s_npz != s_bc:
            raise LoiChuanBi(f"NPZ da doi: sha256 {s_npz[:16]} khac bang chung {s_bc[:16]}")
        tom = doc_json(p_tom)
        nguon = kiem_nguon_bdd(tom, ten, vao, khoa, ch, cwd_giam_sat(tom, doc_json(p_tom.with_name("trang_thai.json"))))
        goc = doc_json(p_goc) if p_goc.exists() else None
        if goc is not None:
            mg = (goc.get("mo_hinh") or {}).get(ten) or {}
            if set(goc.get("mo_hinh") or {}) != {ten} or mg.get("sha256") != khoa["mo_hinh"][ten]["sha256"] \
                    or (goc.get("dataset") or {}).get("gioi_han_anh") != 0:
                raise LoiChuanBi("ket_qua.json goc khong cung model / checkpoint / gioi han 0")
        print(f"[nguon] NPZ {s_npz[:16]} = bang chung; lenh giam sat = lenh ban giao; exit goc {nguon['exit_code']}; "
              f"ket_qua.json goc {'co' if goc is not None else 'khong co'}", flush=True)
    except (LoiChuanBi, FileNotFoundError, ValueError, KeyError) as e:
        print(f"[LOI NGUON] {e}", file=sys.stderr)
        return 2
    d = ch["du_lieu"]["bdd_val"]
    t1 = _dt.datetime.now()
    try:
        danh_tinh, anh, gt = bv.kiem_va_doc_gt(d["nhan"], d["anh"], d["manifest"], d["hash_anh"])
    except LOI_DU_LIEU as e:
        print(f"[LOI DATASET] {e}", file=sys.stderr)
        return 2
    t_kiem = (_dt.datetime.now() - t1).total_seconds()
    dtk = danh_tinh_khoa("bdd_val", danh_tinh)
    khac = sorted(k for k, v in dtk.items() if json.dumps(v, sort_keys=True) !=
                  json.dumps(khoa["du_lieu"]["bdd_val"]["danh_tinh_khoa"].get(k), sort_keys=True))
    if khac:
        print(f"[LOI DATASET] danh tinh du lieu khac khoa: {khac}", file=sys.stderr)
        return 2
    try:
        ten_npz, dd = dg.doc_du_doan(npz, len(anh))
    except (dg.LoiDuLieu, ValueError, KeyError) as e:
        print(f"[LOI NPZ] {e}", file=sys.stderr)
        return 2
    ten_anh = [p.name for p in anh]
    if len(ten_npz) != len(anh) or [str(x) for x in ten_npz] != ten_anh:
        lech = next((i for i, (a, b) in enumerate(zip(ten_npz, ten_anh)) if str(a) != b), min(len(ten_npz), len(anh)))
        print(f"[LOI NPZ] danh sach / thu tu anh NPZ khac adapter: {len(ten_npz)} anh NPZ, {len(anh)} anh adapter, "
              f"lech dau tien o chi so {lech}", file=sys.stderr)
        return 2
    print(f"[du lieu] {len(anh)} anh, danh tinh = khoa, kiem {t_kiem:.0f} s; NPZ dung danh sach + thu tu", flush=True)
    qt = bv.quy_tac_hieu_luc(CONF_VH)
    if json.dumps(qt["du_doan"], sort_keys=True) != json.dumps(khoa["quy_tac"]["du_doan"], sort_keys=True) or \
            qt["van_hanh"] != khoa["quy_tac"]["van_hanh"]:
        print("[LOI KHOA] quy tac du doan / van hanh hien tai khac khoa", file=sys.stderr)
        return 2
    goi_suy_luan = []
    chan = (dg.du_doan_ultralytics, bv.du_doan_ultralytics)

    def cam(*a, **k):
        goi_suy_luan.append(1)
        raise LoiChuanBi("cham lai khong duoc suy luan")
    dg.du_doan_ultralytics = bv.du_doan_ultralytics = cam
    try:
        r = ev.danh_gia(gt, dd, qt["van_hanh"]["conf"])
        cd = bv.chan_doan(gt, dd, qt["van_hanh"]["conf"])
    finally:
        dg.du_doan_ultralytics, bv.du_doan_ultralytics = chan
    if r.get("van_hanh") != qt["van_hanh"] or goi_suy_luan:
        print(f"[LOI] nguong van hanh scorer {r.get('van_hanh')} khac {qt['van_hanh']} / goi suy luan", file=sys.stderr)
        return 3
    kq = {"tao_luc": bay_gio(), "quy_tac": qt, "quy_tac_gt": bv.QUY_TAC_GT,
          "dataset": {"ten": "bdd100k_mot_val_cocovid", "n_anh": len(anh), "gioi_han_anh": 0,
                      "dataset_da_xac_minh": danh_tinh},
          "lenh": [sys.executable, *argv],
          "mo_hinh": {ten: {"checkpoint": nguon["checkpoint_doi_so"], "sha256": khoa["mo_hinh"][ten]["sha256"],
                            "so_anh_da_du_doan": len(dd), "chan_doan": cd, **r,
                            "npz": rel(npz, ch), "npz_sha256": s_npz}}}
    sau = bam_dau_vao([vao, p_bc, p_tom.parent])
    kq["cham_lai"] = {
        "canh_bao": "CHAM LAI tu NPZ da luu, khong suy luan; khong phai bang so sanh detector",
        "thu_muc_vao": rel(vao, ch), "npz_sha256": s_npz, "bang_chung_npz": rel(p_bc, ch),
        "giam_sat_tom_tat": rel(p_tom, ch), "lenh_goc": nguon["lenh"], "exit_goc": nguon["exit_code"],
        "adapter_goc": nguon["adapter"], "cwd_giam_sat": nguon["cwd"],
        "ham": HAM_CHAM_LAI, "goi_suy_luan": len(goi_suy_luan),
        "module_da_nap": {m: m in sys.modules for m in ("ultralytics", "torch")},
        "khoa": {"chuoi": ["khoa_goc", *tg["_cap_phu_luc"]], "sha256sums_sha256": tg["sha256sums_sha256"]},
        "dau_vao_sha256": truoc, "dau_vao_khong_doi": truoc == sau,
        "truong_khong_tu_npz": list(TRUONG_KHONG_TU_NPZ), "thoi_gian_kiem_du_lieu_s": round(t_kiem, 1)}
    if goc is not None:
        kq["cham_lai"]["doi_chieu_ket_qua_goc"] = {"file": rel(p_goc, ch), "sha256": sha(p_goc),
                                                   "truong_khac": doi_chieu_goc(kq, goc, ten)}
    ghi_json_moi(out / "ket_qua.json", kq)
    rows = [{"mo_hinh": ten, "lop": lop, "n_gt": v["n_gt"], "AP50": v["AP50"], "AP50_95": v["AP50_95"],
             "precision_vh": v["van_hanh"]["precision"], "recall_vh": v["van_hanh"]["recall"]} for lop, v in r["theo_lop"].items()]
    with open(out / "theo_lop.csv", "x", newline="", encoding="utf-8") as f:   # cùng cách ghi với adapter BDD
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if truoc != sau:
        print("[LOI] dau vao doi trong luc cham lai", file=sys.stderr)
        return 3
    if goc is not None and kq["cham_lai"]["doi_chieu_ket_qua_goc"]["truong_khac"]:
        print(f"[LOI] cham lai khac ket_qua.json goc o {kq['cham_lai']['doi_chieu_ket_qua_goc']['truong_khac']}",
              file=sys.stderr)
        return 3
    print(f"[DAT] cham lai {ten}: {len(dd)} anh tu NPZ {s_npz[:16]}, mAP50 {r['macro']['mAP50']:.4f}, "
          f"khong suy luan, dau vao khong doi", flush=True)
    return 0


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
    ap.add_argument("--buoc", required=True, choices=["tien-kiem", "lenh-day-du", "khoa", "kiem-khoa", "smoke", "phu-luc",
                                                      "phu-luc-2", "phu-luc-3", "cham-lai-bdd"])
    ap.add_argument("--tap", choices=list(TAP))
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--lan", type=int, default=1, help="smoke / cham-lai-bdd: lan chay lai vao thu muc moi (__lan<N>)")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    ch = ch or cau_hinh_mac_dinh()
    can_tap = a.buoc in ("tien-kiem", "smoke")
    can_model = a.buoc in ("smoke", "cham-lai-bdd")
    if bool(a.tap) != can_tap or bool(a.model) != can_model or a.lan < 1 or (a.lan != 1 and not can_model):
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
              f"{tg['thoi_diem_khoa']}" + (f"; chuoi phu luc dat ({', '.join(tg['_cap_phu_luc'])}), file doi sau khoa: "
                                           f"{', '.join(tg['_phu_luc'])}" if tg["_phu_luc"] else "; khong co phu luc"))
        return 0
    if a.buoc == "phu-luc":
        return buoc_phu_luc(ch)
    if a.buoc == "phu-luc-2":
        return buoc_phu_luc_2(ch)
    if a.buoc == "phu-luc-3":
        return buoc_phu_luc_3(ch)
    if a.buoc == "cham-lai-bdd":
        return buoc_cham_lai_bdd(a.model, a.lan, ch, argv)
    return buoc_smoke(a.tap, a.model, a.lan, ch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
