"""R53A — nghiệm thu bàn giao lượt train mixed5 (YOLOv8n / YOLO11n) từ bằng chứng đã có.

Chỉ ĐỌC bằng chứng train: nhật ký lượt, JSON runner, thư mục giám sát, CSV nvidia-smi, results.csv, checkpoint.
Không train, không inference, không nạp model bằng torch/ultralytics. Metadata checkpoint đọc bằng bộ unpickle
hạn chế (chỉ dựng dict/list/số; mọi lớp khác thay bằng đối tượng giả, không import module nào).

Đối chiếu kế hoạch handoff/logs/review52/02_ke_hoach_luot_dai.md mục 8 (điều kiện kết thúc thành công) và mục 10
(nội dung bàn giao). Mỗi lệnh xử lý một model, ghi ban_giao_<model>.json, rồi dựng lại các bảng gộp
(dev_tung_epoch.csv, checkpoint_sha256.csv, tai_nguyen.csv, nguon_sha256.csv) từ mọi ban_giao_*.json đang có.
--out chỉ nhận ban_giao/ hoặc tests_handoff/ (kiểm trước khi đọc/ghi). Đầu ra cũ được chép sang
<out>/_ban_truoc/<thoi diem>/ trước khi ghi lại, không xóa.

  python src/r53_detector/ban_giao_train.py --model yolov8n
  python src/r53_detector/ban_giao_train.py --model yolo11n

Exit 0 = đạt mọi kiểm; 1 = có lỗi (xem trường "loi" trong JSON); 2 = tham số sai.
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
import codecs  # noqa: E402
import copyreg  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_MAC_DINH = REPO / "results" / "phase2" / "r53_detector" / "ban_giao"
# Chỉ hai vùng ghi của R53A; chuan_bi/, smoke/, full/ thuộc bước khác -> từ chối trước khi tổng hợp hoặc ghi.
VUNG_OUT_CHO_PHEP = (OUT_MAC_DINH, REPO / "results" / "phase2" / "r53_detector" / "tests_handoff")
THU_MUC_BAN_TRUOC = "_ban_truoc"  # bản cũ của đầu ra được chép vào đây trước khi ghi lại, không xóa
MODELS = ("yolov8n", "yolo11n")
TONG_EPOCH = 15
# Kỳ vọng do plan R53 nêu (lấy từ JSON runner + mục 0); script vẫn tự tính lại và báo lệch.
KY_VONG = {
    "yolov8n": {"best_epoch": 15, "best_sha256": "542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694"},
    "yolo11n": {"best_epoch": 13, "best_sha256": "9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb"},
}
NGUONG_KHOANG_TRONG_S = 60.0  # khoảng giữa hai mẫu (chu kỳ 1 s) lớn hơn ngưỡng này = mất mẫu (vd máy ngủ)
DUNG_SAI = 1.01e-5  # results.csv / nhật ký làm tròn 5 chữ số thập phân
KHOA_METRIC = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
               "val/box_loss", "val/cls_loss", "val/dfl_loss"]
TEN_COT_METRIC = {"metrics/precision(B)": "precision", "metrics/recall(B)": "recall", "metrics/mAP50(B)": "map50",
                  "metrics/mAP50-95(B)": "map50_95", "val/box_loss": "val_box_loss", "val/cls_loss": "val_cls_loss",
                  "val/dfl_loss": "val_dfl_loss"}
GHI_CHU_FINAL_EVAL = ("ultralytics final_eval: validate best.pt (FP16 tu dia) sau vong lap. Khoang thoi gian ghi o day "
                      "tinh tu luc het train epoch cuoi, GOM luu checkpoint, don dep cuoi epoch va validate best.pt -> "
                      "KHONG phai thoi gian thuan validation.")


# ----------------------------------------------------------------------------------------------- tiện ích
class SoHash:
    """Băm sha256 một lần mỗi file; giữ danh sách nguồn đã đọc để xuất bảng hash nguồn."""

    def __init__(self, goc):
        self.goc = Path(goc)
        self._bo_nho = {}
        self.nguon = OrderedDict()

    def rel(self, p):
        p = Path(p)
        try:
            return p.resolve().relative_to(self.goc.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def bam(self, p, vai_tro):
        p = Path(p)
        k = str(p.resolve())
        if k not in self._bo_nho:
            h = hashlib.sha256()
            n = 0
            with open(p, "rb") as f:
                for khoi in iter(lambda: f.read(1 << 20), b""):
                    h.update(khoi)
                    n += len(khoi)
            self._bo_nho[k] = (h.hexdigest(), n)
        sha, n = self._bo_nho[k]
        r = self.rel(p)
        if r not in self.nguon:
            self.nguon[r] = {"duong_dan": r, "bytes": n, "sha256": sha, "vai_tro": vai_tro}
        return sha, n


def _iso(t):
    if t is None:
        return None
    return _dt.datetime.fromtimestamp(t, _dt.timezone(_dt.timedelta(hours=7))).isoformat(timespec="seconds")


def _t_iso(s):
    """ISO có múi giờ -> epoch giây; None nếu thiếu/không đọc được."""
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _so(x):
    """float hữu hạn hoặc None (không bao giờ thay dữ liệu thiếu bằng 0)."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _tk(ds):
    ds = [v for v in ds if v is not None]
    if not ds:
        return {"n": 0, "min": None, "trung_vi": None, "max": None}
    return {"n": len(ds), "min": min(ds), "trung_vi": statistics.median(ds), "max": max(ds)}


def _gan(a, b, tol=DUNG_SAI):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def doc_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- đọc metadata checkpoint (an toàn)
class _Gia:
    """Đối tượng giả thay cho mọi lớp/hàm không nằm trong danh sách an toàn (torch, ultralytics, numpy...)."""

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        self.doi_so = a

    def __setstate__(self, state):
        self.trang_thai = state

    def __setitem__(self, k, v):
        self.__dict__.setdefault("muc", {})[k] = v

    def append(self, v):
        self.__dict__.setdefault("ds", []).append(v)

    def extend(self, vs):
        self.__dict__.setdefault("ds", []).extend(vs)


_AN_TOAN = {("collections", "OrderedDict"): OrderedDict, ("builtins", "set"): set,
            ("builtins", "frozenset"): frozenset, ("builtins", "slice"): slice, ("builtins", "complex"): complex,
            ("builtins", "bytearray"): bytearray, ("_codecs", "encode"): codecs.encode,
            ("copyreg", "_reconstructor"): copyreg._reconstructor, ("builtins", "object"): object}


class _UnpickleHanChe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _AN_TOAN:
            return _AN_TOAN[(module, name)]
        return type(f"Gia_{module.replace('.', '_')}_{name}", (_Gia,), {})

    def persistent_load(self, pid):
        return _Gia(pid)


def _thuan(x, sau=0):
    """Chuyển kết quả unpickle sang JSON được; đối tượng giả -> None."""
    if sau > 6:
        return None
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _thuan(v, sau + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_thuan(v, sau + 1) for v in x]
    return None


def doc_metadata_checkpoint(p):
    """Đọc dict cấp cao của checkpoint torch (zip) mà không import torch. Trả dict tóm tắt hoặc {'doc_duoc': False}."""
    try:
        with zipfile.ZipFile(p) as zf:
            ten = [n for n in zf.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
            if len(ten) != 1:
                return {"doc_duoc": False, "ly_do": f"khong thay dung 1 data.pkl ({len(ten)})"}
            du_lieu = zf.read(ten[0])
        ck = _UnpickleHanChe(io.BytesIO(du_lieu)).load()
    except Exception as e:  # noqa: BLE001 — ghi lại lý do, không làm hỏng cả báo cáo
        return {"doc_duoc": False, "ly_do": f"{type(e).__name__}: {e}"}
    if not isinstance(ck, dict):
        return {"doc_duoc": False, "ly_do": f"goc khong phai dict ({type(ck).__name__})"}
    tr = ck.get("train_results")
    ep_tr = None
    if isinstance(tr, dict):
        for k, v in tr.items():
            if str(k).strip() == "epoch" and isinstance(v, (list, tuple)):
                ep_tr = [_thuan(e) for e in v]
    tm = ck.get("train_metrics")
    names = None
    m = ck.get("model")
    st = getattr(m, "trang_thai", None)
    if isinstance(st, dict) and isinstance(st.get("names"), dict):
        names = {str(k): v for k, v in st["names"].items()}
    return {
        "doc_duoc": True,
        "khoa": sorted(str(k) for k in ck.keys()),
        "epoch": _thuan(ck.get("epoch")),
        "best_fitness": _thuan(ck.get("best_fitness")),
        "updates": _thuan(ck.get("updates")),
        "co_optimizer": ck.get("optimizer") is not None,
        "co_ema": ck.get("ema") is not None,
        "date": _thuan(ck.get("date")),
        "version": _thuan(ck.get("version")),
        "train_metrics": _thuan(tm) if isinstance(tm, dict) else None,
        "train_results_epoch": ep_tr,
        "train_args": _thuan(ck.get("train_args")) if isinstance(ck.get("train_args"), dict) else None,
        "names": names,
    }


# ------------------------------------------------------------------------------------------ đọc nguồn
def duong_dan_model(goc, model):
    ten = f"train_mixed5_v1_{model}_day_du"
    kq = Path(goc) / "results" / "phase2" / "mixed5" / "detector_train"
    return {"ten": ten, "run_dir": Path(goc) / "models" / "finetune_mixed5" / ten, "kq_dir": kq,
            "giam_sat": kq / "giam_sat"}


def duong_dan_lan(dd, model, lan):
    goc_ten = dd["ten"] if lan == 0 else f"{dd['ten']}__tiep_tuc_{lan}"
    gs = dd["giam_sat"] / (f"{model}_lan0" if lan == 0 else f"{model}_tiep_tuc_{lan}")
    return {"runner_json": dd["kq_dir"] / f"{goc_ten}.json",
            "nvsmi_csv": dd["kq_dir"] / f"{goc_ten}_tai_nguyen_nvidia_smi.csv",
            "giam_sat": gs}


def doc_nhat_ky(p):
    su_kien, loi = [], []
    for i, dong in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
        if not dong.strip():
            continue
        try:
            su_kien.append(json.loads(dong))
        except json.JSONDecodeError as e:
            loi.append(f"nhat_ky dong {i} khong phai JSON: {e}")
    return su_kien, loi


def doc_results_csv(p):
    with open(p, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        hang = []
        for d in r:
            hang.append({(k or "").strip(): (v or "").strip() for k, v in d.items()})
    return hang


def phan_tich_ram(p):
    """Tóm tắt ram_mau_tho.csv của bộ giám sát (RSS PID chính, RSS tổng gồm con, RAM toàn máy còn trống)."""
    t, rc, rt, sc, rm, pids = [], [], [], [], [], set()
    so_dong = so_hong = 0
    with open(p, "r", encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f):
            so_dong += 1
            tu = _so(d.get("t_unix"))
            vals = [_so(d.get(k)) for k in ("rss_chinh_mib", "rss_tong_mib", "so_con", "ram_he_thong_con_trong_mib")]
            if tu is None or any(v is None for v in vals):
                so_hong += 1
            if tu is not None:
                t.append(tu)
            rc.append(vals[0]); rt.append(vals[1]); sc.append(vals[2]); rm.append(vals[3])
            if d.get("pid_chinh"):
                pids.add(d["pid_chinh"])
    return {"so_mau": so_dong, "so_mau_thieu_truong": so_hong, "pid_chinh": sorted(pids),
            "tu_t": min(t) if t else None, "den_t": max(t) if t else None,
            "rss_chinh_mib": _tk(rc), "rss_tong_mib": _tk(rt), "so_con": _tk(sc),
            "ram_may_con_trong_mib": _tk(rm), "khoang_trong": _khoang_trong(t)}


def phan_tich_nvsmi(p):
    t, mem, util, ram = [], [], [], []
    so_dong = 0
    with open(p, "r", encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f):
            so_dong += 1
            tu = _so(d.get("t_unix"))
            if tu is not None:
                t.append(tu)
            mem.append(_so(d.get("nvidia_smi_memory_used_mib_ca_gpu")))
            util.append(_so(d.get("gpu_util_pct")))
            ram.append(_so(d.get("ram_he_thong_con_trong_mib")))
    return {"so_mau": so_dong, "tu_t": min(t) if t else None, "den_t": max(t) if t else None,
            "vram_ca_gpu_mib": _tk(mem), "gpu_util_pct": _tk(util), "ram_may_con_trong_mib": _tk(ram),
            "khoang_trong": _khoang_trong(t)}


def _khoang_trong(t):
    t = sorted(t)
    if len(t) < 2:
        return {"nguong_s": NGUONG_KHOANG_TRONG_S, "so_khoang": None, "tong_s": None, "max_s": None, "chi_tiet": [],
                "ghi_chu": "duoi 2 moc thoi gian -> khong xac dinh duoc khoang mat mau"}
    gap = [(a, b) for a, b in zip(t, t[1:]) if b - a > NGUONG_KHOANG_TRONG_S]
    return {"nguong_s": NGUONG_KHOANG_TRONG_S, "so_khoang": len(gap),
            "tong_s": round(sum(b - a for a, b in gap), 1) if gap else 0.0,
            "max_s": round(max((b - a for a, b in gap), default=0.0), 1) if gap else None,
            "chi_tiet": [{"tu": _iso(a), "den": _iso(b), "s": round(b - a, 1)} for a, b in gap[:20]]}


def doc_dong_trang_thai_log(p, model):
    """Tìm dòng tổng kết runner ([DAT]/[BI NGAT]/[LOI]...) và dòng '# exit=' cuối log của bộ giám sát."""
    mau = re.compile(r"\[([A-Z ]+)\] day-du " + re.escape(model) + r": (.*)$")
    tong_ket, exit_log = [], None
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for dong in f:
            dong = dong.rstrip("\r\n")
            m = mau.search(dong)
            if m:
                truong = dict(re.findall(r"(loi_callback|loss_huu_han)=(\S+)", m.group(2)))
                tong_ket.append({"nhan": m.group(1), "dong": dong.strip()[:400],
                                 "loi_None": m.group(2).startswith("loi=None "),
                                 "loi_callback": truong.get("loi_callback"),
                                 "loss_huu_han": truong.get("loss_huu_han")})
            if dong.startswith("# exit="):
                exit_log = dong[len("# exit="):].strip()
    return {"tong_ket": tong_ket, "exit_cuoi_log": exit_log}


# ------------------------------------------------------------------------------------------ tổng hợp
def tong_hop(goc, model, ky_vong=None, tong_epoch=TONG_EPOCH, ma_nguon_sha=None):
    """Trả (bao_cao dict, so_hash). Không ghi file."""
    goc = Path(goc)
    ky_vong = KY_VONG.get(model, {}) if ky_vong is None else ky_vong
    sh = SoHash(goc)
    dd = duong_dan_model(goc, model)
    loi, canh_bao, thieu = [], [], []

    def L(s):
        loi.append(s)

    def C(s):
        canh_bao.append(s)

    bc = OrderedDict(loai="R53A ban giao train mixed5", model=model, ten_luot=dd["ten"],
                     tao_luc=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                     pham_vi=("chi doc bang chung train; khong train, khong inference, khong nap model bang "
                              "torch/ultralytics; metadata checkpoint doc bang unpickle han che"),
                     ma_nguon_sha256=ma_nguon_sha)

    # ---- nhật ký lượt
    p_nk = dd["run_dir"] / "nhat_ky_luot.jsonl"
    if not p_nk.is_file():
        L(f"thieu nhat ky luot {sh.rel(p_nk)}")
        bc.update(dat=False, loi=loi, canh_bao=canh_bao, thieu_du_lieu=thieu)
        return bc, sh
    sh.bam(p_nk, "nhat_ky_luot")
    su_kien, l_nk = doc_nhat_ky(p_nk)
    loi.extend(l_nk)
    lans = OrderedDict()
    for ev in su_kien:
        loai, lan = ev.get("loai"), ev.get("lan")
        if loai == "bat_dau":
            if lan in lans:
                L(f"nhat ky: bat_dau lan {lan} xuat hien hai lan")
            lans[lan] = {"bat_dau": ev, "epoch": [], "ket_thuc": None}
        elif loai in ("epoch_da_luu", "ket_thuc"):
            if lan not in lans:
                L(f"nhat ky: su kien {loai} cua lan {lan} truoc bat_dau")
                continue
            if loai == "epoch_da_luu":
                lans[lan]["epoch"].append(ev)
            else:
                if lans[lan]["ket_thuc"] is not None:
                    L(f"nhat ky: lan {lan} co hai ket_thuc")
                lans[lan]["ket_thuc"] = ev
        else:
            C(f"nhat ky: loai su kien la '{loai}' (bo qua)")
    ds_lan = list(lans.keys())
    if ds_lan != list(range(len(ds_lan))):
        L(f"nhat ky: so lan khong lien tuc tu 0: {ds_lan}")
    if not ds_lan:
        L("nhat ky khong co lan chay nao")
        bc.update(dat=False, loi=loi, canh_bao=canh_bao, thieu_du_lieu=thieu)
        return bc, sh
    lan_cuoi = ds_lan[-1]

    # ---- quỹ đạo epoch qua resume
    quy_dao = [(lan, ev.get("epoch")) for lan in ds_lan for ev in lans[lan]["epoch"]]
    ds_epoch = [e for _, e in quy_dao]
    trung = sorted(e for e, c in Counter(ds_epoch).items() if c > 1)
    if trung:
        chi_tiet = {e: [lan for lan, x in quy_dao if x == e] for e in trung}
        L(f"trung/xung dot epoch sau resume: {chi_tiet} (epoch -> cac lan ghi)")
    if ds_epoch != list(range(1, len(ds_epoch) + 1)):
        L(f"quy dao epoch khong tang lien tuc 1..n: {ds_epoch}")
    thieu_epoch = sorted(set(range(1, tong_epoch + 1)) - set(ds_epoch))
    ev_cuoi = lans[lan_cuoi]["epoch"][-1] if lans[lan_cuoi]["epoch"] else None
    dung_som_hop_le = bool(ev_cuoi and ev_cuoi.get("dung") and len(set(ds_epoch)) < tong_epoch
                           and ds_epoch == list(range(1, len(ds_epoch) + 1)))
    du_epoch = (sorted(ds_epoch) == list(range(1, tong_epoch + 1)) and not trung)
    if thieu_epoch and not dung_som_hop_le:
        L(f"thieu epoch {thieu_epoch} (khong co dung som hop le)")
    if not du_epoch:
        L(f"khong dung {tong_epoch} epoch theo tieu chi R53A (co {len(set(ds_epoch))} epoch khac nhau,"
          f" {len(ds_epoch)} su kien)")
    epoch_ev = {}
    for lan, ev in ((lan, ev) for lan in ds_lan for ev in lans[lan]["epoch"]):
        epoch_ev.setdefault(ev.get("epoch"), (lan, ev))

    # ---- từng lần chạy
    lich_su, dev_rows, ck_rows = [], [], []
    so_da_luu_truoc = 0
    ban_sao_truoc = None
    runner_cuoi = None
    for lan in ds_lan:
        rec = lans[lan]
        bd, kt = rec["bat_dau"], rec["ket_thuc"]
        pl = duong_dan_lan(dd, model, lan)
        ep_lan = [ev.get("epoch") for ev in rec["epoch"]]
        h = OrderedDict(lan=lan, che_do=bd.get("che_do_luot"), pid_runner=bd.get("pid"),
                        start_epoch_0=bd.get("start_epoch"), epoch_da_luu=ep_lan)
        # tính liên tục của resume
        if lan == 0:
            if bd.get("che_do_luot") != "moi" or bd.get("start_epoch") != 0:
                L(f"lan 0: che_do_luot={bd.get('che_do_luot')} start_epoch={bd.get('start_epoch')} (ky vong moi/0)")
        else:
            if bd.get("che_do_luot") != "tiep_tuc":
                L(f"lan {lan}: che_do_luot={bd.get('che_do_luot')} (ky vong tiep_tuc)")
            if bd.get("start_epoch") != so_da_luu_truoc:
                L(f"lan {lan}: start_epoch {bd.get('start_epoch')} != so epoch da luu truoc do {so_da_luu_truoc}")
            tt = bd.get("tiep_tuc") or {}
            chon = (tt.get("chon") or {}).get("sha256")
            if ban_sao_truoc is not None and chon != ban_sao_truoc:
                L(f"lan {lan}: checkpoint resume {chon} != ban sao epoch cuoi lan truoc {ban_sao_truoc}")
            kk = bd.get("kiem_khoi_phuc") or {}
            hong = [k for k, v in kk.items() if not (isinstance(v, dict) and v.get("dat") is True)]
            if not kk or hong:
                L(f"lan {lan}: kiem_khoi_phuc khong dat/thieu: {hong if kk else 'thieu'}")
            h["resume"] = {"checkpoint_nguon": (tt.get("chon") or {}).get("duong_dan"), "checkpoint_nguon_sha256": chon,
                           "epoch_bat_dau_tiep": tt.get("epoch_bat_dau_tiep"),
                           "kiem_khoi_phuc_dat": (not hong) if kk else None, "so_muc_kiem_khoi_phuc": len(kk),
                           "nguon_sao_luu_sha256": (tt.get("sao_luu") or {}).get("nguon_sha256")}
            p_nguon = dd["run_dir"] / "khoi_phuc" / f"nguon_tiep_tuc_{lan}.pt"
            if p_nguon.is_file():
                s, n = sh.bam(p_nguon, f"checkpoint nguon resume lan {lan}")
                ky = (tt.get("sao_luu") or {}).get("nguon_sha256")
                if s != ky:
                    L(f"hash lech: {sh.rel(p_nguon)} {s} != nhat ky {ky}")
                ck_rows.append(OrderedDict(model=model, vai_tro=f"nguon_tiep_tuc_{lan}", epoch=so_da_luu_truoc,
                                           duong_dan=sh.rel(p_nguon), bytes=n, sha256_thuc=s,
                                           sha256_nhat_ky=ky, sha256_runner_json=None, khop=s == ky,
                                           co_optimizer=None, epoch_trong_checkpoint=None,
                                           ghi_chu="ban sao checkpoint dung de resume (co optimizer)"))
            else:
                thieu.append(f"lan {lan}: khong co {sh.rel(p_nguon)}")
        # kết thúc lần chạy
        if kt is None:
            (L if lan == lan_cuoi else C)(f"lan {lan}: khong co su kien ket_thuc (tien trinh chet khong ghi)")
        else:
            if kt.get("epoch_da_luu_lan_nay") != ep_lan:
                L(f"lan {lan}: ket_thuc.epoch_da_luu_lan_nay {kt.get('epoch_da_luu_lan_nay')} != su kien {ep_lan}")
            if lan != lan_cuoi and kt.get("trang_thai") not in ("bi_ngat",):
                C(f"lan {lan}: ket thuc '{kt.get('trang_thai')}' truoc khi resume")
        h["ket_thuc"] = None if kt is None else {k: kt.get(k) for k in
                                                 ("trang_thai", "loi", "kiem_sau_train_dat", "exit_code", "t_iso")}
        h["bat_dau_iso"] = bd.get("t_iso")
        h["thoi_luong_nhat_ky_s"] = (round(kt["t"] - bd["t"], 1) if kt and _so(kt.get("t")) and _so(bd.get("t"))
                                     else None)
        if rec["epoch"] and kt and _so(kt.get("t")):
            h["mat_sau_epoch_luu_cuoi_s"] = round(kt["t"] - rec["epoch"][-1]["t"], 1)
        # JSON runner
        rj = None
        if pl["runner_json"].is_file():
            sh.bam(pl["runner_json"], f"JSON runner lan {lan}")
            rj = doc_json(pl["runner_json"])
            if rj.get("model") != model or rj.get("ten") != dd["ten"]:
                L(f"lan {lan}: JSON runner model/ten lech ({rj.get('model')}, {rj.get('ten')})")
            ep_rj = [e.get("epoch") for e in rj.get("epoch") or []]
            if ep_rj != ep_lan:
                L(f"lan {lan}: epoch trong JSON runner {ep_rj} != nhat ky {ep_lan}")
            ep_lddu = (rj.get("luot_day_du") or {}).get("epoch_da_luu_lan_nay")
            if ep_lddu is not None and ep_lddu != ep_lan:
                L(f"lan {lan}: luot_day_du.epoch_da_luu_lan_nay {ep_lddu} != nhat ky {ep_lan}")
            nke = {e.get("epoch"): e for e in rj.get("nhat_ky_epoch") or []}
            for ev in rec["epoch"]:
                x = nke.get(ev.get("epoch"))
                if x is None or x.get("last_sha256") != ev.get("last_sha256"):
                    L(f"lan {lan} epoch {ev.get('epoch')}: last_sha256 JSON runner "
                      f"{None if x is None else x.get('last_sha256')} != nhat ky {ev.get('last_sha256')}")
            h["runner"] = {"duong_dan": sh.rel(pl["runner_json"]), "loi": rj.get("loi"),
                           "loss_huu_han": rj.get("loss_huu_han"), "thoi_gian_train_goi_s": _so(rj.get("thoi_gian_s")),
                           "kiem_dataset_s": _so((rj.get("dataset") or {}).get("thoi_gian_s"))}
        else:
            (L if lan == lan_cuoi else thieu.append)(f"lan {lan}: thieu JSON runner {sh.rel(pl['runner_json'])}")
        if lan == lan_cuoi:
            runner_cuoi = rj
        ep_rj_map = {e.get("epoch"): e for e in (rj or {}).get("epoch") or []}
        # bộ giám sát
        gs = pl["giam_sat"]
        h["giam_sat"] = OrderedDict(thu_muc=sh.rel(gs))
        tom = trg = None
        for ten_f in ("tom_tat.json", "trang_thai.json", "log.txt", "ram_mau_tho.csv"):
            if (gs / ten_f).is_file():
                sh.bam(gs / ten_f, f"giam sat lan {lan}: {ten_f}")
            else:
                (L if (lan == lan_cuoi and ten_f != "ram_mau_tho.csv") else thieu.append)(
                    f"lan {lan}: thieu {sh.rel(gs / ten_f)}")
        if (gs / "tom_tat.json").is_file():
            tom = doc_json(gs / "tom_tat.json")
        if (gs / "trang_thai.json").is_file():
            trg = doc_json(gs / "trang_thai.json")
        if tom is not None:
            h["giam_sat"].update(exit_code=tom.get("exit_code"), bat_dau=tom.get("bat_dau"),
                                 ket_thuc=tom.get("ket_thuc"), thoi_gian_s=_so(tom.get("thoi_gian_s")),
                                 pid=tom.get("pid"), ctrl_c=len(tom.get("ctrl_c") or []),
                                 loi_in_man_hinh=tom.get("loi_in_man_hinh"))
            if tom.get("pid") != bd.get("pid"):
                L(f"lan {lan}: PID giam sat {tom.get('pid')} != PID runner trong nhat ky {bd.get('pid')}")
            if kt is not None and tom.get("exit_code") != kt.get("exit_code"):
                L(f"lan {lan}: exit giam sat {tom.get('exit_code')} != exit nhat ky {kt.get('exit_code')}")
        if trg is not None:
            h["giam_sat"]["trang_thai"] = trg.get("trang_thai")
            if tom is not None and trg.get("exit_code") != tom.get("exit_code"):
                L(f"lan {lan}: trang_thai.json exit {trg.get('exit_code')} != tom_tat {tom.get('exit_code')}")
        if (gs / "log.txt").is_file():
            lg = doc_dong_trang_thai_log(gs / "log.txt", model)
            h["giam_sat"]["log"] = lg
        # tài nguyên
        tn = OrderedDict()
        if (gs / "ram_mau_tho.csv").is_file():
            ram = phan_tich_ram(gs / "ram_mau_tho.csv")
            tn["ram_giam_sat"] = ram
            if ram["pid_chinh"] and str(bd.get("pid")) not in ram["pid_chinh"]:
                L(f"lan {lan}: PID trong ram_mau_tho.csv {ram['pid_chinh']} khong chua PID runner {bd.get('pid')}")
            tr = (tom or {}).get("ram") or {}
            if tr:
                tn["khop_tom_tat_giam_sat"] = {
                    "rss_chinh_max": _gan((tr.get("rss_chinh_mib") or {}).get("max"), ram["rss_chinh_mib"]["max"], 0.051),
                    "rss_tong_max": _gan((tr.get("rss_tong_mib") or {}).get("max"), ram["rss_tong_mib"]["max"], 0.051),
                    "ram_may_min": _gan((tr.get("ram_he_thong_con_trong_mib") or {}).get("min"),
                                        ram["ram_may_con_trong_mib"]["min"], 0.51)}
                if not all(tn["khop_tom_tat_giam_sat"].values()):
                    C(f"lan {lan}: so lieu RAM tinh lai lech tom_tat.json {tn['khop_tom_tat_giam_sat']}")
        else:
            tn["ram_giam_sat"] = None
        if pl["nvsmi_csv"].is_file():
            sh.bam(pl["nvsmi_csv"], f"nvidia-smi lan {lan}")
            nv = phan_tich_nvsmi(pl["nvsmi_csv"])
            tn["nvidia_smi"] = nv
            vj = (rj or {}).get("vram_nvidia_smi") or {}
            if vj.get("max_mib") is not None and nv["vram_ca_gpu_mib"]["max"] is not None \
                    and not _gan(vj["max_mib"], nv["vram_ca_gpu_mib"]["max"], 0.51):
                C(f"lan {lan}: VRAM max CSV {nv['vram_ca_gpu_mib']['max']} != JSON runner {vj['max_mib']}")
        else:
            tn["nvidia_smi"] = None
            thieu.append(f"lan {lan}: thieu CSV nvidia-smi {sh.rel(pl['nvsmi_csv'])}")
        ta = [_so(e.get("torch_max_allocated_mib")) for e in ep_rj_map.values()]
        trs = [_so(e.get("torch_max_reserved_mib")) for e in ep_rj_map.values()]
        tn["torch_epoch"] = {"max_allocated_mib": _tk(ta)["max"], "max_reserved_mib": _tk(trs)["max"],
                             "so_epoch_co_so_lieu": len([v for v in ta if v is not None])}
        h["tai_nguyen"] = tn
        if tn["ram_giam_sat"] is None:
            thieu.append(f"lan {lan}: khong co mau RAM (RSS PID/tien trinh con, RAM toan may) -> de trong, khong dien 0")
        # dev từng epoch
        for ev in rec["epoch"]:
            e = ev.get("epoch")
            m = ev.get("metrics") or {}
            x = ep_rj_map.get(e) or {}
            r = OrderedDict(model=model, epoch=e, lan=lan, luu_luc=ev.get("t_iso"))
            for k in KHOA_METRIC:
                r[TEN_COT_METRIC[k]] = _so(m.get(k))
            r.update(fitness=_so(ev.get("fitness")), best_fitness_sau_epoch=_so(ev.get("best_fitness")),
                     best_ghi_epoch_nay=ev.get("best_ghi_epoch_nay"), dung=ev.get("dung"),
                     mosaic=ev.get("mosaic_thuc_te"), val_epoch_s=_so(x.get("val_s")), train_epoch_s=_so(x.get("train_s")),
                     loss_huu_han=x.get("loss_huu_han"))
            if x:
                lech = [k for k in KHOA_METRIC if not _gan((x.get("metrics") or {}).get(k), m.get(k))]
                if lech:
                    L(f"epoch {e}: metric JSON runner lech nhat ky {lech}")
                if x.get("loss_huu_han") is not True:
                    L(f"epoch {e}: loss khong huu han theo JSON runner")
            r["khop_runner_json"] = bool(x) and not [k for k in KHOA_METRIC
                                                       if not _gan((x.get("metrics") or {}).get(k), m.get(k))]
            r["khop_results_csv"] = None
            dev_rows.append(r)
            # checkpoint epoch (bản lưu trong epoch, CÒN optimizer)
            p_bs = dd["run_dir"] / (ev.get("ban_sao") or f"khoi_phuc/last_e{e:03d}.pt")
            if p_bs.is_file():
                s, n = sh.bam(p_bs, f"checkpoint luu trong epoch {e}")
                md = doc_metadata_checkpoint(p_bs)
                khop = s == ev.get("ban_sao_sha256") == ev.get("last_sha256")
                if not khop:
                    L(f"hash lech: {sh.rel(p_bs)} {s} != nhat ky ban_sao {ev.get('ban_sao_sha256')} / last "
                      f"{ev.get('last_sha256')}")
                if md.get("doc_duoc") and md.get("epoch") != e - 1:
                    L(f"{sh.rel(p_bs)}: epoch trong checkpoint {md.get('epoch')} != {e - 1} (0-based)")
                ck_rows.append(OrderedDict(model=model, vai_tro="luu_trong_epoch", epoch=e, duong_dan=sh.rel(p_bs),
                                           bytes=n, sha256_thuc=s, sha256_nhat_ky=ev.get("ban_sao_sha256"),
                                           sha256_runner_json=(nke.get(e) or {}).get("last_sha256") if rj else None,
                                           khop=khop, co_optimizer=md.get("co_optimizer"),
                                           epoch_trong_checkpoint=md.get("epoch"),
                                           ghi_chu=("ban last.pt luc cuoi epoch (con optimizer/EMA, epoch 0-based)"
                                                    if md.get("doc_duoc") else f"metadata: {md.get('ly_do')}")))
            else:
                L(f"thieu checkpoint epoch {e}: {sh.rel(p_bs)}")
        so_da_luu_truoc += len(ep_lan)
        if rec["epoch"]:
            ban_sao_truoc = rec["epoch"][-1].get("ban_sao_sha256")
        lich_su.append(h)

    # tạm dừng giữa các lần
    for a, b in zip(lich_su, lich_su[1:]):
        ta, tb = _t_iso((a.get("giam_sat") or {}).get("ket_thuc")), _t_iso((b.get("giam_sat") or {}).get("bat_dau"))
        na = _t_iso((a.get("ket_thuc") or {}).get("t_iso"))
        nb = _t_iso(b.get("bat_dau_iso"))
        b["tam_dung_truoc_lan"] = {
            "giam_sat_ket_thuc_lan_truoc": (a.get("giam_sat") or {}).get("ket_thuc"),
            "giam_sat_bat_dau_lan_nay": (b.get("giam_sat") or {}).get("bat_dau"),
            "khoang_giam_sat_s": round(tb - ta, 1) if ta is not None and tb is not None else None,
            "khoang_nhat_ky_ket_thuc_den_bat_dau_s": round(nb - na, 1) if na is not None and nb is not None else None,
            "ghi_chu": "khoang nhat ky gom ca kiem dataset/khoi phuc truoc su kien bat_dau cua lan moi"}

    # ---- results.csv khớp nhật ký
    p_rc = dd["run_dir"] / "results.csv"
    rc_ket = {"duong_dan": sh.rel(p_rc)}
    if p_rc.is_file():
        sh.bam(p_rc, "results.csv cuoi")
        hang = doc_results_csv(p_rc)
        ep_rc = [int(float(r.get("epoch"))) for r in hang if _so(r.get("epoch")) is not None]
        rc_ket.update(so_hang=len(hang), epoch=ep_rc)
        if ep_rc != list(range(1, len(ep_rc) + 1)) or len(ep_rc) != len(set(ds_epoch)):
            L(f"results.csv epoch {ep_rc} khong khop quy dao nhat ky {ds_epoch}")
        lech = []
        by_e = {int(float(r["epoch"])): r for r in hang if _so(r.get("epoch")) is not None}
        for r in dev_rows:
            x = by_e.get(r["epoch"])
            if x is None:
                r["khop_results_csv"] = False
                continue
            bad = [TEN_COT_METRIC[k] for k in KHOA_METRIC if not _gan(_so(x.get(k)), r[TEN_COT_METRIC[k]])]
            r["khop_results_csv"] = not bad
            if bad:
                lech.append({"epoch": r["epoch"], "cot": bad})
        rc_ket["lech"] = lech
        if lech:
            L(f"results.csv lech nhat ky: {lech}")
        rc_ket["ghi_chu"] = ("cot 'time' cua results.csv cong don trong tung lan chay (dat lai khi resume), "
                             "khong dung lam thoi gian epoch")
    else:
        L(f"thieu {sh.rel(p_rc)}")
    for ten_f in ("args.yaml",):
        if (dd["run_dir"] / ten_f).is_file():
            sh.bam(dd["run_dir"] / ten_f, ten_f)
        else:
            thieu.append(f"thieu {sh.rel(dd['run_dir'] / ten_f)}")

    # ---- best
    fit = [(r["fitness"], r["epoch"]) for r in dev_rows if r["fitness"] is not None]
    best_fit = None
    if fit:
        mx = max(f for f, _ in fit)
        best_fit = min(e for f, e in fit if f == mx)  # ultralytics chỉ cập nhật khi fitness lớn hơn hẳn
    ghi_best = [r["epoch"] for r in dev_rows if r["best_ghi_epoch_nay"]]
    best_nk = ghi_best[-1] if ghi_best else None
    best = OrderedDict(epoch_theo_fitness_max=best_fit, epoch_theo_su_kien_ghi_best_cuoi=best_nk,
                       epoch_ky_vong=ky_vong.get("best_epoch"),
                       fitness=next((r["fitness"] for r in dev_rows if r["epoch"] == best_fit), None),
                       dinh_nghia_fitness="fitness = mAP50-95(B) theo nhat ky (fitness == metrics/mAP50-95(B) moi epoch)")
    if any(not _gan(r["fitness"], r["map50_95"]) for r in dev_rows):
        C("fitness khong trung mAP50-95 o mot so epoch")
    if best_fit != best_nk:
        L(f"best theo fitness e{best_fit} != best theo su kien ghi best e{best_nk}")
    if ky_vong.get("best_epoch") is not None and best_fit != ky_vong["best_epoch"]:
        L(f"best e{best_fit} != ky vong e{ky_vong['best_epoch']}")
    if best_nk in epoch_ev:
        ev_b = epoch_ev[best_nk][1]
        best["best_sha256_luc_ghi_con_optimizer"] = ev_b.get("best_sha256")
        best["ban_sao_epoch"] = ev_b.get("ban_sao")
        if ev_b.get("best_sha256") != ev_b.get("ban_sao_sha256"):
            L(f"best e{best_nk}: best_sha256 luc ghi != ban sao last cung epoch")

    # ---- checkpoint cuối (đã bỏ optimizer)
    ck_cuoi = OrderedDict()
    for vai in ("best", "last"):
        p = dd["run_dir"] / "weights" / f"{vai}.pt"
        o = OrderedDict(duong_dan=sh.rel(p))
        if not p.is_file():
            L(f"thieu checkpoint cuoi {sh.rel(p)}")
            ck_cuoi[vai] = o
            continue
        s, n = sh.bam(p, f"checkpoint cuoi {vai}.pt")
        rjc = ((runner_cuoi or {}).get("checkpoint") or {}).get(vai) or {}
        md = doc_metadata_checkpoint(p)
        o.update(bytes=n, sha256_thuc=s, sha256_runner_json=rjc.get("sha256"), bytes_runner_json=rjc.get("bytes"),
                 khop_runner_json=s == rjc.get("sha256") and n == rjc.get("bytes"),
                 runner_nap_lai=rjc.get("nap_lai"), runner_dung_5_lop=rjc.get("dung_5_lop"),
                 runner_train_args_lech_cau_hinh=rjc.get("train_args_lech_cau_hinh"), metadata=md)
        if not o["khop_runner_json"]:
            L(f"hash lech: {sh.rel(p)} {s}/{n} B != JSON runner {rjc.get('sha256')}/{rjc.get('bytes')} B")
        if vai == "best" and ky_vong.get("best_sha256") and s != ky_vong["best_sha256"]:
            L(f"hash lech: best.pt {s} != ky vong plan {ky_vong['best_sha256']}")
        if not (rjc.get("nap_lai") is True and rjc.get("dung_5_lop") is True and rjc.get("train_args_lech_cau_hinh") == {}):
            L(f"{vai}.pt: runner khong xac nhan nap lai/5 lop/tham so trung cau hinh")
        if md.get("doc_duoc"):
            if md.get("co_optimizer") or md.get("epoch") != -1:
                L(f"{vai}.pt: khong phai ban cuoi da bo optimizer (co_optimizer={md.get('co_optimizer')}, "
                  f"epoch={md.get('epoch')})")
            nm = md.get("names")
            if nm is not None and nm != {"0": "car", "1": "truck", "2": "bus", "3": "motorcycle", "4": "bicycle"}:
                L(f"{vai}.pt: ten lop trong checkpoint {nm}")
            tre = md.get("train_results_epoch")
            o["train_results_epoch_cuoi"] = int(tre[-1]) if tre and _so(tre[-1]) is not None else None
            o["train_results_ghi_chu"] = ("ultralytics final_eval chep train_results cua last.pt sang best.pt -> "
                                          "khong dung lam bang chung epoch cua best.pt")
            muc_tieu = best_fit if vai == "best" else (max(ds_epoch) if ds_epoch else None)
            # (1) train_metrics luu cung checkpoint = metric val cua epoch da ghi checkpoint
            tm = md.get("train_metrics") or {}
            o["fitness_theo_train_metrics"] = tm.get("fitness")
            khop_tm = [r["epoch"] for r in dev_rows
                       if tm and _gan(tm.get("fitness"), r["fitness"])
                       and all(_gan(tm.get(k), r[TEN_COT_METRIC[k]]) for k in KHOA_METRIC if k in tm)]
            o["epoch_khop_train_metrics"] = khop_tm
            # (2) 'date' luu checkpoint so voi thoi diem su kien epoch_da_luu (ghi ngay sau khi luu)
            t_ck = _t_iso(md.get("date"))
            gan_nhat = None
            if t_ck is not None:
                ung = [(ev.get("t") - t_ck, e) for e, (_, ev) in epoch_ev.items()
                       if _so(ev.get("t")) is not None and 0 <= ev.get("t") - t_ck <= 600]
                gan_nhat = min(ung)[1] if ung else None
            o["epoch_theo_thoi_diem_luu"] = gan_nhat
            o["epoch_xac_dinh"] = muc_tieu if (khop_tm == [muc_tieu] and gan_nhat == muc_tieu) else None
            if khop_tm != [muc_tieu]:
                L(f"{vai}.pt: train_metrics khop epoch {khop_tm} != [{muc_tieu}]")
            if gan_nhat != muc_tieu:
                L(f"{vai}.pt: thoi diem luu {md.get('date')} ung voi epoch {gan_nhat} != {muc_tieu}")
            tk = (runner_cuoi or {}).get("train_kwargs") or {}
            ta = md.get("train_args") or {}
            lech_args = {k: [v, ta.get(k)] for k, v in tk.items() if str(ta.get(k)) != str(v)} if ta else None
            o["train_args_lech_train_kwargs"] = lech_args
            if lech_args:
                C(f"{vai}.pt: train_args trong checkpoint khac train_kwargs runner: {lech_args}")
        else:
            thieu.append(f"{vai}.pt: khong doc duoc metadata ({md.get('ly_do')}); dua vao kiem nap lai cua runner")
        ck_cuoi[vai] = o
        ck_rows.append(OrderedDict(model=model, vai_tro=f"cuoi_{vai}", epoch=o.get("epoch_xac_dinh"),
                                   duong_dan=sh.rel(p), bytes=n, sha256_thuc=s, sha256_nhat_ky=None,
                                   sha256_runner_json=rjc.get("sha256"), khop=o["khop_runner_json"],
                                   co_optimizer=md.get("co_optimizer"), epoch_trong_checkpoint=md.get("epoch"),
                                   ghi_chu="ban cuoi da bo optimizer (strip_optimizer, epoch=-1); hash khac ban luu "
                                           "trong epoch"))
    if best_nk in epoch_ev and ck_cuoi.get("best", {}).get("sha256_thuc"):
        best["best_pt_cuoi_sha256_da_bo_optimizer"] = ck_cuoi["best"]["sha256_thuc"]
        best["bang_chung"] = [
            f"fitness max o e{best_fit} (dev_tung_epoch.csv, khop results.csv/JSON runner)",
            f"su kien epoch_da_luu cuoi co best_ghi_epoch_nay=true la e{best_nk}",
            f"best.pt cuoi: train_metrics chi khop epoch {ck_cuoi['best'].get('epoch_khop_train_metrics')} "
            f"(fitness {ck_cuoi['best'].get('fitness_theo_train_metrics')}); thoi diem luu "
            f"{(ck_cuoi['best'].get('metadata') or {}).get('date')} ung voi su kien epoch_da_luu "
            f"e{ck_cuoi['best'].get('epoch_theo_thoi_diem_luu')}",
            f"best_sha256 luc ghi (con optimizer) = ban sao khoi_phuc e{best_nk}: "
            f"{best.get('best_sha256_luc_ghi_con_optimizer')}; best.pt cuoi da bo optimizer nen hash khac"]

    # ---- validation cuối (final_eval) tách khỏi metric từng epoch
    dgc = (runner_cuoi or {}).get("danh_gia_cuoi_ultralytics")
    if dgc:
        m = dgc.get("metrics") or {}
        bf = next((r for r in dev_rows if r["epoch"] == best_fit), {})
        final = OrderedDict(
            nguon=sh.rel(duong_dan_lan(dd, model, lan_cuoi)["runner_json"]) + "#danh_gia_cuoi_ultralytics",
            checkpoint="weights/best.pt", metrics=m, trainer_epoch_tam=dgc.get("trainer_epoch_tam"),
            khoang_sau_train_epoch_cuoi_s=_so(dgc.get("thoi_gian_tu_het_train_epoch_cuoi_s")),
            ghi_chu=GHI_CHU_FINAL_EVAL,
            so_voi_val_epoch_best={"epoch": best_fit,
                                   "chenh_map50": (round(m["metrics/mAP50(B)"] - bf["map50"], 6)
                                                   if m.get("metrics/mAP50(B)") is not None and bf.get("map50") is not None
                                                   else None),
                                   "chenh_map50_95": (round(m["metrics/mAP50-95(B)"] - bf["map50_95"], 6)
                                                      if m.get("metrics/mAP50-95(B)") is not None
                                                      and bf.get("map50_95") is not None else None),
                                   "ghi_chu": "val trong epoch dung model EMA FP32 trong bo nho; final_eval nap best.pt "
                                              "FP16 tu dia -> co the lech nho; chi mo ta"})
    else:
        final = None
        L("thieu danh_gia_cuoi_ultralytics trong JSON runner lan cuoi")

    # ---- mục 8
    kt_c = lans[lan_cuoi]["ket_thuc"] or {}
    gs_c = lich_su[-1].get("giam_sat") or {}
    tk_log = ((gs_c.get("log") or {}).get("tong_ket") or [])
    dat_log = [t for t in tk_log if t["nhan"] == "DAT"]
    loss_ok = runner_cuoi is not None and runner_cuoi.get("loss_huu_han") is True and all(
        r.get("loss_huu_han") is True for r in dev_rows if r["lan"] == lan_cuoi)
    muc8 = OrderedDict()
    muc8["giam_sat_exit_0"] = {"dat": gs_c.get("exit_code") == 0 and gs_c.get("trang_thai") == "ket_thuc"
                               and (gs_c.get("log") or {}).get("exit_cuoi_log") == "0",
                               "bang_chung": {"tom_tat.exit_code": gs_c.get("exit_code"),
                                              "trang_thai.json": gs_c.get("trang_thai"),
                                              "log.txt '# exit='": (gs_c.get("log") or {}).get("exit_cuoi_log")},
                               "ghi_chu": "chay_giam_sat.py tra ve dung exit code cua tien trinh con"}
    muc8["runner_DAT_loi_None_callback_loss"] = {
        "dat": bool(dat_log) and dat_log[-1]["loi_None"] and dat_log[-1]["loi_callback"] == "None"
        and dat_log[-1]["loss_huu_han"] == "True" and runner_cuoi is not None and runner_cuoi.get("loi") is None
        and loss_ok,
        "bang_chung": {"dong_log": dat_log[-1]["dong"] if dat_log else None,
                       "json_loi": None if runner_cuoi is None else runner_cuoi.get("loi"),
                       "json_loss_huu_han": None if runner_cuoi is None else runner_cuoi.get("loss_huu_han")}}
    muc8["last_best_nap_duoc_5_lop_tham_so"] = {
        "dat": all((ck_cuoi.get(v) or {}).get("runner_nap_lai") is True and (ck_cuoi.get(v) or {}).get("runner_dung_5_lop")
                   is True and (ck_cuoi.get(v) or {}).get("runner_train_args_lech_cau_hinh") == {} for v in ("best", "last")),
        "bang_chung": "JSON runner lan cuoi: checkpoint.{best,last}.nap_lai/dung_5_lop/train_args_lech_cau_hinh "
                      "(runner nap lai sau train); R53A khong nap lai bang torch, chi doc metadata",
        "metadata_doc_lai": {v: {"names": ((ck_cuoi.get(v) or {}).get("metadata") or {}).get("names"),
                                 "co_optimizer": ((ck_cuoi.get(v) or {}).get("metadata") or {}).get("co_optimizer")}
                             for v in ("best", "last")}}
    muc8["nhat_ky_ket_thuc_hoan_tat"] = {
        "dat": kt_c.get("trang_thai") == "hoan_tat" and kt_c.get("kiem_sau_train_dat") is True
        and kt_c.get("exit_code") == 0 and kt_c.get("loi") is None,
        "bang_chung": {k: kt_c.get(k) for k in ("trang_thai", "kiem_sau_train_dat", "exit_code", "loi", "t_iso")}}
    muc8["du_epoch_hoac_dung_som_hop_le"] = {"dat": (du_epoch or dung_som_hop_le) and not trung,
                                             "bang_chung": {"so_epoch": len(set(ds_epoch)), "tong_epoch": tong_epoch,
                                                            "dung_epoch_cuoi": None if ev_cuoi is None
                                                            else ev_cuoi.get("dung"),
                                                            "dung_som_hop_le": dung_som_hop_le}}
    for k, v in muc8.items():
        if not v["dat"]:
            L(f"muc 8 khong dat: {k}")

    # ---- toàn lượt (không cộng peak)
    def _lay(h, *ks):
        x = h
        for k in ks:
            x = (x or {}).get(k) if isinstance(x, dict) else None
        return x

    def _cuc(ks, ham):
        vs = [(_lay(h, *ks), h["lan"]) for h in lich_su if _lay(h, *ks) is not None]
        if not vs:
            return {"gia_tri": None, "lan": None}
        v = ham(x for x, _ in vs)
        return {"gia_tri": v, "lan": [lan for x, lan in vs if x == v]}

    t0 = _t_iso(_lay(lich_su[0], "giam_sat", "bat_dau"))
    t1 = _t_iso(_lay(lich_su[-1], "giam_sat", "ket_thuc"))
    tg_chay = [_lay(h, "giam_sat", "thoi_gian_s") for h in lich_su]
    tam = [_lay(h, "tam_dung_truoc_lan", "khoang_giam_sat_s") for h in lich_su[1:]]
    toan = OrderedDict(
        so_lan_chay=len(lich_su), so_lan_resume=len(lich_su) - 1,
        exit_tung_lan=[_lay(h, "ket_thuc", "exit_code") for h in lich_su],
        trang_thai_tung_lan=[_lay(h, "ket_thuc", "trang_thai") for h in lich_su],
        bat_dau=_lay(lich_su[0], "giam_sat", "bat_dau"), ket_thuc=_lay(lich_su[-1], "giam_sat", "ket_thuc"),
        dong_ho_tu_dau_den_cuoi_s=round(t1 - t0, 1) if t0 is not None and t1 is not None else None,
        tong_thoi_luong_chay_s=(round(sum(tg_chay), 1) if all(v is not None for v in tg_chay) else None),
        tong_tam_dung_giua_lan_s=(round(sum(tam), 1) if tam and all(v is not None for v in tam) else (0.0 if not tam else None)),
        tong_khoang_mat_mau_trong_lan_s=(round(sum(_lay(h, "tai_nguyen", "ram_giam_sat", "khoang_trong", "tong_s")
                                                   for h in lich_su), 1)
                                         if all(_lay(h, "tai_nguyen", "ram_giam_sat", "khoang_trong", "tong_s") is not None
                                                for h in lich_su) else None),
        tong_val_epoch_s=round(sum(r["val_epoch_s"] for r in dev_rows), 2)
        if dev_rows and all(r["val_epoch_s"] is not None for r in dev_rows) else None,
        tong_train_epoch_s=round(sum(r["train_epoch_s"] for r in dev_rows), 2)
        if dev_rows and all(r["train_epoch_s"] is not None for r in dev_rows) else None,
        rss_chinh_max_mib=_cuc(("tai_nguyen", "ram_giam_sat", "rss_chinh_mib", "max"), max),
        rss_tong_max_mib=_cuc(("tai_nguyen", "ram_giam_sat", "rss_tong_mib", "max"), max),
        so_con_max=_cuc(("tai_nguyen", "ram_giam_sat", "so_con", "max"), max),
        ram_may_con_trong_min_mib=_cuc(("tai_nguyen", "ram_giam_sat", "ram_may_con_trong_mib", "min"), min),
        vram_nvidia_smi_max_mib=_cuc(("tai_nguyen", "nvidia_smi", "vram_ca_gpu_mib", "max"), max),
        torch_max_allocated_mib=_cuc(("tai_nguyen", "torch_epoch", "max_allocated_mib"), max),
        torch_max_reserved_mib=_cuc(("tai_nguyen", "torch_epoch", "max_reserved_mib"), max),
        ghi_chu=("Cuc tri toan luot = max/min qua cac lan (kem lan dat), KHONG cong peak. Chi cong thoi luong. "
                 "RSS = working set Windows cua PID runner; rss_tong = PID + moi tien trinh con (dem trung trang chia "
                 "se). RAM may = psutil available toan may. VRAM nvidia-smi = memory.used ca GPU (gom desktop); "
                 "torch_max_* = allocator PyTorch cua tien trinh train theo epoch. dong_ho = tu bat dau giam sat lan "
                 "dau den ket thuc giam sat lan cuoi."))

    # ---- bảng tài nguyên
    tn_rows = []
    for h in lich_su:
        r_ = _lay(h, "tai_nguyen", "ram_giam_sat") or {}
        nv = _lay(h, "tai_nguyen", "nvidia_smi") or {}
        tn_rows.append(OrderedDict(
            model=model, pham_vi="lan", lan=h["lan"], pid_runner=h.get("pid_runner"),
            trang_thai=_lay(h, "ket_thuc", "trang_thai"), exit_code=_lay(h, "ket_thuc", "exit_code"),
            exit_giam_sat=_lay(h, "giam_sat", "exit_code"), bat_dau=_lay(h, "giam_sat", "bat_dau"),
            ket_thuc=_lay(h, "giam_sat", "ket_thuc"), thoi_luong_giam_sat_s=_lay(h, "giam_sat", "thoi_gian_s"),
            thoi_gian_train_goi_s=_lay(h, "runner", "thoi_gian_train_goi_s"),
            tam_dung_truoc_lan_s=_lay(h, "tam_dung_truoc_lan", "khoang_giam_sat_s"),
            epoch_da_luu=" ".join(str(e) for e in h["epoch_da_luu"]),
            so_mau_ram=r_.get("so_mau"), khoang_mat_mau_ram_s=_lay(r_, "khoang_trong", "tong_s"),
            so_khoang_mat_mau_ram=_lay(r_, "khoang_trong", "so_khoang"),
            rss_chinh_trung_vi_mib=_lay(r_, "rss_chinh_mib", "trung_vi"), rss_chinh_max_mib=_lay(r_, "rss_chinh_mib", "max"),
            rss_tong_trung_vi_mib=_lay(r_, "rss_tong_mib", "trung_vi"), rss_tong_max_mib=_lay(r_, "rss_tong_mib", "max"),
            so_con_max=_lay(r_, "so_con", "max"), ram_may_con_trong_min_mib=_lay(r_, "ram_may_con_trong_mib", "min"),
            so_mau_nvidia_smi=nv.get("so_mau"), vram_nvidia_smi_max_mib=_lay(nv, "vram_ca_gpu_mib", "max"),
            gpu_util_trung_vi_pct=_lay(nv, "gpu_util_pct", "trung_vi"),
            torch_max_allocated_mib=_lay(h, "tai_nguyen", "torch_epoch", "max_allocated_mib"),
            torch_max_reserved_mib=_lay(h, "tai_nguyen", "torch_epoch", "max_reserved_mib"),
            nguon=f"{_lay(h, 'giam_sat', 'thu_muc')}/ram_mau_tho.csv; "
                  f"{sh.rel(duong_dan_lan(dd, model, h['lan'])['nvsmi_csv'])}; JSON runner epoch[]",
            thieu=("; ".join(x for x in thieu if x.startswith(f"lan {h['lan']}:")) or None)))
    tn_rows.append(OrderedDict(
        model=model, pham_vi="toan_luot", lan=None, pid_runner=None, trang_thai=_lay(lich_su[-1], "ket_thuc", "trang_thai"),
        exit_code=_lay(lich_su[-1], "ket_thuc", "exit_code"), exit_giam_sat=_lay(lich_su[-1], "giam_sat", "exit_code"),
        bat_dau=toan["bat_dau"], ket_thuc=toan["ket_thuc"], thoi_luong_giam_sat_s=toan["tong_thoi_luong_chay_s"],
        thoi_gian_train_goi_s=None, tam_dung_truoc_lan_s=toan["tong_tam_dung_giua_lan_s"],
        epoch_da_luu=" ".join(str(e) for e in ds_epoch), so_mau_ram=None,
        khoang_mat_mau_ram_s=toan["tong_khoang_mat_mau_trong_lan_s"], so_khoang_mat_mau_ram=None,
        rss_chinh_trung_vi_mib=None, rss_chinh_max_mib=toan["rss_chinh_max_mib"]["gia_tri"],
        rss_tong_trung_vi_mib=None, rss_tong_max_mib=toan["rss_tong_max_mib"]["gia_tri"],
        so_con_max=toan["so_con_max"]["gia_tri"], ram_may_con_trong_min_mib=toan["ram_may_con_trong_min_mib"]["gia_tri"],
        so_mau_nvidia_smi=None, vram_nvidia_smi_max_mib=toan["vram_nvidia_smi_max_mib"]["gia_tri"],
        gpu_util_trung_vi_pct=None, torch_max_allocated_mib=toan["torch_max_allocated_mib"]["gia_tri"],
        torch_max_reserved_mib=toan["torch_max_reserved_mib"]["gia_tri"],
        nguon="cuc tri qua cac lan (max/min, khong cong); thoi luong = tong cac lan",
        thieu=None))

    muc10 = OrderedDict(
        epoch_thoi_gian_exit="lich_su_chay[].epoch_da_luu / giam_sat / ket_thuc; toan_luot",
        checkpoint_sha256="checkpoint_cuoi; checkpoint_sha256 (gom ban luu trong epoch con optimizer)",
        dev_tung_epoch_va_danh_gia_cuoi="dev_tung_epoch (validation trong vong train) va danh_gia_cuoi_ultralytics (tach rieng)",
        tai_nguyen="tai_nguyen (RSS PID + con, RAM toan may, VRAM nvidia-smi + torch)",
        lich_su_resume="lich_su_chay[].resume / tam_dung_truoc_lan")

    bc.update(dat=not loi, loi=loi, canh_bao=canh_bao, thieu_du_lieu=thieu, muc_8=muc8, muc_10=muc10,
              quy_dao_epoch=[list(x) for x in quy_dao], so_epoch=len(set(ds_epoch)), best=best,
              checkpoint_cuoi=ck_cuoi, danh_gia_cuoi_ultralytics=final, results_csv=rc_ket,
              lich_su_chay=lich_su, toan_luot=toan, dev_tung_epoch=dev_rows, tai_nguyen=tn_rows,
              checkpoint_sha256=ck_rows)
    return bc, sh


# ------------------------------------------------------------------------------------------ ghi đầu ra
def _ghi_csv(p, rows, cot):
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cot, lineterminator="\n", extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cot})
    os.replace(tmp, p)


def _ghi_json(p, obj):
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, p)


def kiem_out(out):
    """Trả đường dẫn đã resolve nếu nằm trong ban_giao/ hoặc tests_handoff/; ngược lại ValueError (chưa ghi gì)."""
    p = Path(out).resolve()
    for vung in VUNG_OUT_CHO_PHEP:
        v = vung.resolve()
        if p == v or v in p.parents:
            return p
    raise ValueError(f"--out {p} nam ngoai vung cho phep: " + ", ".join(str(v.resolve()) for v in VUNG_OUT_CHO_PHEP))


def _luu_ban_truoc(out, ten_files):
    """Chép đầu ra đang có sang out/_ban_truoc/<thoi diem>/ trước khi ghi lại; trả thư mục chép hoặc None."""
    co = [out / t for t in ten_files if (out / t).is_file()]
    if not co:
        return None
    dich = out / THU_MUC_BAN_TRUOC / (_dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}")
    dich.mkdir(parents=True, exist_ok=False)
    for p in co:
        shutil.copy2(p, dich / p.name)
        if hashlib.sha256((dich / p.name).read_bytes()).digest() != hashlib.sha256(p.read_bytes()).digest():
            raise RuntimeError(f"chep ban truoc lech hash: {p}")
    return dich


def ghi_dau_ra(out, bc, sh):
    out = kiem_out(out)
    out.mkdir(parents=True, exist_ok=True)
    _luu_ban_truoc(out, [f"ban_giao_{bc['model']}.json", "dev_tung_epoch.csv", "checkpoint_sha256.csv",
                         "tai_nguyen.csv", "nguon_sha256.csv"])
    bc["nguon_sha256"] = list(sh.nguon.values())
    _ghi_json(out / f"ban_giao_{bc['model']}.json", bc)
    # dựng lại bảng gộp từ mọi ban_giao_*.json (thứ tự model cố định)
    tat_ca = []
    for m in MODELS + tuple(sorted({p.stem[len("ban_giao_"):] for p in out.glob("ban_giao_*.json")} - set(MODELS))):
        p = out / f"ban_giao_{m}.json"
        if p.is_file():
            tat_ca.append(doc_json(p))
    for ten, khoa in (("dev_tung_epoch.csv", "dev_tung_epoch"), ("checkpoint_sha256.csv", "checkpoint_sha256"),
                      ("tai_nguyen.csv", "tai_nguyen")):
        rows = [r for b in tat_ca for r in b.get(khoa) or []]
        cot = list((bc.get(khoa) or [{}])[0].keys()) if bc.get(khoa) else (list(rows[0].keys()) if rows else ["model"])
        _ghi_csv(out / ten, rows, cot)
    rows = [OrderedDict(model=b["model"], **r) for b in tat_ca for r in b.get("nguon_sha256") or []]
    _ghi_csv(out / "nguon_sha256.csv", rows, ["model", "duong_dan", "bytes", "sha256", "vai_tro"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", default=str(OUT_MAC_DINH), help="thu muc dau ra trong results/phase2/r53_detector/{ban_giao,tests_handoff} (mac dinh ban_giao)")
    a = ap.parse_args(argv)
    try:
        out = kiem_out(a.out)
    except ValueError as e:
        print(f"[LOI] {e}", file=sys.stderr)
        return 2
    ma = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bc, sh = tong_hop(REPO, a.model, ma_nguon_sha={"src/r53_detector/ban_giao_train.py": ma})
    ghi_dau_ra(out, bc, sh)
    print(f"[{'DAT' if bc['dat'] else 'KHONG DAT'}] ban giao {a.model}: {len(bc['loi'])} loi, "
          f"{len(bc['canh_bao'])} canh bao, {len(bc['thieu_du_lieu'])} muc thieu du lieu -> {out}")
    for x in bc["loi"]:
        print(f"  LOI: {x}")
    for x in bc["canh_bao"]:
        print(f"  CANH BAO: {x}")
    for x in bc["thieu_du_lieu"]:
        print(f"  THIEU: {x}")
    if bc.get("best"):
        print(f"  best e{bc['best'].get('epoch_theo_fitness_max')} fitness {bc['best'].get('fitness')}; "
              f"so epoch {bc.get('so_epoch')}; quy dao {bc.get('quy_dao_epoch')}")
    return 0 if bc["dat"] else 1


if __name__ == "__main__":
    sys.exit(main())
