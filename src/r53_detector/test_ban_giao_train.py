"""Test R53A ban_giao_train.py trên fixture tổng hợp (không đọc dữ liệu thật, không torch, không GPU).

Fixture: một lượt 3 epoch có một lần resume (lan 0: e1-e2 bị ngắt; lan 1: e3 hoàn tất), dựng dưới
results/phase2/r53_detector/tests_handoff/lan_<thoi diem>_pid<pid>/fixture/<ten_test>/ với cùng bố cục thư mục như
repo thật. Mỗi lần chạy test dùng thư mục lan_* mới; không xóa hay ghi đè fixture/kết quả của lần trước (kể cả bố
cục cũ tests_handoff/fixture/, dau_ra_*/, ket_qua_test.json). Test cuối băm lại mọi file có trước lần chạy để kiểm.

  python src/r53_detector/test_ban_giao_train.py
Exit 0 khi mọi test đạt.
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

import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ban_giao_train as bg  # noqa: E402

TEST_DIR = bg.REPO / "results" / "phase2" / "r53_detector" / "tests_handoff"
R53_DIR = TEST_DIR.parent
# Gán trong main(): thư mục riêng của lần chạy này và ảnh chụp hash mọi file có trước nó.
LAN_DIR = None
FIX_DIR = None
ANH_TRUOC = None
DAU_MOC = ".fixture_r53a"
MODEL = "yolov8n"
T0 = 1792150000.0
TZ = _dt.timezone(_dt.timedelta(hours=7))


def iso(t):
    return _dt.datetime.fromtimestamp(t, TZ).isoformat(timespec="seconds")


class _MoHinhGia:
    """Pickle thành tham chiếu lớp lạ -> bộ unpickle hạn chế phải thay bằng đối tượng giả."""

    def __init__(self):
        self.names = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}


class _ReduceNguyHiem:
    def __reduce__(self):
        return (os.getcwd, ())


def ghi_ckpt(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("arch/data.pkl", pickle.dumps(d, protocol=2))
        zf.writestr("arch/version", "3\n")
    p.write_bytes(buf.getvalue())
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metric(e):
    f = round(0.1 * e, 5)
    return {"metrics/precision(B)": 0.5, "metrics/recall(B)": 0.4, "metrics/mAP50(B)": round(f + 0.15, 5),
            "metrics/mAP50-95(B)": f, "val/box_loss": 1.0, "val/cls_loss": 0.6, "val/dfl_loss": 0.9}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def chup_hash(goc):
    """{duong dan tuong doi: sha256} của mọi file dưới goc."""
    return {p.relative_to(goc).as_posix(): sha256(p) for p in sorted(Path(goc).rglob("*")) if p.is_file()}


def tao_thu_muc_lan(goc):
    """Thư mục mới cho một lần chạy test; tên đã tồn tại thì thêm hậu tố _k, không tái dùng, không xóa."""
    ten = _dt.datetime.now().strftime("lan_%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"
    for k in range(100):
        p = Path(goc) / (ten if k == 0 else f"{ten}_{k}")
        try:
            p.mkdir(parents=True, exist_ok=False)
            return p
        except FileExistsError:
            continue
    raise FileExistsError(f"khong tao duoc thu muc lan moi duoi {goc}")


def tao_fixture(ten, sua=None):
    goc = FIX_DIR / ten
    if goc.exists():
        raise FileExistsError(f"{goc} da ton tai -> khong xoa, khong ghi de")
    goc.mkdir(parents=True)
    (goc / DAU_MOC).write_text("fixture test R53A\n", encoding="utf-8")
    dd = bg.duong_dan_model(goc, MODEL)
    run = dd["run_dir"]
    lan_cua = {1: 0, 2: 0, 3: 1}
    pid = {0: 1111, 1: 2222}
    t_ep = {1: T0 + 100, 2: T0 + 200, 3: T0 + 1300}
    t_bd = {0: T0, 1: T0 + 1200}
    t_kt = {0: T0 + 250, 1: T0 + 1400}
    train_args = {"epochs": 3, "batch": 8}
    # checkpoint lưu trong epoch (còn optimizer)
    h = {}
    ck = {}
    for e in (1, 2, 3):
        d = {"epoch": e - 1, "best_fitness": 0.1 * e, "model": _MoHinhGia(), "ema": "ema", "updates": 10 * e,
             "optimizer": {"state": e}, "train_args": train_args, "train_metrics": {**metric(e), "fitness": metric(e)["metrics/mAP50-95(B)"]},
             "train_results": {"epoch": list(range(1, e + 1))}, "date": iso(t_ep[e] - 5), "version": "8.4.141"}
        ck[e] = d
        h[e] = ghi_ckpt(run / "khoi_phuc" / f"last_e{e:03d}.pt", d)
    shutil.copyfile(run / "khoi_phuc" / "last_e002.pt", run / "khoi_phuc" / "nguon_tiep_tuc_1.pt")
    cuoi = dict(ck[3], epoch=-1, optimizer=None, ema=None, best_fitness=None, updates=None)
    hb = ghi_ckpt(run / "weights" / "best.pt", cuoi)
    hl = ghi_ckpt(run / "weights" / "last.pt", cuoi)
    nb = (run / "weights" / "best.pt").stat().st_size
    spec = {"nhat_ky": [], "runner": {}, "results": [], "giam_sat": {}, "nvsmi": {}}
    for lan in (0, 1):
        bd = {"loai": "bat_dau", "t": t_bd[lan], "t_iso": iso(t_bd[lan]), "che_do_luot": "moi" if lan == 0 else "tiep_tuc",
              "lan": lan, "pid": pid[lan], "start_epoch": 0 if lan == 0 else 2,
              "kiem_khoi_phuc": None if lan == 0 else {"ema_updates": {"dat": True}, "start_epoch": {"dat": True}},
              "tiep_tuc": None if lan == 0 else {"chon": {"sha256": h[2], "duong_dan": "khoi_phuc/last_e002.pt"},
                                                  "epoch_bat_dau_tiep": 3, "sao_luu": {"nguon_sha256": h[2]}}}
        spec["nhat_ky"].append(bd)
        eps = [e for e in (1, 2, 3) if lan_cua[e] == lan]
        for e in eps:
            m = metric(e)
            spec["nhat_ky"].append({"loai": "epoch_da_luu", "t": t_ep[e], "t_iso": iso(t_ep[e]), "lan": lan, "epoch": e,
                                    "last_sha256": h[e], "ban_sao": f"khoi_phuc/last_e{e:03d}.pt", "ban_sao_sha256": h[e],
                                    "best_ghi_epoch_nay": True, "best_sha256": h[e], "fitness": m["metrics/mAP50-95(B)"],
                                    "best_fitness": m["metrics/mAP50-95(B)"], "dung": e == 3, "mosaic_thuc_te": True,
                                    "metrics": m})
            spec["results"].append({"epoch": e, **m})
        cuoi_lan = lan == 1
        spec["nhat_ky"].append({"loai": "ket_thuc", "t": t_kt[lan], "t_iso": iso(t_kt[lan]), "lan": lan,
                                "trang_thai": "hoan_tat" if cuoi_lan else "bi_ngat",
                                "loi": None if cuoi_lan else "KeyboardInterrupt", "kiem_sau_train_dat": cuoi_lan,
                                "epoch_da_luu_lan_nay": eps, "exit_code": 0 if cuoi_lan else 130})
        rj = {"model": MODEL, "ten": dd["ten"], "loi": None if cuoi_lan else "KeyboardInterrupt", "loss_huu_han": True,
              "thoi_gian_s": 90.0, "dataset": {"thoi_gian_s": 1.0}, "vram_nvidia_smi": {"max_mib": 2000 + lan},
              "luot_day_du": {"epoch_da_luu_lan_nay": eps},
              "epoch": [{"epoch": e, "train_s": 80.0, "val_s": 7.5, "loss_huu_han": True, "metrics": metric(e),
                         "torch_max_allocated_mib": 1500 + e, "torch_max_reserved_mib": 1800 + e} for e in eps],
              "nhat_ky_epoch": [{"epoch": e, "last_sha256": h[e]} for e in eps]}
        if cuoi_lan:
            rj["train_kwargs"] = train_args
            rj["danh_gia_cuoi_ultralytics"] = {"metrics": {"metrics/mAP50(B)": 0.4501, "metrics/mAP50-95(B)": 0.3001},
                                               "trainer_epoch_tam": 3, "thoi_gian_tu_het_train_epoch_cuoi_s": 20.0}
            rj["checkpoint"] = {v: {"sha256": hh, "bytes": nb, "nap_lai": True, "dung_5_lop": True,
                                    "train_args_lech_cau_hinh": {}} for v, hh in (("best", hb), ("last", hl))}
        spec["runner"][lan] = rj
        ec = 0 if cuoi_lan else 130
        ts = [t_bd[lan] - 2 + i for i in range(5)] + ([t_bd[lan] + 103] if cuoi_lan else [])
        spec["giam_sat"][lan] = {
            "tom_tat": {"pid": pid[lan], "exit_code": ec, "bat_dau": iso(t_bd[lan] - 2), "ket_thuc": iso(t_kt[lan] + 2),
                        "thoi_gian_s": t_kt[lan] - t_bd[lan] + 4, "ctrl_c": [] if cuoi_lan else [{"tin_hieu": 2}],
                        "loi_in_man_hinh": 0,
                        "ram": {"rss_chinh_mib": {"max": 500.0 + lan}, "rss_tong_mib": {"max": 1500.0 + lan},
                                "ram_he_thong_con_trong_mib": {"min": 300.0}}},
            "trang_thai": {"trang_thai": "ket_thuc", "pid": pid[lan], "exit_code": ec},
            "log": ("dong dau\n"
                    + (f"[+ 10.0s] [DAT] day-du {MODEL}: loi=None loi_callback=None loss_huu_han=True checkpoint=[]\n"
                       if cuoi_lan else
                       f"[+ 10.0s] [BI NGAT] day-du {MODEL}: loi=KeyboardInterrupt loi_callback=None loss_huu_han=None\n")
                    + f"# exit={ec}\n"),
            "ram": [{"t_unix": t, "t_tuong_doi_s": 0, "pid_chinh": pid[lan], "rss_chinh_mib": 400.0 + lan + (100 if i == 2 else 0),
                     "so_con": 2, "rss_con_mib": 1, "rss_tong_mib": 1400.0 + lan + (100 if i == 2 else 0),
                     "ram_he_thong_con_trong_mib": 300.0 if i == 3 else 900.0, "chi_tiet_con": "", "ghi_chu": ""}
                    for i, t in enumerate(ts)]}
        spec["nvsmi"][lan] = [{"t_unix": t_bd[lan] + i, "nvidia_smi_memory_used_mib_ca_gpu": 1000 if i else 2000 + lan,
                               "gpu_util_pct": 95, "ram_he_thong_con_trong_mib": 800} for i in range(3)]
    if sua:
        sua(spec, goc)
    # ghi
    with open(run / "nhat_ky_luot.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for ev in spec["nhat_ky"]:
            f.write(json.dumps(ev) + "\n")
    cot = ["epoch"] + bg.KHOA_METRIC
    with open(run / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        w.writerows(spec["results"])
    (run / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")
    for lan, rj in spec["runner"].items():
        pl = bg.duong_dan_lan(dd, MODEL, lan)
        pl["runner_json"].parent.mkdir(parents=True, exist_ok=True)
        pl["runner_json"].write_text(json.dumps(rj), encoding="utf-8")
        if spec["nvsmi"].get(lan) is not None:
            with open(pl["nvsmi_csv"], "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(spec["nvsmi"][lan][0].keys()))
                w.writeheader()
                w.writerows(spec["nvsmi"][lan])
        g = spec["giam_sat"][lan]
        pl["giam_sat"].mkdir(parents=True, exist_ok=True)
        (pl["giam_sat"] / "tom_tat.json").write_text(json.dumps(g["tom_tat"]), encoding="utf-8")
        (pl["giam_sat"] / "trang_thai.json").write_text(json.dumps(g["trang_thai"]), encoding="utf-8")
        (pl["giam_sat"] / "log.txt").write_text(g["log"], encoding="utf-8")
        if g.get("ram") is not None:
            with open(pl["giam_sat"] / "ram_mau_tho.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(g["ram"][0].keys()))
                w.writeheader()
                w.writerows(g["ram"])
    return goc, {"h": h, "hb": hb}


def chay(goc, ky_vong=None):
    bc, sh = bg.tong_hop(goc, MODEL, ky_vong={"best_epoch": 3} if ky_vong is None else ky_vong, tong_epoch=3)
    return bc, sh


def co_loi(bc, *cum):
    return any(all(c in x for c in cum) for x in bc["loi"])


# ------------------------------------------------------------------------------------------------ test
def test_hop_le():
    goc, hh = tao_fixture("hop_le")
    bc, sh = chay(goc, {"best_epoch": 3, "best_sha256": hh["hb"]})
    assert bc["dat"], bc["loi"]
    assert bc["loi"] == [] and bc["canh_bao"] == [], (bc["loi"], bc["canh_bao"])
    assert bc["quy_dao_epoch"] == [[0, 1], [0, 2], [1, 3]]
    assert bc["best"]["epoch_theo_fitness_max"] == 3 and bc["best"]["epoch_theo_su_kien_ghi_best_cuoi"] == 3
    assert all(v["dat"] for v in bc["muc_8"].values())
    b = bc["checkpoint_cuoi"]["best"]
    assert b["epoch_xac_dinh"] == 3 and b["epoch_khop_train_metrics"] == [3] and b["epoch_theo_thoi_diem_luu"] == 3
    assert b["metadata"]["names"]["0"] == "car" and b["metadata"]["co_optimizer"] is False
    vt = [r["vai_tro"] for r in bc["checkpoint_sha256"]]
    assert vt.count("luu_trong_epoch") == 3 and "nguon_tiep_tuc_1" in vt and "cuoi_best" in vt
    assert all(r["co_optimizer"] for r in bc["checkpoint_sha256"] if r["vai_tro"] == "luu_trong_epoch")
    assert all(r["khop_results_csv"] and r["khop_runner_json"] for r in bc["dev_tung_epoch"])
    lan1 = bc["lich_su_chay"][1]
    assert lan1["tam_dung_truoc_lan"]["khoang_giam_sat_s"] == 946.0, lan1["tam_dung_truoc_lan"]
    tl = bc["toan_luot"]
    # peak toàn lượt = max qua các lần (1400 + lan + 100 ở mẫu thứ 3), không phải tổng các peak
    assert tl["rss_tong_max_mib"] == {"gia_tri": 1501.0, "lan": [1]}, tl["rss_tong_max_mib"]
    assert tl["rss_chinh_max_mib"]["gia_tri"] == 501.0
    assert tl["ram_may_con_trong_min_mib"]["gia_tri"] == 300.0 and tl["ram_may_con_trong_min_mib"]["lan"] == [0, 1]
    assert tl["vram_nvidia_smi_max_mib"] == {"gia_tri": 2001.0, "lan": [1]}
    # khoảng mất mẫu 101 s trong lan 1 (mẫu t+2 -> t+103, giả lập máy ngủ); lan 0 không mất mẫu
    assert tl["tong_khoang_mat_mau_trong_lan_s"] == 101.0, tl["tong_khoang_mat_mau_trong_lan_s"]
    assert bc["lich_su_chay"][1]["tai_nguyen"]["ram_giam_sat"]["khoang_trong"]["so_khoang"] == 1
    assert bc["danh_gia_cuoi_ultralytics"]["ghi_chu"] == bg.GHI_CHU_FINAL_EVAL
    assert "khoang_sau_train_epoch_cuoi_s" in bc["danh_gia_cuoi_ultralytics"]
    assert not any("val" in k for k in bc["danh_gia_cuoi_ultralytics"] if k.endswith("_s")), "final_eval khong goi la val"
    # ghi đầu ra
    out = LAN_DIR / "dau_ra_hop_le"
    bg.ghi_dau_ra(out, bc, sh)
    ten_ra = ("ban_giao_yolov8n.json", "dev_tung_epoch.csv", "checkpoint_sha256.csv", "tai_nguyen.csv", "nguon_sha256.csv")
    for f in ten_ra:
        assert (out / f).is_file(), f
    assert not (out / bg.THU_MUC_BAN_TRUOC).exists(), "lan ghi dau khong co ban truoc"
    with open(out / "dev_tung_epoch.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3
    # ghi lại cùng thư mục: bản cũ được chép sang _ban_truoc/<thoi diem>/ nguyên byte, không mất
    h1 = {f: sha256(out / f) for f in ten_ra}
    bc["canh_bao"] = ["ghi lai de kiem _ban_truoc"]
    bg.ghi_dau_ra(out, bc, sh)
    cu = [p for p in (out / bg.THU_MUC_BAN_TRUOC).iterdir() if p.is_dir()]
    assert len(cu) == 1, cu
    assert {f: sha256(cu[0] / f) for f in ten_ra} == h1
    assert sha256(out / "ban_giao_yolov8n.json") != h1["ban_giao_yolov8n.json"]


def test_thieu_epoch():
    def sua(spec, goc):
        spec["nhat_ky"] = [ev for ev in spec["nhat_ky"] if not (ev["loai"] == "epoch_da_luu" and ev["epoch"] == 2)]
        for ev in spec["nhat_ky"]:
            if ev["loai"] == "ket_thuc" and ev["lan"] == 0:
                ev["epoch_da_luu_lan_nay"] = [1]
        spec["results"] = [r for r in spec["results"] if r["epoch"] != 2]
    goc, _ = tao_fixture("thieu_epoch", sua)
    bc, _ = chay(goc)
    assert not bc["dat"]
    assert co_loi(bc, "thieu epoch [2]"), bc["loi"]
    assert co_loi(bc, "quy dao epoch khong tang"), bc["loi"]
    assert not bc["muc_8"]["du_epoch_hoac_dung_som_hop_le"]["dat"]


def test_trung_epoch_sau_resume():
    def sua(spec, goc):
        # lan 1 ghi lại epoch 2 (xung đột với epoch 2 của lan 0) trước epoch 3
        ev2 = next(ev for ev in spec["nhat_ky"] if ev["loai"] == "epoch_da_luu" and ev["epoch"] == 2)
        i3 = next(i for i, ev in enumerate(spec["nhat_ky"]) if ev["loai"] == "epoch_da_luu" and ev["epoch"] == 3)
        spec["nhat_ky"].insert(i3, dict(ev2, lan=1, t=ev2["t"] + 1050))
    goc, _ = tao_fixture("trung_epoch", sua)
    bc, _ = chay(goc)
    assert not bc["dat"]
    assert co_loi(bc, "trung/xung dot epoch sau resume", "2: [0, 1]"), bc["loi"]
    assert not bc["muc_8"]["du_epoch_hoac_dung_som_hop_le"]["dat"]


def test_start_epoch_resume_lech():
    def sua(spec, goc):
        for ev in spec["nhat_ky"]:
            if ev["loai"] == "bat_dau" and ev["lan"] == 1:
                ev["start_epoch"] = 1
    goc, _ = tao_fixture("start_lech", sua)
    bc, _ = chay(goc)
    assert co_loi(bc, "lan 1: start_epoch 1 != so epoch da luu truoc do 2"), bc["loi"]


def test_hash_lech():
    goc, hh = tao_fixture("hash_lech")
    run = bg.duong_dan_model(goc, MODEL)["run_dir"]
    p = run / "khoi_phuc" / "last_e001.pt"
    p.write_bytes(p.read_bytes() + b"\0")
    # best.pt bị thay sau khi runner ghi JSON
    pb = run / "weights" / "best.pt"
    pb.write_bytes(pb.read_bytes() + b"\0")
    bc, _ = chay(goc, {"best_epoch": 3, "best_sha256": hh["hb"]})
    assert not bc["dat"]
    assert co_loi(bc, "hash lech", "last_e001.pt"), bc["loi"]
    assert co_loi(bc, "hash lech", "best.pt", "JSON runner"), bc["loi"]
    assert co_loi(bc, "hash lech: best.pt", "ky vong plan"), bc["loi"]
    r = next(r for r in bc["checkpoint_sha256"] if r["vai_tro"] == "luu_trong_epoch" and r["epoch"] == 1)
    assert r["khop"] is False


def test_ket_thuc_loi():
    def sua(spec, goc):
        for ev in spec["nhat_ky"]:
            if ev["loai"] == "ket_thuc" and ev["lan"] == 1:
                ev.update(trang_thai="loi", loi="RuntimeError: x", kiem_sau_train_dat=False, exit_code=1)
        spec["runner"][1]["loi"] = "RuntimeError: x"
        g = spec["giam_sat"][1]
        g["tom_tat"]["exit_code"] = 1
        g["trang_thai"]["exit_code"] = 1
        g["log"] = f"[+ 1.0s] [LOI] day-du {MODEL}: loi=RuntimeError loi_callback=None loss_huu_han=True\n# exit=1\n"
    goc, _ = tao_fixture("ket_thuc_loi", sua)
    bc, _ = chay(goc)
    assert not bc["dat"]
    m8 = bc["muc_8"]
    assert not m8["giam_sat_exit_0"]["dat"] and not m8["runner_DAT_loi_None_callback_loss"]["dat"]
    assert not m8["nhat_ky_ket_thuc_hoan_tat"]["dat"]
    assert co_loi(bc, "muc 8 khong dat: nhat_ky_ket_thuc_hoan_tat"), bc["loi"]


def test_thieu_ket_thuc_lan_cuoi():
    def sua(spec, goc):
        spec["nhat_ky"] = [ev for ev in spec["nhat_ky"] if not (ev["loai"] == "ket_thuc" and ev["lan"] == 1)]
    goc, _ = tao_fixture("thieu_ket_thuc", sua)
    bc, _ = chay(goc)
    assert co_loi(bc, "lan 1: khong co su kien ket_thuc"), bc["loi"]
    assert not bc["muc_8"]["nhat_ky_ket_thuc_hoan_tat"]["dat"]


def test_thieu_du_lieu_tai_nguyen():
    def sua(spec, goc):
        spec["giam_sat"][0]["ram"] = None
        spec["giam_sat"][1]["ram"] = None
        spec["nvsmi"][0] = None
    goc, _ = tao_fixture("thieu_tai_nguyen", sua)
    bc, sh = chay(goc)
    assert bc["dat"], bc["loi"]  # thiếu tài nguyên được ghi rõ, không làm hỏng mục 8
    assert any("lan 0: thieu CSV nvidia-smi" in x for x in bc["thieu_du_lieu"]), bc["thieu_du_lieu"]
    assert any("lan 1: khong co mau RAM" in x for x in bc["thieu_du_lieu"]), bc["thieu_du_lieu"]
    r0, r1, tl = bc["tai_nguyen"]
    assert r0["rss_chinh_max_mib"] is None and r0["ram_may_con_trong_min_mib"] is None and r0["vram_nvidia_smi_max_mib"] is None
    assert r1["vram_nvidia_smi_max_mib"] == 2001.0 and r1["rss_tong_max_mib"] is None
    assert tl["rss_tong_max_mib"] is None and tl["khoang_mat_mau_ram_s"] is None
    assert bc["toan_luot"]["vram_nvidia_smi_max_mib"] == {"gia_tri": 2001.0, "lan": [1]}
    assert r0["thieu"] and "nvidia-smi" in r0["thieu"]
    out = LAN_DIR / "dau_ra_thieu_tai_nguyen"
    bg.ghi_dau_ra(out, bc, sh)
    with open(out / "tai_nguyen.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["rss_chinh_max_mib"] == "" and rows[0]["vram_nvidia_smi_max_mib"] == "", rows[0]
    assert all(rows[i]["rss_tong_max_mib"] != "0" for i in range(3))


def test_results_csv_lech():
    def sua(spec, goc):
        spec["results"][1]["metrics/mAP50(B)"] = 0.99
    goc, _ = tao_fixture("results_lech", sua)
    bc, _ = chay(goc)
    assert co_loi(bc, "results.csv lech nhat ky"), bc["loi"]


def test_best_lech_ky_vong():
    goc, _ = tao_fixture("best_lech")
    bc, _ = chay(goc, {"best_epoch": 2})
    assert co_loi(bc, "best e3 != ky vong e2"), bc["loi"]


def test_thieu_dong_DAT():
    def sua(spec, goc):
        spec["giam_sat"][1]["log"] = "khong co dong tong ket\n# exit=0\n"
    goc, _ = tao_fixture("thieu_dat", sua)
    bc, _ = chay(goc)
    assert not bc["muc_8"]["runner_DAT_loi_None_callback_loss"]["dat"]
    assert co_loi(bc, "muc 8 khong dat: runner_DAT"), bc["loi"]


def test_unpickle_khong_goi_ham_la():
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    p = FIX_DIR / "ck_nguy_hiem.pt"
    ghi_ckpt(p, {"epoch": 1, "optimizer": _ReduceNguyHiem(), "date": "x"})
    md = bg.doc_metadata_checkpoint(p)
    assert md["doc_duoc"] and md["epoch"] == 1 and md["co_optimizer"] is True
    with zipfile.ZipFile(p) as zf:
        obj = bg._UnpickleHanChe(io.BytesIO(zf.read("arch/data.pkl"))).load()
    assert isinstance(obj["optimizer"], bg._Gia), type(obj["optimizer"])  # os.getcwd không được gọi
    p2 = FIX_DIR / "khong_phai_zip.pt"
    p2.write_bytes(b"khong phai zip")
    assert bg.doc_metadata_checkpoint(p2)["doc_duoc"] is False


def test_chay_lai_khong_xoa_fixture_cu():
    """Mô phỏng hai lần chạy test liên tiếp: lần sau không xóa/ghi đè fixture của lần trước."""
    global FIX_DIR
    fix_goc = FIX_DIR
    con = LAN_DIR / "mo_phong_chay_lai"
    a, b = tao_thu_muc_lan(con), tao_thu_muc_lan(con)
    assert a != b
    try:
        FIX_DIR = a / "fixture"
        goc_a, _ = tao_fixture("hop_le")
        h_a = chup_hash(goc_a)
        FIX_DIR = b / "fixture"
        tao_fixture("hop_le")
        assert chup_hash(goc_a) == h_a
        # cùng tên trong cùng lần chạy -> từ chối, không xóa
        try:
            tao_fixture("hop_le")
            raise AssertionError("tao_fixture ghi de fixture dang co")
        except FileExistsError:
            pass
        assert chup_hash(goc_a) == h_a
    finally:
        FIX_DIR = fix_goc


def test_out_ngoai_vung_bi_tu_choi():
    tu_choi = [R53_DIR, R53_DIR / "chuan_bi", R53_DIR / "smoke", R53_DIR / "full" / "detrac_test" / "yolov8n_mixed5",
               R53_DIR / "ban_giao_khac", R53_DIR / "tests_handoff_x", bg.OUT_MAC_DINH / ".." / "smoke",
               TEST_DIR / ".." / "chuan_bi", bg.REPO / "results", bg.REPO.parent / "_r53_ngoai_repo"]
    for p in tu_choi:
        try:
            bg.kiem_out(p)
            raise AssertionError(f"kiem_out chap nhan {p}")
        except ValueError:
            pass
    for p in (bg.OUT_MAC_DINH, bg.OUT_MAC_DINH / "x", TEST_DIR, LAN_DIR / "dau_ra", bg.OUT_MAC_DINH / "a" / ".." / "b"):
        assert bg.kiem_out(p) == Path(p).resolve(), p
    # ghi_dau_ra tự kiểm cổng trước mkdir
    p = R53_DIR / "smoke" / f"_kiem_F2_ghi_dau_ra_{LAN_DIR.name}"
    try:
        bg.ghi_dau_ra(p, {"model": MODEL}, None)
        raise AssertionError("ghi_dau_ra chap nhan out ngoai vung")
    except ValueError:
        pass
    assert not p.exists()
    # CLI: exit 2 trước khi tổng hợp hay ghi; không tạo thư mục/file nào
    ten_goc_truoc = sorted(x.name for x in R53_DIR.iterdir())
    for p in (R53_DIR / "smoke" / f"_kiem_F2_{LAN_DIR.name}", R53_DIR / "chuan_bi" / f"_kiem_F2_{LAN_DIR.name}",
              R53_DIR / "full" / f"_kiem_F2_{LAN_DIR.name}", R53_DIR):
        r = subprocess.run([sys.executable, str(Path(bg.__file__)), "--model", MODEL, "--out", str(p)],
                           cwd=str(bg.REPO), capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert r.returncode == 2, (p, r.returncode, r.stdout, r.stderr)
        assert "nam ngoai vung cho phep" in r.stderr and r.stdout == "", (r.stdout, r.stderr)
        if p != R53_DIR:
            assert not p.exists(), p
    assert sorted(x.name for x in R53_DIR.iterdir()) == ten_goc_truoc
    assert not (R53_DIR / f"ban_giao_{MODEL}.json").exists()


def test_giu_bang_chung_luot_truoc():
    """Chạy cuối: mọi file có trong tests_handoff/ trước lần chạy này còn nguyên byte; file mới chỉ nằm trong LAN_DIR."""
    sau = chup_hash(TEST_DIR)
    mat = sorted(set(ANH_TRUOC) - set(sau))
    doi = sorted(k for k in ANH_TRUOC if k in sau and sau[k] != ANH_TRUOC[k])
    assert not mat and not doi, {"mat": mat[:10], "doi": doi[:10]}
    ngoai = sorted(k for k in set(sau) - set(ANH_TRUOC) if not k.startswith(LAN_DIR.name + "/"))
    assert not ngoai, ngoai[:10]


def main():
    global LAN_DIR, FIX_DIR, ANH_TRUOC
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    ANH_TRUOC = chup_hash(TEST_DIR)
    LAN_DIR = tao_thu_muc_lan(TEST_DIR)
    FIX_DIR = LAN_DIR / "fixture"
    print(f"thu muc lan chay: {LAN_DIR}; {len(ANH_TRUOC)} file co san trong tests_handoff/ (giu nguyen)")
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    tests.sort(key=lambda kv: kv[0] == "test_giu_bang_chung_luot_truoc")  # ổn định: test này chạy cuối
    kq = []
    for ten, fn in tests:
        try:
            fn()
            kq.append({"test": ten, "dat": True})
            print(f"[DAT] {ten}")
        except Exception as e:  # noqa: BLE001
            kq.append({"test": ten, "dat": False, "loi": f"{type(e).__name__}: {e}"})
            print(f"[KHONG DAT] {ten}: {type(e).__name__}: {e}")
            traceback.print_exc()
    so_dat = sum(r["dat"] for r in kq)
    p_kq = LAN_DIR / "ket_qua_test.json"
    with open(p_kq, "x", encoding="utf-8", newline="\n") as f:  # "x": không ghi đè
        f.write(json.dumps(
            {"script": "src/r53_detector/test_ban_giao_train.py",
             "ban_giao_train_sha256": hashlib.sha256(Path(bg.__file__).read_bytes()).hexdigest(),
             "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "luc": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "thu_muc_lan": LAN_DIR.relative_to(bg.REPO).as_posix(),
             "so_file_co_truoc_lan_chay": len(ANH_TRUOC),
             "sha256_anh_chup_truoc": hashlib.sha256(json.dumps(ANH_TRUOC, sort_keys=True).encode()).hexdigest(),
             "so_test": len(kq), "so_dat": so_dat, "ket_qua": kq}, ensure_ascii=False, indent=1) + "\n")
    print(f"{so_dat}/{len(kq)} test dat -> {p_kq}")
    return 0 if so_dat == len(kq) else 1


if __name__ == "__main__":
    sys.exit(main())
