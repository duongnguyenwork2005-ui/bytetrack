"""
test_danh_gia_detector_detrac_test.py -- Test adapter UA-DETRAC test (ke hoach R47 moc B).

Chay tren FIXTURE TU TAO trong thu muc tam (2 video, 5 anh JPEG 960x540, XML nho, manifest); fingerprint cua fixture dung
CHINH script moc A `src/fingerprint_detrac_test.py` (cung dinh dang voi du lieu that). Du doan GIA LAP: thay
`du_doan_ultralytics` cua adapter bang ham tra du doan dinh san -- khong nap model, khong GPU. Cac ca D1-D5, D3, D4, B,
D5b, L, N di qua entry point that `main(argv)`; ca D1-D5/D3/D4 doc lai ca JSON lan CSV. Ca X va ca "lop ngoai tap co GT"
goi ham truc tiep (`doc_gt_xml`, `chuan_hoa_d3`). Gioi han (review Codex 2026-10-10): test cat khung kiem bo dem va so
GT/FN, chua assert toa do sau cat; test D5 kiem tong TP/FP o IoU 0,5, chua kiem rieng tung detection hay nguong IoU khac;
nhanh luot cu (chi co ket_qua.json) chi kiem duong thanh cong; chua co test tu choi cau hinh du doan khac.

Nhom kiem:
  D1  box van -> vung bo qua cho moi lop: detection car trung box van khong thanh FP; van khong vao n_gt car.
  D2  box others -> vung bo qua: detection truck trung others khong thanh FP; truck khong co GT.
  D3  chi car, bus co AP; 'mAP (car, bus)'; truck / motorcycle / bicycle = N/A o JSON va CSV; so lieu tho o truong/file rieng;
      lop ngoai tap ma co GT -> loi.
  D4  anh khong co <frame> bi loai, duoc liet ke, khong duoc dua di du doan.
  D5  detection chua ghep nam trong ignored_region (IoA >= nguong) bi bo qua; GT car nam trong ignored_region van duoc cham.
  K   kiem truoc du doan: sai sha256 checkpoint, anh doi noi dung, XML doi, sai kich thuoc ky vong -> exit 2, khong du doan,
      khong tao thu muc ket qua; thu muc ket qua da co -> exit 1; --chi-kiem khong ghi.
  B   bien cat khung: box mot phan ngoai khung van la GT (dem bi cat); box < 1 px sau cat bi bo (dem rieng).
  D5b bien IoA: detection chua ghep co IoA = 0,50 voi vung bo qua bi bo qua, IoA = 0,49 la FP (o diem van hanh IoU 0,5).
  L   danh sach anh co thu tu luu cung du doan; --cham-lai cho cung so lieu; danh sach bi sua thu tu -> exit 2.
  N   (sua sau review Codex 2026-10-10) nguon_du_doan_<ten>.json ghi truoc khi cham: scorer loi van cham lai duoc ma khong
      du doan lai; du lieu hien tai khac du lieu luc du doan (anh doi + fingerprint dung lai) -> exit 2; NPZ bi doi -> exit 2;
      luot cua ban dau (chi co ket_qua.json) van cham lai duoc va van kiem danh tinh du lieu.
  X   vehicle_type la -> loi doc XML.

    python src/test_danh_gia_detector_detrac_test.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import danh_gia_detector_detrac_test as dg  # noqa: E402

W, H = 960, 540
VIDEOS = ["MVI_00001", "MVI_00002"]
# (video, so anh tren dia, {frame: [(loai, left, top, width, height)]}, ignored_region)
XML = {
    "MVI_00001": (3, {1: [("car", 20, 20, 100, 100), ("bus", 200, 200, 120, 100), ("van", 300, 20, 80, 80)],
                      2: [("others", 20, 200, 100, 100), ("car", 440, 20, 80, 80)]},   # frame 3: KHONG co -> D4
                  [(400, 0, 240, 120)]),
    "MVI_00002": (2, {1: [("car", 50, 50, 100, 100)],
                      2: [("car", -20, 250, 60, 60),        # bien: cat vao khung -> GT (0, 250, 40, 310), dem bi cat
                          ("car", 959.5, 10, 5, 5),         # bien: < 1 px sau cat -> bo, dem rieng
                          ("van", 600, 300, 100, 100)]}, []),   # D5 bien IoA (xem DU_DOAN)
}
# du doan gia lap theo rel: (xyxy, conf, cls)  -- cls: 0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle
DU_DOAN = {
    "MVI_00001/img00001.jpg": [((20, 20, 120, 120), 0.9, 0),      # TP car
                               ((300, 20, 380, 100), 0.8, 0),     # D1: car trung box van -> bo qua
                               ((200, 200, 320, 300), 0.9, 2)],   # TP bus
    "MVI_00001/img00002.jpg": [((20, 200, 120, 300), 0.7, 1),     # D2: truck trung box others -> bo qua
                               ((440, 20, 520, 100), 0.95, 0),    # D5: GT car trong ignored_region van duoc cham -> TP
                               ((560, 20, 620, 80), 0.6, 0)],     # D5: car chua ghep, nam trong ignored_region -> bo qua
    "MVI_00002/img00001.jpg": [((50, 50, 150, 150), 0.9, 0),      # TP car
                               ((400, 200, 500, 300), 0.5, 0)],   # FP car (khong vung bo qua)
    "MVI_00002/img00002.jpg": [((100, 100, 200, 200), 0.6, 1),    # truck (lop khong GT) -> chi vao so lieu tho
                               ((550, 300, 650, 400), 0.4, 0),    # D5 bien: IoA voi box van = 0,50 >= 0,5 -> bo qua
                               ((549, 300, 649, 400), 0.4, 0)],   # D5 bien: IoA = 0,49 < 0,5 -> FP
}


def tao_fixture(goc: Path) -> dict:
    from PIL import Image
    anh, xml = goc / "anh", goc / "xml"
    for v, (n, frames, ign) in XML.items():
        (anh / v).mkdir(parents=True)
        for i in range(1, n + 1):
            Image.new("RGB", (W, H), (i * 40 % 256, 80, 120)).save(anh / v / f"img{i:05d}.jpg", quality=60)
        dong = [f'<?xml version="1.0" encoding="utf-8"?>', f'<sequence name="{v}">',
                '<sequence_attribute camera_state="unstable" sence_weather="cloudy"/>', '<ignored_region>']
        dong += [f'<box left="{x}" top="{y}" width="{w}" height="{h}"/>' for x, y, w, h in ign]
        dong.append("</ignored_region>")
        for num, tg in frames.items():
            dong.append(f'<frame density="{len(tg)}" num="{num}"><target_list>')
            for k, (loai, x, y, w, h) in enumerate(tg, 1):
                dong.append(f'<target id="{k}"><box left="{x}" top="{y}" width="{w}" height="{h}"/>'
                            f'<attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" vehicle_type="{loai}"/></target>')
            dong.append("</target_list></frame>")
        dong.append("</sequence>")
        xml.mkdir(exist_ok=True)
        (xml / f"{v}.xml").write_text("\n".join(dong) + "\n", encoding="utf-8")
    mf = goc / "manifest.json"
    mf.write_text(json.dumps({"name": "fixture", "videos": VIDEOS, "n_videos": len(VIDEOS)}) + "\n", encoding="utf-8")
    fp = goc / "fp"
    p = subprocess.run([sys.executable, str(HERE / "fingerprint_detrac_test.py"), "--manifest", str(mf), "--anh", str(anh),
                        "--xml", str(xml), "--out", str(fp), "--doi-chieu-d4", "", "--doi-chieu-theo-video", "",
                        "--so-anh-ky-vong", "5"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"fingerprint fixture exit {p.returncode}: {p.stderr[-800:]}")
    model = goc / "model.pt"
    model.write_bytes(b"checkpoint gia lap")
    return {"anh": anh, "xml": xml, "manifest": mf, "fp": fp, "model": model, "sha": dg.sha256_file(model)}


class Gia:
    """Thay `du_doan_ultralytics`: ghi lai anh duoc yeu cau, tra du doan dinh san."""
    def __init__(self):
        self.goi = []

    def __call__(self, weights, anh, qt, batch=16, device="0"):
        self.goi.append([f"{p.parent.name}/{p.name}" for p in anh])
        ra = []
        for p in anh:
            ds = DU_DOAN.get(f"{p.parent.name}/{p.name}", [])
            ra.append({"xyxy": np.array([d[0] for d in ds], float).reshape(-1, 4), "conf": np.array([d[1] for d in ds], float),
                       "cls": np.array([d[2] for d in ds], int)})
        return ra, {"gia_lap": True}


def chay(fx: dict, out: Path | None, *them, gia: Gia | None = None, fp: Path | None = None) -> tuple[int, str]:
    fp = fp or fx["fp"]
    argv = ["--xml", str(fx["xml"]), "--anh", str(fx["anh"]), "--manifest", str(fx["manifest"]),
            "--fingerprint", str(fp / "detrac_test_fingerprint.json"), "--hash-anh", str(fp / "detrac_test_sha256.csv"),
            "--loai-d4", str(fp / "detrac_test_loai_D4.csv"), *them]
    if out is not None:
        argv += ["--out", str(out)]
    goc = dg.du_doan_ultralytics
    dg.du_doan_ultralytics = gia or Gia()
    e = io.StringIO()
    try:
        with contextlib.redirect_stderr(e), contextlib.redirect_stdout(io.StringIO()):
            ma = dg.main(argv)
    finally:
        dg.du_doan_ultralytics = goc
    return ma, e.getvalue()


def doc_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =========================================================================== cac ca
def ca_d1_d5_d3_d4_qua_entry_point(fx, tmp):
    gia = Gia()
    out = tmp / "luot1"
    ma, err = chay(fx, out, "--model", f"m={fx['model']}", "--sha256", f"m={fx['sha']}", gia=gia)
    assert ma == 0, (ma, err)
    # D4: anh khong co <frame> khong duoc dua di du doan va duoc liet ke
    assert gia.goi == [["MVI_00001/img00001.jpg", "MVI_00001/img00002.jpg", "MVI_00002/img00001.jpg", "MVI_00002/img00002.jpg"]], gia.goi
    kq = json.loads((out / "ket_qua.json").read_text(encoding="utf-8"))
    dt_ = kq["dataset"]["dataset_da_xac_minh"]
    assert dt_["loai_D4"] == ["MVI_00001/img00003.jpg"], dt_["loai_D4"]
    # D1: van khong vao car; bien cat khung: box cat vao khung van la GT, box < 1 px sau cat bi bo (dem rieng)
    assert dt_["dem"]["gt_theo_lop"] == {"car": 4, "truck": 0, "bus": 1, "motorcycle": 0, "bicycle": 0}, dt_["dem"]
    assert (dt_["dem"]["box_gt_bi_cat_vao_khung"], dt_["dem"]["box_gt_nho_hon_1px_sau_cat"]) == (1, 1), dt_["dem"]
    assert dt_["dem"]["vung_bo_qua_tu_van"] == 2 and dt_["dem"]["vung_bo_qua_tu_others"] == 1
    d3 = kq["mo_hinh"]["m"]["ket_qua_D3"]
    car, bus = d3["theo_lop"]["car"], d3["theo_lop"]["bus"]
    # D1 + D5: TP car 3 (ke ca GT trong ignored_region); FP car 2 = det thua o MVI_00002/1 + det IoA 0,49 (bien D5);
    # det trung van, det chua ghep trong ignored_region va det IoA 0,50 bi bo qua; GT bi cat khung khong co det -> FN
    assert (car["van_hanh"]["TP"], car["van_hanh"]["FP"], car["van_hanh"]["FN"], car["n_gt"]) == (3, 2, 1, 4), car["van_hanh"]
    assert (bus["van_hanh"]["TP"], bus["van_hanh"]["FP"], bus["n_gt"]) == (1, 0, 1), bus["van_hanh"]
    # D3: N/A cho lop ngoai tap; so lieu tho rieng (D2: truck trung others bi bo qua -> FP tho = 1, tu anh khong GT)
    assert d3["lop_danh_gia"] == ["car", "bus"]
    assert all(d3["theo_lop"][x] == "N/A" for x in ("truck", "motorcycle", "bicycle"))
    tho = d3["so_lieu_tho_khong_phai_metric"]
    assert "khong phai metric" in tho["ghi_chu"] and tho["theo_lop"]["truck"]["van_hanh"]["FP"] == 1
    m = d3["mAP (car, bus)"]
    assert abs(m["mAP50"] - (car["AP50"] + bus["AP50"]) / 2) < 1e-12
    # CSV
    rows = doc_csv(out / "theo_lop.csv")
    tr = next(r for r in rows if r["lop"] == "truck")
    assert tr["AP50"] == "N/A" and tr["n_gt"] == "N/A", tr
    assert next(r for r in rows if r["lop"] == "car")["n_gt"] == "4"
    tong = doc_csv(out / "tong_hop.csv")
    assert tong[0]["chi_so"] == "mAP (car, bus)" and abs(float(tong[0]["mAP50"]) - m["mAP50"]) < 1e-9
    th = doc_csv(out / "so_lieu_tho_khong_phai_metric.csv")
    assert {r["lop"] for r in th} == {"truck", "motorcycle", "bicycle"} and all(r["ghi_chu"] == "tho, khong phai metric" for r in th)
    # L: danh sach anh co thu tu luu cung du doan
    ds = json.loads((out / "danh_sach_anh.json").read_text(encoding="utf-8"))
    assert ds["rel"] == gia.goi[0] and (out / "du_doan_m.npz").is_file()
    return out


def ca_cham_lai_cung_so_va_phat_hien_danh_sach_bi_sua(fx, tmp):
    out = ca_d1_d5_d3_d4_qua_entry_point(fx, tmp / "nguon")
    gia = Gia()
    ma, err = chay(fx, tmp / "cham_lai", "--cham-lai", str(out), gia=gia)
    assert ma == 0 and gia.goi == [], (ma, err, gia.goi)          # khong du doan lai
    a = json.loads((out / "ket_qua.json").read_text(encoding="utf-8"))["mo_hinh"]["m"]["ket_qua_D3"]
    b = json.loads((tmp / "cham_lai" / "ket_qua.json").read_text(encoding="utf-8"))["mo_hinh"]["m"]["ket_qua_D3"]
    assert a == b
    ds = json.loads((out / "danh_sach_anh.json").read_text(encoding="utf-8"))
    ds["rel"] = list(reversed(ds["rel"]))
    (out / "danh_sach_anh.json").write_text(json.dumps(ds), encoding="utf-8")
    ma, err = chay(fx, tmp / "cham_lai_2", "--cham-lai", str(out))
    assert ma == 2 and "danh sach anh" in err, (ma, err)


def dung_fingerprint(fx, out: Path) -> Path:
    p = subprocess.run([sys.executable, str(HERE / "fingerprint_detrac_test.py"), "--manifest", str(fx["manifest"]), "--anh",
                        str(fx["anh"]), "--xml", str(fx["xml"]), "--out", str(out), "--doi-chieu-d4", "",
                        "--doi-chieu-theo-video", "", "--so-anh-ky-vong", "5"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    return out


def ca_cham_lai_tu_choi_du_lieu_khac(fx, tmp):
    """Codex R47 issue 1: anh doi noi dung + fingerprint dung lai (cung rel/thu tu/kich thuoc) -> khong cham NPZ cu."""
    from PIL import Image
    out = ca_d1_d5_d3_d4_qua_entry_point(fx, tmp / "nguon")
    p = fx["anh"] / "MVI_00002" / "img00002.jpg"
    goc = p.read_bytes()
    try:
        Image.new("RGB", (W, H), (7, 7, 7)).save(p, quality=60)
        fp2 = dung_fingerprint(fx, tmp / "fp2")
        gia = Gia()
        ma, err = chay(fx, tmp / "cham", "--cham-lai", str(out), gia=gia, fp=fp2)
        assert ma == 2 and "du lieu hien tai khac" in err and not (tmp / "cham").exists() and gia.goi == [], (ma, err)
    finally:
        p.write_bytes(goc)


def ca_phuc_hoi_khi_buoc_cham_loi(fx, tmp):
    """Codex R47 issue 2: scorer loi SAU khi NPZ da luu -> --cham-lai van cham duoc, khong du doan lai."""
    goc = dg.danh_gia

    def hong(*a, **k):
        raise RuntimeError("scorer gia lap loi")
    dg.danh_gia = hong
    try:
        try:
            chay(fx, tmp / "luot", "--model", f"m={fx['model']}", "--sha256", f"m={fx['sha']}")
        except RuntimeError:
            pass
        else:
            raise AssertionError("scorer loi phai lam luot dung")
    finally:
        dg.danh_gia = goc
    luot = tmp / "luot"
    assert (luot / "du_doan_m.npz").is_file() and (luot / "nguon_du_doan_m.json").is_file() and not (luot / "ket_qua.json").exists()
    gia = Gia()
    ma, err = chay(fx, tmp / "cham", "--cham-lai", str(luot), gia=gia)
    assert ma == 0 and gia.goi == [], (ma, err)
    d3 = json.loads((tmp / "cham" / "ket_qua.json").read_text(encoding="utf-8"))["mo_hinh"]["m"]["ket_qua_D3"]
    assert d3["theo_lop"]["car"]["van_hanh"]["TP"] == 3


def ca_cham_lai_tu_choi_npz_bi_doi(fx, tmp):
    out = ca_d1_d5_d3_d4_qua_entry_point(fx, tmp / "nguon")
    npz = out / "du_doan_m.npz"
    npz.write_bytes(npz.read_bytes() + b"x")
    ma, err = chay(fx, tmp / "cham", "--cham-lai", str(out))
    assert ma == 2 and "da doi sau khi luu" in err, (ma, err)


def ca_cham_lai_luot_ban_dau_khong_co_nguon(fx, tmp):
    """Luot chay bang ban dau (chi co ket_qua.json): van cham lai duoc, van kiem danh tinh du lieu."""
    out = ca_d1_d5_d3_d4_qua_entry_point(fx, tmp / "nguon")
    (out / "nguon_du_doan_m.json").unlink()
    gia = Gia()
    ma, err = chay(fx, tmp / "cham", "--cham-lai", str(out), gia=gia)
    assert ma == 0 and gia.goi == [], (ma, err)
    kq = json.loads((tmp / "cham" / "ket_qua.json").read_text(encoding="utf-8"))
    assert kq["mo_hinh"]["m"]["nguon_du_doan"]["tu_ket_qua_json"] is True


def ca_sai_sha256_checkpoint(fx, tmp):
    gia = Gia()
    ma, err = chay(fx, tmp / "x", "--model", f"m={fx['model']}", "--sha256", "m=" + "0" * 64, gia=gia)
    assert ma == 2 and gia.goi == [] and not (tmp / "x").exists(), (ma, err)


def ca_anh_doi_noi_dung(fx, tmp):
    p = fx["anh"] / "MVI_00002" / "img00001.jpg"
    goc = p.read_bytes()
    try:
        p.write_bytes(goc[:-2] + bytes([goc[-2] ^ 1]) + goc[-1:])
        gia = Gia()
        ma, err = chay(fx, tmp / "x", "--model", f"m={fx['model']}", "--sha256", f"m={fx['sha']}", gia=gia)
        assert ma == 2 and "noi dung anh" in err and gia.goi == [] and not (tmp / "x").exists(), (ma, err)
    finally:
        p.write_bytes(goc)


def ca_xml_doi(fx, tmp):
    p = fx["xml"] / "MVI_00001.xml"
    goc = p.read_bytes()
    try:
        p.write_bytes(goc.replace(b'vehicle_type="van"', b'vehicle_type="car"'))
        ma, err = chay(fx, tmp / "x", "--model", f"m={fx['model']}", "--sha256", f"m={fx['sha']}")
        assert ma == 2 and "khac fingerprint" in err, (ma, err)
    finally:
        p.write_bytes(goc)


def ca_sai_kich_thuoc_ky_vong(fx, tmp):
    ma, err = chay(fx, tmp / "x", "--model", f"m={fx['model']}", "--sha256", f"m={fx['sha']}", "--kich-thuoc", "1280x720")
    assert ma == 2 and "kich thuoc" in err, (ma, err)


def ca_thu_muc_ket_qua_da_co(fx, tmp):
    (tmp / "co").mkdir()
    ma, _ = chay(fx, tmp / "co", "--model", f"m={fx['model']}", "--sha256", f"m={fx['sha']}")
    assert ma == 1


def ca_chi_kiem_khong_ghi(fx, tmp):
    gia = Gia()
    ma, err = chay(fx, None, "--chi-kiem", gia=gia)
    assert ma == 0 and gia.goi == [], (ma, err)


def ca_vehicle_type_la(fx, tmp):
    p = tmp / "la.xml"
    p.write_text('<sequence name="x"><frame num="1"><target_list><target id="1"><box left="1" top="1" width="5" height="5"/>'
                 '<attribute vehicle_type="truck"/></target></target_list></frame></sequence>', encoding="utf-8")
    try:
        dg.doc_gt_xml(p)
    except dg.LoiDuLieu as e:
        assert "truck" in str(e)
    else:
        raise AssertionError("vehicle_type la phai bi tu choi")


def ca_d3_lop_ngoai_tap_co_gt_la_loi(fx, tmp):
    tl = {c: {"n_gt": 0, "AP50": float("nan"), "AP75": float("nan"), "AP50_95": float("nan"),
              "van_hanh": {"precision": float("nan"), "recall": float("nan")}} for c in dg.LOP_5}
    tl["car"] = dict(tl["car"], n_gt=5, AP50=0.5, AP75=0.4, AP50_95=0.3, van_hanh={"precision": 0.5, "recall": 0.5})
    tl["bus"] = dict(tl["car"])
    tl["truck"] = dict(tl["car"])
    try:
        dg.chuan_hoa_d3({"theo_lop": tl, "macro": {"lop_khong_co_gt": [], "mAP50": 0.5, "mAP50_95": 0.3}, "n_anh": 1, "van_hanh": {}})
    except dg.LoiDuLieu as e:
        assert "truck" in str(e)
    else:
        raise AssertionError("lop ngoai tap D3 co GT phai la loi")


CAC_CA = [ca_d1_d5_d3_d4_qua_entry_point, ca_cham_lai_cung_so_va_phat_hien_danh_sach_bi_sua, ca_cham_lai_tu_choi_du_lieu_khac,
          ca_phuc_hoi_khi_buoc_cham_loi, ca_cham_lai_tu_choi_npz_bi_doi, ca_cham_lai_luot_ban_dau_khong_co_nguon, ca_sai_sha256_checkpoint,
          ca_anh_doi_noi_dung, ca_xml_doi, ca_sai_kich_thuoc_ky_vong, ca_thu_muc_ket_qua_da_co, ca_chi_kiem_khong_ghi,
          ca_vehicle_type_la, ca_d3_lop_ngoai_tap_co_gt_la_loi]


def main() -> int:
    dat = 0
    with tempfile.TemporaryDirectory() as td:
        fx = tao_fixture(Path(td) / "fx")
        for i, ca in enumerate(CAC_CA):
            tmp = Path(td) / f"ca{i}"
            tmp.mkdir()
            try:
                ca(fx, tmp)
                dat += 1
                print(f"PASS {ca.__name__}")
            except Exception:  # noqa: BLE001
                print(f"FAIL {ca.__name__}\n{traceback.format_exc()}")
    print(f"{dat}/{len(CAC_CA)}")
    return 0 if dat == len(CAC_CA) else 1


if __name__ == "__main__":
    raise SystemExit(main())
