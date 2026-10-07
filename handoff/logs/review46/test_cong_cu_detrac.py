# -*- coding: utf-8 -*-
"""Vong 46: test cho cong cu phan tich UA-DETRAC (R45-F3, R45-F4) + doi chieu hanh vi evaluator ma tai lieu D3 / D5 phai
mo ta dung (R45-F1, R45-F2). CPU, fixture tong hop trong thu muc tam; KHONG doc dataset that, KHONG nap model.

    python handoff/logs/review46/test_cong_cu_detrac.py [--cong-cu-dir <thu muc chua phan_tich_detrac_test.py +
        dem_theo_phuong_an.py>] [--src <thu muc chua eval_detector_5lop.py>]

Mac dinh --cong-cu-dir = thu muc cua file nay (ban SUA vong 46); chay voi handoff/logs/review45 (ban cu) de chung minh
test bat loi cu. Nhom:
  X  (R45-F4, doc XML) X1 fixture cua reviewer (sequence_attribute tu dong, mot dong); X2 fixture SAT cau truc XML that
     (khai bao xml, thut le, ignored_region, frame / target / attribute / occlusion); X3 XML KHONG co sequence_attribute;
     X4 qua `main()` (repo gia + JPEG nho): cot camera_state / weather cua theo_video.csv = gia tri XML, hoac ghi ro
     "khong co phan tu" / "khong co thuoc tinh" -- khong de trong, khong dien gia tri suy doan.
  D  (R45-F3, dem vung) D1 fixture cua reviewer (1 video, 2 anh, 1 <frame>, 2 vung tinh): giu -> 4 luot, loai -> 2;
     D2 hai video + box van / others: moi phuong an D1 x D2 x D4 = cong thuc tinh tay.
  E  (DOI CHIEU HANH VI evaluator, khong phai sua ma) E1 fixture D3 cua reviewer: lop khong GT van co FP = 1, precision = 0,
     AP = null; macro chi 2 lop. E2 phan vi du D5: bo GT nam trong vung ignore co the lam AP50 GIAM (0,5050 -> 0)."""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path

DAY = Path(__file__).resolve().parent
REPO = DAY.parents[2]
_KQ: list[tuple[str, bool]] = []


def kiem(ten: str, ok: bool, chi_tiet: str = "") -> None:
    _KQ.append((ten, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {ten}" + (f"  ({chi_tiet})" if chi_tiet else ""))


def nap(ten: str, p: Path):
    spec = importlib.util.spec_from_file_location(ten, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def chay_main(m, argv: list[str]) -> int:
    cu = sys.argv[:]
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return m.main()
    finally:
        sys.argv = cu


XML_REVIEWER = ('<sequence><sequence_attribute camera_state="stable" sence_weather="cloudy"/>'
                '<ignored_region><box left="1" top="1" width="10" height="10"/>'
                '<box left="20" top="20" width="10" height="10"/></ignored_region>'
                '<frame num="1" density="1"><target_list><target id="1">'
                '<box left="40" top="40" width="10" height="10"/>'
                '<attribute vehicle_type="car"/></target></target_list></frame></sequence>')

XML_THAT = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="{ten}">
   {seq_attr}
   <ignored_region>
      <box left="335.75" top="52.75" width="256.5" height="117.5"/>
      <box left="0.5" top="296.75" width="223.75" height="120.5"/>
      <box left="690.75" top="116.75" width="269.75" height="94.5"/>
   </ignored_region>
   <frame density="2" num="1">
      <target_list>
         <target id="1">
            <box left="745.6" top="357.33" width="148.2" height="115.14"/>
            <attribute orientation="222.06" speed="11.52" trajectory_length="54" truncation_ratio="0" vehicle_type="car"/>
         </target>
         <target id="2">
            <box left="100.5" top="200.25" width="60" height="40"/>
            <attribute orientation="10" speed="3" trajectory_length="20" truncation_ratio="0.1" vehicle_type="van"/>
            <occlusion>
               <region_overlap left="120" top="210" width="10" height="5" occlusion_id="1" occlusion_status="0"/>
            </occlusion>
         </target>
      </target_list>
   </frame>
   <frame density="1" num="3">
      <target_list>
         <target id="3">
            <box left="400" top="300" width="80" height="60"/>
            <attribute orientation="0" speed="0" trajectory_length="5" truncation_ratio="0" vehicle_type="others"/>
         </target>
      </target_list>
   </frame>
</sequence>
"""


def nhom_x(cc: Path, td: Path) -> None:
    pt = nap("pt_detrac", cc / "phan_tich_detrac_test.py")
    p1 = td / "x1.xml"
    p1.write_text(XML_REVIEWER, encoding="utf-8")
    a1 = pt.doc_xml(p1)["attr"]
    kiem("X1 fixture reviewer: sequence_attribute tu dong -> doc du camera_state / sence_weather",
         a1 == {"camera_state": "stable", "sence_weather": "cloudy"}, f"attr {a1}")
    p2 = td / "x2.xml"
    p2.write_text(XML_THAT.format(ten="MVI_X2", seq_attr='<sequence_attribute camera_state="unstable" '
                                                     'sence_weather="sunny"/>'), encoding="utf-8")
    d2 = pt.doc_xml(p2)
    loai = sorted(t["type"] for f in d2["frames"].values() for t in f["targets"])
    kiem("X2 fixture sat XML that: thuoc tinh + 3 vung + 2 frame + loai box doc dung",
         d2["attr"] == {"camera_state": "unstable", "sence_weather": "sunny"} and len(d2["ign"]) == 3
         and sorted(d2["frames"]) == [1, 3] and loai == ["car", "others", "van"], f"attr {d2['attr']}, loai {loai}")
    p3 = td / "x3.xml"
    p3.write_text(XML_THAT.format(ten="MVI_X3", seq_attr=""), encoding="utf-8")
    d3 = pt.doc_xml(p3)
    kiem("X3 XML KHONG co sequence_attribute -> attr rong VA co danh dau 'khong co phan tu'",
         d3["attr"] == {} and d3.get("co_sequence_attribute") is False, f"attr {d3['attr']}, "
                                                                        f"co_sequence_attribute {d3.get('co_sequence_attribute')}")
    # X4: qua main() tren repo gia
    goc = td / "repo_x"
    xd = goc / "data/extracted/DETRAC-Test-Annotations-XML"
    xd.mkdir(parents=True)
    (xd / "MVI_A.xml").write_text(XML_THAT.format(ten="MVI_A", seq_attr='<sequence_attribute camera_state="unstable" '
                                                                      'sence_weather="sunny"/>'), encoding="utf-8")
    (xd / "MVI_B.xml").write_text(XML_THAT.format(ten="MVI_B", seq_attr=""), encoding="utf-8")
    (xd / "MVI_C.xml").write_text(XML_THAT.format(ten="MVI_C", seq_attr='<sequence_attribute camera_state="stable"/>'),
                                  encoding="utf-8")
    from PIL import Image
    for v in ("MVI_A", "MVI_B", "MVI_C"):
        d = goc / "data/extracted/DETRAC-Images" / v
        d.mkdir(parents=True)
        for i in (1, 2, 3):
            Image.new("RGB", (96, 54), (i * 40, 80, 120)).save(d / f"img{i:05d}.jpg")
    mf = goc / "results/phase2/manifests/detrac_test.json"
    mf.parent.mkdir(parents=True)
    mf.write_text(json.dumps({"videos": ["MVI_A", "MVI_B", "MVI_C"]}), encoding="utf-8")
    pt.REPO = goc
    pt.XML_DIR, pt.IMG_DIR = xd, goc / "data/extracted/DETRAC-Images"
    pt.IGR_DIR, pt.MANIFEST = goc / "khong_co_igrs", mf
    out = td / "out_x4"
    rc = chay_main(pt, ["phan_tich_detrac_test.py", "--out", str(out)])
    hang = {r["video"]: r for r in csv.DictReader(open(out / "theo_video.csv", encoding="utf-8"))} if rc == 0 else {}
    o = {v: (r["camera_state"], r["weather"]) for v, r in hang.items()}
    kiem("X4 main(): cot camera_state / weather = gia tri XML; thieu phan tu / thieu thuoc tinh ghi RO, khong de trong",
         rc == 0 and o.get("MVI_A") == ("unstable", "sunny")
         and all(x and "khong co phan tu" in x for x in o.get("MVI_B", ("", "")))
         and o.get("MVI_C", ("", ""))[0] == "stable" and "khong co thuoc tinh" in o.get("MVI_C", ("", ""))[1],
         f"rc {rc}, {o}")


def nhom_d(cc: Path, td: Path) -> None:
    dm = nap("dem_pa", cc / "dem_theo_phuong_an.py")

    def repo(ten: str, videos: dict) -> Path:
        goc = td / ten
        xd = goc / "data/extracted/DETRAC-Test-Annotations-XML"
        xd.mkdir(parents=True)
        for v, (xml, n_anh) in videos.items():
            (xd / f"{v}.xml").write_text(xml, encoding="utf-8")
            d = goc / "data/extracted/DETRAC-Images" / v
            d.mkdir(parents=True)
            for i in range(1, n_anh + 1):
                (d / f"img{i:05d}.jpg").write_bytes(b"chi dem ten file; khong giai ma")
        mf = goc / "results/phase2/manifests/detrac_test.json"
        mf.parent.mkdir(parents=True)
        mf.write_text(json.dumps({"videos": sorted(videos)}), encoding="utf-8")
        return goc

    # D1: dung fixture reviewer
    goc = repo("repo_d1", {"fixture": (XML_REVIEWER, 2)})
    dm.REPO = goc
    rc = chay_main(dm, ["dem_theo_phuong_an.py", "--out", str(goc / "counts.json")])
    c = json.loads((goc / "counts.json").read_text(encoding="utf-8")) if rc == 0 else {}
    giu, loai = c.get("D1_van=bo_qua|D2_others=bo_qua|D4=giu_173_am_tinh", {}), \
        c.get("D1_van=bo_qua|D2_others=bo_qua|D4=loai_173", {})
    kiem("D1 fixture reviewer (2 anh, 1 <frame>, 2 vung tinh): giu -> 2 anh, 4 luot vung; loai -> 1 anh, 2 luot",
         rc == 0 and giu.get("so_anh_cham") == 2 and giu.get("vung_bo_qua_cong_moi_anh") == 4
         and loai.get("so_anh_cham") == 1 and loai.get("vung_bo_qua_cong_moi_anh") == 2,
         f"rc {rc}, giu {giu.get('so_anh_cham')} anh / {giu.get('vung_bo_qua_cong_moi_anh')} luot, "
         f"loai {loai.get('so_anh_cham')} / {loai.get('vung_bo_qua_cong_moi_anh')}")
    # D2: 2 video. MVI_A: 3 vung tinh, frame 1 (car + van), frame 3 (others), 3 anh -> anh 2 khong <frame>.
    #     MVI_B: 0 vung tinh, cung frame, 4 anh -> anh 2, 4 khong <frame>.
    goc = repo("repo_d2", {"MVI_A": (XML_THAT.format(ten="MVI_A", seq_attr=""), 3),
                           "MVI_B": (XML_THAT.format(ten="MVI_B", seq_attr="").replace(
                               XML_THAT.split("<ignored_region>")[1].split("</ignored_region>")[0], ""), 4)})
    dm.REPO = goc
    rc = chay_main(dm, ["dem_theo_phuong_an.py", "--out", str(goc / "counts.json")])
    c = json.loads((goc / "counts.json").read_text(encoding="utf-8")) if rc == 0 else {}
    # tinh tay: 4 anh co <frame> (2 / video); vung tinh tren anh co frame = 3 + 3 + 0 + 0 = 6; van 2, others 2 (1 / video);
    # anh khong <frame>: MVI_A 1 anh x 3 vung = 3, MVI_B 2 anh x 0 = 0.
    mong = {}
    for pv, van_vung in (("bo_qua", 2), ("car", 0)):
        for po, oth_vung in (("bo_qua", 2), ("truck", 0)):
            for d4, them_anh, them_vung in (("loai_173", 0, 0), ("giu_173_am_tinh", 3, 3)):
                mong[f"D1_van={pv}|D2_others={po}|D4={d4}"] = (4 + them_anh, 6 + van_vung + oth_vung + them_vung)
    thuc = {k: (v.get("so_anh_cham"), v.get("vung_bo_qua_cong_moi_anh")) for k, v in c.items() if k in mong}
    kiem("D2 hai video + van / others: so anh + luot vung moi phuong an D1 x D2 x D4 = tinh tay",
         rc == 0 and thuc == mong, f"rc {rc}; lech: {[(k, thuc.get(k), mong[k]) for k in mong if thuc.get(k) != mong[k]]}")


def nhom_e(src: Path) -> None:
    import numpy as np
    sys.path.insert(0, str(src))
    ev = nap("ev_5lop", src / "eval_detector_5lop.py")

    def gt(boxes, cls, ign=()):
        return {"xyxy": np.asarray(boxes, float).reshape(-1, 4), "cls": np.asarray(cls, int),
                "ign": np.asarray(ign, float).reshape(-1, 4)}

    def pr(boxes, cls, conf):
        return {"xyxy": np.asarray(boxes, float).reshape(-1, 4), "cls": np.asarray(cls, int),
                "conf": np.asarray(conf, float)}
    r = ev._json_safe(ev.danh_gia([gt([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 2])],
                                  [pr([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], [0, 2, 1], [.9, .8, .7])]))
    t = r["theo_lop"]["truck"]
    kiem("E1 (doi chieu) lop khong GT: scorer VAN dem FP = 1, precision = 0, AP = null; macro 2 lop, mAP50 = 1",
         t["AP50"] is None and t["van_hanh"]["FP"] == 1 and t["van_hanh"]["precision"] == 0
         and r["macro"]["so_lop_co_gt"] == 2 and r["macro"]["mAP50"] == 1, json.dumps(t["van_hanh"]))
    ign = [[0, 0, 10, 10]]
    p = pr(ign, [0], [.9])
    truoc = ev._json_safe(ev.danh_gia([gt([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 0], ign)], [p]))["theo_lop"]["car"]
    sau = ev._json_safe(ev.danh_gia([gt([[20, 0, 30, 10]], [0], ign)], [p]))["theo_lop"]["car"]
    kiem("E2 (doi chieu) phan vi du D5: chuyen GT trong vung ignore thanh ignore lam AP50 GIAM (khong tang chac chan)",
         truoc["AP50"] > sau["AP50"] == 0 and truoc["van_hanh"]["recall"] == 0.5 and sau["van_hanh"]["recall"] == 0,
         f"giu GT: AP50 {truoc['AP50']:.10f} R {truoc['van_hanh']['recall']}; chuyen: AP50 {sau['AP50']} "
         f"R {sau['van_hanh']['recall']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cong-cu-dir", default=str(DAY))
    ap.add_argument("--src", default=str(REPO / "src"))
    a = ap.parse_args()
    cc = Path(a.cong_cu_dir).resolve()
    print(f"cong cu: {cc}; evaluator: {Path(a.src).resolve()}")
    with tempfile.TemporaryDirectory(prefix="test_cong_cu_detrac_") as td:
        for ten, fn in (("X", lambda: nhom_x(cc, Path(td))), ("D", lambda: nhom_d(cc, Path(td))),
                        ("E", lambda: nhom_e(Path(a.src).resolve()))):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                kiem(f"nhom {ten}: loi khong mong doi {type(e).__name__}: {e}", False)
    n = sum(ok for _, ok in _KQ)
    print(f"\nKET QUA: {n}/{len(_KQ)} PASS")
    return 0 if n == len(_KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
