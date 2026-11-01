"""Test bo doc log trtexec tren FIXTURE TU SOAN (dinh dang TRT 8.2, so khong phai do that).
Chay: python handoff/logs/review63/cong_cu/tests/test_phan_tich_trtexec.py"""
import ast
import json
import os
import shutil
import sys
import tempfile

DAY = os.path.dirname(os.path.abspath(__file__))
CONG_CU = os.path.dirname(DAY)
sys.path.insert(0, CONG_CU)
import phan_tich_trtexec as pt  # noqa: E402


def test_tom_tat_co_du_thong_ke():
    kq = pt.doc_log(os.path.join(DAY, "fixture_trtexec82_A.log"))
    tk = kq["thong_ke_ms"]
    assert tk["GPU Compute Time"]["median"] == 68.5
    assert tk["GPU Compute Time"]["p90"] == 69.7
    assert tk["Latency"]["mean"] == 72.2
    assert tk["H2D Latency"]["median"] == 2.2 and tk["D2H Latency"]["median"] == 1.3
    assert tk["Enqueue Time"]["median"] == 2.0
    assert "End-to-End Host Latency" in tk
    assert kq["don"]["Throughput"]["gia_tri"] == 14.5012
    assert kq["exit"] == 0 and kq["loi"] == []
    assert kq["lenh"].startswith("FIXTURE")


def test_profile_lop_bo_dong_total():
    kq = pt.doc_log(os.path.join(DAY, "fixture_trtexec82_E.log"))
    assert kq["so_lop_profile"] == 3
    assert kq["lop_top"][0]["lop"].startswith("Conv_3") and kq["lop_top"][0]["tb_ms"] == 5.0
    assert all(x["lop"] != "Total" for x in kq["lop_top"])


def test_export_times_va_main_json():
    tmp = tempfile.mkdtemp()
    try:
        log = os.path.join(tmp, "A_mac_dinh.log")
        shutil.copy(os.path.join(DAY, "fixture_trtexec82_A.log"), log)
        times = [{"startInMs": i * 70.0, "inMs": 2.0, "computeMs": 68.0 + (i % 3), "outMs": 1.0,
                  "latencyMs": 71.0 + (i % 3), "endToEndMs": 137.0} for i in range(10)]
        with open(os.path.join(tmp, "A_mac_dinh.times.json"), "w") as f:
            json.dump(times, f)
        out = os.path.join(tmp, "tt.json")
        assert pt.main([log, "--json", out]) == 0
        d = json.load(open(out))[0]["export_times"]
        assert d["n"] == 10 and d["computeMs"]["median"] == 69.0
        assert "startInMs" not in d and "endToEndMs" in d
    finally:
        shutil.rmtree(tmp)


def test_thieu_gpu_compute_tra_ve_1():
    tmp = tempfile.mkdtemp()
    try:
        log = os.path.join(tmp, "rong.log")
        open(log, "w").write("[I] nothing\n[exit=1]\n")
        assert pt.main([log]) == 1
    finally:
        shutil.rmtree(tmp)


def test_exit_khac_0_hoac_co_loi_la_khong_hop_le():
    tmp = tempfile.mkdtemp()
    try:
        goc = open(os.path.join(DAY, "fixture_trtexec82_A.log")).read()
        for ten, noi_dung in (("exit1.log", goc.replace("[exit=0]", "[exit=1]")),
                              ("thieu_exit.log", goc.replace("[exit=0]\n", "")),
                              ("co_loi.log", goc.replace("&&&& PASSED", "[E] Error Code 1: Cuda Runtime\n&&&& PASSED"))):
            p = os.path.join(tmp, ten)
            open(p, "w").write(noi_dung)
            assert pt.main([p]) == 1, ten
    finally:
        shutil.rmtree(tmp)


def test_cu_phap_python36_cho_script_chay_tren_nano():
    for ten in ("phan_tich_trtexec.py", "bench_engine_pipeline.py"):
        p = os.path.join(CONG_CU, ten)
        if os.path.exists(p):
            ast.parse(open(p, encoding="utf-8").read(), filename=ten, feature_version=(3, 6))


if __name__ == "__main__":
    n = 0
    for ten, f in sorted(globals().items()):
        if ten.startswith("test_") and callable(f):
            f(); n += 1; print("PASS", ten)
    print("%d/%d PASS" % (n, n))
