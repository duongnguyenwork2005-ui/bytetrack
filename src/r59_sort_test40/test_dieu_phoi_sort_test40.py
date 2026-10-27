"""
test_dieu_phoi_sort_test40.py -- fixture CPU cua R59 (khong GPU, khong detector, khong doc du lieu that ngoai protocol).

Phan 0: tests/ va <lan> kiem junction/symlink TRUOC lan ghi dau tien (tao_lan_tests); junction that trong tests/<lan>/jx
-> tao_lan_tests va Ctx fixture cua wrapper (tien trinh con) deu tu choi, dich khong co file moi.
Phan 1 (trong tien trinh, instance THAT cua SORT / ByteTrack): K1 cau hinh hieu luc + Q/R/P0 / lop bo loc / dt tu
instance, bo dem buoc CTRV duoi skip; K3 nguong float32 0.25 / ngay tren / ngay duoi qua runner, qua SORT.update that
(ca 2 bo loc) va tang ByteTrack; cat [:300] sau loc (480 det) -> SORT.update nhan du 300; K4 frame khong detection
dau/giua/cuoi, giu track o tuoi 30, xoa khi > 30, reset ID moi video; khac vong doi du kien (track chua xac nhan).
Phan 2 (tien trinh con, R59_FIXTURE): du lieu TONG HOP trong tests/<lan>/fx/ (GT, cache, khoa/phien/nguon R56 gia lap tao
bang chinh helper R54); chay chuan-bi -> khoa -> kiem-khoa -> smoke -> cham-nguon -> dong-goi-a -> mo-phien (chan thieu
approve / khoa sai / goi A lech hash, thieu file, marker hong) -> cham truoc full (chan thieu dau ra) -> full bi chan khi
goi A bi doi sau mo-phien -> full (ngat sau marker, marker cat, gioi han 0 s, tiep tuc)
-> cham x6 -> lap-tracker / cham-lai -> hau-kiem -> tong-hop; khoa sai (sua byte MOT tong hop) -> kiem-khoa chan.
Ghi results/phase2/r59_sort_test40/tests/<lan>/ket_qua_test.json. Exit 0 khi moi test dat.

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r59_sort_test40/test_dieu_phoi_sort_test40.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import dieu_phoi_sort_test40 as D  # noqa: E402

W, R55 = D.W, D.R55
sha, rel = D.sha, D.rel
TESTS = D.VUNG / "tests"
VIDEOS = ["MVI_39031", "MVI_39051"]
N = 130
D4 = {"MVI_39051": [60, 61, 62]}
KET_QUA: list[dict] = []


def test(ten):
    def deco(f):
        t0 = time.time()
        try:
            chi_tiet = f()
            KET_QUA.append({"test": ten, "dat": True, "chi_tiet": chi_tiet, "giay": round(time.time() - t0, 1)})
            print(f"[DAT] {ten}")
        except Exception as e:  # noqa: BLE001
            KET_QUA.append({"test": ten, "dat": False, "loi": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-1500:]})
            print(f"[HONG] {ten}: {type(e).__name__}: {e}")
        return f
    return deco


def kiem(dk, msg):
    if not dk:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Phan 1: instance that
# ---------------------------------------------------------------------------
def chay_phan_1(ctx):
    import numpy as np
    import det_cache
    import run_tracking as rt
    reg = rt.tracker_registry()
    ov = {"track_buffer": 30}

    def tracker(ten):
        return rt.make_tracker(D.cau_hinh_sort(ctx, ctx.bo_loc[ten])[0], reg)

    def bt(loai):
        return rt.make_tracker(rt.load_tracker_cfg(W.yaml_arg(ctx.gt["tracker"]["yaml_bytetrack"][loai]), ov), reg)

    @test("K1 cau hinh SORT hieu luc tu cfg + instance (det_thresh/max_age/iou/min_hits, khong assoc_log mixin)")
    def _():
        ch, lech = D.cau_hinh_hieu_luc_sort(ctx)
        kiem(not lech, lech)
        return {k: v["instance"] for k, v in ch.items()}

    @test("K1 Q/R/P0 + lop bo loc + dt: SORT == ByteTrack o cung trang thai tham chieu (trung bit)")
    def _():
        bi, lech = D.so_bo_loc_instance(ctx)
        kiem(not lech, lech)
        kiem(all(x["trung_bit"] for b in bi.values() for x in b["trang_thai_tham_chieu"].values()), "khong trung bit")
        return {k: {"lop": v["lop_sort"], "buoc": list(v["trang_thai_tham_chieu"])} for k, v in bi.items()}

    @test("K1 bo dem buoc/quan sat CTRV duoi skip: n_frames SORT == ByteTrack (mat frame 2-3 -> 3)")
    def _():
        db = D.dem_buoc_ctrv(ctx)
        kiem(db["trung_n_frames"] and db["sort"]["n_frames"] == [3], db)
        return db

    @test("K3 nguong float32: == 0.25 bi loai (runner + SORT), ngay tren giu, ngay duoi loai; tang ByteTrack")
    def _():
        f32 = np.float32
        c = [f32(0.25), np.nextafter(f32(0.25), f32(1)), np.nextafter(f32(0.25), f32(0)), f32(0.2)]
        arr = np.array([[10 + 100 * i, 10, 90 + 100 * i, 90, c[i], 2] for i in range(4)], dtype=np.float32)
        loc = det_cache.filter_dets(arr, 0.25, np.array([2, 3, 5, 7]), 300, None)
        kiem(len(loc) == 1 and loc[0, 4] == c[1], f"runner giu {loc[:, 4].tolist()}")
        tr_s = tracker("sort-cv30")
        out = {}
        for ten, a in (("sau_runner", loc), ("khong_runner", arr)):
            b = D._boxes(a)
            n_sort = int((np.asarray(b.conf, dtype=np.float64) > tr_s.det_thresh).sum())
            _, _, cao, thap = bt("cv")._split_detections(b)
            out[ten] = {"sort": n_sort, "bt_cao": int(np.asarray(cao).sum()), "bt_thap": int(np.asarray(thap).sum())}
        kiem(out["sau_runner"] == {"sort": 1, "bt_cao": 1, "bt_thap": 0}, out)
        # khong qua runner: ByteTrack nhan == 0.25 vao tang cao (>=), SORT loai (>) -> khac biet bi runner che (c25)
        kiem(out["khong_runner"] == {"sort": 1, "bt_cao": 2, "bt_thap": 2}, out)
        # SORT.update THAT (ca 2 bo loc), frame 1 (frame_id <= min_hits -> xuat moi track vua sinh): cot 7 = chi so
        # detection dau vao duoc giu; chi box ngay tren 0.25 sinh track, diem xuat == float32 goc
        for ten in ("sort-cv30", "sort-ekf30"):
            for nhan, a, idx in (("khong_runner", arr, [1]), ("sau_runner", loc, [0])):
                tr = tracker(ten)
                o = tr.update(D._boxes(a), None, n_steps=1)
                kiem(len(tr.tracks) == 1 and o[:, 7].astype(int).tolist() == idx and o[0, 5] == c[1],
                     f"{ten} {nhan}: {len(tr.tracks)} track, xuat {o.tolist()}")
                out[f"update_{ten}_{nhan}"] = {"so_track": len(tr.tracks), "idx_giu": idx, "diem": float(o[0, 5])}
        return out

    @test("K3 cat [:300] sau loc conf/lop, giu thu tu NMS; SORT.update nhan du 300 (ca 2 bo loc)")
    def _():
        n = 480
        conf = np.linspace(0.95, 0.26, n).astype(np.float32)
        cls = np.array([[2, 3, 5, 7, 0][i % 5] for i in range(n)], np.float32)
        conf[::7] = np.float32(0.25)                          # dung nguong: bi loai
        arr = np.array([[20 * (i % 40), 20 * (i // 40), 20 * (i % 40) + 15, 20 * (i // 40) + 15, conf[i], cls[i]]
                        for i in range(n)], np.float32)
        ky_vong = arr[(arr[:, 4] > 0.25) & np.isin(arr[:, 5].astype(np.int64), [2, 3, 5, 7])]
        kiem(len(ky_vong) > 300, f"fixture chi co {len(ky_vong)} det hop le")
        loc = det_cache.filter_dets(arr, 0.25, np.array([2, 3, 5, 7]), 300, None)
        kiem(len(loc) == 300 and loc.dtype == np.float32 and np.array_equal(loc, ky_vong[:300]), "cat [:300] / thu tu")
        out = {"hop_le_truoc_cat": int(len(ky_vong)), "sau_cat": int(len(loc))}
        for ten in ("sort-cv30", "sort-ekf30"):
            tr = tracker(ten)
            o = tr.update(D._boxes(loc), None, n_steps=1)
            kiem(len(o) == 300 and sorted(o[:, 7].astype(int).tolist()) == list(range(300)), f"{ten} xuat {len(o)}")
            out[ten] = len(o)
        return out

    def chay(tr, khung):
        ra = []
        for arr in khung:
            o = tr.update(D._boxes(arr), None, n_steps=1)
            ra.append(len(o))
        return ra

    @test("K4 frame rong dau/giua/cuoi; giu track o time_since_update 30, xoa o frame hut thu 31 (ca 2 SORT)")
    def _():
        out = {}
        hop = [[300.0, 200.0, 380.0, 280.0, 0.9, 2.0]]
        for ten in ("sort-cv30", "sort-ekf30"):
            tr = tracker(ten)
            ra = chay(tr, [[]] * 3 + [hop] * 5)              # frame 1-3 rong, 4-8 co det
            # sort_tracker.py (nhu KalmanBoxTracker goc): track sinh o frame 4 voi hit_streak 0, +1 moi update ghep ->
            # hit_streak 3 o frame 7; frame_id 4 > min_hits nen khong co ngoai le dau video -> xuat tu frame 7
            kiem(ra[:3] == [0, 0, 0] and ra[3:6] == [0, 0, 0] and ra[6:] == [1, 1], f"{ten} xuat {ra}")
            ra2 = chay(tr, [[]] * 30)                         # frame 9-38 rong
            kiem(sum(ra2) == 0 and len(tr.tracks) == 1 and tr.tracks[0].time_since_update == 30, f"{ten} tuoi 30")
            chay(tr, [[]])                                    # frame 39: hut thu 31 -> xoa
            kiem(len(tr.tracks) == 0, f"{ten}: chua xoa khi time_since_update > 30")
            ra3 = chay(tr, [[]] * 3)                          # cuoi video rong: khong loi
            out[ten] = {"xuat_dau": ra, "giu_30": True, "xoa_31": True, "cuoi_rong": ra3}
        return out

    @test("K4 reset ID moi video qua run_tracking.track_video (ID bat dau 1 o moi video)")
    def _():
        class VD:
            orig_shape = W.IMG_HW

            def __init__(self, n): self.n = n

            def iter_frames(self):
                for i in range(1, self.n + 1):
                    yield i, (i - 1) / 25.0, False, np.array([[50.0 + 2 * i, 60.0, 130.0 + 2 * i, 140.0, 0.9, 2.0]], np.float32)

            def __len__(self): return self.n

        out = {}
        for ten in ("sort-cv30", "sort-ekf30"):
            cfg = D.cau_hinh_sort(ctx, ctx.bo_loc[ten])[0]
            ids = []
            for _ in range(2):
                df, _, dt = rt.track_video(VD(10), cfg, 1 / 25.0, "skip", None, "v", reg)
                ids.append(sorted(set(int(x) for x in df["id"])))
                kiem(sorted(set(dt["n_steps"])) == [1] and len(dt) == 10, "n_steps / timeline")
            kiem(ids == [[1], [1]], f"{ten} ids {ids}")
            out[ten] = ids
        return out

    @test("Khac vong doi DU KIEN (khong phai tham so): track chua xac nhan - ByteTrack xoa sau 1 frame hut, SORT giu")
    def _():
        hop = [[300.0, 200.0, 380.0, 280.0, 0.9, 2.0]]
        tb = bt("cv")
        ts = tracker("sort-cv30")
        for tr in (tb, ts):
            tr.update(D._boxes([]), None, n_steps=1)          # frame 1 rong -> det o frame 2 sinh track chua xac nhan
            tr.update(D._boxes(hop), None, n_steps=1)
            tr.update(D._boxes([]), None, n_steps=1)
        n_b = len(tb.tracked_stracks) + len(tb.lost_stracks)
        kiem(n_b == 0 and len(ts.tracks) == 1, f"bytetrack {n_b}, sort {len(ts.tracks)}")
        return {"bytetrack_con": n_b, "sort_con": len(ts.tracks), "phan_loai": "vong_doi_du_kien"}


# ---------------------------------------------------------------------------
# Phan 2: du lieu tong hop + luong A/B qua tien trinh con
# ---------------------------------------------------------------------------
def tao_fixture(fx: Path) -> dict:
    import numpy as np
    import run_tracking as rt
    gt_that = json.loads(D.GIAO_THUC.read_text(encoding="utf-8"))
    ng = fx / "nguon"
    n56, td = gt_that["nguon_r56"], gt_that["tap_du_lieu"]
    gtd = ng / td["thu_muc_gt"]
    rng = np.random.default_rng(0)
    gt_files = []
    for vi, v in enumerate(VIDEOS):
        (gtd / v / "gt").mkdir(parents=True)
        rows = []
        for f in range(1, N + 1):
            if f in D4.get(v, []):
                continue
            for o in range(3):
                x, y = 50 + 160 * o + (2 + o) * f * (1 if vi == 0 else 0.5), 100 + 90 * o
                rows.append(f"{f},{o + 1},{x:.2f},{y:.2f},70.0,60.0,1,1,1.0\n")
        (gtd / v / "gt" / "gt.txt").write_bytes("".join(rows).encode())
        (gtd / v / "seqinfo.ini").write_bytes(
            f"[Sequence]\r\nname={v}\r\nimDir=img1\r\nframeRate=25\r\nseqLength={N}\r\nimWidth=960\r\nimHeight=540\r\nimExt=.jpg\r\n".encode())
        gt_files += [gtd / v / "gt" / "gt.txt", gtd / v / "seqinfo.ini"]
    p_d4 = ng / td["loai_d4"]
    p_d4.parent.mkdir(parents=True)
    p_d4.write_text("rel,video,frame\n" + "".join(f"{v}/img{f:05d}.jpg,{v},{f}\n" for v, fs in D4.items() for f in fs),
                    encoding="utf-8")
    p_man = ng / td["manifest"]
    p_man.parent.mkdir(parents=True)
    p_man.write_text(json.dumps({"videos": VIDEOS}), encoding="utf-8")
    kd = ng / n56["khoa"]
    kd.mkdir(parents=True)
    kj = {"buoc": "R56_LOCK", "fixture": False, "file": [{"rel": rel(p), "vai_tro": "gt", "sha256": sha(p)} for p in gt_files]}
    K = W.ghi_json(kd / "khoa.json", kj, [kd])
    (kd / "SHA256SUMS").write_text(f"{K}  khoa.json\n", encoding="utf-8")
    ph = ng / n56["phien"]
    ph.mkdir(parents=True)
    (ph / "phien.json").write_text(json.dumps({"khoa": kd.name, "khoa_sha256": K}), encoding="utf-8")
    gia = types.SimpleNamespace(fixture=False)
    man = {}
    for di, det in enumerate(D.DETECTORS):
        mv = {}
        for vi, v in enumerate(VIDEOS):
            lan = ph / "cache" / det / v / "lan_0001"
            lan.mkdir(parents=True)
            data, npf = [], []
            for f in range(1, N + 1):
                rong = (vi == 0 and (f <= 2 or 50 <= f <= 52 or f >= N - 1))
                fr = []
                if not rong:
                    for o in range(3):
                        if rng.random() < 0.1:
                            continue
                        x = 50 + 160 * o + (2 + o) * f * (1 if vi == 0 else 0.5) + rng.normal(0, 1.5)
                        y = 100 + 90 * o + rng.normal(0, 1.5)
                        fr.append([x, y, x + 70, y + 60, 0.5 + 0.45 * rng.random(), [2, 7, 3][(o + di) % 3]])
                    fr.append([700, 400, 760, 450, 0.25, 2])           # conf DUNG nguong: bi loai
                    fr.append([600, 50, 650, 90, 0.2, 2])              # diem thap: bi loai o c25
                    fr.append([800, 300, 850, 340, 0.9, 0])            # lop ngoai [2,3,5,7]
                fr.sort(key=lambda r: -r[4])
                data += fr
                npf.append(len(fr))
            a = np.array(data, dtype=np.float32).reshape(-1, 6)
            np.savez(lan / f"{v}.npz", frame_idx=np.arange(1, N + 1, dtype=np.int32), n_per_frame=np.array(npf, np.int32),
                     unreadable=np.zeros(N, bool), truncated=np.zeros(N, bool), time_s=np.arange(N) / 25.0, data=a,
                     orig_shape=np.array(W.IMG_HW, np.int32), cls_goc=a[:, 5].astype(np.int16))
            (lan / f"{v}.json").write_text(json.dumps({"spec_sha256": "fx"}), encoding="utf-8")
            (lan / "meta.json").write_text(json.dumps({"conf": 0.05, "classes": None, "max_det": 3000, "spec_sha256": "fx"}),
                                           encoding="utf-8")
            D.ghi_marker(gia, "cache", lan, {"loai": "cache", "detector": det, "video": v}, {})
            mv[v] = {"lan": rel(lan), "npz_sha256": sha(lan / f"{v}.npz"), "json_sha256": sha(lan / f"{v}.json"),
                     "meta_sha256": sha(lan / "meta.json"), "marker_sha256": sha(lan / D.MARKER), "n_frames": N}
        md = ph / "xac_nhan_cache" / det / "lan_0001"
        md.mkdir(parents=True)
        s = W.ghi_json(md / "manifest_cache.json", {"detector": det, "rang_buoc": {"khoa_sha256": K}, "video": mv}, [md])
        (md / "SHA256SUMS").write_text(f"{s}  manifest_cache.json\n", encoding="utf-8")
        D.ghi_marker(gia, "xn", md, {"loai": "xac-nhan-cache", "detector": det, "khoa_sha256": K}, {})
        man[det] = (s, mv)
    reg = rt.tracker_registry()
    si = {v: N for v in VIDEOS}
    summary, bang = {}, []
    for r in gt_that["ma_tran"]["runs"]:
        dc, det = r["doi_chung"], r["detector"]
        cfg = rt.load_tracker_cfg(W.yaml_arg(gt_that["tracker"]["yaml_bytetrack"][dc.rsplit("-", 1)[1]]), {"track_buffer": 30})
        ms, mv = man[det]
        mot = {}
        for v in VIDEOS:
            tl = ph / "tracking" / dc / v / "lan_0001"
            vd = W.mo_cache(W.tuyet_doi(mv[v]["lan"]), v, 0.25, [2, 3, 5, 7], 300)
            info = W.chay_tracker(tl, v, vd, cfg, None, reg, False, False)
            D.ghi_marker(gia, "tr", tl, {"loai": "tracking", "run": dc, "detector": det, "cond": "c25", "conf_min": 0.25,
                                         "video": v, "khoa_sha256": K, "npz_sha256": mv[v]["npz_sha256"],
                                         "manifest_cache_sha256": ms}, {"info": info})
            mot[v] = tl
        cl = ph / "cham" / dc / "lan_0001"
        trk = cl / "trackers" / "DETRAC-test" / dc / "data"
        trk.mkdir(parents=True)
        for v, tl in mot.items():
            (trk / f"{v}.txt").write_bytes((tl / "data" / f"{v}.txt").read_bytes())
        ev = W.cham(gtd.parent, cl / "trackers", "DETRAC-test", dc, si, cl / "trackeval", [cl, gtd.parent])
        D.ghi_marker(gia, "cham", cl, {"loai": "cham", "run": dc, "split": "DETRAC-test", "khoa_sha256": K,
                                       "manifest_cache_sha256": ms, "gt_sha256_ds": "fx", "seq_info": si,
                                       "mot_sha256_ds": W.sha256_ds((v, sha(trk / f"{v}.txt")) for v in VIDEOS)},
                     {"combined": ev["combined"], "file_metric": ev["file"], "mot_lan": {v: rel(t) for v, t in mot.items()}})
        sp = cl / "trackeval" / dc / "pedestrian_summary.txt"
        summary[dc] = {"file": rel(sp), "sha256": sha(sp)}
        c = ev["combined"]
        bang.append({"run": dc, "HOTA": round(c["HOTA"] * 100, 3), "DetA": round(c["DetA"] * 100, 3),
                     "AssA": round(c["AssA"] * 100, 3), "MOTA": round(c["MOTA"] * 100, 3), "IDF1": round(c["IDF1"] * 100, 3),
                     "IDSW": int(c["IDSW"]), "CLR_FP": int(c["FP"])})
    pn = ng / n56["nguon_json"]
    pn.parent.mkdir(parents=True)
    pn.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    pb = ng / n56["bang_r56"]
    W.ghi_bang_csv(pb, bang, [pb.parent])
    gt = json.loads(json.dumps(gt_that))
    gt["tap_du_lieu"].update({"so_video": 2, "so_anh": 2 * N, "so_anh_D4": 3, "manifest_sha256": sha(p_man),
                              "loai_d4_sha256": sha(p_d4)})
    gt["nguon_r56"].update({"khoa_json_sha256": K, "nguon_json_sha256": sha(pn), "bang_r56_sha256": sha(pb)})
    (fx / "giao_thuc.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    (fx / "fixture.json").write_text(json.dumps({"giao_thuc": "giao_thuc.json", "goc_nguon": "nguon"}), encoding="utf-8")
    return {"K": K, "nguon": ng}


LOG_CON: list[dict] = []


def con(fx: Path, *a, ngat: str | None = None) -> tuple[int, str]:
    env = dict(os.environ)
    env[D.ENV_FIXTURE] = str(fx / "fixture.json")
    env.pop(D.ENV_NGAT, None)
    if ngat:
        env[D.ENV_NGAT] = ngat
    for k in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        env.pop(k, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    p = subprocess.run([sys.executable, str(D.FILE_MA), *a], cwd=str(D.GOC), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    LOG_CON.append({"argv": list(a), "ngat": ngat, "exit": p.returncode, "stdout_cuoi": p.stdout[-1200:],
                    "stderr_cuoi": p.stderr[-1200:]})
    return p.returncode, p.stdout + p.stderr


def lay_id(out: str, khoa: str) -> str:
    for ln in out.splitlines():
        if ln.startswith(khoa + "="):
            return ln.split("=", 1)[1].strip()
    raise AssertionError(f"khong thay {khoa}= trong dau ra: {out[-800:]}")


def them_byte(b: bytes) -> bytes:
    return b + b" "


def cat_nua(b: bytes) -> bytes:
    return b[: len(b) // 2]


def doi_tam(f: Path, moi, ham) -> dict:
    """Doi tam file TONG HOP f (moi=None -> xoa, nguoc lai f <- moi(bytes)), goi ham(), khoi phuc nguyen byte."""
    b = f.read_bytes()
    try:
        if moi is None:
            f.unlink()
        else:
            f.write_bytes(moi(b))
        ham()
    finally:
        f.write_bytes(b)
    kiem(f.read_bytes() == b, f"khong khoi phuc {rel(f)}")
    return {"file": rel(f), "kieu": "xoa" if moi is None else moi.__name__, "exit": 2}


def chay_phan_2(lan: Path):
    fx = lan / "fx"
    st = {}

    @test("Fixture: tao du lieu tong hop + nguon R56 gia lap bang helper R54 (tracker ByteTrack that, W.cham that)")
    def _():
        st.update(tao_fixture(fx))
        return {"khoa_r56_gia_lap": st["K"]}

    def buoc(ten, *a, ky_vong=0, id_=None, ngat=None):
        ma, out = con(fx, *a, ngat=ngat)
        kiem(ma == ky_vong, f"{ten}: exit {ma} != {ky_vong}: {out[-1500:]}")
        return lay_id(out, id_) if id_ else out

    @test("A: chuan-bi -> khoa -> kiem-khoa -> smoke -> cham-nguon -> dong-goi-a (fixture)")
    def _():
        cb = buoc("chuan-bi", "--buoc", "chuan-bi", id_="CHUAN_BI_ID")
        cbj = json.loads((fx / "chuan_bi" / cb / "chuan_bi.json").read_text(encoding="utf-8"))
        kiem(cbj["k2"]["tom_tat"]["so_dat"] == 6 and cbj["k2"]["tom_tat"]["d4_danh_sach"] == 3, cbj["k2"])
        td = fx / "tests" / "lan_stub"
        td.mkdir(parents=True)
        (td / "ket_qua_test.json").write_text(json.dumps({"dat": True, "so_test": 0, "ma_sha256": {
            "dieu_phoi": sha(D.FILE_MA), "test": sha(D.FILE_TEST), "giao_thuc": sha(D.GIAO_THUC)}}), encoding="utf-8")
        buoc("khoa thieu tests", "--buoc", "khoa", "--chuan-bi", cb, ky_vong=2)
        k = buoc("khoa", "--buoc", "khoa", "--chuan-bi", cb, "--tests", "lan_stub", id_="KHOA_ID")
        buoc("kiem-khoa", "--buoc", "kiem-khoa", "--khoa", k)
        buoc("kiem-khoa sai ten", "--buoc", "kiem-khoa", "--khoa", "khoa_khong_co", ky_vong=2)
        sm = buoc("smoke", "--buoc", "smoke", "--khoa", k, id_="SMOKE_ID")
        buoc("cham-nguon sai run", "--buoc", "cham-nguon", "--khoa", k, "--run", "r54-yolov8n_coco-c25-ekf", ky_vong=2)
        cn = buoc("cham-nguon", "--buoc", "cham-nguon", "--khoa", k, "--run", D.CHAM_NGUON_RUN, id_="CHAM_NGUON_ID")
        ga = buoc("dong-goi-a", "--buoc", "dong-goi-a", "--khoa", k, "--smoke", sm, "--cham-nguon", cn, id_="GOI_A_ID")
        st.update({"khoa": k, "goi": ga, "smoke": sm, "cham_nguon": cn})
        sk = json.loads((fx / "smoke" / sm / "ket_qua_smoke.json").read_text(encoding="utf-8"))
        kiem(all(not x["co_assoc_log"] and not x["co_pf_diag"] for x in sk["run"].values()), "smoke co assoc_log/pf_diag")
        return {"chuan_bi": cb, "khoa": k, "smoke": sm, "cham_nguon": cn, "goi_a": ga}

    @test("B: chan thieu approve A / review khong rang buoc / khoa sai; cham truoc full bi chan, khong marker")
    def _():
        k = st["khoa"]
        (fx / "review_revise.json").write_text(json.dumps({"verdict": "revise", "ghi_chu": k}), encoding="utf-8")
        (fx / "review_lac.json").write_text(json.dumps({"verdict": "approve", "ghi_chu": "khong nhac khoa"}), encoding="utf-8")
        (fx / "review_ok.json").write_text(json.dumps({"verdict": "approve", "ghi_chu": f"R59A {k} {st['goi']}"}), encoding="utf-8")
        buoc("mo-phien thieu review", "--buoc", "mo-phien", "--khoa", k, ky_vong=2)
        buoc("mo-phien revise", "--buoc", "mo-phien", "--khoa", k, "--review-a", str(fx / "review_revise.json"), ky_vong=2)
        buoc("mo-phien lac", "--buoc", "mo-phien", "--khoa", k, "--review-a", str(fx / "review_lac.json"), ky_vong=2)
        buoc("mo-phien khoa sai", "--buoc", "mo-phien", "--khoa", "khoa_sai", "--review-a", str(fx / "review_ok.json"), ky_vong=2)
        # toan ven goi A (file liet ke trong goi_a.json, ngoai khoa): lech hash smoke / thieu file cham-nguon / marker goi hong
        f_sm = fx / "smoke" / st["smoke"] / "ket_qua_smoke.json"
        f_cn = fx / "kiem_duong_cham" / st["cham_nguon"] / "ket_qua_cham_nguon.json"
        f_mk = fx / "chuan_bi" / st["goi"] / D.MARKER
        mo = ("--buoc", "mo-phien", "--khoa", k, "--review-a", str(fx / "review_ok.json"))
        ca = {}
        for ten, f, moi in (("lech hash smoke", f_sm, them_byte), ("thieu file cham-nguon", f_cn, None),
                            ("marker goi A hong", f_mk, cat_nua)):
            ca[ten] = doi_tam(f, moi, lambda t=ten: buoc(f"mo-phien {t}", *mo, ky_vong=2))
        kiem(not (fx / "thuc_thi").exists() or not any((fx / "thuc_thi").iterdir()), "phien bi tao khi bi chan")
        p = buoc("mo-phien", *mo, id_="PHIEN_ID")
        st["phien"] = p
        buoc("cham truoc full", "--buoc", "cham", "--phien", p, "--run", D.RUNS[0], ky_vong=2)
        kiem(not (fx / "thuc_thi" / p / "cham").exists(), "cham tao dau ra khi thieu MOT")
        # sau mo-phien: goi A bi doi -> full / tiep tuc bi chan truoc khi ghi
        for ten, f, moi in (("full lech hash smoke", f_sm, them_byte), ("full thieu file cham-nguon", f_cn, None)):
            ca[ten] = doi_tam(f, moi, lambda t=ten: buoc(t, "--buoc", "full", "--phien", p, ky_vong=2))
        kiem(not (fx / "thuc_thi" / p / "tracking").exists(), "full ghi tracking khi goi A khong nguyen ven")
        return {"phien": p, "goi_a_bi_chan": ca}

    @test("B: full bi ngat sau marker / marker cat / gioi han 0 s (exit 10) roi tiep tuc; giu lan do dang/hong")
    def _():
        p = st["phien"]
        buoc("full ngat", "--buoc", "full", "--phien", p, ngat="full-sau-marker:3", ky_vong=9)
        buoc("full marker cat", "--buoc", "full", "--phien", p, ngat="full-marker-cat:1", ky_vong=9)
        buoc("full gioi han 0", "--buoc", "full", "--phien", p, "--gioi-han-giay", "0", ky_vong=D.MA_THOAT_CHUA_XONG)
        buoc("full", "--buoc", "full", "--phien", p)
        out = buoc("full lai (tai su dung)", "--buoc", "full", "--phien", p)
        kiem("moi 0, tai su dung 12" in out, out[-600:])
        tt = []
        for x in (fx / "thuc_thi" / p / "tracking").rglob("lan_*"):
            m = R55.doc_marker(x)
            tt.append("hop_le" if m else ("hong" if (x / D.MARKER).exists() else "do_dang"))
        kiem(tt.count("hong") == 1 and tt.count("hop_le") == 12, f"trang thai lan {sorted(tt)}")
        return {"lan": {s: tt.count(s) for s in set(tt)}}

    @test("B: cham x6 -> lap-tracker / cham-lai (trung byte) -> hau-kiem -> tong-hop; kiem-khoa cuoi")
    def _():
        p = st["phien"]
        for r in D.RUNS:
            buoc(f"cham {r}", "--buoc", "cham", "--phien", p, "--run", r)
        buoc("lap sai video", "--buoc", "lap-tracker", "--phien", p, "--run", D.LAP_RUNS[0], "--video", "MVI_39051", ky_vong=2)
        for r in D.LAP_RUNS:
            buoc(f"lap {r}", "--buoc", "lap-tracker", "--phien", p, "--run", r, "--video", D.LAP_VIDEO)
            buoc(f"cham-lai {r}", "--buoc", "cham-lai", "--phien", p, "--run", r)
        buoc("hau-kiem", "--buoc", "hau-kiem", "--phien", p)
        buoc("tong-hop", "--buoc", "tong-hop", "--phien", p)
        bang =D.doc_csv(fx / "tong_hop" / p / "doi_chieu_6_cap.csv")
        kiem(len(bang) == 6 and [x["run_sort"] for x in bang] == list(D.RUNS), "bang 6 cap")
        buoc("kiem-khoa cuoi", "--buoc", "kiem-khoa", "--khoa", st["khoa"])
        return {"bang_dong_1": bang[0]}

    @test("Khoa sai: sua byte MOT nguon (tong hop) -> kiem-khoa exit 2; khoi phuc -> exit 0")
    def _():
        f = next((st["nguon"]).rglob("tracking/*/MVI_39031/lan_0001/data/MVI_39031.txt"))
        b = f.read_bytes()
        try:
            f.write_bytes(b + b"\n")
            buoc("kiem-khoa lech", "--buoc", "kiem-khoa", "--khoa", st["khoa"], ky_vong=2)
        finally:
            f.write_bytes(b)
        buoc("kiem-khoa khoi phuc", "--buoc", "kiem-khoa", "--khoa", st["khoa"])
        return {"file": rel(f)}


def tao_lan_tests(goc: Path) -> Path:
    """Chan junction/symlink TRUOC lan ghi dau tien: goc (moi thanh phan da ton tai, ke ca thu muc cha) phai
    resolve == abspath va nam LEXICAL trong vung R59 (R55.kiem_doc_lap; W.tao_phien resolve ca goc lan dich nen
    khong tu chan duoc)."""
    R55.kiem_doc_lap(goc, D.VUNG)
    lan = W.tao_phien(goc, None, "lan")
    R55.kiem_doc_lap(lan, goc)
    return lan


def kiem_junction(lan: Path):
    @test("Chan junction truoc ghi: tests/<lan>/jx/cong -> jx/dich; tao_lan_tests + Ctx fixture (tien trinh con) tu choi, dich khong co file moi")
    def _():
        import _winapi
        dich, cong = lan / "jx" / "dich", lan / "jx" / "cong"
        dich.mkdir(parents=True)
        (dich / "fixture.json").write_text(json.dumps({"giao_thuc": "giao_thuc.json", "goc_nguon": "nguon"}), encoding="utf-8")
        truoc = sorted(p.name for p in dich.rglob("*"))
        _winapi.CreateJunction(str(dich), str(cong))
        try:
            kiem(cong.resolve() != Path(os.path.abspath(cong)), "junction khong duoc tao")
            bi_chan = []
            for g in (cong, cong / "tests"):
                try:
                    tao_lan_tests(g)
                    raise AssertionError(f"tao_lan_tests({rel(g)}) khong bi chan")
                except W.LoiDauVao as e:
                    bi_chan.append(str(e)[:120])
            ma, out = con(cong, "--buoc", "chuan-bi")
            kiem(ma == 2 and "junction" in out, f"Ctx qua junction: exit {ma}: {out[-600:]}")
            sau = sorted(p.name for p in dich.rglob("*"))
            kiem(sau == truoc, f"dich co file moi {sau}")
        finally:
            os.rmdir(cong)                                    # chi go diem noi, khong dong vao dich
        kiem(not cong.exists() and (dich / "fixture.json").is_file(), "go junction")
        return {"bi_chan": bi_chan, "ctx_exit": ma, "dich": sau}


def main() -> int:
    t0 = time.time()
    lan = tao_lan_tests(TESTS)
    W.dat_moi_truong(lan)
    for k in (D.ENV_FIXTURE, D.ENV_NGAT):
        os.environ.pop(k, None)
    kiem_junction(lan)
    ctx = D.Ctx()
    chay_phan_1(ctx)
    chay_phan_2(lan)
    dat = all(x["dat"] for x in KET_QUA)
    kq = {"dat": dat, "so_test": len(KET_QUA), "so_dat": sum(x["dat"] for x in KET_QUA), "ket_qua": KET_QUA,
          "tien_trinh_con": LOG_CON, "ma_sha256": {"dieu_phoi": sha(D.FILE_MA), "test": sha(D.FILE_TEST),
                                                  "giao_thuc": sha(D.GIAO_THUC)},
          "python": sys.executable, "giay": round(time.time() - t0, 1), "luc": D.bay_gio()}
    W.ghi_json(lan / "ket_qua_test.json", kq, [lan])
    print(f"[test] {kq['so_dat']}/{kq['so_test']} dat -> {rel(lan)}; TESTS_ID={lan.name}")
    return 0 if dat else 1


if __name__ == "__main__":
    sys.exit(main())
