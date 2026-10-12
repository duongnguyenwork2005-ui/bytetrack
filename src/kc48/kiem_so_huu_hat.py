"""
kiem_so_huu_hat.py -- R48A (M1): kiem THUC NGHIEM so huu trang thai particle cua PF-CTRV. CHI DOC.

Cau hoi: moi track PF co mang particle / log-trong so / rng RIENG khong, hay co trang thai mutable bi chia se
nham giua cac track? Phan biet hai loai chia se:
  - CO Y (thiet ke): mot instance PFCTRV cho moi track cua MOT tracker (cau hinh, f, Q, R, bo dem chan doan);
  - NHAM (loi): hai track tro cung vung nho particle / logw / rng, hoac buoc xu ly track nay lam doi track kia.

Cach kiem -- KHONG sua ma tracker. Script chi boc QUAN SAT o muc instance trong tien trinh cua chinh no (goi
dung ham goc voi dung doi so, tra lai dung gia tri; cung kieu `_PredictTimer` cua run_tracking.py):
  A. Chay PFByteTracker that (load_tracker_cfg + make_tracker cua src/run_tracking.py) tren mot canh tong hop
     nho, tat dinh: xe lon + xe nho sat mep, xe re, xe mat detection 8 frame roi xuat hien lai, xe moi sinh
     giua chung (track chua xac nhan, nam ngoai pool predict), xe nho co detection diem thap (vong ghep 2).
  B. Boc tr.multi_predict, pf.sample_initial, pf.reseed_motion, pf.update_particles, pf.resample_if_needed:
     chup (ban sao particles, logw, trang thai rng) cua MOI track khac truoc moi lan goi roi kiem khong doi;
     kiem mang tra ve khong dung chung bo nho (np.shares_memory) voi mang cua track khac.
  C. Sau moi frame: kiem tung cap track (np.shares_memory, id), rng va PFTrackView rieng, PFCTRV chung,
     logw chuan hoa, N khong doi. Ngay sau multi_predict: mean (-> tlwh -> IoU) == estimate_state(particle
     cua CHINH track).
  D. Muc API: nhanh suy bien cua update_particles ghi TAI CHO vao X -> chi mang cua track so huu bi doi.
  E. Thuoc tinh cua PFCTRV dung chung bi doi trong luc chay: chi duoc la bo dem chan doan.

Ghi results/phase2/kc48_pf_code/so_huu_hat.json. Exit 0: moi phep kiem dat va du do phu; 1: co phep kiem
hong; 2: thieu do phu (canh khong kich hoat du cac nhanh can kiem) hoac backend khong ho tro.

    python src/kc48/kiem_so_huu_hat.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import ul_backend  # noqa: E402
from pf_ctrv import PFCTRV, PFTrackView, _logsumexp  # noqa: E402
from run_tracking import load_tracker_cfg, make_tracker  # noqa: E402
from tracker_ctrv import PFSTrack  # noqa: E402
from ul_backend import STrack, TrackState  # noqa: E402

YAML_MAC_DINH = "configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml"
OUT_MAC_DINH = "results/phase2/kc48_pf_code/so_huu_hat.json"
VIDEO_SALT = "kc48_so_huu_hat"
ANH_HW = (540, 960)
SO_FRAME = 45
MAT_DO = range(15, 23)          # frame 15..22: xe D khong co detection (8 frame)
MAT_DO_NGAN = range(20, 23)     # frame 20..22: xe E khong co detection (3 frame) -> phu nhanh re_activate
CONF_THAP = range(30, 34)       # frame 30..33: xe may sat bus co conf 0.20 (< track_high_thresh 0.25)
THEO_DOI = {"xe_D_mat_do": {"v_px_buoc": 3.0, "theta_rad": 0.0, "omega": 0.0},     # chuyen dong that (GT tong hop)
            "xe_E_mat_ngan": {"v_px_buoc": 2.0, "theta_rad": float(np.pi), "omega": 0.0}}
TEN_TRANG_THAI = {1: "Tracked", 2: "Lost", 3: "Removed"}
BO_DEM_CHAN_DOAN = {"n_update", "n_resample", "n_degenerate", "diag_rows"}
HAM_BOC = ("sample_initial", "reseed_motion", "update_particles", "resample_if_needed")
FILE_HASH = ("src/pf_ctrv.py", "src/tracker_ctrv.py", "src/ekf_ctrv.py", "src/run_tracking.py",
             "src/kc48/kiem_so_huu_hat.py")


def canh(f: int) -> list[tuple[str, list[float]]]:
    """Detection tong hop cua frame f (1..SO_FRAME): [(ten, [x1, y1, x2, y2, conf, cls])]. Tat dinh."""
    out = []

    def hop(ten, cx, cy, w, h, conf, cls):
        out.append((ten, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, cls]))

    hop("bus_lon", 200 + 4 * f, 300, 220, 120, 0.90, 5)
    hop("xe_may_sat_bus", 260 + 4 * f, 380, 24, 40, 0.20 if f in CONF_THAP else 0.85, 3)   # cham mep duoi bus
    if f not in MAT_DO:
        hop("xe_D_mat_do", 120 + 3 * f, 200, 50, 40, 0.90, 2)
    th0, om, v = np.pi / 2, 0.04, 5.0                                   # cung tron, R = v / omega = 125 px
    hop("xe_re", 700 + v / om * (np.sin(th0 + om * f) - np.sin(th0)),
        150 + v / om * (np.cos(th0) - np.cos(th0 + om * f)), 60, 45, 0.80, 2)
    hop("xe_may_xa", 900 - 3 * f, 470, 20, 36, 0.70, 3)
    if f >= 10:
        hop("xe_moi_sinh", 600 + 2 * (f - 10), 420, 70, 50, 0.85, 2)
    # Them sau lan chay dau (xe D doi ID nen nhanh re_activate chua duoc phu). Dat CUOI danh sach de track_id
    # (va do do rng) cua 5 xe xuat hien tu frame 1 giu nguyen nhu lan chay dau.
    if f not in MAT_DO_NGAN:
        hop("xe_E_mat_ngan", 820 - 2 * f, 90, 80, 60, 0.90, 2)
    return out


def _iou_tlwh_xyxy(tlwh, xyxy) -> float:
    ax1, ay1 = float(tlwh[0]), float(tlwh[1])
    ax2, ay2 = ax1 + float(tlwh[2]), ay1 + float(tlwh[3])
    bx1, by1, bx2, by2 = (float(v) for v in xyxy[:4])
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _tim_track(tr, track_id):
    for t in list(tr.tracked_stracks) + list(tr.lost_stracks) + list(tr.removed_stracks):
        if int(t.track_id) == track_id:
            return t
    return None


def _spread_xy(X: np.ndarray, logw: np.ndarray) -> float:
    w = np.exp(logw - _logsumexp(logw))
    mx, my = w @ X[:, PFCTRV.CX], w @ X[:, PFCTRV.CY]
    return float(np.sqrt(w @ ((X[:, PFCTRV.CX] - mx) ** 2 + (X[:, PFCTRV.CY] - my) ** 2)))


class GiamSat:
    """Boc QUAN SAT quanh cac buoc PF cua MOT tracker. Khong doi doi so / gia tri tra ve."""

    def __init__(self, tr):
        self.tr = tr
        self.pf: PFCTRV = tr.kalman_filter
        self.loi: list[str] = []
        self.dem: Counter = Counter()
        self.frame = 0
        self.pool_lon_nhat = 0
        self.ghi_update: list[dict] = []
        self.ghi_mat_do: list[dict] = []
        self._chu_update = None
        self._ghi_tam: dict = {}
        self._goc_multi_predict = tr.multi_predict
        tr.multi_predict = self._multi_predict
        for ten in HAM_BOC:
            setattr(self.pf, ten, getattr(self, "_" + ten))

    # ---------------- tien ich ----------------
    def _loi(self, su_kien: str, msg: str) -> None:
        self.loi.append(f"frame {self.frame} [{su_kien}] {msg}")

    def cac_track(self) -> list:
        out = []
        for t in list(self.tr.tracked_stracks) + list(self.tr.lost_stracks):
            if getattr(t, "particles", None) is not None and all(t is not o for o in out):
                out.append(t)
        return out

    def chup(self) -> dict:
        return {id(t): (t, t.particles.copy(), t.logw.copy(), t.rng.bit_generator.state) for t in self.cac_track()}

    def kiem_khong_doi(self, snap: dict, bo_qua: set, su_kien: str) -> None:
        for k, (t, X, lw, st) in snap.items():
            if k in bo_qua:
                continue
            self.dem[f"{su_kien}: track khac da kiem khong doi"] += 1
            if not (np.array_equal(t.particles, X) and np.array_equal(t.logw, lw)
                    and t.rng.bit_generator.state == st):
                self._loi(su_kien, f"track {t.track_id} bi doi trong luc xu ly track khac")

    def kiem_mang_moi(self, mang: np.ndarray, chu, su_kien: str) -> None:
        """`mang` (vua tao / tra ve cho track `chu`) khong dung chung bo nho voi mang cua track khac."""
        for t in self.cac_track():
            if t is chu:
                continue
            for ten, b in (("particles", t.particles), ("logw", t.logw)):
                self.dem[f"{su_kien}: cap mang da kiem shares_memory"] += 1
                if mang is b or np.shares_memory(mang, b):
                    self._loi(su_kien, f"mang cua track {getattr(chu, 'track_id', 'moi')} dung chung bo nho voi "
                                       f"{ten} cua track {t.track_id}")

    def kiem_cap(self, tracks: list, su_kien: str) -> None:
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                for na, xa in (("particles", a.particles), ("logw", a.logw)):
                    for nb, xb in (("particles", b.particles), ("logw", b.logw)):
                        self.dem[f"{su_kien}: cap mang da kiem shares_memory"] += 1
                        if xa is xb or np.shares_memory(xa, xb):
                            self._loi(su_kien, f"{na} track {a.track_id} dung chung bo nho voi {nb} track {b.track_id}")
                self.dem[f"{su_kien}: cap track da kiem rng/PFTrackView"] += 1
                if a.rng is b.rng:
                    self._loi(su_kien, f"track {a.track_id} va {b.track_id} dung chung MOT rng")
                if a.kalman_filter is b.kalman_filter:
                    self._loi(su_kien, f"track {a.track_id} va {b.track_id} dung chung MOT PFTrackView")

    def tim_chu(self, X: np.ndarray):
        for t in self.cac_track():
            if t.particles is X:
                return t
        return None

    # ---------------- cac ham boc ----------------
    def _sample_initial(self, mean, cov, rng):
        snap = self.chup()
        X = PFCTRV.sample_initial(self.pf, mean, cov, rng)
        self.dem["sample_initial (track moi)"] += 1
        for t in self.cac_track():
            if t.rng is rng:
                self._loi("sample_initial", f"rng cua track moi chinh la rng cua track {t.track_id}")
        self.kiem_khong_doi(snap, set(), "sample_initial")
        self.kiem_mang_moi(X, None, "sample_initial")
        return X

    def _reseed_motion(self, X, speed, heading, h, rng):
        chu = self.tim_chu(X)
        if chu is None:
            self._loi("reseed_motion", "khong tim thay track so huu mang X")
        elif rng is not chu.rng:
            self._loi("reseed_motion", f"track {chu.track_id}: rng truyen vao khong phai cua chinh track")
        snap = self.chup()
        Xn = PFCTRV.reseed_motion(self.pf, X, speed, heading, h, rng)
        self.dem["reseed_motion (khoi tao 2 frame)"] += 1
        self.kiem_khong_doi(snap, {id(chu)}, "reseed_motion")
        self.kiem_mang_moi(Xn, chu, "reseed_motion")
        return Xn

    def _update_particles(self, X, logw, z, conf, rng, diag_key=None):
        chu = self.tim_chu(X)
        if chu is None:
            self._loi("update", "khong tim thay track so huu mang X")
        elif rng is not chu.rng or logw is not chu.logw:
            self._loi("update", f"track {chu.track_id}: rng / logw truyen vao khong phai cua chinh track")
        tt = None if chu is None else ("Tracked" if chu.state == TrackState.Tracked else "Lost")
        snap = self.chup()
        n_deg0 = self.pf.n_degenerate
        self._chu_update, self._ghi_tam = chu, {}
        try:
            out = PFCTRV.update_particles(self.pf, X, logw, z, conf, rng, diag_key=diag_key)
        finally:
            self._chu_update = None
        Xo, lwo, _m, _P = out
        self.dem["update"] += 1
        if tt == "Lost":
            self.dem["update khi track dang LOST (re_activate)"] += 1
        self.kiem_khong_doi(snap, {id(chu)}, "update")
        self.kiem_mang_moi(Xo, chu, "update")
        self.kiem_mang_moi(lwo, chu, "update")
        self.ghi_update.append({"frame": self.frame, "track_id": None if chu is None else int(chu.track_id),
                                "trang_thai_truoc": tt, "conf": None if conf is None else float(conf),
                                "suy_bien": bool(self.pf.n_degenerate > n_deg0), **self._ghi_tam})
        return out

    def _resample_if_needed(self, X, logw, rng):
        chu = self._chu_update
        w = np.exp(logw - _logsumexp(logw))
        ess_ratio = float(1.0 / np.sum(w * w) / len(w))          # ESS/N TRUOC resample (trong so da nhan likelihood)
        Xo, lwo, did = PFCTRV.resample_if_needed(self.pf, X, logw, rng)
        self._ghi_tam = {"ess_ratio_truoc_resample": ess_ratio, "resample": bool(did)}
        if not did:
            self.dem["khong resample (ESS >= nguong)"] += 1
            return Xo, lwo, did
        self.dem["resample"] += 1
        n = len(X)
        if np.shares_memory(Xo, X):
            self._loi("resample", "mang sau resample dung chung bo nho voi mang truoc resample")
        if len(Xo) != n or not np.allclose(lwo, -np.log(n)):
            self._loi("resample", "sau resample: so particle doi hoac trong so khong deu 1/N")
        if self.pf.roughening == 0.0:
            hang = {r.tobytes() for r in X}
            self.dem["resample: particle da kiem to tien thuoc dam may cua chinh track"] += n
            if not all(r.tobytes() in hang for r in Xo):
                self._loi("resample", "co particle sau resample KHONG lay tu dam may cua chinh track")
            self._ghi_tam["ty_le_to_tien_khac_nhau_sau_resample"] = len({r.tobytes() for r in Xo}) / n
        self.kiem_mang_moi(Xo, chu, "resample")
        return Xo, lwo, did

    def _multi_predict(self, tracks):
        pool = [t for t in tracks if getattr(t, "particles", None) is not None]
        self.pool_lon_nhat = max(self.pool_lon_nhat, len(pool))
        snap = self.chup()
        truoc = {id(t): (id(t.particles), t.logw.copy(), t.state, _spread_xy(t.particles, t.logw)) for t in pool}
        self._goc_multi_predict(tracks)
        self.dem["multi_predict"] += 1
        self.dem["multi_predict: track da predict"] += len(pool)
        self.kiem_khong_doi(snap, {id(t) for t in pool}, "multi_predict (track ngoai pool)")
        for t in pool:
            idp, lw, st, sp0 = truoc[id(t)]
            if id(t.particles) != idp:
                self.dem["multi_predict: particles la mang moi"] += 1
            if not np.array_equal(t.logw, lw):
                self._loi("multi_predict", f"track {t.track_id}: predict lam doi trong so")
            m, _ = self.pf.estimate_state(t.particles, t.logw)
            self.dem["multi_predict: mean da kiem == estimate_state(particle cua chinh track)"] += 1
            if not np.array_equal(m, t.mean):
                self._loi("multi_predict", f"track {t.track_id}: mean dung cho association khac estimate_state")
            if st != TrackState.Tracked:
                self.dem["predict track LOST (khong co do)"] += 1
                self.ghi_mat_do.append({"frame": self.frame, "track_id": int(t.track_id), "spread_xy_truoc": sp0,
                                        "spread_xy_sau": _spread_xy(t.particles, t.logw),
                                        "logw_khong_doi": bool(np.array_equal(t.logw, lw))})
        self.kiem_cap(pool, "multi_predict (pool)")

    # ---------------- sau moi frame ----------------
    def kiem_sau_frame(self, n: int) -> None:
        ts = self.cac_track()
        self.kiem_cap(ts, "sau frame")
        for t in ts:
            self.dem["sau frame: track da kiem bat bien"] += 1
            kf = t.kalman_filter
            ok = (isinstance(kf, PFTrackView) and kf.track is t and kf.pf is self.pf
                  and t.particles.shape == (n, PFCTRV.NDIM) and t.logw.shape == (n,)
                  and bool(np.isfinite(t.particles).all()) and bool(np.isfinite(t.logw).all())
                  and abs(_logsumexp(t.logw)) < 1e-9)
            if not ok:
                self._loi("sau frame", f"track {t.track_id}: vi pham bat bien (PFTrackView rieng / PFCTRV chung / "
                                       f"shape (N,9),(N,) / huu han / sum w = 1)")


def kiem_api_suy_bien(seed: int) -> tuple[dict, list[str]]:
    """Muc API: nhanh suy bien ghi TAI CHO vao X cua track A; track B (cung PFCTRV) phai khong doi."""
    loi = []
    pf = PFCTRV(num_particles=64, degenerate_maha2=1e4)
    pf.set_seed(seed, VIDEO_SALT + "_api")
    rng_a, rng_b = pf.make_rng(1), pf.make_rng(2)
    m_a, P_a = pf.initiate(np.array([300.0, 200.0, 0.5, 40.0]))
    m_b, P_b = pf.initiate(np.array([330.0, 205.0, 2.0, 110.0]))
    Xa, Xb = pf.sample_initial(m_a, P_a, rng_a), pf.sample_initial(m_b, P_b, rng_b)
    la, lb = np.full(64, -np.log(64)), np.full(64, -np.log(64))
    Xb0, lb0, st_b0 = Xb.copy(), lb.copy(), rng_b.bit_generator.state
    Xa_vao = Xa
    Xa_ra, la_ra, mean, _ = pf.update_particles(Xa, la, np.array([5000.0, 5000.0, 0.5, 40.0]), None, rng_a)
    kq = {
        "n_degenerate": pf.n_degenerate,
        "X_ra_la_X_vao (ghi tai cho)": bool(Xa_ra is Xa_vao),
        "mean_A_sau": [float(mean[0]), float(mean[1])],
        "track_B_khong_doi": bool(np.array_equal(Xb, Xb0) and np.array_equal(lb, lb0)
                                  and rng_b.bit_generator.state == st_b0),
        "shares_memory(A, B)": bool(np.shares_memory(Xa_ra, Xb) or np.shares_memory(la_ra, lb)),
        "trong_so_A_deu": bool(np.allclose(la_ra, -np.log(64))),
    }
    if kq["n_degenerate"] != 1:
        loi.append("api_suy_bien: nhanh suy bien khong duoc kich hoat")
    if not kq["track_B_khong_doi"] or kq["shares_memory(A, B)"]:
        loi.append("api_suy_bien: ghi tai cho vao track A lam doi / dung chung voi track B")
    if not kq["trong_so_A_deu"]:
        loi.append("api_suy_bien: trong so A sau suy bien khong deu")
    return kq, loi


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else f"LOI git (exit {r.returncode}): {r.stderr.strip()}"
    except Exception as e:                                          # noqa: BLE001
        return f"LOI git: {e}"


def _loi_goi_bo_loc() -> dict:
    """Doan ma cua STrack.update / re_activate (backend dang chay) goi bo loc: co truyen confidence khong."""
    out = {"nguon_STrack": inspect.getsourcefile(STrack)}
    for ten in ("update", "re_activate"):
        dong = inspect.getsource(getattr(STrack, ten)).splitlines()
        i = next((k for k, l in enumerate(dong) if "kalman_filter.update(" in l), None)
        goi = " ".join(l.strip() for l in dong[i:i + 3]) if i is not None else None
        out[ten] = {"loi_goi": goi, "co_truyen_confidence": None if goi is None else ("score" in goi or "conf" in goi)}
    return out


def _bang(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--yaml", default=YAML_MAC_DINH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_MAC_DINH)
    a = ap.parse_args()
    t0 = time.perf_counter()
    if ul_backend.backend_name() != "ultralytics":
        print("Script nay chi chay backend mac dinh 'ultralytics' (dung Boxes cua ultralytics).")
        return 2
    from ultralytics.engine.results import Boxes   # noqa: PLC0415

    cfg = load_tracker_cfg(ROOT / a.yaml)
    tr = make_tracker(cfg)
    tr.set_seed(a.seed, VIDEO_SALT)
    pf = tr.kalman_filter
    n = pf.num_particles
    vars0 = {k: (v.copy() if isinstance(v, (np.ndarray, list)) else v) for k, v in vars(pf).items()}
    gs = GiamSat(tr)

    id_theo_doi_tuong: dict[str, set] = {}
    id_dau: dict[str, int] = {}
    theo_doi: dict[str, list] = {k: [] for k in THEO_DOI}
    update_conf_thap = 0
    for f in range(1, SO_FRAME + 1):
        gs.frame = f
        dets = canh(f)
        arr = np.array([r for _, r in dets], dtype=np.float32).reshape(-1, 6)
        out = tr.update(Boxes(arr, ANH_HW).cpu().numpy(), None)
        gs.kiem_sau_frame(n)
        for row in out:                     # result = [x1, y1, x2, y2, track_id, score, cls, idx]
            ten = dets[int(row[7])][0]
            id_theo_doi_tuong.setdefault(ten, set()).add(int(row[4]))
            id_dau.setdefault(ten, int(row[4]))
            if f in CONF_THAP and ten == "xe_may_sat_bus" and abs(float(row[5]) - 0.20) < 1e-3:
                update_conf_thap += 1       # track duoc cap nhat bang detection diem thap (vong ghep 2)
        # Nhat ky track DAU TIEN cua xe D / E (chi mo ta mot ca tong hop; khong so sanh bo loc o moc nay)
        for ten in THEO_DOI:
            t = _tim_track(tr, id_dau[ten]) if ten in id_dau else None
            if t is None:
                continue
            det = next((r for nm, r in dets if nm == ten), None)
            m = t.mean
            theo_doi[ten].append({
                "frame": f, "track_id": int(t.track_id), "trang_thai": TEN_TRANG_THAI.get(int(t.state), str(t.state)),
                "co_detection": det is not None, "cx": float(m[0]), "cy": float(m[1]), "v": float(m[2]),
                "theta": float(m[3]), "omega": float(m[4]), "h": float(m[6]),
                "iou_bbox_track_voi_detection": None if det is None else round(_iou_tlwh_xyxy(t.tlwh, det), 4),
                "det_cx": None if det is None else (det[0] + det[2]) / 2,
                "spread_xy": _spread_xy(t.particles, t.logw)})

    api_kq, api_loi = kiem_api_suy_bien(a.seed)
    doi = sorted(k for k, v in vars(pf).items() if k not in HAM_BOC and (k not in vars0 or not _bang(vars0[k], v)))
    loi = list(gs.loi) + api_loi
    if not set(doi) <= BO_DEM_CHAN_DOAN:
        loi.append(f"PFCTRV dung chung: thuoc tinh ngoai bo dem chan doan bi doi trong luc chay: {doi}")

    id_E = sorted(id_theo_doi_tuong.get("xe_E_mat_ngan", set()))
    do_phu = {
        "multi_predict voi pool >= 2 track": gs.dem["multi_predict"] > 0 and gs.pool_lon_nhat >= 2,
        "co update": gs.dem["update"] > 0,
        "co resample": gs.dem["resample"] > 0,
        "co predict track LOST": gs.dem["predict track LOST (khong co do)"] > 0,
        "co update khi track dang LOST (re_activate)": gs.dem["update khi track dang LOST (re_activate)"] > 0,
        "co track ngoai pool khi predict (chua xac nhan)":
            gs.dem["multi_predict (track ngoai pool): track khac da kiem khong doi"] > 0,
        "co khoi tao 2 frame (reseed_motion)": gs.dem["reseed_motion (khoi tao 2 frame)"] > 0,
        "xe E (mat 3 frame) giu 1 ID qua doan mat do": len(id_E) == 1,
        "nhanh suy bien (API) duoc kich hoat": api_kq["n_degenerate"] == 1,
    }
    if loi:
        ket_luan, code = "HONG: co phep kiem so huu khong dat", 1
    elif not all(do_phu.values()):
        ket_luan, code = "THIEU DO PHU: canh khong kich hoat du nhanh can kiem", 2
    else:
        ket_luan, code = "DAT: khong phat hien chia se nham trang thai particle / trong so / rng giua cac track", 0

    lenh = "C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_so_huu_hat.py"
    if sys.argv[1:]:
        lenh += " " + " ".join(sys.argv[1:])
    kq = {
        "muc_dich": "R48A/M1: kiem thuc nghiem so huu trang thai particle cua PF-CTRV (chi doc, khong sua ma tracker)",
        "lenh": lenh,
        "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"head": _git("rev-parse", "HEAD"),
                "status_ma_lien_quan (rong = khop HEAD)": _git("status", "--porcelain", "--", "src/pf_ctrv.py",
                                                              "src/tracker_ctrv.py", "src/ekf_ctrv.py",
                                                              "src/run_tracking.py", "third_party/ultralytics_trackers")},
        "sha256_lf": {p: ul_backend.sha256_lf(ROOT / p) for p in FILE_HASH + (a.yaml,)},
        "moi_truong": {"python": platform.python_version(), "backend": ul_backend.fingerprint(),
                       "STrack_goi_bo_loc": _loi_goi_bo_loc()},
        "cau_hinh": {"yaml": a.yaml, "seed": a.seed, "video_salt": VIDEO_SALT, "num_particles": n,
                     "resample_ess_ratio": pf.resample_ess_ratio, "estimate": pf.estimate,
                     "roughening": pf.roughening, "degenerate_maha2": pf.degenerate_maha2,
                     "track_high_thresh": cfg.track_high_thresh, "track_low_thresh": cfg.track_low_thresh,
                     "match_thresh": cfg.match_thresh, "track_buffer": cfg.track_buffer, "fuse_score": cfg.fuse_score},
        "canh": {"so_frame": SO_FRAME, "anh_hw": list(ANH_HW), "frame_xe_D_mat_do": [MAT_DO.start, MAT_DO.stop - 1],
                 "frame_xe_may_conf_0.20": [CONF_THAP.start, CONF_THAP.stop - 1],
                 "doi_tuong_frame_1": [[t, r] for t, r in canh(1)], "doi_tuong_frame_10": [[t, r] for t, r in canh(10)]},
        "id_theo_doi_tuong": {k: sorted(v) for k, v in sorted(id_theo_doi_tuong.items())},
        "update_tu_detection_conf_thap (vong ghep 2)": update_conf_thap,
        "nhat_ky_xe_mat_do": {
            "ghi_chu": ("track DAU TIEN cua moi xe; mean sau update (uoc luong TRUOC resample) hoac sau predict khi "
                        "LOST; IoU giua bbox track (tlwh tu mean) va detection cung frame. Ghep tang 1 dung "
                        "cost = 1 - IoU * score (fuse_score) <= match_thresh 0.8. Mot ca tong hop, mot seed: "
                        "chi mo ta, khong phai so sanh bo loc."),
            "chuyen_dong_that": THEO_DOI, "ban_ghi": theo_doi},
        "dem_su_kien": dict(sorted(gs.dem.items())),
        "pool_predict_lon_nhat": gs.pool_lon_nhat,
        "do_phu": do_phu,
        "api_suy_bien": api_kq,
        "PFCTRV_dung_chung": {"thuoc_tinh_bi_doi_trong_luc_chay": doi, "cho_phep": sorted(BO_DEM_CHAN_DOAN),
                              "pf_stats": tr.pf_stats(),
                              "PFSTrack.n_multi_predict_calls (bo dem cap lop)": getattr(PFSTrack,
                                                                                         "n_multi_predict_calls", None)},
        "ghi_update": gs.ghi_update,
        "ghi_predict_track_lost": gs.ghi_mat_do,
        "loi": loi,
        "ket_luan": ket_luan,
        "exit": code,
        "thoi_gian_s": round(time.perf_counter() - t0, 3),
    }
    out_path = ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kq, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"HEAD {kq['git']['head']}  backend {kq['moi_truong']['backend'].get('backend')}  N = {n}")
    for k, v in kq["dem_su_kien"].items():
        print(f"  {v:8d}  {k}")
    for k, v in do_phu.items():
        print(f"  [{'x' if v else ' '}] {k}")
    print(f"  ID theo doi tuong: {kq['id_theo_doi_tuong']}")
    for ten, ds in theo_doi.items():
        for g in ds:
            if g["frame"] in (MAT_DO.start - 1, MAT_DO.stop, MAT_DO_NGAN.start - 1, MAT_DO_NGAN.stop):
                print(f"  {ten} f{g['frame']:02d} id {g['track_id']} {g['trang_thai']:8s} cx {g['cx']:7.1f} "
                      f"(det {g['det_cx']}) v {g['v']:5.2f} theta {g['theta']:6.3f} omega {g['omega']:7.4f} "
                      f"IoU {g['iou_bbox_track_voi_detection']}")
    print(f"  API suy bien: {api_kq}")
    print(f"  PFCTRV: thuoc tinh bi doi = {doi}")
    for e in loi[:20]:
        print(f"  LOI: {e}")
    print(f"{ket_luan} -> {out_path.relative_to(ROOT)} (exit {code}, {kq['thoi_gian_s']} s)")
    return code


if __name__ == "__main__":
    sys.exit(main())
