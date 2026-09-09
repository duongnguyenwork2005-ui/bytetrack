"""
test_tracker_lifecycle.py -- Regression test di THAT qua vong doi cua tracker.

KHAC GI VOI test_init_regression.py
    test_init_regression.py goi TRUC TIEP cac ham cua bo loc (initiate,
    try_initiate_from_motion, predict, update). Nho vay no kiem tra duoc phan
    TOAN HOC, nhung KHONG chung minh duoc rang duong chay that cua tracker
    (CTRVSTrack.activate -> update -> mark_lost -> predict -> re_activate) noi
    dung cac manh do lai voi nhau.

    File nay di dung duong do: moi khang dinh deu di qua phuong thuc cua
    CTRVSTrack / UKFSTrack, khong goi tat try_initiate_from_motion.

LOI DUOC BAO VE (xem FIX_REPORT.md muc 3)
    Tracker cu luon dat `_motion_initialized = True` ngay ca khi bo loc BAO
    khong khoi tao duoc (vat the gan nhu dung yen). Hau qua: xe dung yen luc dau
    roi moi chay se khong bao gio duoc khoi tao huong -- UKF ket han o
    v = 0,14 px/frame sau 6 frame chay that (dung phai ~8).
    Ngoai ra `n_frames` phai la khoang cach frame THAT giua hai quan sat, va moc
    tinh dich chuyen phai la vi tri QUAN SAT (`_last_obs_xy`), khong phai vi tri
    trong `mean` (da qua predict).

LUU Y VE TEN LOP
    Lop EKF ten la `CTRVSTrack` (khong phai `EKFSTrack`); lop UKF la `UKFSTrack`.

KHONG can dataset, YOLO hay GPU. Deterministic, chay trong duoi mot giay.

    python src/test_tracker_lifecycle.py       # in bao cao tung phep kiem tra
    pytest src/test_tracker_lifecycle.py -q    # neu co pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from tracker_ctrv import CTRVSTrack, UKFSTrack  # noqa: E402

from ultralytics.trackers.basetrack import TrackState  # noqa: E402

#: (lop STrack, nhan de in bao cao)
TRACK_CLASSES = [(CTRVSTrack, "EKF/CTRVSTrack"), (UKFSTrack, "UKF/UKFSTrack")]

EIG_TOL = 1e-8          # dung sai cho tri rieng am do sai so lam tron float64
START_XY = (100.0, 200.0)
BOX_W, BOX_H = 20.0, 40.0
#: Toc do that trong moi kich ban: |(3, 4)| = 5 px/frame
TRUE_SPEED = 5.0

_RESULTS: list[tuple[str, bool]] = []


def _check(name: str, ok: bool, detail: str = ""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def make_detection(cls, cx: float, cy: float, w: float = BOX_W, h: float = BOX_H):
    """Detection toi thieu dung giao dien STrack cua ultralytics 8.4.141.

    `STrack.__init__` nhan xywh gom 5 phan tu: (cx, cy, w, h, chi_so_detection).
    Tra ve mot STrack CHUA activate -- dung lam `new_track` cho update/re_activate.
    """
    return cls(np.array([cx, cy, w, h, 0.0], dtype=float), 0.9, 0)


def cov_problems(P: np.ndarray) -> list[str]:
    """Liet ke moi van de cua ma tran hiep phuong sai. Rong = hop le."""
    out = []
    if P is None:
        return ["covariance la None"]
    P = np.asarray(P, dtype=float)
    if not np.all(np.isfinite(P)):
        out.append("co phan tu khong huu han")
        return out
    asym = float(np.abs(P - P.T).max())
    if asym > 1e-9:
        out.append(f"bat doi xung {asym:.2e}")
    eig = float(np.linalg.eigvalsh(0.5 * (P + P.T)).min())
    if eig < -EIG_TOL:
        out.append(f"tri rieng am {eig:.3e}")
    return out


def cov_summary(P: np.ndarray) -> str:
    P = np.asarray(P, dtype=float)
    return (f"tri rieng min {float(np.linalg.eigvalsh(0.5 * (P + P.T)).min()):+.3e}, "
            f"bat doi xung {float(np.abs(P - P.T).max()):.1e}")


# ---------------------------------------------------------------------------
# Kich ban chinh: activate -> update (dung yen) -> mat track -> re_activate
# ---------------------------------------------------------------------------
def run_lifecycle(cls, gap: int) -> dict:
    """Chay TRON VEN vong doi tracker, tra ve trang thai tai tung moc.

    Moi buoc deu goi phuong thuc THAT cua tracker. `gap` la so frame track bi
    mat dau truoc khi duoc re_activate.

    Quang duong khi re_activate ti le theo `gap` -- (3*gap, 4*gap) -- nen toc do
    THAT luon bang TRUE_SPEED du `gap` bang bao nhieu. Nho vay co the kiem tra
    `n_frames` co duoc chia dung khong ma khong phu thuoc gia tri `gap` cu the.

    SPY. Ta boc `try_initiate_from_motion` cua CHINH INSTANCE bo loc bang mot
    wrapper ghi lai roi UY QUYEN cho ham that. Tracker van chay y nguyen (van di
    qua re_activate, khong goi tat), nhung ta quan sat duoc gia tri NGAY TAI
    THOI DIEM SEED -- truoc khi buoc KF update cua `super().re_activate()` lam
    nhieu no. Nho vay khang dinh duoc CHINH XAC n_frames, ref_pos va van toc
    seed, thay vi phai noi long nguong.
    """
    IDX = cls.filter_class
    kf = cls.filter_class()
    cx0, cy0 = START_XY
    steps: dict = {}

    seed_calls: list[dict] = []
    _real_try_init = kf.try_initiate_from_motion

    def spy(mean, covariance, measurement, n_frames=1.0, ref_pos=None):
        out_mean, out_cov, ok = _real_try_init(
            mean, covariance, measurement, n_frames=n_frames, ref_pos=ref_pos)
        seed_calls.append(dict(n_frames=float(n_frames), ref_pos=ref_pos, ok=bool(ok),
                               v=float(out_mean[IDX.V]),
                               theta_deg=float(np.rad2deg(out_mean[IDX.THETA]))))
        return out_mean, out_cov, ok

    kf.try_initiate_from_motion = spy
    steps["seed_calls"] = seed_calls

    # --- 1. Activate tu detection dau tien ---
    track = make_detection(cls, cx0, cy0)
    track.activate(kf, frame_id=1)
    steps["after_activate"] = dict(
        motion_initialized=track._motion_initialized,
        last_obs=track._last_obs_xy,
        state=track.state,
        cov_bad=cov_problems(track.covariance),
    )

    # --- 2. Quan sat tiep theo dich chuyen QUA NHO (0,1 px < nguong 0,5) ---
    cx1, cy1 = cx0 + 0.1, cy0
    track.predict()
    track.update(make_detection(cls, cx1, cy1), frame_id=2)
    steps["after_static_update"] = dict(
        motion_initialized=track._motion_initialized,
        last_obs=track._last_obs_xy,
        v=float(track.mean[IDX.V]),
        cov_bad=cov_problems(track.covariance),
    )

    # --- 3. Mat track, predict qua `gap` frame ---
    track.mark_lost()
    for _ in range(gap):
        track.predict()
    steps["after_lost"] = dict(
        state=track.state,
        motion_initialized=track._motion_initialized,
        mean_finite=bool(np.all(np.isfinite(track.mean))),
        cov_bad=cov_problems(track.covariance),
    )

    # --- 4. re_activate voi chuyen dong DU LON ---
    cx2, cy2 = cx1 + 3.0 * gap, cy1 + 4.0 * gap
    frame_reactivate = 2 + gap
    track.re_activate(make_detection(cls, cx2, cy2), frame_id=frame_reactivate)
    steps["after_reactivate"] = dict(
        motion_initialized=track._motion_initialized,
        last_obs=track._last_obs_xy,
        expected_last_obs=(cx2, cy2),
        v=float(track.mean[IDX.V]),
        theta_deg=float(np.rad2deg(track.mean[IDX.THETA])),
        state=track.state,
        frame_id=track.frame_id,
        mean_finite=bool(np.all(np.isfinite(track.mean))),
        cov_bad=cov_problems(track.covariance),
        cov_desc=cov_summary(track.covariance),
    )
    return steps


def test_activate_does_not_claim_motion_initialized():
    """Vua activate xong thi chua the biet huong -> co phai la False."""
    for cls, lab in TRACK_CLASSES:
        s = run_lifecycle(cls, gap=10)["after_activate"]
        _check(f"1. Sau activate: chua khoi tao chuyen dong [{lab}]",
               s["motion_initialized"] is False
               and s["state"] == TrackState.Tracked
               and not s["cov_bad"],
               f"_motion_initialized = {s['motion_initialized']} (phai False), "
               f"state = Tracked, covariance: "
               f"{'OK' if not s['cov_bad'] else s['cov_bad']}")


def test_tiny_motion_does_not_mark_initialized():
    """LOI DUOC BAO VE: dich chuyen 0,1 px < nguong -> KHONG duoc danh dau xong."""
    thr = config.CTRV_MIN_SPEED_FOR_HEADING
    for cls, lab in TRACK_CLASSES:
        s = run_lifecycle(cls, gap=10)["after_static_update"]
        _check(f"2. Update voi dich chuyen qua nho -> van chua khoi tao [{lab}]",
               s["motion_initialized"] is False and not s["cov_bad"],
               f"dich chuyen 0.1 px < nguong {thr} px/frame\n"
               f"_motion_initialized = {s['motion_initialized']} (phai False), "
               f"v = {s['v']:.4f}")


def test_last_obs_tracks_observation_not_prediction():
    """`_last_obs_xy` phai bam theo vi tri QUAN SAT, khong phai vi tri predict."""
    for cls, lab in TRACK_CLASSES:
        st = run_lifecycle(cls, gap=10)
        after_static = st["after_static_update"]["last_obs"]
        after_re = st["after_reactivate"]
        want_static = (START_XY[0] + 0.1, START_XY[1])
        want_re = after_re["expected_last_obs"]
        ok = (np.allclose(after_static, want_static, atol=1e-9)
              and np.allclose(after_re["last_obs"], want_re, atol=1e-9))
        _check(f"3. _last_obs_xy cap nhat dung bang vi tri quan sat [{lab}]", ok,
               f"sau update dung yen : {tuple(np.round(after_static, 4))} "
               f"(mong {tuple(np.round(want_static, 4))})\n"
               f"sau re_activate     : {tuple(np.round(after_re['last_obs'], 4))} "
               f"(mong {tuple(np.round(want_re, 4))})")


def test_reactivate_initializes_motion():
    """Sau re_activate voi chuyen dong du lon -> phai khoi tao duoc huong."""
    for cls, lab in TRACK_CLASSES:
        s = run_lifecycle(cls, gap=10)["after_reactivate"]
        _check(f"4. re_activate voi chuyen dong du lon -> da khoi tao [{lab}]",
               s["motion_initialized"] is True
               and s["state"] == TrackState.Tracked
               and s["frame_id"] == 12,
               f"_motion_initialized = {s['motion_initialized']} (phai True), "
               f"state = Tracked, frame_id = {s['frame_id']} (mong 12)")


def test_seed_uses_exact_frame_gap_and_observed_ref():
    """Tai DUNG thoi diem seed: n_frames, ref_pos va van toc phai CHINH XAC.

    Day la phep kiem chat nhat: spy bat gia tri ngay khi
    `try_initiate_from_motion` tra ve, TRUOC khi buoc KF update cua
    `super().re_activate()` lam nhieu. Nho vay khang dinh duoc dang thuc chinh
    xac, khong phai noi long nguong.

    Neu `n_frames` bi truyen nham thanh 1 thi van toc seed se la 5*gap
    (50 hoac 100) chu khong phai 5.
    """
    for cls, lab in TRACK_CLASSES:
        for gap in (10, 20):
            st = run_lifecycle(cls, gap=gap)
            calls = st["seed_calls"]
            # Chi lan seed THANH CONG moi dang xet; lan o buoc dung yen tra ve ok=False
            ok_calls = [c for c in calls if c["ok"]]
            want_ref = (START_XY[0] + 0.1, START_XY[1])
            # Dung sai theo FLOAT32, khong phai float64: `STrack` cua ultralytics
            # luu `_tlwh` o float32, nen 100.1 duoc luu thanh 100.0999984741211.
            # Sai lech ~1.5e-6 la do KIEU DU LIEU, khong phai do tinh toan sai.
            # Nguong nay van chat hon nhieu bac so voi loi can bat (van toc sai
            # se la 5*gap = 50 hoac 100, chu khong phai lech o chu so thu 7).
            good = (len(ok_calls) == 1
                    and ok_calls[0]["n_frames"] == float(gap)
                    and ok_calls[0]["ref_pos"] is not None
                    and np.allclose(ok_calls[0]["ref_pos"], want_ref, atol=1e-4)
                    and abs(ok_calls[0]["v"] - TRUE_SPEED) < 1e-5)
            c = ok_calls[0] if ok_calls else {}
            _check(f"5. Seed dung n_frames/ref_pos/van toc, gap={gap} [{lab}]", good,
                   f"so lan seed thanh cong = {len(ok_calls)} (mong 1)\n"
                   f"n_frames = {c.get('n_frames')} (mong {float(gap)}; "
                   f"neu nham thanh 1.0 thi van toc se ra {TRUE_SPEED * gap})\n"
                   f"ref_pos  = {c.get('ref_pos')} (mong {want_ref} = vi tri QUAN SAT)\n"
                   f"v seed   = {c.get('v'):.9f} (mong {TRUE_SPEED}, dung sai float32 1e-5)"
                   if ok_calls else f"khong co lan seed thanh cong nao")


def test_velocity_after_update_is_gap_invariant():
    """Sau `re_activate`, van toc phai BAT BIEN theo `gap`.

    `super().re_activate()` chay them mot buoc KF update voi innovation rat lon
    (vi tri du doan sau cac buoc predict mu cach quan sat hang chuc pixel), nen
    `mean[V]` bi keo len khoang 2 lan gia tri seed. Day la hanh vi DUNG cua KF
    khi innovation lon, khong phai loi -- do do KHONG rang buoc gia tri tuyet doi.

    Cai VAN phai dung la: quang duong ti le theo `gap` nen toc do that luon = 5,
    vay ket qua phai gan nhu KHONG DOI khi `gap` doi. Neu `n_frames` bi nham
    thanh 1 thi hai lan se ra ~50 va ~100 -- lech nhau rat xa.
    """
    for cls, lab in TRACK_CLASSES:
        v10 = run_lifecycle(cls, gap=10)["after_reactivate"]["v"]
        v20 = run_lifecycle(cls, gap=20)["after_reactivate"]["v"]
        wrong10, wrong20 = TRUE_SPEED * 10, TRUE_SPEED * 20
        gap_invariant = abs(v10 - v20) < 1.5
        far_from_wrong = v10 < wrong10 / 2 and v20 < wrong20 / 2
        _check(f"6. Van toc sau update bat bien theo gap [{lab}]",
               gap_invariant and far_from_wrong,
               f"gap=10 -> v = {v10:.4f}   (neu chia nham n_frames=1 se ra ~{wrong10})\n"
               f"gap=20 -> v = {v20:.4f}   (neu chia nham n_frames=1 se ra ~{wrong20})\n"
               f"chenh giua hai lan = {abs(v10 - v20):.4f} px/frame (mong < 1.5)\n"
               f"Gia tri tuyet doi ~2x van toc seed la do KF update voi innovation "
               f"lon -- hanh vi dung, xem docstring.")


def test_heading_learned_on_vertical_component():
    """Huong phai hoc dung. Dich chuyen (3, 4) -> theta = 53,13 do.

    Kiem tra ca thanh phan DOC: loi cu chi lo ra khi xe di theo truc y, vi
    theta khoi tao = 0 tinh co dung voi truc x.
    """
    want = float(np.rad2deg(np.arctan2(4.0, 3.0)))     # 53.13 do
    for cls, lab in TRACK_CLASSES:
        th = run_lifecycle(cls, gap=10)["after_reactivate"]["theta_deg"]
        d = (th - want + 180.0) % 360.0 - 180.0
        _check(f"7. Hoc dung huong co thanh phan doc [{lab}]", abs(d) < 15.0,
               f"theta = {th:.2f} do (mong ~{want:.2f} do), lech {d:+.2f} do")


def test_state_valid_through_whole_lifecycle():
    """mean/covariance huu han, P doi xung va khong co tri rieng am, o MOI moc."""
    for cls, lab in TRACK_CLASSES:
        st = run_lifecycle(cls, gap=20)
        # Chi duyet cac moc trang thai (dict); bo qua khoa "seed_calls" (list)
        moc = {k: v for k, v in st.items() if isinstance(v, dict)}
        bad = {k: v["cov_bad"] for k, v in moc.items() if v.get("cov_bad")}
        finite_ok = all(v.get("mean_finite", True) for v in moc.values())
        _check(f"8. P hop le va mean huu han o moi moc vong doi [{lab}]",
               not bad and finite_ok,
               f"moc bi loi: {bad if bad else 'khong co'}\n"
               f"P sau re_activate: {st['after_reactivate']['cov_desc']}")


def test_never_initialized_if_always_static():
    """Neu KHONG BAO GIO du dich chuyen thi khong duoc bia ra huong."""
    for cls, lab in TRACK_CLASSES:
        IDX = cls.filter_class
        kf = cls.filter_class()
        cx, cy = START_XY
        track = make_detection(cls, cx, cy)
        track.activate(kf, frame_id=1)
        for i in range(2, 12):                  # 10 frame, moi frame dich 0,1 px
            cx += 0.1
            track.predict()
            track.update(make_detection(cls, cx, cy), frame_id=i)
        _check(f"9. Dung yen suot -> khong bia ra huong [{lab}]",
               track._motion_initialized is False
               and abs(float(track.mean[IDX.V])) < config.CTRV_MIN_SPEED_FOR_HEADING
               and not cov_problems(track.covariance),
               f"_motion_initialized = {track._motion_initialized} (phai False), "
               f"v = {float(track.mean[IDX.V]):.4f}\n"
               f"P: {cov_summary(track.covariance)}")


def main() -> int:
    print("=" * 78)
    print("REGRESSION TEST -- vong doi THAT cua tracker (activate/update/")
    print("                   mark_lost/predict/re_activate)")
    print("=" * 78)
    for fn in (test_activate_does_not_claim_motion_initialized,
               test_tiny_motion_does_not_mark_initialized,
               test_last_obs_tracks_observation_not_prediction,
               test_reactivate_initializes_motion,
               test_seed_uses_exact_frame_gap_and_observed_ref,
               test_velocity_after_update_is_gap_invariant,
               test_heading_learned_on_vertical_component,
               test_state_valid_through_whole_lifecycle,
               test_never_initialized_if_always_static):
        print(f"\n--- {fn.__name__} ---")
        fn()
    n_ok = sum(1 for _, ok in _RESULTS if ok)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 0 if n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
