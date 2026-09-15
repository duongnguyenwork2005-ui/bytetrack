"""
lost_recovery.py -- (MOC 2, C2) Cac thay doi NHO trong co che ghep, CHI ap dung cho track LOST, giong nhau
cho moi bo loc (CV / EKF / UKF / PF). Mac dinh TAT -> hanh vi BIT-IDENTICAL BYTETracker goc
(test_lost_recovery.py). Ung vien nao duoc chay do PREREG_moc2.md quyet dinh SAU phan tich nguyen nhan.

Khoa .yaml (mac dinh tat):
    lost_no_fuse: false        (a) vong 1: hang chi phi cua track LOST = 1 - IoU (KHONG nhan conf cua det).
                                   Giả thuyết: det tai xuat hien conf thap; fuse_score day cost > match_thresh
                                   du IoU du (nguyen nhan `score_fusion_reject`).
    lost_recover: none|iou|maha (b)/(c) VONG GHEP THU 3 (hook _post_first_association, TRUOC vong 2 va TRUOC khi
                                   sinh track moi): track LOST chua ghep x det HIGH chua ghep.
                                   iou : cost = 1 - IoU, nhan khi IoU >= lost_recover_iou (mac dinh 0.05)
                                   maha: gate Mahalanobis VI TRI theo hiep phuong sai CUA CHINH BO LOC
                                         (gating_distance(only_position=True) <= lost_recover_maha2, mac dinh
                                         chi2(2 dof, 95%) = 5.991), xep hang theo khoang cach chuan hoa
                                         d_norm = |tam det - tam du doan| / (w + h)/2 cua det; nhan khi trong gate.
                                   Giả thuyết: det con du (chua ai lay, sap thanh ID moi) nhung IoU voi du doan
                                   < 0.2 (`prediction_drift`) hoac bi cost fused loai; cua so bat dinh cua bo loc
                                   la gate tu nhien (khac GD1: gate theo so frame + IoA bao hoa -> IDSW no).
    lost_recover_iou: 0.05
    lost_recover_maha2: 5.991
    lost_recover_max_lost: 0   (0 = khong gioi han; >0: chi track co n_lost <= gia tri nay)

GHI CHU: ket qua ghep o vong 3 la re_activate (giu ID) -> refind; ghep nham do bang failure_analysis
(refind_quality) - ca phuc hoi dung VA ghep nham deu phai bao cao.
"""
from __future__ import annotations

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching


class LostRecoveryMixin:
    """Dat TRUOC lop tracker (va truoc LostAwareAssociation) trong MRO."""

    # ---- (a) khong fuse score cho hang LOST ----
    def get_dists(self, tracks, detections):
        dists = super().get_dists(tracks, detections)
        if bool(getattr(self.args, "lost_no_fuse", False)) and bool(getattr(self.args, "fuse_score", True)) \
                and len(tracks) and len(detections):
            lost = [i for i, t in enumerate(tracks) if t.state != TrackState.Tracked]
            if lost:
                raw = matching.iou_distance(tracks, detections)          # 1 - IoU, khong fuse
                dists[np.array(lost), :] = raw[np.array(lost), :]
        return dists

    # ---- (b)/(c) vong ghep thu 3 cho track LOST ----
    def _post_first_association(self, strack_pool, detections, u_track, u_detection, activated, refind):
        u_track, u_detection = super()._post_first_association(strack_pool, detections, u_track, u_detection,
                                                               activated, refind)
        mode = str(getattr(self.args, "lost_recover", "none")).lower()
        if mode in ("none", "false", "0", "") or not len(u_track) or not len(u_detection):
            return u_track, u_detection
        max_lost = int(getattr(self.args, "lost_recover_max_lost", 0) or 0)
        lost_idx = [i for i in u_track if strack_pool[i].state == TrackState.Lost
                    and (max_lost <= 0 or (self.frame_id - strack_pool[i].end_frame) <= max_lost)]
        if not lost_idx:
            return u_track, u_detection
        lost_tracks = [strack_pool[i] for i in lost_idx]
        dets = [detections[j] for j in u_detection]
        if mode == "iou":
            thr = float(getattr(self.args, "lost_recover_iou", 0.05))
            cost = matching.iou_distance(lost_tracks, dets)                  # 1 - IoU
            matches, u_l, u_d = matching.linear_assignment(cost, thresh=1.0 - thr + 1e-9)
        elif mode == "maha":
            gate = float(getattr(self.args, "lost_recover_maha2", 5.991))
            det_xyah = np.array([d.tlwh_to_xyah(d.tlwh) for d in dets], dtype=np.float64)
            cost = np.ones((len(lost_tracks), len(dets)), dtype=np.float32)
            for i, t in enumerate(lost_tracks):
                kf = t.kalman_filter
                m2 = kf.gating_distance(t.mean, t.covariance, det_xyah, only_position=True)
                pc = np.asarray(t.tlwh, float)
                pcx, pcy = pc[0] + pc[2] / 2, pc[1] + pc[3] / 2
                for j, d in enumerate(dets):
                    if m2[j] <= gate:
                        b = np.asarray(d.tlwh, float)
                        d_norm = np.hypot(b[0] + b[2] / 2 - pcx, b[1] + b[3] / 2 - pcy) / max((b[2] + b[3]) / 2, 1e-6)
                        cost[i, j] = min(float(d_norm) / 4.0, 0.999)        # trong gate: cost < 1
            matches, u_l, u_d = matching.linear_assignment(cost, thresh=0.9999)
        else:
            raise ValueError(f"lost_recover: {mode!r} (none | iou | maha)")
        for il, jd in matches:
            self._apply_match(lost_tracks[il], dets[jd], activated, refind)     # re_activate -> refind, giu ID
        matched_tr = {lost_idx[il] for il, _ in matches}
        matched_de = {u_detection[jd] for _, jd in matches}
        u_track = [i for i in u_track if i not in matched_tr]
        u_detection = [j for j in u_detection if j not in matched_de]
        return u_track, u_detection


def with_lost_recovery(tracker_cls):
    if issubclass(tracker_cls, LostRecoveryMixin):
        return tracker_cls
    return type(f"Recover{tracker_cls.__name__}", (LostRecoveryMixin, tracker_cls), {})
