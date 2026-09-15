"""
trackeval_ext.py -- Mo rong TrackEval MotChallenge2DBox: VUNG BO QUA (crowd-ignore)
theo giao thuc BDD100K, dung cho split BDD-* xuat boi mot_datasets/bdd100k_mot.py.

===========================================================================
VAN DE
===========================================================================
MotChallenge2DBox (trackeval 1.3.0, datasets/mot_challenge_2d_box.py) KHONG co vung bo
qua: `gt_crowd_ignore_regions` luon rong (dong 275). Cach duy nhat de "bo qua" mot box GT
la gan no lop distractor (2/7/8/12): tracker det KHOP (IoU >= 0.5, Hungarian) voi box do
bi loai (dong 382-389). Nhung box `crowd` cua BDD100K la mot box LON bao ca nhom xe: mot
detection cua MOT xe ben trong co IoA ~ 1 nhung IoU << 0.5 voi box crowd -> KHONG bi loai
-> bi tinh la FP sai. Giao thuc BDD100K (trackeval/datasets/bdd100k.py dong 251-263):
    "For unmatched tracker dets, remove those that are greater than 50% within a crowd
     ignore region"  (IoA = dien tich giao / dien tich detection > 0.5).

===========================================================================
CACH LAM
===========================================================================
GT xuat boi bdd100k_mot.export_motchallenge ghi vung bo qua voi class = 13 ('crowd' trong
bang class_name_to_class_id cua TrackEval) va conf = 0.
    _load_raw_file(is_gt=True): tach cac dong class 13 ra khoi gt_* va dua vao
        gt_crowd_ignore_regions[t] (xywh) - TRUOC khi TrackEval tinh similarity, nen
        chung khong tham gia ghep cap.
    get_preprocessed_seq_data: giu nguyen 4 buoc cua MOT17 (loai det khop distractor,
        bo GT zero-marked / khac lop) va THEM: det CHUA KHOP co IoA > 0.5 voi vung bo qua
        -> loai (dung ham _calculate_box_ious(..., do_ioa=True) cua TrackEval).
Voi GT khong co class 13 (UA-DETRAC), lop nay cho ket qua TRUNG KHIT lop goc
(test_trackeval_ext.py). Dung qua run_trackeval.py --dataset-class mot_crowd.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackeval import _timing
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval.utils import TrackEvalException

CROWD_CLASS_ID = 13


class MotChallengeCrowdIgnore(MotChallenge2DBox):
    """MotChallenge2DBox + vung bo qua (class 13) loai FP theo IoA > 0.5 (giao thuc BDD100K)."""

    IGNORE_IOA_THRESHOLD = 0.5

    def _load_raw_file(self, tracker, seq, is_gt):
        raw = super()._load_raw_file(tracker, seq, is_gt)
        if not is_gt:
            return raw
        for t in range(raw["num_timesteps"]):
            classes = raw["gt_classes"][t]
            if len(classes) == 0:
                continue
            crowd = classes == CROWD_CLASS_ID
            if not crowd.any():
                continue
            raw["gt_crowd_ignore_regions"][t] = np.atleast_2d(raw["gt_dets"][t][crowd])
            keep = ~crowd
            raw["gt_dets"][t] = raw["gt_dets"][t][keep]
            raw["gt_ids"][t] = raw["gt_ids"][t][keep]
            raw["gt_classes"][t] = classes[keep]
            raw["gt_extras"][t]["zero_marked"] = raw["gt_extras"][t]["zero_marked"][keep]
        return raw

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Ban sao get_preprocessed_seq_data cua MotChallenge2DBox + buoc loai FP trong vung bo qua."""
        self._check_unique_ids(raw_data)

        distractor_class_names = ["person_on_vehicle", "static_person", "distractor", "reflection"]
        if self.benchmark == "MOT20":
            distractor_class_names.append("non_mot_vehicle")
        distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ["gt_ids", "tracker_ids", "gt_dets", "tracker_dets", "tracker_confidences", "similarity_scores"]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids, unique_tracker_ids = [], []
        num_gt_dets = num_tracker_dets = 0
        n_removed_ignore = 0
        for t in range(raw_data["num_timesteps"]):
            gt_ids = raw_data["gt_ids"][t]
            gt_dets = raw_data["gt_dets"][t]
            gt_classes = raw_data["gt_classes"][t]
            gt_zero_marked = raw_data["gt_extras"][t]["zero_marked"]
            tracker_ids = raw_data["tracker_ids"][t]
            tracker_dets = raw_data["tracker_dets"][t]
            tracker_classes = raw_data["tracker_classes"][t]
            tracker_confidences = raw_data["tracker_confidences"][t]
            similarity_scores = raw_data["similarity_scores"][t]
            ignore_regions = raw_data["gt_crowd_ignore_regions"][t]

            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    "Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at "
                    "timestep %i." % (np.max(tracker_classes), raw_data["seq"], t))

            to_remove_tracker = np.array([], int)
            matched_cols = np.array([], int)
            if self.do_preproc and self.benchmark != "MOT15" and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
                if len(invalid_classes) > 0:
                    raise TrackEvalException("Attempting to evaluate using invalid gt classes: "
                                             + " ".join(str(x) for x in invalid_classes))
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]
                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                to_remove_tracker = match_cols[is_distractor_class]
                matched_cols = match_cols

            # --- THEM (giao thuc BDD100K): det CHUA KHOP nam > 50% trong vung bo qua -> loai ---
            if self.do_preproc and tracker_ids.shape[0] > 0 and ignore_regions is not None and len(ignore_regions) > 0:
                unmatched = np.setdiff1d(np.arange(tracker_ids.shape[0]), matched_cols)
                if len(unmatched):
                    ioa = self._calculate_box_ious(tracker_dets[unmatched], np.atleast_2d(ignore_regions),
                                                   box_format="xywh", do_ioa=True)
                    inside = np.any(ioa > self.IGNORE_IOA_THRESHOLD + np.finfo("float").eps, axis=1)
                    extra = unmatched[inside]
                    n_removed_ignore += len(extra)
                    to_remove_tracker = np.unique(np.concatenate([to_remove_tracker, extra])).astype(int)

            data["tracker_ids"][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data["tracker_confidences"][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            if self.do_preproc and self.benchmark != "MOT15":
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & (np.equal(gt_classes, cls_id))
            else:
                gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            data["gt_ids"][t] = gt_ids[gt_to_keep_mask]
            data["gt_dets"][t] = gt_dets[gt_to_keep_mask, :]
            data["similarity_scores"][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[data["tracker_ids"][t]].astype(int)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]
        data["num_removed_in_ignore_regions"] = n_removed_ignore
        self._check_unique_ids(data, after_preproc=True)
        return data
