"""Summarize the locked 40-video test experiment without running evaluation.

Source CSVs are read-only. TrackEval bootstrap replicates recombine sequence
sufficient statistics, using the installed official combine_sequences formulas;
they are not averages of per-video HOTA/IDF1. All comparisons resample the same
videos for both configurations. See the generated metadata.json for units,
rosters, source hashes, and numerical validation against the original results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from trackeval.metrics import CLEAR, HOTA, Identity


KEY = ["video", "track_id", "event_id"]
EVENT_FIELDS = KEY + ["start_frame", "end_frame", "length_frames", "has_full",
                      "eligible", "exclusion_reason"]
OUTCOMES = ["preserved", "switched", "lost_after", "no_match_before"]
REASONS = ["gt_missing_before", "gt_missing_after"]
TRACK_METRICS = ["HOTA", "IDF1", "AssA", "DetA", "LocA", "MOTA", "FP", "FN",
                 "IDSW", "GT_dets"]
COUNT_METRICS = {"FP", "FN", "IDSW", "GT_dets", "events_total", "events_eligible",
                 "events_excluded", "id_continuous_count"}
COUNT_METRICS.update(REASONS)
COUNT_METRICS.update(f"{outcome}_count" for outcome in OUTCOMES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def assert_roster(frame: pd.DataFrame, column: str, videos: list[str], label: str) -> None:
    require(not frame[column].duplicated().any(), f"Duplicate video rows: {label}")
    require(set(frame[column]) == set(videos), f"Incomplete video roster: {label}")


def assert_close(actual, expected, label: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12,
                               err_msg=label)


def sample_weights(n_videos: int, n_boot: int, seed: int) -> np.ndarray:
    # Drawing a rectangular array consumes the same RNG sequence as the existing
    # bootstrap_diff_by_video loop of rng.choice(videos, size=len(videos)).
    rng = np.random.default_rng(seed)
    picks = rng.choice(n_videos, size=(n_boot, n_videos), replace=True)
    weights = np.zeros((n_boot, n_videos), dtype=np.int64)
    np.add.at(weights, (np.arange(n_boot)[:, None], picks), 1)
    return weights


def alpha_columns(field: str) -> list[str]:
    return [f"{field}___{alpha}" for alpha in range(5, 100, 5)]


def pooled_tracking(detail: pd.DataFrame, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized equivalent of official HOTA/CLEAR/Identity sequence combine."""
    tp_video = detail[alpha_columns("HOTA_TP")].to_numpy(dtype=float)
    tp = weights @ tp_video
    fn = weights @ detail[alpha_columns("HOTA_FN")].to_numpy(dtype=float)
    fp = weights @ detail[alpha_columns("HOTA_FP")].to_numpy(dtype=float)
    assa = (weights @ (detail[alpha_columns("AssA")].to_numpy(dtype=float) * tp_video)
            / np.maximum(1, tp))
    loca = np.maximum(1e-10, weights @ (
        detail[alpha_columns("LocA")].to_numpy(dtype=float) * tp_video)) / np.maximum(1e-10, tp)
    deta = tp / np.maximum(1, tp + fn + fp)
    sums = {field: weights @ detail[field].to_numpy(dtype=float)
            for field in ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "IDTP", "IDFN", "IDFP"]}
    return {
        "HOTA": np.sqrt(deta * assa).mean(axis=1),
        "IDF1": sums["IDTP"] / np.maximum(
            1, sums["IDTP"] + 0.5 * sums["IDFN"] + 0.5 * sums["IDFP"]),
        "AssA": assa.mean(axis=1), "DetA": deta.mean(axis=1), "LocA": loca.mean(axis=1),
        "MOTA": (sums["CLR_TP"] - sums["CLR_FP"] - sums["IDSW"]) / np.maximum(
            1, sums["CLR_TP"] + sums["CLR_FN"]),
        "FP": sums["CLR_FP"], "FN": sums["CLR_FN"], "IDSW": sums["IDSW"],
        "GT_dets": sums["CLR_TP"] + sums["CLR_FN"],
    }


def official_tracking(detail: pd.DataFrame, picks: list[int]) -> dict[str, float]:
    """Independent validation via installed TrackEval APIs, including repeats."""
    hota = HOTA()
    clear = CLEAR({"PRINT_CONFIG": False})
    identity = Identity({"PRINT_CONFIG": False})
    h_res, c_res, i_res = {}, {}, {}
    for copy_id, row_idx in enumerate(picks):
        row = detail.iloc[row_idx]
        h_res[copy_id] = {field: row[alpha_columns(field)].to_numpy(dtype=float)
                          for field in hota.integer_array_fields +
                          ["AssA", "AssRe", "AssPr", "LocA"]}
        c_res[copy_id] = {field: float(row[field]) for field in clear.summed_fields}
        i_res[copy_id] = {field: float(row[field]) for field in identity.integer_fields}
    h, c, i = (hota.combine_sequences(h_res), clear.combine_sequences(c_res),
               identity.combine_sequences(i_res))
    return {"HOTA": float(h["HOTA"].mean()), "IDF1": float(i["IDF1"]),
            "AssA": float(h["AssA"].mean()), "DetA": float(h["DetA"].mean()),
            "LocA": float(h["LocA"].mean()), "MOTA": float(c["MOTA"]),
            "FP": float(c["CLR_FP"]), "FN": float(c["CLR_FN"]),
            "IDSW": float(c["IDSW"]), "GT_dets": float(c["CLR_TP"] + c["CLR_FN"])}


def outcome_statistics(events: pd.DataFrame, roster: list[str],
                       eligible_roster: list[str], all_weights: np.ndarray,
                       eligible_weights: np.ndarray) -> tuple[dict, dict]:
    def counts(mask: pd.Series, videos: list[str]) -> np.ndarray:
        return events.loc[mask].groupby("video").size().reindex(videos, fill_value=0).to_numpy()

    eligible = events.eligible
    gt_counts = {
        "events_total": counts(pd.Series(True, index=events.index), roster),
        "events_eligible": counts(eligible, roster),
        "events_excluded": counts(~eligible, roster),
        **{reason: counts(events.exclusion_reason.eq(reason), roster) for reason in REASONS},
    }
    observed = {field: float(value.sum()) for field, value in gt_counts.items()}
    bootstrap = {field: all_weights @ value for field, value in gt_counts.items()}
    denominator = counts(eligible, eligible_roster)
    sampled_denominator = eligible_weights @ denominator
    for outcome in OUTCOMES + ["id_continuous"]:
        mask = events.id_continuous if outcome == "id_continuous" else events.outcome.eq(outcome)
        numerator = counts(eligible & mask, eligible_roster)
        name = "id_match" if outcome == "preserved" else (
            "id_continuous" if outcome == "id_continuous" else f"{outcome}_rate")
        observed[f"{outcome}_count"] = float(numerator.sum())
        observed[name] = float(numerator.sum() / denominator.sum())
        bootstrap[f"{outcome}_count"] = eligible_weights @ numerator
        bootstrap[name] = bootstrap[f"{outcome}_count"] / sampled_denominator
    return observed, bootstrap


def ci_row(metric: str, observed: float, samples: np.ndarray,
           eligible_videos: int, n_boot: int, seed: int) -> dict:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    event_rate_or_outcome = metric not in TRACK_METRICS and metric not in {
        "events_total", "events_eligible", "events_excluded", *REASONS}
    return dict(metric=metric, estimate=observed, ci_low=float(lo), ci_high=float(hi),
                unit="count" if metric in COUNT_METRICS else "proportion",
                n_videos=eligible_videos if event_rate_or_outcome else 40,
                n_boot=n_boot, seed=seed, ci_method="video_cluster_percentile_95")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/test_eval/manifest.json")
    parser.add_argument("--out-dir", default="results/test_eval/summary")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    require(args.n_boot > 0, "n-boot must be positive")
    root = Path(__file__).resolve().parents[1]
    manifest_path = (root / args.manifest).resolve()
    out_dir = (root / args.out_dir).resolve()
    require(out_dir == root / "results/test_eval/summary", "Only the dedicated new summary directory may be written")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    videos = manifest["danh_sach_video"]
    require(len(videos) == len(set(videos)) == manifest["so_video_test"] == 40,
            "The locked manifest must contain exactly 40 distinct videos")
    require(manifest["motion_model"] == ["cv", "ekf_ctrv", "ukf_ctrv"], "Unexpected model roster")
    # These are the literal templates in the locked manifest, not guessed paths.
    require(manifest["thu_muc_ket_qua"] == {
        "tracking": "data/processed/trackers/DETRAC-test/testA-<mm>, testB-<mm>",
        "trackeval": "results/trackeval/DETRAC-test/test{A,B}-<mm>/",
        "occlusion_eval": "results/test_eval/{A,B}/"}, "Unexpected manifest path templates")
    trackers = {f"test{run}-{model}": (run, model) for run in ["A", "B"]
                for model in manifest["motion_model"]}
    source_paths = [manifest_path]
    paths = {}
    for run in ["A", "B"]:
        paths[run] = {kind: root / f"results/test_eval/{run}/{kind}_DETRAC-test.csv"
                      for kind in ["events", "event_outcomes", "paired", "bootstrap"]}
        source_paths.extend(paths[run].values())
    for tracker in trackers:
        paths[tracker] = {
            "detailed": root / f"data/processed/trackers/DETRAC-test/{tracker}/pedestrian_detailed.csv",
            **{kind: root / f"results/trackeval/DETRAC-test/{tracker}/{kind}_metrics.csv"
               for kind in ["combined", "per_video"]}}
        source_paths.extend(paths[tracker].values())
    missing = [str(path.relative_to(root)) for path in source_paths if not path.is_file()]
    require(not missing, "Missing evaluation inputs; no summary written: " + ", ".join(missing))
    hashes_before = {str(path.relative_to(root)): sha256(path) for path in source_paths}

    loaded = {name: {kind: read_csv(path) for kind, path in entries.items()}
              for name, entries in paths.items()}
    reference_events = loaded["A"]["events"]
    require(not reference_events.duplicated(KEY).any(), "Duplicate event keys")
    require(set(reference_events.video).issubset(videos), "Unexpected event videos")
    require(reference_events.eligible.dtype == bool, "Eligibility must be boolean")
    require((reference_events.eligible == (reference_events.gt_before & reference_events.gt_after)).all(),
            "GT eligibility inconsistent with before/after fields")
    for run in ["A", "B"]:
        events = loaded[run]["events"]
        pd.testing.assert_frame_equal(events, reference_events, check_dtype=False)
        paired = loaded[run]["paired"]
        require(not paired.duplicated(KEY).any(), f"Duplicate paired event keys in {run}")
        pd.testing.assert_frame_equal(paired[KEY], events.loc[events.eligible, KEY].reset_index(drop=True))
        require(set(loaded[run]["event_outcomes"].tracker) ==
                {f"test{run}-{model}" for model in manifest["motion_model"]},
                f"Incomplete event tracker roster in {run}")
    eligible_roster = loaded["A"]["paired"].video.unique().tolist()
    require(len(eligible_roster) == 39, "Expected existing 39-video eligible-event roster")
    require(eligible_roster == loaded["B"]["paired"].video.unique().tolist(), "Paired video orders differ")
    weights = sample_weights(len(videos), args.n_boot, args.seed)
    eligible_weights = sample_weights(len(eligible_roster), args.n_boot, args.seed)
    observed_by_tracker, bootstrap_by_tracker = {}, {}
    integrity_rows = []
    for tracker, (run, model) in trackers.items():
        data = loaded[tracker]
        detail_all = data["detailed"]
        detail = detail_all.loc[detail_all.seq != "COMBINED"]
        assert_roster(detail, "seq", videos, tracker + " detailed")
        detail = detail.set_index("seq").loc[videos].reset_index()
        per_video = data["per_video"]
        assert_roster(per_video, "video", videos, tracker + " per_video")
        per_video = per_video.set_index("video").loc[videos]
        require(len(data["combined"]) == 1 and data["combined"].iloc[0].video == "COMBINED_SEQ",
                f"Invalid combined CSV for {tracker}")
        require(len(detail_all.loc[detail_all.seq == "COMBINED"]) == 1,
                f"Invalid detailed combined row for {tracker}")
        point = pooled_tracking(detail, np.ones((1, len(videos))))
        samples = pooled_tracking(detail, weights)
        individual = pooled_tracking(detail, np.eye(len(videos)))
        official = official_tracking(detail, list(range(len(videos))))
        for metric in TRACK_METRICS:
            assert_close(point[metric][0], official[metric], tracker + " official " + metric)
            if metric in data["combined"].columns:
                assert_close(point[metric][0], data["combined"].iloc[0][metric], tracker + " combined " + metric)
                assert_close(individual[metric], per_video[metric], tracker + " per_video " + metric)
        for metric in ["HOTA", "AssA", "DetA", "LocA"]:
            assert_close(point[metric][0], detail_all.loc[detail_all.seq == "COMBINED"].iloc[0][metric + "___AUC"],
                         tracker + " detailed combined " + metric)
        # Verify true resampling with duplicate sequences via the official API.
        for replicate in [0, args.n_boot - 1]:
            picks = np.repeat(np.arange(len(videos)), weights[replicate]).tolist()
            actual = official_tracking(detail, picks)
            for metric in TRACK_METRICS:
                assert_close(samples[metric][replicate], actual[metric], tracker + " bootstrap " + metric)

        outcomes = loaded[run]["event_outcomes"]
        events = outcomes.loc[outcomes.tracker == tracker].reset_index(drop=True)
        require(not events.duplicated(KEY).any(), "Duplicate outcome keys: " + tracker)
        pd.testing.assert_frame_equal(events[EVENT_FIELDS], reference_events[EVENT_FIELDS], check_dtype=False)
        require(events.id_continuous.dtype == bool, "id_continuous must be boolean: " + tracker)
        require(events.loc[events.eligible, "outcome"].isin(OUTCOMES).all(), "Unscored eligible event: " + tracker)
        require(events.loc[~events.eligible, "exclusion_reason"].isin(REASONS).all(), "Unknown exclusion: " + tracker)
        require(events.loc[events.eligible, "exclusion_reason"].eq("").all(), "Excluded eligible event: " + tracker)
        require(events.loc[~events.eligible, "outcome"].eq("").all(), "Scored excluded event: " + tracker)
        require((~events.id_continuous | events.outcome.eq("preserved")).all(), "Continuous non-preserved event: " + tracker)
        paired = loaded[run]["paired"]
        ev = events.loc[events.eligible].reset_index(drop=True)
        require(ev.outcome.equals(paired[f"out__{tracker}"]), "Paired outcome mismatch: " + tracker)
        assert_close(ev.outcome.eq("preserved").astype(float), paired[f"pres__{tracker}"], "Paired preserved")
        assert_close(ev.id_continuous.astype(float), paired[f"cont__{tracker}"], "Paired continuous")
        event_observed, event_samples = outcome_statistics(events, videos, eligible_roster, weights, eligible_weights)
        observed_by_tracker[tracker] = {**{key: float(value[0]) for key, value in point.items()}, **event_observed}
        bootstrap_by_tracker[tracker] = {**samples, **event_samples}
        integrity_rows.append(dict(tracker=tracker, trackeval_videos=len(detail),
                                   event_videos=events.video.nunique(), eligible_videos=len(eligible_roster),
                                   event_keys_unique=True, event_keys_and_gt_eligibility_identical=True,
                                   paired_outcomes_identical=True, pooled_matches_official=True,
                                   per_video_and_combined_match_csv=True,
                                   bootstrap_replicates_match_official=True))

    config_rows, config_ci_rows, contrast_rows = [], [], []
    for tracker, (run, model) in trackers.items():
        info = dict(tracker=tracker, run=run, model=model, conf=manifest["dieu_kien"][run]["conf"])
        config_rows.append({**info, **observed_by_tracker[tracker]})
        for metric, value in observed_by_tracker[tracker].items():
            config_ci_rows.append({**info, **ci_row(metric, value, bootstrap_by_tracker[tracker][metric],
                                                   len(eligible_roster), args.n_boot, args.seed)})
    comparisons = [(f"test{run}-{model}", f"test{run}-cv")
                   for run in ["A", "B"] for model in ["ekf_ctrv", "ukf_ctrv"]]
    comparisons += [(f"testB-{model}", f"testA-{model}") for model in manifest["motion_model"]]
    old_ci_validation = []
    for treatment, baseline in comparisons:
        for metric in observed_by_tracker[treatment]:
            delta = observed_by_tracker[treatment][metric] - observed_by_tracker[baseline][metric]
            boot_delta = bootstrap_by_tracker[treatment][metric] - bootstrap_by_tracker[baseline][metric]
            # Existing event routine computes (sum_a-sum_b)/n. Using count
            # differences also reproduces its floating-point operation order.
            if metric == "id_match" or metric == "id_continuous" or metric.endswith("_rate"):
                count_metric = {"id_match": "preserved_count", "id_continuous": "id_continuous_count"}.get(
                    metric, metric.removesuffix("_rate") + "_count")
                denominator = loaded["A"]["paired"].groupby("video").size().reindex(eligible_roster).to_numpy()
                boot_delta = (bootstrap_by_tracker[treatment][count_metric] -
                              bootstrap_by_tracker[baseline][count_metric]) / (eligible_weights @ denominator)
            row = {"contrast": treatment + " minus " + baseline, "treatment": treatment,
                   "baseline": baseline, **ci_row(metric, delta, boot_delta, len(eligible_roster), args.n_boot, args.seed)}
            row["ci_method"] = "paired_video_cluster_percentile_95"
            row["ci_contains_zero"] = row["ci_low"] <= 0 <= row["ci_high"]
            contrast_rows.append(row)
            if metric == "id_match" and treatment[:5] == baseline[:5] and args.n_boot == 5000 and args.seed == 0:
                run = trackers[treatment][0]
                old = loaded[run]["bootstrap"]
                old = old.loc[(old.tracker == treatment) & (old.baseline == baseline)]
                require(len(old) == 1, "Existing bootstrap comparison missing")
                old = old.iloc[0]
                assert_close([row["estimate"], row["ci_low"], row["ci_high"]],
                             [old["diff"], old.ci_low, old.ci_high], "Existing id_match bootstrap")
                require(float(old.ci_low) == row["ci_low"] and float(old.ci_high) == row["ci_high"],
                        "Existing bootstrap CI not reproduced exactly")
                old_ci_validation.append(dict(treatment=treatment, baseline=baseline,
                                               exact_ci_match=True, n_videos=int(old.n_videos),
                                               n_events=int(old.n_events)))
    hashes_after = {str(path.relative_to(root)): sha256(path) for path in source_paths}
    require(hashes_before == hashes_after, "Source evaluation files changed while summarizing")
    metric_sources = {}
    for metric in [HOTA, CLEAR, Identity]:
        module_path = Path(inspect.getfile(metric))
        metric_sources[metric.__name__] = {"path": str(module_path), "sha256": sha256(module_path)}
    metadata = {
        "manifest": str(manifest_path.relative_to(root)), "analysis_script": "src/summarize_test_evaluation.py",
        "analysis_script_sha256": sha256(Path(__file__)),
        "inference_commit": manifest["git"]["commit"], "n_boot": args.n_boot, "seed": args.seed,
        "ci": "Two-sided percentile 95%; numpy.percentile with default linear method; no multiplicity adjustment",
        "pairing": "Same video multiplicities for treatment and baseline in each contrast",
        "tracking": {"roster": videos, "n_videos": len(videos),
                     "method": "Pool sequence sufficient statistics using official HOTA/CLEAR/Identity combine formulas, then average HOTA submetrics over 19 alpha values (0.05 to 0.95). Identity assignment remains within each sampled sequence copy.",
                     "validation": "All 6 configurations match official combine_sequences, exported per-video/combined metrics, detailed combined AUC; first and last bootstrap replicate independently match official API."},
        "occlusion_rates_and_outcome_counts": {"roster": eligible_roster, "n_videos": len(eligible_roster),
                                               "method": "Sample eligible-event videos; event-weighted sums / eligible-event counts, retaining each sampled video's complete event cluster.",
                                               "existing_bootstrap_validation": old_ci_validation},
        "gt_event_counts": {"roster": videos, "n_videos": len(videos),
                            "method": "All manifest videos, including the video without eligible events; sum event totals, eligible counts and exclusion reasons."},
        "units": "Ratios are fractions (multiply by 100 for percent; ratio contrasts by 100 for percentage points). FP/FN/IDSW and event counts are counts. Count bootstrap intervals describe counts in a resampled roster of the same size.",
        "scope": "Intervals reflect between-video sampling variation conditional on these fixed runs; not training or detector randomness. Config intervals are single-config cluster bootstrap; contrasts are paired cluster bootstrap.",
        "numpy": np.__version__, "pandas": pd.__version__, "trackeval": importlib.metadata.version("trackeval"),
        "trackeval_metric_sources": metric_sources, "source_sha256": hashes_before,
        "raw_detailed_copies": {tracker: f"trackeval_detailed/{tracker}.csv" for tracker in trackers},
        "source_files_unchanged_during_analysis": True,
    }
    # Write only after all input and numerical validations have succeeded.
    (out_dir / "trackeval_detailed").mkdir(parents=True, exist_ok=True)
    for tracker in trackers:
        shutil.copyfile(paths[tracker]["detailed"], out_dir / "trackeval_detailed" / f"{tracker}.csv")
    pd.DataFrame(config_rows).to_csv(out_dir / "configuration_metrics.csv", index=False)
    pd.DataFrame(config_ci_rows).to_csv(out_dir / "configuration_bootstrap.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out_dir / "paired_contrasts_bootstrap.csv", index=False)
    pd.DataFrame(integrity_rows).to_csv(out_dir / "validation.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(config_rows).to_string(index=False))
    print(f"Validated 6 configurations and {len(comparisons)} paired contrasts; wrote {out_dir}")


if __name__ == "__main__":
    main()
