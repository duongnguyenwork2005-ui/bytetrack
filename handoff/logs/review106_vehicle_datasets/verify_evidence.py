"""Independent aggregation and lineage checks for the R106 evidence bundle."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_r106 as audit
from trackeval.metrics import HOTA, Identity, CLEAR


def main():
    verification = audit.read_json(HERE / 'verification.json')
    for path, digest in verification['inputs'].items():
        assert audit.sha(audit.ROOT / path) == digest, path
    assert verification['clip_filter_checks'] == 156
    sources = audit.read_json(HERE / 'sources_manifest.json')
    for record in sources:
        if 'file' in record:
            path = HERE / record['file']
            assert path.stat().st_size == record['bytes'] and audit.sha(path) == record['sha256']
    kitti = (HERE / 'source_kitti_tracking.txt').read_text(encoding='utf-8')
    assert 'href="user_login.php" target="_blank">Download training labels' in kitti
    assert all(x['status'] == 404 for x in sources if x['name'] in ('sherbrooke_author', 'urbantracker_author_alt'))
    gram = audit.read_json(HERE / 'gram_manifest.json')
    for key in ('page', 'annotation_kit'):
        record = gram[key]
        assert record['status'] == 200 and audit.sha(HERE / record['file']) == record['sha256']
    download_bytes = sum(r.get('bytes', 0) for r in sources) + sum(gram[k]['bytes'] for k in ('page', 'annotation_kit'))
    assert download_bytes <= 25 * 1024 * 1024
    sample = audit.read_json(HERE / 'gram_sample_gt_audit.json')
    assert sample['frames'] == 250 and not sample['duplicate_frame_ids']
    assert audit.sha(HERE / 'gram_Urban1_prefix250_gt.csv') == sample['derived_gt_sha256']
    lineage, direct = [], []
    checkpoint = audit.sha(audit.ROOT / 'models/yolov8n.pt')
    for name, split, prefix, sequences in audit.cohorts():
        cache, common, cache_markers = None, None, {}
        for method in audit.METHODS:
            run = f'{prefix}-yolov8n_coco-c25-{method}'
            folder = audit.ROOT / 'data/processed/trackers' / split / run
            meta = audit.read_json(folder / 'run_meta.json')
            assert meta['status'] == 'complete' and meta['elapsed_mode'] == 'steps'
            assert meta['seed'] is None and meta['max_det'] == 300
            cfg = meta['tracker_cfg']
            core = {key: cfg[key] for key in ('track_high_thresh', 'track_low_thresh', 'new_track_thresh',
                                             'track_buffer', 'match_thresh', 'fuse_score')}
            if common is None:
                common = core
            assert common == core
            assert meta.get('conf_min', meta.get('threshold')) == .25
            if name == 'HELMET':
                assert meta['detector'] == 'yolov8n_coco' and meta['classes'] == [3]
                assert set(meta['videos']) == set(sequences)
                for seq in sequences:
                    certificate_path = audit.ROOT / 'handoff/logs/external_eval/r66/helmet/full/units' / run / seq / 'unit.json'
                    assert audit.sha(certificate_path) == meta['units'][seq]
                    certificate = audit.read_json(certificate_path)
                    marker = certificate['cache_marker_sha256']
                    if seq not in cache_markers:
                        cache_markers[seq] = marker
                    assert marker == cache_markers[seq]
                    mot = folder / 'data' / f'{seq}.txt'
                    assert audit.sha(mot) == certificate['sha256'][f'data/{seq}.txt']
                phase = audit.read_json(audit.ROOT / 'handoff/logs/external_eval/r66/helmet/full/phase_inputs.json')
                cache_id = phase['cache_ids']['yolov8n_coco']
            else:
                cache_id = meta['cache_id']
                assert meta['cache_meta']['weights_sha256'] == checkpoint
                assert meta['cache_meta']['conf'] == .05
                if cache is None:
                    cache = meta['cache_meta']
                assert cache == meta['cache_meta']
            cache_meta = audit.read_json(audit.ROOT / 'data/interim/det_cache' / cache_id / 'meta.json')
            assert cache_meta['weights_sha256'] == checkpoint
            score_folder = audit.ROOT / 'results/trackeval' / split / run
            raw = audit.read_json(score_folder / 'raw_fields.json')
            # LocA was not saved in historical sufficient statistics. Neutral fill is
            # confined to that unused localization field; HOTA/DetA/AssA do not depend on it.
            hota_input = {s: {**{k: np.asarray(v) if isinstance(v, list) else v for k, v in r.items()},
                              'LocA': np.zeros(19)} for s, r in raw.items()}
            h = HOTA().combine_sequences(hota_input)
            i = Identity({'PRINT_CONFIG': False}).combine_sequences(raw)
            combined = audit.combine(raw)
            assert abs(h['HOTA'].mean() - combined['HOTA']) < 1e-12
            assert abs(i['IDF1'] - combined['IDF1']) < 1e-12
            direct.append({'dataset': name, 'filter': method, 'HOTA': float(h['HOTA'].mean()), 'IDF1': float(i['IDF1'])})
            lineage.append(dict(dataset=name, filter=method, cache_id=cache_id,
                                checkpoint_sha256=checkpoint, tracker_cfg=cfg,
                                yaml_hash=meta.get('yaml_sha256', meta.get('tracker_yaml_sha256_12')),
                                core_settings_equal=True,
                                evidence='stored run metadata + cache metadata; no image/NPZ rehash'))
    # Association event locations: a change after an empty prediction frame must
    # occur on the returning match, with one FN and no event on initialization.
    data = dict(num_timesteps=4, num_gt_ids=1, num_tracker_ids=2, num_gt_dets=4, num_tracker_dets=3,
                gt_ids=[np.array([0]) for _ in range(4)],
                tracker_ids=[np.array([0]), np.array([], dtype=int), np.array([1]), np.array([1])],
                similarity_scores=[np.ones((1, 1)), np.empty((1, 0)), np.ones((1, 1)), np.ones((1, 1))])
    matched, changed = audit.trace(data, (4, 1))
    reference = CLEAR({'PRINT_CONFIG': False}).eval_sequence(data)
    assert changed[:, 0].tolist() == [False, False, True, False]
    assert (~matched).sum() == reference['CLR_FN'] == 1 and changed.sum() == reference['IDSW'] == 1
    # Scenario summaries must agree with the per-clip counts and ALL with COMBINED.
    with (HERE / 'scenario_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    with (HERE / 'combined_verified.csv').open() as stream:
        combined = list(csv.DictReader(stream))
    for result in combined:
        row = next(r for r in summary if (r['dataset'], r['filter'], r['group']) ==
                   (result['dataset'], result['filter'], 'all'))
        assert int(row['IDSW']) == int(result['IDSW']) and int(row['FN']) == int(result['FN'])
    audit.sha(Path(__file__))
    audit.write_json('integrity_verified.json', dict(lineage=lineage, trackeval_direct=direct,
                                                    clear_gap_fixture=True, sources_checked=len(sources),
                                                    download_bytes=download_bytes, gram_gt_only_checked=True,
                                                    inputs=audit.INPUTS))
    print('VERIFIED: input/source hashes; 12 run/cache lineages; 12 direct TrackEval combinations; CLEAR gap fixture; 15 ALL/COMBINED count checks; GRAM GT sample; downloads <=25 MiB')


if __name__ == '__main__':
    main()
