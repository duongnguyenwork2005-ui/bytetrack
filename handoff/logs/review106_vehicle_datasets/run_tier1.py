"""Record observed earlier commands and capture the final deterministic integrity check."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'


def main():
    entries = [
        dict(command=f'{PYTHON} handoff/logs/review106_vehicle_datasets/audit_r106.py gt', exit=0,
             evidence='gt_cohort.json; gt_summary.csv; cohort_*.npz',
             observed='52 clips, 1093 clip-local IDs; GT-only phase completed before audit.',
             capture='Observed tool exit 0 in this session; original stdout not redirected.'),
        dict(command=f'{PYTHON} handoff/logs/review106_vehicle_datasets/audit_r106.py audit', exit=0,
             evidence='verification.json; combined_verified.csv; scenario_per_clip.csv; scenario_summary.csv',
             observed='156 clip/filter CLEAR parity checks; 15 COMBINED results.',
             capture='Observed tool session 43373 completion exit 0; original stdout not redirected.'),
        dict(command=f'{PYTHON} handoff/logs/review106_vehicle_datasets/fetch_sources.py', exit=0,
             evidence='sources_manifest.json; source_*.txt',
             observed='12 source requests completed: 10 HTTP 200, 2 HTTP 404. HTTP 200 is not dataset download success.',
             capture='Observed tool exit 0; this script records HTTP failures as evidence, not process failure.'),
        dict(command=f'{PYTHON} handoff/logs/review106_vehicle_datasets/verify_evidence.py', exit=1,
             evidence='Initial assertion at units == meta[units] failed.',
             observed='Verifier incorrectly compared whole HELMET unit certificate hashes across filters; these include filter settings. Corrected to compare cache_marker_sha256 per clip and verify each filter certificate/MOT hash.',
             capture='Observed initial tool exit 1; no tracking/data change.'),
        dict(command='urllib HEAD https://gram.web.uah.es/data/datasets/rtm/GRAM-RTMv4.zip', exit=1,
             evidence='gram_manifest.json prior_head', observed='HTTP 403 for HEAD; public GET with the task user agent subsequently returned 200.',
             capture='Observed tool exit 1; no authentication or access-gate bypass.'),
        dict(command=f'{PYTHON} handoff/logs/review106_vehicle_datasets/check_gram.py', exit=0,
             evidence='gram_manifest.json; gram_annotation_kit.zip; gram_page.txt',
             observed='Public page and bounded annotation/devkit archive downloaded; kit code never executed.',
             capture='Observed tool session 31128 completion exit 0.'),
        dict(command=f'{PYTHON} handoff/logs/review106_vehicle_datasets/audit_gram_sample.py', exit=0,
             evidence='gram_sample_gt_audit.json; gram_Urban1_prefix250_gt.csv',
             observed='250 selected XML member CRC/shape/ID checks; 752 boxes, 8 IDs, no duplicate frame/ID; zero turn25 boxes; no tracking score.',
             capture='Observed tool exit 0; original stdout not redirected.'),
    ]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [PYTHON, str(HERE / 'verify_evidence.py')]
    start = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    (HERE / 'tier1_integrity.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    entries.append(dict(command=command, started_utc=start, exit=result.returncode,
                        evidence='integrity_verified.json; tier1_integrity.log', capture='subprocess captured stdout/stderr'))
    record = dict(checks=entries, final_success=result.returncode == 0,
                  not_run=['new detector inference/GPU/training', 'tracker rerun', 'new dataset tracking',
                           'independent Codex review (orchestrator responsibility)', 'rehash original images/NPZ archives',
                           'full GRAM archive CRC (selected XML CRC only)'],
                  process_policy='All child subprocesses waited; no background jobs opened.')
    (HERE / 'tier1_checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result.stdout, end='')
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
