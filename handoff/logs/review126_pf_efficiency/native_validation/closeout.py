"""Final integrity check for R126 speed/quality and the rejected R127 idea."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
R127 = HERE.parent.parent/'review127_pf_stratified'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    final = read(HERE/'final_verification.json')
    for p,h in final['sha256'].items():
        assert sha(p) == h, p
    all_timing = read(HERE/'all_filters_timing.json')
    assert all_timing['prefix_parity'] == 24
    rows = all_timing['rows']
    assert len(rows) == len({(r['repeat'],r['video'],r['mode']) for r in rows}) == 24
    for mode, mean in all_timing['mean_update_ms'].items():
        matching = [r['mean_update_ms'] for r in rows if r['mode']==mode]
        assert len(matching) == 6 and abs(sum(matching)/6-mean) < 1e-10
    for p,h in read(HERE/'all_filters_timing_inputs.json')['sha256'].items():
        assert sha(p) == h, p
    dev = read(R127/'dev_summary.json')
    assert dev['pass'] is False and not any(dev['gates'].values())
    assert read(R127/'verification.json')['units'] == 36
    for p,h in read(R127/'inputs.json')['sha256'].items():
        assert sha(p) == h, p
    frozen = read(R127/'inputs.json')
    outputs = []
    for seed in range(3):
        outputs += [R127/f'dev/evaluation/cartesian-s{seed}/raw_fields.json',
                    R127/f'dev/evaluation/cartesian-s{seed}/combined.json']
        for video in frozen['frames']:
            folder = R127/f'dev/tracking/DETRAC-all/cartesian-s{seed}'
            mot, marker = folder/f'data/{video}.txt', folder/f'meta/{video}.json'
            saved = read(marker)
            assert saved['frames'] == frozen['frames'][video] and saved['mot_sha256'] == sha(mot)
            outputs += [mot, marker]
    outputs += [Path(__file__), HERE/'all_filters_timing.json', HERE/'all_filters_timing_inputs.json',
                HERE/'final_verification.json', R127/'inputs.json', R127/'smoke.json',
                R127/'dev_summary.json', R127/'verification.json']
    report = {'status': 'PASS', 'R126_quality_units': 600, 'R126_main_serial_timings': 18,
              'R126_all_filter_timings': 24, 'R127_quality_units': 36,
              'R127_selected': False, 'R127_BDD_started': False,
              'all_filters_mean_update_ms': all_timing['mean_update_ms'],
              'sha256': {str(p): sha(p) for p in outputs}}
    (HERE/'closeout_verification.json').write_text(json.dumps(report, indent=2))
    print('CLOSEOUT PASS', report['R126_quality_units'], report['R127_quality_units'], flush=True)


if __name__ == '__main__':
    main()
