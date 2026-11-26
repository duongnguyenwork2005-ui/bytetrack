"""Locate hot functions; profiled wall times are not speed measurements."""
from pathlib import Path
import cProfile
import io
import json
import pstats
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'review125_pf_bdd'))
import run as job


def main():
    names = json.loads((job.HERE / 'inputs.json').read_text())['timing_videos']
    for mode in ('pf100', 'classic500', 'kf'):
        profiler = cProfile.Profile()
        for name in names:
            profiler.runcall(job.replay, name, mode, None if mode == 'kf' else 0, 200)
        profiler.dump_stats(HERE / f'{mode}.prof')
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('cumtime').print_stats(35)
        (HERE / f'profile_{mode}.txt').write_text(text.getvalue(), encoding='utf-8')
        print(mode, text.getvalue()[:6500], flush=True)


if __name__ == '__main__':
    main()
