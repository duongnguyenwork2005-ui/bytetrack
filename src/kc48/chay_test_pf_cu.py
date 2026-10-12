"""
chay_test_pf_cu.py -- R48C: chay cac test cua src/test_pf_ctrv.py (goi dung ham goc, khong sua file goc), TRU test 11.

`test_11_runner_seeds` ghi fixture vao data/interim/det_cache/fixture_pf_seed_test va
data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1} -- ngoai write_paths cua R48C. Cach chay test nay dang cho
nguoi dung quyet dinh (de xuat da gui), nen o day chi BAO la CHUA CHAY. Cac test con lai chay theo thu tu dinh nghia
trong file goc, cung cach bat loi nhu main() goc. Exit 0 khi moi test da chay deu dat.

    python src/kc48/chay_test_pf_cu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import test_pf_ctrv as T  # noqa: E402

BO_QUA = {"test_11_runner_seeds": "ghi ngoai write_paths (data/interim/det_cache, data/processed/trackers); "
                                  "cho nguoi dung quyet dinh"}


def main() -> int:
    T.tracker_ctrv.register()
    tests = sorted((f for n, f in vars(T).items() if n.startswith("test_") and callable(f)),
                   key=lambda f: f.__code__.co_firstlineno)
    failed, ran = 0, []
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        if t.__name__ in BO_QUA:
            print(f"  [CHUA CHAY] {BO_QUA[t.__name__]}")
            continue
        ran.append(t.__name__)
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  !! {e}")
    n_ok = sum(ok for _, ok in T._RESULTS)
    print("\n" + "=" * 78)
    print(f"KET QUA: {len(ran)}/{len(tests)} test da chay, {n_ok}/{len(T._RESULTS)} phep kiem tra PASS, "
          f"{failed} test co loi; chua chay: {', '.join(sorted(BO_QUA))}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
