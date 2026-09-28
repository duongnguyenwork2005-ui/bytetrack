"""
nano_replay.py -- Replay det cache nam NGOAI thu muc mac dinh bang CHINH `run_tracking.main()` (review 9f210e2, N21-F2).

`nano_live.py --save-cache ID --cache-root R` ghi cache vao R/ID; `run_tracking.py` chi tim duoi
`det_cache.CACHE_ROOT` mac dinh. Thay vi sua runner dung chung, wrapper nay dat `det_cache.CACHE_ROOT = R` (moi cho
dung cache -- `run_tracking.main`, `det_cache.open_cache`, `run_fingerprint.cache_fingerprint` -- deu doc thuoc tinh
module nay LUC GOI) roi chay `run_tracking.main()` voi phan doi so con lai, KHONG doi gi khac.

    python src/nano_replay.py --cache-root "/duong dan/co dau cach" -- --cache-id ID --split-name S ... (doi so run_tracking)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_cache  # noqa: E402
import run_tracking  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "--cache-root" or "--" not in argv:
        print("dung: nano_replay.py --cache-root <thu muc> -- <doi so run_tracking.py>")
        return 2
    root = Path(argv[1]).resolve()
    rest = argv[argv.index("--") + 1:]
    if not root.is_dir():
        print(f"[LOI] cache-root khong ton tai: {root}")
        return 2
    det_cache.CACHE_ROOT = root
    print(f"[nano_replay] det_cache.CACHE_ROOT = {root}")
    sys.argv = ["run_tracking.py", *rest]
    return run_tracking.main()


if __name__ == "__main__":
    raise SystemExit(main())
