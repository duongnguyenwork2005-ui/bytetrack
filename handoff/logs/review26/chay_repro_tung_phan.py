# -*- coding: utf-8 -*-
"""Vong 26: goi TUNG HAM cua script Codex `CODEX_REPRO_20e66e9_ui.py` (KHONG sua script) bang interpreter hien tai.
Ly do: tren Nano khong co interpreter nao vua co OpenCV vua >= 3.7 -- `closed_preview` (N25-F1) can cv2 -> Python 3.6 he
thong (OpenCV 3.2, dung bo ve that tren thiet bi); `picker_close` (N25-F2) import nano_launcher (cu phap 3.8) -> env `nano`.
Cu phap Python 3.6.

    python chay_repro_tung_phan.py <CODEX_REPRO_20e66e9_ui.py> <closed_preview|picker_close> <goc_repo> <ket_qua.json>
exit 1 = loi tai hien (pass false), 0 = dat / skip.
"""
import importlib.util
import json
import os
import sys


def main():
    script, ham, repo, out = sys.argv[1:5]
    sys.path.insert(0, os.path.join(os.path.abspath(repo), "src"))
    spec = importlib.util.spec_from_file_location("codex_repro", script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = getattr(m, ham)()
    r["chay_boi"] = {"python": sys.version.split()[0], "ham": ham, "repo": os.path.abspath(repo)}
    s = json.dumps(r, ensure_ascii=False, indent=2)
    print(s)
    with open(out, "w") as f:
        f.write(s + "\n")
    return 1 if r.get("pass") is False else 0


if __name__ == "__main__":
    sys.exit(main())
