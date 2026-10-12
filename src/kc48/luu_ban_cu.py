"""
luu_ban_cu.py -- R48D vong sua: sao luu dau ra R48D ban truoc vong sua (kem sha256) truoc khi tinh lai ghi de.
Chay mot lan; da co manifest thi khong sao lai (giu nguyen ban cu dau tien).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def luu(files: list[Path], dich: Path) -> dict:
    mf = dich / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text(encoding="utf-8"))
    dich.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in files:
        if not p.exists():
            continue
        q = dich / p.name
        shutil.copy2(p, q)
        h1, h2 = (hashlib.sha256(x.read_bytes()).hexdigest() for x in (p, q))
        assert h1 == h2, p
        rows.append({"nguon": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": h1, "byte": p.stat().st_size})
    m = {"mo_ta": "ban R48D truoc vong sua F1-F3 (Codex review_003), sao nguyen byte truoc khi tinh lai",
         "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "file": rows}
    mf.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m
