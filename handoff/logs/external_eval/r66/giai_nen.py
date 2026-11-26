"""R66: giai nen chi doc cac zip dataset moi sang o roi F: (anh giai nen = sinh lai duoc, LUU_TRU muc 5).
Moi zip -> F:/LABOTARY_data/<ten>/; kiem sha256 zip voi sha256_zip.txt truoc; chong path traversal; danh dau hoan tat
bang <ten>/.giai_nen_xong.json (so file, byte, sha256 zip). Da co marker dung -> bo qua."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

GOC = Path("D:/LABOTARY")
DICH = Path("F:/LABOTARY_data")
L = Path(__file__).resolve().parent
JOBS = {"sherbrooke_frames.zip": "sherbrooke", "sherbrooke_annotations.zip": "sherbrooke",
        "TrafficMOT-20261102T061326Z-1-001.zip": "trafficmot", "VisDrone2019-MOT-val.zip": "visdrone_mot",
        "VisDrone2019-MOT-test-dev.zip": "visdrone_mot"}
sha_ky_vong = {}
for dong in (L / "sha256_zip.txt").read_text().splitlines():
    h, p = dong.split(None, 1)
    sha_ky_vong[Path(p.lstrip("*")).name] = h


def sha(p: Path) -> str:
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            x.update(b)
    return x.hexdigest()


for z, ten in JOBS.items():
    out = DICH / ten
    mk = out / f".giai_nen_xong_{Path(z).stem}.json"
    h = sha(GOC / z)
    if h != sha_ky_vong[z]:
        sys.exit(f"[DUNG] sha256 {z} khac sha256_zip.txt")
    if mk.exists() and json.loads(mk.read_text())["sha256_zip"] == h:
        print("[bo qua]", z); continue
    out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(GOC / z)
    n = b = 0
    for i in zf.infolist():
        dst = (out / i.filename).resolve()
        if not str(dst).startswith(str(out.resolve())):
            sys.exit(f"[DUNG] duong dan ngoai thu muc: {i.filename}")
        if i.is_dir():
            continue
        zf.extract(i, out)
        n += 1; b += i.file_size
    mk.write_text(json.dumps({"zip": z, "sha256_zip": h, "so_file": n, "byte": b}), encoding="utf-8")
    print(z, "->", out, n, "file", round(b / 1e9, 2), "GB", flush=True)
print("xong")
