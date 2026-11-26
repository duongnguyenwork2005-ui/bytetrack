"""Kiem ke chi doc cac zip dataset moi tai (02/11): cau truc, so file theo duoi, thu muc cap 1-3, README/LICENSE/seqinfo
(doc noi dung text nho), kiem CRC toan bo (testzip). Ghi JSON ra thu muc chi dinh."""
import collections
import json
import sys
import zipfile
from pathlib import Path

ZIPS = ["sherbrooke_annotations.zip", "sherbrooke_frames.zip", "TrafficMOT-20261102T061326Z-1-001.zip",
        "VisDrone2019-MOT-val.zip", "VisDrone2019-MOT-test-dev.zip"]
GOC = Path("D:/LABOTARY")
out = Path(sys.argv[1])
kq = {}
for ten in ZIPS:
    p = GOC / ten
    z = zipfile.ZipFile(p)
    infos = z.infolist()
    duoi = collections.Counter(Path(i.filename).suffix.lower() for i in infos if not i.is_dir())
    cap = collections.Counter("/".join(i.filename.split("/")[:3]) for i in infos if not i.is_dir())
    text = {}
    for i in infos:
        n = i.filename.lower()
        if (any(k in n for k in ("readme", "license", "licence", "seqinfo", ".ini", "classes", "label"))
                and i.file_size < 200_000 and Path(n).suffix in (".txt", ".md", ".ini", ".json", ".names", "")):
            if len(text) < 25:
                text[i.filename] = z.read(i).decode("utf-8", "replace")[:3000]
    vi_du = [i.filename for i in infos if not i.is_dir()][:15]
    print(ten, "files", sum(duoi.values()), "testzip...", flush=True)
    loi = z.testzip()
    kq[ten] = {"byte_zip": p.stat().st_size, "so_file": sum(duoi.values()),
               "byte_giai_nen": sum(i.file_size for i in infos), "duoi": dict(duoi.most_common(15)),
               "thu_muc_cap3": dict(sorted(cap.items())[:80]), "so_thu_muc_cap3": len(cap), "vi_du": vi_du,
               "text_nho": text, "crc_loi_dau_tien": loi}
    print(ten, "CRC", "OK" if loi is None else f"LOI {loi}", flush=True)
out.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
print("xong", out)
