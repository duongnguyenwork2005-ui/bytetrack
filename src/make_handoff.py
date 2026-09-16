"""
make_handoff.py -- Dong goi handoff/phase2_review_latest.zip cho nguoi review KHONG co may (Moc 2, muc 7).

GOM (giu nguyen duong dan tuong doi tu goc repo):
    - CODEX_HANDOFF.md, PHASE2_REPORT.md, README.md, requirements.txt
    - file GIT-TRACKED trong src/ (tru src/demo/), configs/, scripts/, data/fixtures/, handoff/logs, handoff/patches
    - results/phase2/** (csv/json/md/txt/yaml/log), results/trackeval/<split>/p2-*/{per_video,combined}_metrics.csv + raw_fields.json
    - data/processed/trackers/<split>/p2-*/{summary.csv, run_meta.json} (KHONG gom data/*.txt, assoc_log, timing)
    - data/interim/det_cache/<cache_id>/{meta.json, *.json} + *.npz neu tong dung luong npz <= --cache-max-mb
      (cache detection la du lieu DAN XUAT tu weights + anh, cho phep chay lai tracking khong can GPU)
    - handoff/versions.json (pip freeze, python, GPU, OS), handoff/large_inputs.md, handoff/INDEX.md (tao o day)
    - handoff/patches/moc2_<base>..<head>.diff + commits.txt (tao o day; --base = commit bat dau moc 2)
LOAI: dataset (data/raw, data/extracted), weights (models/), venv, .git, video, outputs/, ZIP cu, log
parquet assoc_log (giu tai may; trich trong results/phase2/failure_analysis/*/event_window_log.csv) va
FILE RIENG CUA NGUOI DUNG - danh sach tuong minh trong USER_FILES (`.gitignore`, `src/demo/**`, `live.bat`,
`outputs/**`). Loi P2 cua review doc lap: ban truoc van dong goi `.gitignore` du mo ta noi da loai.

TINH NHAT QUAN SNAPSHOT: ZIP phai duoc dung SAU commit cuoi. Neu con file (thuoc pham vi dong goi) khac
HEAD thi mac dinh TU CHOI dung goi (`--allow-dirty` de bo qua, khi do danh sach dirty duoc ghi vao
versions.json va INDEX.md). Kem `handoff/FILE_HASHES.csv` (sha256 tung file trong goi) va
`handoff/external/bytetrack.yaml` (ban ultralytics dung trong LOCK) de doi chieu hash.

Kiem tra sau khi tao: zipfile.testzip(), file bat buoc co mat, moi duong dan tuong doi trong CODEX_HANDOFF.md
(dang [..](path) hoac `path`) ton tai trong ZIP hoac duoc liet ke la "chi tren may".

    python src/make_handoff.py --base e353b84
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "handoff" / "phase2_review_latest.zip"
REQUIRED = ["CODEX_HANDOFF.md", "PHASE2_REPORT.md", "README.md", "requirements.txt", "handoff/versions.json",
            "handoff/INDEX.md", "handoff/large_inputs.md", "handoff/logs/tests_summary.csv", "handoff/logs/tests_latest.log",
            "configs/phase2/PREREG_moc2.md", "configs/phase2/LOCK_moc2.md", "handoff/FILE_HASHES.csv",
            "results/phase2/REVIEW_moc2_codex.md"]

#: File RIENG CUA NGUOI DUNG (khong bao gio dong goi, ke ca khi git dang theo doi)
USER_FILES = (".gitignore", "live.bat")
USER_PREFIXES = ("src/demo/", "outputs/")
RESULT_EXT = {".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".log"}


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace")


def tracked_files() -> list[str]:
    return [ln for ln in git("ls-files").splitlines() if ln]


def collect(cache_max_mb: float, splits: list[str]) -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    notes: list[str] = []
    tracked = tracked_files()
    for f in tracked:
        p = Path(f)
        top = p.parts[0] if p.parts else ""
        if f in USER_FILES or f.startswith(USER_PREFIXES):
            continue                                      # file rieng cua nguoi dung -> khong dong goi
        if f in ("CODEX_HANDOFF.md", "PHASE2_REPORT.md", "README.md", "requirements.txt"):
            files.add(p)
        elif top == "third_party" and "_testdeps" not in f:
            files.add(p)                                  # ma SORT upstream da ghim + PROVENANCE (tham chieu kiem thu)
        elif top == "src" and not f.startswith("src/demo/"):
            files.add(p)
        elif top in ("configs", "scripts"):
            files.add(p)
        elif f.startswith("data/fixtures/"):
            files.add(p)
        elif f.startswith("handoff/logs/") or f.startswith("handoff/patches/"):
            files.add(p)
        elif f.startswith("results/phase2/") and p.suffix.lower() in RESULT_EXT:
            files.add(p)
    # results/phase2 chua commit (vua sinh) - gom moi csv/json/md
    for p in (ROOT / "results" / "phase2").rglob("*"):
        if p.is_file() and p.suffix.lower() in RESULT_EXT and "prefix" not in p.parts and "fixed" not in p.parts:
            files.add(p.relative_to(ROOT))
    for split in splits:
        for d in (ROOT / "results" / "trackeval" / split).glob("p2-*"):
            for name in ("per_video_metrics.csv", "combined_metrics.csv", "raw_fields.json"):
                if (d / name).exists():
                    files.add((d / name).relative_to(ROOT))
        for d in (ROOT / "data" / "processed" / "trackers" / split).glob("p2-*"):
            for name in ("summary.csv", "run_meta.json"):
                if (d / name).exists():
                    files.add((d / name).relative_to(ROOT))
    # cache detection
    for cdir in (ROOT / "data" / "interim" / "det_cache").glob("*"):
        if not cdir.is_dir():
            continue
        for p in cdir.glob("*.json"):
            files.add(p.relative_to(ROOT))
        npz = sorted(cdir.glob("*.npz"))
        mb = sum(p.stat().st_size for p in npz) / 1e6
        if npz and mb <= cache_max_mb:
            for p in npz:
                files.add(p.relative_to(ROOT))
            notes.append(f"cache {cdir.name}: {len(npz)} npz ({mb:.1f} MB) DA GOM")
        elif npz:
            notes.append(f"cache {cdir.name}: {len(npz)} npz ({mb:.1f} MB) KHONG gom (> {cache_max_mb} MB) - chi tren may")
    for h in ("handoff/versions.json", "handoff/INDEX.md", "handoff/large_inputs.md", "handoff/FILE_HASHES.csv",
              "handoff/external/bytetrack.yaml", "CODEX_HANDOFF.md", "PHASE2_REPORT.md", "README.md",
              "requirements.txt"):
        if (ROOT / h).exists():                     # ke ca khi chua commit (tai lieu ban giao)
            files.add(Path(h))
    for p in (ROOT / "handoff" / "logs").glob("*"):
        if p.is_file():
            files.add(p.relative_to(ROOT))
    for p in (ROOT / "handoff" / "patches").glob("*"):
        if p.is_file():
            files.add(p.relative_to(ROOT))
    return sorted(files, key=str), notes


def dirty_files(exclude_user: bool = True) -> list[str]:
    """File TRACKED dang khac HEAD. Mac dinh bo qua file rieng cua nguoi dung."""
    out = []
    for ln in git("status", "--porcelain", "--untracked-files=no").splitlines():
        if not ln.strip():
            continue
        path = ln[3:].strip().replace("\\", "/")
        if exclude_user and (path in USER_FILES or path.startswith(USER_PREFIXES)):
            continue
        out.append(ln.rstrip())
    return out


def write_external() -> None:
    """Chep file cau hinh NGOAI repo duoc LOCK tham chieu (bytetrack.yaml cua ultralytics) de doi chieu hash."""
    d = ROOT / "handoff" / "external"
    d.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.checks import check_yaml
        src = Path(check_yaml("bytetrack.yaml"))
        (d / "bytetrack.yaml").write_bytes(src.read_bytes())
        (d / "README.md").write_text(
            "File cua thu vien ultralytics (khong thuoc repo), duoc `configs/phase2/LOCK_moc2.md` tham chieu.\n"
            f"Nguon: {src}\nPhien ban ultralytics: xem handoff/versions.json\n"
            "Chep vao day de nguoi review doi chieu duoc hash ghi trong lock_moc2.json.\n", encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        (d / "README.md").write_text(f"Khong chep duoc bytetrack.yaml: {e}\n", encoding="utf-8")


def write_file_hashes(files: list[Path]) -> None:
    import csv
    out = ROOT / "handoff" / "FILE_HASHES.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "sha256"])
        for p in files:
            fp = ROOT / p
            if fp.is_file() and p.as_posix() != "handoff/FILE_HASHES.csv":
                import hashlib
                h = hashlib.sha256(fp.read_bytes()).hexdigest()
                w.writerow([p.as_posix(), fp.stat().st_size, h])


def write_versions(dirty: list[str] | None = None) -> None:
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "git_head": git("rev-parse", "HEAD").strip(),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "git_dirty_packaged_files": dirty or [],
            "note_user_files": "Cac file rieng cua nguoi dung (.gitignore, src/demo/**, live.bat, outputs/**) "
                               "KHONG duoc dong goi va KHONG tinh vao trang thai dirty o tren."}
    try:
        import torch
        info["torch"] = torch.__version__; info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"? ({e})"
    for mod in ("ultralytics", "trackeval", "numpy", "scipy", "pandas", "fastparquet", "yaml", "cv2"):
        try:
            m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"? ({e})"
    try:
        info["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception as e:  # noqa: BLE001
        info["pip_freeze"] = [f"? {e}"]
    (ROOT / "handoff" / "versions.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def write_patches(base: str) -> None:
    d = ROOT / "handoff" / "patches"; d.mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "--short", "HEAD").strip()
    (d / f"moc2_{base}..{head}.diff").write_text(git("diff", f"{base}..HEAD", "--", ".", ":(exclude)results", ":(exclude)handoff/logs"),
                                                 encoding="utf-8")
    (d / "commits.txt").write_text(git("log", "--format=%h %ci %s", f"{base}..HEAD"), encoding="utf-8")
    (d / "README.md").write_text(f"Diff cua moc 2: {base}..{head} (khong gom results/ va handoff/logs; xem commits.txt).\n"
                                 "Ap dung: git checkout <base> && git apply moc2_*.diff\n", encoding="utf-8")


def check_links(zf: zipfile.ZipFile, doc: str) -> list[str]:
    names = set(zf.namelist())
    text = (ROOT / doc).read_text(encoding="utf-8")
    refs = set(re.findall(r"\]\(([^)#\s]+)\)", text)) | set(re.findall(r"`((?:src|configs|results|data|handoff|scripts)/[^`\s]+)`", text))
    big = (ROOT / "handoff" / "large_inputs.md")
    big_txt = big.read_text(encoding="utf-8") if big.exists() else ""
    missing = []
    for r in refs:
        if r.startswith("http") or "{" in r or "*" in r or "<" in r:
            continue                                  # mau nhieu file / glob / placeholder - kiem tra thu cong
        r2 = r.rstrip("/")
        if r2 in names or any(n.startswith(r2 + "/") for n in names):
            continue
        if r2 in big_txt or r2 + "/" in big_txt:      # da khai bao la "chi tren may"
            continue
        missing.append(r)
    return sorted(missing)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="e353b84", help="commit truoc moc 2 (de tao diff)")
    ap.add_argument("--cache-max-mb", type=float, default=60.0)
    ap.add_argument("--splits", nargs="*", default=["DETRAC-all", "DETRAC-test"])
    ap.add_argument("--allow-dirty", action="store_true",
                    help="van dong goi khi con file (thuoc pham vi goi) khac HEAD; danh sach duoc ghi vao versions.json")
    a = ap.parse_args()
    (ROOT / "handoff").mkdir(exist_ok=True)
    dirty = dirty_files()
    if dirty and not a.allow_dirty:
        print("[LOI] Con file khac HEAD (khong tinh file rieng cua nguoi dung) -> commit truoc khi dong goi, "
              "de snapshot trong ZIP trung voi commit bao cao:")
        for ln in dirty:
            print("   ", ln)
        print("   (hoac chay lai voi --allow-dirty)")
        return 2
    write_external()
    write_versions(dirty)
    write_patches(a.base)
    files, notes = collect(a.cache_max_mb, a.splits)
    write_file_hashes(files)
    files, _ = collect(a.cache_max_mb, a.splits)          # lay them FILE_HASHES.csv vua ghi
    # INDEX
    idx = ["# INDEX - handoff/phase2_review_latest.zip", "",
           f"Tao: {time.strftime('%Y-%m-%d %H:%M:%S')}  HEAD {git('rev-parse', '--short', 'HEAD').strip()}",
           f"File tracked khac HEAD luc dong goi (khong tinh file rieng cua nguoi dung): "
           f"{dirty if dirty else 'khong'}", "",
           "sha256 tung file: handoff/FILE_HASHES.csv", "",
           "## Ghi chu dong goi", *[f"- {n}" for n in notes],
           "- KHONG gom: data/raw, data/extracted (UA-DETRAC), models/*.pt, outputs/, .git, venv, ZIP cu, file rieng chua commit "
           "cua nguoi dung (src/demo/*, live.bat), data/processed/trackers/*/data/*.txt + assoc_log/*.parquet + timing/*.csv "
           "(log day du tai may; trich quanh su kien trong results/phase2/failure_analysis/*/event_window_log.csv),",
           "  results/phase2/shared_kalman/{prefix,fixed}/ (output tung video - tom tat trong diff_summary.csv).",
           "- Xem handoff/large_inputs.md de biet cach lay dau vao lon va lenh dung.", "", "## Danh sach file"]
    total = 0
    for p in files:
        fp = ROOT / p
        if fp.exists():
            total += fp.stat().st_size
            idx.append(f"- {p.as_posix()} ({fp.stat().st_size / 1e3:.1f} kB)")
    idx.append(""); idx.append(f"Tong {len(files)} file, {total / 1e6:.1f} MB (chua nen)")
    (ROOT / "handoff" / "INDEX.md").write_text("\n".join(idx), encoding="utf-8")
    files, _ = collect(a.cache_max_mb, a.splits)          # lay lai INDEX vua ghi
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            fp = ROOT / p
            if fp.exists() and fp.is_file():
                zf.write(fp, p.as_posix())
    with zipfile.ZipFile(ZIP) as zf:
        bad = zf.testzip()
        names = set(zf.namelist())
        missing_req = [r for r in REQUIRED if r not in names]
        missing_links = check_links(zf, "CODEX_HANDOFF.md") if "CODEX_HANDOFF.md" in names else ["CODEX_HANDOFF.md"]
    print(f"ZIP: {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB, {len(names)} file); testzip: {'OK' if bad is None else bad}")
    print(f"file bat buoc thieu: {missing_req or 'khong'}")
    print(f"duong dan trong CODEX_HANDOFF.md khong co trong ZIP: {len(missing_links)}")
    for m in missing_links:
        print(f"   - {m}")
    return 0 if (bad is None and not missing_req) else 1


if __name__ == "__main__":
    raise SystemExit(main())
