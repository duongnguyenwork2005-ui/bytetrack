"""
download_data.py — Tai bo du lieu UA-DETRAC tu Google Drive.

VI SAO KHONG DUNG `wget` THUAN?
-------------------------------
Google Drive khong tra ve file truc tiep cho file lon (> ~100 MB). No tra ve mot
trang HTML "Google Drive can't scan this file for viruses" kem mot form chua
`confirm token`. `wget` thuan se luu chinh trang HTML do thanh file .zip
=> giai nen se bao "not a zip file".
Script nay dung thu vien `gdown`, thu vien tu dong:
  1) goi request dau tien, doc cookie + confirm token,
  2) goi request thu hai kem token -> nhan duoc byte stream thuc su.

Sau khi tai, script LUON kiem tra file co phai zip hop le khong
(zipfile.is_zipfile + testzip tren toan bo entry) truoc khi bao thanh cong.

CACH DUNG
---------
    python src/download_data.py                        # tai tat ca muc da co file_id
    python src/download_data.py --only train_annotations_xml
    python src/download_data.py --images-id <GOOGLE_DRIVE_FILE_ID>
    python src/download_data.py --check-only           # chi kiem tra file da co
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Tien ich kiem tra file tai ve
# ---------------------------------------------------------------------------
def looks_like_html(path: Path, nbytes: int = 2048) -> bool:
    """Doc vai KB dau file, xem co phai trang HTML cua Google Drive khong.

    Day la trieu chung dien hinh cua loi 'virus scan confirmation page':
    file .zip that ra la text/html.
    """
    if not path.exists():
        return False
    head = path.open("rb").read(nbytes).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def validate_zip(path: Path, deep: bool = True) -> tuple[bool, str]:
    """Kiem tra tinh toan ven cua file zip.

    Returns
    -------
    (ok, message)
        ok = True  -> file zip hop le, message chua so entry va dung luong.
        ok = False -> message giai thich loi cu the.
    """
    if not path.exists():
        return False, f"Khong tim thay file: {path}"

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb < 0.05:
        return False, f"File qua nho ({size_mb:.3f} MB) - gan nhu chac chan tai loi."

    if looks_like_html(path):
        return False, (
            "File tai ve la trang HTML (khong phai zip). Day la trang xac nhan "
            "virus-scan cua Google Drive, hoac link da het han / bi gioi han quota."
        )

    if not zipfile.is_zipfile(path):
        return False, f"File khong co chu ky zip hop le ({size_mb:.1f} MB)."

    try:
        with zipfile.ZipFile(path) as zf:
            n = len(zf.infolist())
            if deep:
                # testzip() doc CRC toan bo entry -> phat hien file tai thieu byte.
                bad = zf.testzip()
                if bad is not None:
                    return False, f"Entry hong trong zip: {bad}"
        return True, f"Zip hop le: {n} entry, {size_mb:.1f} MB."
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {e}"


# ---------------------------------------------------------------------------
# Tai file
# ---------------------------------------------------------------------------
def download_from_drive(file_id: str, dest: Path, verify_ssl: bool = True) -> tuple[bool, str]:
    """Tai 1 file Google Drive ve `dest` bang gdown (co xu ly confirm token).

    Ghi chu ve API cua gdown: ban >= 6.0 da BO tham so `fuzzy` va nhan truc tiep
    `id=`. Ham nay do signature luc runtime nen chay duoc voi ca gdown 4/5/6.

    `verify_ssl=False` tuong duong `wget --no-check-certificate` (chi dung khi
    may cua ban bi loi certificate, vi du dung proxy noi bo).
    """
    try:
        import gdown
    except ImportError:
        return False, "Chua cai gdown. Chay: python -m pip install gdown"

    import inspect
    params = inspect.signature(gdown.download).parameters

    kwargs = {"output": str(dest), "quiet": False, "resume": True, "verify": verify_ssl}
    if "id" in params:
        # gdown >= 6: truyen thang file id, khong can dung URL.
        kwargs["id"] = file_id
        shown = f"drive file id = {file_id}"
    else:
        # gdown cu: truyen URL, co the bat che do fuzzy de chap nhan link /file/d/<id>/view
        kwargs["url"] = f"https://drive.google.com/uc?id={file_id}"
        if "fuzzy" in params:
            kwargs["fuzzy"] = True
        shown = kwargs["url"]

    # Loai bo cac kwargs ma phien ban gdown hien tai khong ho tro.
    kwargs = {k: v for k, v in kwargs.items() if k in params}

    print(f"[download] {shown}")
    print(f"           -> {dest}")
    try:
        out = gdown.download(**kwargs)
    except Exception as e:  # noqa: BLE001 - muon bao loi that cho nguoi dung
        return False, f"gdown nem exception: {type(e).__name__}: {e}"

    if out is None:
        return False, (
            "gdown tra ve None - thuong do: link khong public, "
            "vuot quota tai trong ngay, hoac file_id sai."
        )
    return True, str(out)


def handle_item(key: str, spec: dict, check_only: bool = False) -> bool:
    """Xu ly 1 muc trong config.DRIVE_FILES: kiem tra -> tai -> validate."""
    dest = config.RAW_DIR / spec["filename"]
    print("\n" + "=" * 72)
    print(f"[{key}] {spec['note']}")
    print("=" * 72)

    # 1) Neu file da ton tai va hop le -> bo qua.
    if dest.exists():
        ok, msg = validate_zip(dest)
        if ok:
            print(f"  [SKIP] Da co san va hop le. {msg}")
            return True
        print(f"  [WARN] File da ton tai nhung KHONG hop le: {msg}")
        if check_only:
            return False
        print("  [INFO] Se xoa va tai lai.")
        dest.unlink()

    if check_only:
        print("  [MISS] Chua co file.")
        return False

    # 2) Chua co file_id -> khong doan bua, bao cho nguoi dung.
    if not spec.get("file_id"):
        print(f"  [ERROR] Chua co Google Drive file_id cho '{key}'.")
        print(f"          Hay dien vao config.DRIVE_FILES['{key}']['file_id'],")
        print(f"          hoac tai thu cong va dat file vao: {dest}")
        return False

    # 3) Tai + validate.
    ok, msg = download_from_drive(spec["file_id"], dest)
    if not ok:
        print(f"  [ERROR] Tai that bai: {msg}")
        return False

    ok, msg = validate_zip(dest)
    print(f"  [{'OK' if ok else 'ERROR'}] {msg}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Tai UA-DETRAC tu Google Drive")
    ap.add_argument("--only", choices=list(config.DRIVE_FILES), default=None,
                    help="Chi tai 1 muc")
    ap.add_argument("--images-id", default=None,
                    help="Google Drive file id cua DETRAC-Train-Images.zip")
    ap.add_argument("--check-only", action="store_true",
                    help="Chi kiem tra file da co trong data/raw, khong tai")
    args = ap.parse_args()

    if args.images_id:
        config.DRIVE_FILES["train_images"]["file_id"] = args.images_id

    keys = [args.only] if args.only else list(config.DRIVE_FILES)
    results = {k: handle_item(k, config.DRIVE_FILES[k], args.check_only) for k in keys}

    print("\n" + "=" * 72)
    print("TONG KET")
    print("=" * 72)
    for k, ok in results.items():
        print(f"  {'OK   ' if ok else 'FAIL '} {k:<24} -> {config.RAW_DIR / config.DRIVE_FILES[k]['filename']}")

    if not all(results.values()):
        print("\nCo muc that bai. Xem huong dan tai thu cong trong README.md muc "
              "'Tai du lieu thu cong'.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
