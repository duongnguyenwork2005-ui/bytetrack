"""Read-only readiness check for the SVMOT and UTUAV downloads.

This does not extract datasets, convert annotations, or run tracking.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import shutil
import sys
import zipfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "SVMOT": "svmot_data_v1.0.zip",
    "UTUAV images B/C": "UTUAV_B_C_3and6class.tar.gz",
    "UTUAV MOT labels": "UTUAV_MOT_VOC_3_6classes.tar.gz",
}
MODULES = ("numpy", "scipy", "cv2", "torch", "ultralytics", "trackeval", "yaml")
TRACKERS = {
    "KF/CV": "configs/phase2/trackers/bytetrack_cv.yaml",
    "EKF/CTRV": "configs/bytetrack_ekf_ctrv.yaml",
    "UKF/CTRV": "configs/bytetrack_ukf_ctrv.yaml",
    "PF/CTRV": "configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml",
}


def report(label: str, state: str, detail: str) -> None:
    print(f"[{state}] {label}: {detail}")


def check_archive(label: str, path: Path, full_check: bool = False) -> bool:
    if not path.is_file():
        report(label, "WAIT", f"chưa có tệp: {path}")
        return False
    size = path.stat().st_size
    if path.name.endswith(".zip"):
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if not entries:
                    raise zipfile.BadZipFile("archive rỗng")
                total = sum(entry.file_size for entry in entries)
                if full_check:
                    bad_member = archive.testzip()
                    if bad_member is not None:
                        raise zipfile.BadZipFile(f"CRC lỗi ở {bad_member}")
                report(label, "OK", f"ZIP đọc được danh mục {len(entries)} mục; nén {size:,} B, giải nén ước theo header {total:,} B")
                return True
        except (OSError, zipfile.BadZipFile, zlib.error) as exc:
            report(label, "WAIT", f"ZIP chưa hoàn chỉnh hoặc lỗi: {exc}")
            return False
    try:
        with path.open("rb") as stream:
            signature = stream.read(2)
    except OSError as exc:
        report(label, "WAIT", f"không đọc được: {exc}")
        return False
    if signature != b"\x1f\x8b":
        report(label, "WAIT", "không có header gzip hợp lệ")
        return False
    if full_check:
        try:
            with gzip.open(path, "rb") as stream:
                while stream.read(8 * 1024 * 1024):
                    pass
        except (OSError, EOFError, zlib.error) as exc:
            report(label, "WAIT", f"gzip chưa hoàn chỉnh hoặc CRC lỗi: {exc}")
            return False
        report(label, "OK", f"gzip đọc hết luồng và CRC hợp lệ; {size:,} B")
        return True
    report(label, "CHECK", f"gzip header hợp lệ, {size:,} B; phải kiểm CRC/toàn bộ tar sau khi tải xong")
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads",
                        help="thư mục chứa các tệp tải về; mặc định là Downloads")
    parser.add_argument("--env-only", action="store_true", help="chỉ kiểm môi trường, chưa kiểm archive")
    parser.add_argument("--full-check", action="store_true", help="đọc hết ZIP/gzip để kiểm CRC; có thể mất nhiều phút")
    args = parser.parse_args()

    report("Python", "OK", f"{sys.version.split()[0]} ({sys.executable})")
    missing_modules = [name for name in MODULES if importlib.util.find_spec(name) is None]
    report("Thư viện", "OK" if not missing_modules else "MISSING",
           "đủ NumPy/SciPy/OpenCV/Torch/Ultralytics/TrackEval/YAML" if not missing_modules
           else ", ".join(missing_modules))
    missing_assets = []
    for label, relative in TRACKERS.items():
        if not (ROOT / relative).is_file():
            missing_assets.append(label)
    if not (ROOT / "models/yolov8n.pt").is_file():
        missing_assets.append("YOLOv8n checkpoint")
    report("Checkpoint và tracker", "OK" if not missing_assets else "MISSING",
           "KF/EKF/UKF/PF + YOLOv8n có sẵn" if not missing_assets else ", ".join(missing_assets))
    free = shutil.disk_usage(args.downloads_dir if args.downloads_dir.exists() else ROOT).free
    report("Dung lượng trống", "INFO", f"{free / 2**30:.1f} GiB tại ổ kiểm tra; chưa tính nhu cầu giải nén")

    if not args.env_only:
        states = {}
        for label, filename in FILES.items():
            states[label] = check_archive(label, args.downloads_dir / filename, full_check=args.full_check)
        if not states["UTUAV MOT labels"]:
            print("Lưu ý: cần gói nhãn MOT UTUAV để chấm IDF1/HOTA/IDSW.")
        if not args.full_check:
            print("Danh mục ZIP và header gzip không chứng minh toàn bộ dữ liệu nguyên vẹn; dùng --full-check sau khi tải xong.")
        print("Tiền kiểm không giải nén hoặc chạy benchmark.")
    return 0 if not missing_modules and not missing_assets else 1


if __name__ == "__main__":
    raise SystemExit(main())
