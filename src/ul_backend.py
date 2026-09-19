"""
ul_backend.py -- Chon NGUON cua cac lop nen ByteTrack: goi `ultralytics` da cai, hay ban VENDOR
khong phu thuoc torch (`third_party/ultralytics_trackers/`).

LY DO: `tracker_ctrv.py` ke thua `STrack` / `BYTETracker` NGAY LUC IMPORT. Muon chay cung mot ma CTRV
tren mot moi truong KHONG co torch (Jetson Nano) thi phai doi NGUON cua hai lop nen do - chu khong
phai viet lai tracker. Nho vay hai backend dung CHUNG mot ban ma CTRV/CV, va phep kiem parity moi co
y nghia (chi khac lop nen, khong khac thuat toan).

CHON BACKEND (doc MOT LAN, truoc khi import `tracker_ctrv`):
    - bien moi truong `BYTETRACK_BACKEND` = "ultralytics" (mac dinh) | "np"
    - hoac goi `use_backend("np")` truoc khi import bat ky module tracker nao

    BYTETRACK_BACKEND=np python src/run_tracking.py ...

Backend "np" KHONG import torch/ultralytics duoi bat ky hinh thuc nao - `src/test_bytetrack_np_parity.py`
chung minh dieu do bang mot tien trinh CHAN san hai package do trong `sys.modules`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "third_party"

ENV_VAR = "BYTETRACK_BACKEND"
BACKENDS = ("ultralytics", "np")

_loaded: dict | None = None


def use_backend(name: str) -> None:
    """Dat backend cho tien trinh nay. Phai goi TRUOC khi import module tracker."""
    if name not in BACKENDS:
        raise ValueError(f"backend {name!r} khong hop le; chon mot trong {BACKENDS}")
    if _loaded is not None and _loaded["backend"] != name:
        raise RuntimeError(
            f"backend da duoc nap la {_loaded['backend']!r}, khong doi sang {name!r} giua chung: cac lop "
            "tracker da ke thua lop nen cua backend cu. Dat bien moi truong BYTETRACK_BACKEND roi chay "
            "mot tien trinh moi.")
    os.environ[ENV_VAR] = name


def backend_name() -> str:
    return os.environ.get(ENV_VAR, "ultralytics")


def _load() -> dict:
    global _loaded
    if _loaded is not None:
        return _loaded
    name = backend_name()
    if name not in BACKENDS:
        raise ValueError(f"{ENV_VAR}={name!r} khong hop le; chon mot trong {BACKENDS}")

    if name == "np":
        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))
        from ultralytics_trackers import basetrack as _bt          # noqa: PLC0415
        from ultralytics_trackers import byte_tracker as _byte      # noqa: PLC0415
        from ultralytics_trackers.utils import matching as _match    # noqa: PLC0415
        from ultralytics_trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: PLC0415
    else:
        from ultralytics.trackers import basetrack as _bt           # noqa: PLC0415
        from ultralytics.trackers import byte_tracker as _byte       # noqa: PLC0415
        from ultralytics.trackers.utils import matching as _match     # noqa: PLC0415
        from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: PLC0415

    _loaded = {"backend": name, "BaseTrack": _bt.BaseTrack, "TrackState": _bt.TrackState,
               "BYTETracker": _byte.BYTETracker, "STrack": _byte.STrack, "matching": _match,
               "KalmanFilterXYAH": KalmanFilterXYAH}
    return _loaded


def __getattr__(attr: str):
    """Cho phep `from ul_backend import BYTETracker, STrack, TrackState, matching`."""
    d = _load()
    if attr in d:
        return d[attr]
    raise AttributeError(attr)


class _Cfg(dict):
    """Thay `IterableSimpleNamespace` cua ultralytics: truy cap bang thuoc tinh, lap duoc theo cap.

    `BYTETracker` chi doc cau hinh bang `args.<ten>`; `run_tracking` con lap `for k, v in cfg`.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def __iter__(self):
        return iter(self.items())


#: TEN TRAN duoc phep tra ban vendor (tuong duong `check_yaml` cua ultralytics tra file trong package).
#: Moi duong dan CO THU MUC deu phai ton tai that - xem `resolve_tracker_yaml_np`.
VENDORED_YAML = ("bytetrack.yaml",)


def resolve_tracker_yaml_np(yaml_path) -> Path:
    """Giai duong dan cau hinh tracker cho backend `np`. Tra ve DUONG DAN THUC SE DUOC NAP.

    Quy tac (review vong 5, P2-4 - truoc day bo phan thu muc roi am tham lay ban vendor):
      - `bytetrack.yaml` (TEN TRAN, nam trong VENDORED_YAML) -> ban vendor trong third_party/.
      - moi duong dan khac (tuong doi co thu muc, hoac tuyet doi) -> phai TON TAI THAT, khong thi
        `FileNotFoundError`. KHONG bao gio bo phan thu muc de tra ve cau hinh mac dinh.
    """
    p = Path(yaml_path)
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(f"khong co file cau hinh tracker: {p}")
        return p
    if p.parent == Path("."):                       # ten tran
        if p.name in VENDORED_YAML:
            vend = VENDOR_DIR / "ultralytics_trackers" / p.name
            if not vend.is_file():
                raise FileNotFoundError(f"thieu ban vendor cua {p.name}: {vend}")
            return vend
        cand = ROOT / p
        if cand.is_file():
            return cand
        raise FileNotFoundError(
            f"ten cau hinh tracker {yaml_path!r} khong co ban vendor (chi ho tro {VENDORED_YAML}) "
            f"va khong co {cand}")
    cand = ROOT / p                                 # duong dan CO thu muc -> phai ton tai that
    if not cand.is_file():
        raise FileNotFoundError(
            f"khong co file cau hinh tracker: {cand} (duong dan co thu muc thi KHONG duoc thay bang "
            f"ban vendor - tranh am tham chay cau hinh mac dinh)")
    return cand


def load_tracker_cfg_np(yaml_path, overrides: dict | None = None) -> _Cfg:
    """Nap cau hinh tracker KHONG can ultralytics (dung cho backend `np`), tu DUNG file resolver tra ve."""
    import yaml as _yaml

    p = resolve_tracker_yaml_np(yaml_path)
    d = _yaml.safe_load(p.read_text(encoding="utf-8"))
    for k, v in (overrides or {}).items():
        d[k] = v
    return _Cfg(d)


def as_mapping(cfg) -> dict:
    """Cau hinh HIEU LUC duoi dang mapping, dung cho CA HAI backend.

    Review vong 5 (P1-B): runner truoc day dung `vars(cfg)`. Voi `IterableSimpleNamespace` cua ultralytics
    thi dung, nhung `_Cfg` ke thua `dict` nen `vars()` = `{}` => metadata rong VA `input_fingerprint` bo
    qua toan bo tham so tracker => doi YAML khong lam mat hieu luc tai su dung ket qua cu.
    """
    if isinstance(cfg, dict):
        return dict(cfg)
    d = vars(cfg)
    if d:
        return dict(d)
    # phong truong hop lop cau hinh khong dung __dict__ (vd dung __slots__)
    return {k: getattr(cfg, k) for k in dir(cfg)
            if not k.startswith("_") and not callable(getattr(cfg, k, None))}


#: QUY UOC HASH DUY NHAT cho ma vendor (review vong 5, muc 5 + phat hien them):
#: sha256 cua NOI DUNG DA CHUAN HOA XUONG DONG VE LF.
#: Ly do: `core.autocrlf=true` tren Windows lam byte tren dia khac byte git/ZIP, va lam
#: `fingerprint()` KHAC NHAU giua Windows (CRLF) va Nano (LF) => runner chay lai vo co va provenance
#: khong so duoc giua hai may. Quy uoc nay ap dung THONG NHAT cho PROVENANCE.json, fingerprint() va test.
HASH_CONVENTION = "sha256 cua noi dung da chuan hoa xuong dong ve LF"


def sha256_lf(path) -> str:
    """sha256 theo QUY UOC LF (xem HASH_CONVENTION) - on dinh tren moi nen tang."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


#: Cac module vendor duoc tinh vao danh tinh backend.
VENDOR_MODULES = ("ultralytics_trackers/basetrack.py", "ultralytics_trackers/byte_tracker.py",
                  "ultralytics_trackers/np_compat.py", "ultralytics_trackers/utils/matching.py",
                  "ultralytics_trackers/utils/kalman_filter.py", "ultralytics_trackers/utils/stracks.py")


def fingerprint() -> dict:
    """Danh tinh backend de dua vao provenance cua runner (doi backend => phai chay lai)."""
    import importlib.metadata as md

    name = backend_name()
    out = {"backend": name}
    if name == "np":
        out["hash_convention"] = HASH_CONVENTION
        mods = {}
        for f in VENDOR_MODULES:
            p = VENDOR_DIR / f
            mods[f] = (sha256_lf(p)[:16] if p.exists() else "MISSING")
        out["vendor_modules"] = mods
    else:
        try:
            out["ultralytics"] = md.version("ultralytics")
        except Exception:                                  # noqa: BLE001
            out["ultralytics"] = "?"
    for dep in ("numpy", "scipy", "lap"):
        try:
            out[dep] = md.version(dep)
        except Exception:                                  # noqa: BLE001
            out[dep] = None
    return out
