"""
run_state.py -- CHINH SACH RESUME dung chung cho CA HAI duong chay tracking:
`run_experiment.stage_track()` va CLI `run_tracking.py`.

LY DO TACH RA (review vong 6, finding 2): guard da duoc sua o `stage_track()` tu vong 2, nhung CLI
rieng van ghi `run_meta.json` MOI truoc vong lap roi bo qua video chi vi file tracking ton tai
(`--skip-existing`). Hau qua da do duoc: sua YAML 0.25 -> 0.95 roi chay lai voi `--skip-existing`
=> giu nguyen 4 dong cu NHUNG metadata ghi cau hinh MOI 0.95. Hai duong chay phai dung CHUNG mot
chinh sach, khong phai hai ban cai dat.

NGUYEN TAC
  1. Chi tai su dung mot video khi CO BANG CHUNG HOAN TAT (`video_meta/<video>.json` ghi NGUYEN TU
     sau khi video chay xong, `complete: true`) VA dau van tay dau vao TRUNG KHIT.
     Thieu / hong metadata => KHONG tai su dung (khong mac nhien tuong thich).
  2. Truoc khi ghi de bat ky dau ra nao: `run_meta.json` mang `status="running"`. Mot lan chay bi
     ngat khong the duoc xac nhan boi metadata hoan tat cu.
  3. KHONG ghi de provenance (`created`/`git_commit`) cua phan duoc tai su dung bang cau hinh cua lan
     goi hien tai; dau van tay doi => bat dau provenance moi va luu ban cu vao `previous`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["atomic_write_json", "load_json", "video_meta_path", "can_reuse_video",
           "mark_run_started", "mark_run_finished", "write_video_meta", "plan_videos"]


def atomic_write_json(path: Path, obj: dict) -> None:
    """Ghi JSON nguyen tu: ghi file tam roi doi ten (tranh file metadata bi cat doi khi bi ngat)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    """None neu thieu HOAC hong (thieu/hong => KHONG duoc coi la tuong thich)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def video_meta_path(out_root: Path, video: str) -> Path:
    return out_root / "video_meta" / f"{video}.json"


def can_reuse_video(out_root: Path, video: str, fingerprint: str) -> tuple[bool, str]:
    """Chi tai su dung mot video khi CO BANG CHUNG HOAN TAT va dau van tay khop."""
    if not (out_root / "data" / f"{video}.txt").exists():
        return False, "chua co file tracking"
    vm = load_json(video_meta_path(out_root, video))
    if vm is None:
        return False, "thieu/hong video_meta"
    if not vm.get("complete"):
        return False, "video_meta chua danh dau hoan tat"
    if vm.get("input_fingerprint") != fingerprint:
        return False, "dau van tay khac"
    return True, ""


def plan_videos(out_root: Path, videos: list[str], fingerprint: str,
                skip_existing: bool) -> tuple[list[str], list[str], dict[str, str]]:
    """Chia danh sach video thanh (tai_su_dung, phai_chay, ly_do).

    `skip_existing=False` => chay lai TAT CA (khong xet bang chung).
    """
    reuse, run, reasons = [], [], {}
    for v in videos:
        if not skip_existing:
            run.append(v)
            continue
        ok, why = can_reuse_video(out_root, v, fingerprint)
        if ok:
            reuse.append(v)
        else:
            run.append(v)
            reasons[v] = why
    return reuse, run, reasons


def write_video_meta(out_root: Path, video: str, fingerprint: str, extra: dict | None = None) -> None:
    """Bang chung HOAN TAT cua mot video - ghi NGUYEN TU SAU khi video do chay xong."""
    atomic_write_json(video_meta_path(out_root, video),
                      {"video": video, "complete": True, "input_fingerprint": fingerprint,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})})


def mark_run_started(path: Path, meta: dict) -> dict:
    """Danh dau run DANG CHAY truoc khi ghi de bat ky dau ra nao.

    Giu `created`/`git_commit` GOC khi dau van tay khong doi (provenance cua phan thuc su tai su dung);
    neu dau van tay doi (hoac metadata cu thieu/hong) thi bat dau provenance moi.
    """
    old = load_json(path)
    m = dict(meta)
    same = old is not None and old.get("input_fingerprint") == meta["input_fingerprint"]
    if same:
        m["created"] = old.get("created", m["created"])
        m["git_commit"] = old.get("git_commit", m["git_commit"])
        m["resumed"] = list(old.get("resumed", []))
    elif old is not None:
        m["previous"] = {"created": old.get("created"), "git_commit": old.get("git_commit"),
                         "input_fingerprint": old.get("input_fingerprint"), "status": old.get("status")}
    m["status"] = "running"
    m["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(path, m)
    return m


def mark_run_finished(path: Path, meta: dict, ran: int, reused: int, videos: list[str],
                      git_commit: str = "?") -> None:
    """Ket thuc mot lan chay. Neu tren dia da co metadata CUNG dau van tay (vd lan nay khong chay video nao
    nen `mark_run_started` khong duoc goi) thi GIU provenance goc va noi tiep danh sach `resumed`."""
    old = load_json(path)
    m = dict(meta)
    if old is not None and old.get("input_fingerprint") == m.get("input_fingerprint"):
        m["created"] = old.get("created", m.get("created"))
        m["git_commit"] = old.get("git_commit", m.get("git_commit"))
        m["resumed"] = list(old.get("resumed", [])) if not m.get("resumed") else m["resumed"]
        if old.get("previous") and not m.get("previous"):
            m["previous"] = old["previous"]
    m["status"] = "complete"
    m["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    m["n_videos_ran"] = ran
    m["n_videos_reused"] = reused
    m["videos"] = list(videos)
    m.setdefault("resumed", []).append({"at": m["finished_at"], "git_commit": git_commit,
                                        "n_videos_ran": ran, "n_videos_reused": reused})
    atomic_write_json(path, m)
