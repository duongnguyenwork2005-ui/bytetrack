"""
test_runner_backend_np.py -- Backend NumPy phai chay dung QUA TOAN BO RUNNER, khong chi o tang
`track_video()`. Day la cac phep kiem Codex yeu cau o review vong 5 (P1-A, P1-B).

Moi phep deu di qua CLI `run_tracking.py` hoac `run_experiment.stage_track()` THAT, trong tien trinh
CHAN san `torch` va `ultralytics` o `sys.meta_path`. Du lieu nam trong THU MUC TAM -> khong dung cham
cache hay ket qua nghien cuu.

  1. CLI `run_tracking.py` chay duoc voi BYTETRACK_BACKEND=np khi KHONG co torch/Ultralytics (P1-A:
     truoc day `resolve_tracker_yaml` luon goi `check_yaml` cua ultralytics de hash YAML).
  2. `run_experiment.stage_track()` chay duoc trong cung dieu kien.
  3. `tracker_cfg` trong `run_meta.json` KHONG rong, tren CA HAI backend (P1-B: `vars(_Cfg)` = {}).
  4. Kich ban Codex da do - 4 frame, moi frame 1 detection conf 0.9:
       a. `new_track_thresh=0.25` -> 4 dong tracking
       b. SUA CHINH FILE YAML do thanh 0.95 -> resume phai CHAY LAI va cho ket qua cua cau hinh MOI
          (khong giu 4 dong cu)
       c. chay them lan nua khong doi gi -> TAI SU DUNG
  5. Doi backend qua runner: cung backend -> tai su dung; doi backend -> THUC SU chay lai
     (kiem so video da chay + metadata, khong dung `--dry-run` hay so hash lam bang chung).
  6. `assoc_log=True` chay duoc tren backend NumPy (P2-3) va KHONG doi output tracking.

Review vong 6 bo sung:
  7. CLI `--skip-existing` dung CHUNG chinh sach resume voi `stage_track` (F2): sua YAML tai CUNG
     duong dan roi chay lai voi `--skip-existing` phai CHAY LAI va cho ket qua cua cau hinh MOI,
     KHONG giu output cu roi gan metadata moi; khong doi gi -> tai su dung; thieu/hong `video_meta`
     -> chay lai.
  8. `run_meta.json` luu BACKEND TRACKER tuong minh (F3) - truoc day backend chi nam trong doi tuong
     da hash nen `grep '"backend": "np"'` luon tra 0 ke ca voi run NumPy that.

    python src/test_runner_backend_np.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
_RESULTS: list[tuple[str, bool]] = []

#: Chan torch/ultralytics NGAY dau tien trinh con.
BLOCK = '''
import sys
class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in ("torch", "ultralytics"):
            return self
    def load_module(self, name):
        raise ImportError("CHAN (test runner np): " + name)
sys.meta_path.insert(0, _Block())
'''

YAML_TMPL = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: {thresh}
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-8:]:
            print(f"         {line}")
    assert ok, name


def make_cache(tmp: Path, cache_id: str, video: str, n_frames: int = 4) -> Path:
    """Cache tong hop: `n_frames` frame, moi frame 1 detection conf 0.9 (kich ban cua Codex)."""
    cdir = tmp / "det_cache" / cache_id
    cdir.mkdir(parents=True, exist_ok=True)
    # schema npz DUNG NHU cache that: frame_idx, n_per_frame, unreadable, truncated, time_s,
    # data (n,6) = [x1, y1, x2, y2, conf, cls], orig_shape
    rows = []
    for i in range(n_frames):
        x = 100.0 + 10 * i
        rows.append([x, 100.0, x + 50.0, 180.0, 0.9, 2.0])
    data = np.asarray(rows, dtype=np.float32)
    np.savez_compressed(
        cdir / f"{video}.npz",
        frame_idx=np.arange(1, n_frames + 1, dtype=np.int32),
        n_per_frame=np.ones(n_frames, dtype=np.int32),
        unreadable=np.zeros(n_frames, dtype=bool),
        truncated=np.zeros(n_frames, dtype=bool),
        time_s=(np.arange(n_frames, dtype=np.float64) / 25.0),
        data=data, orig_shape=np.array([540, 960], dtype=np.int32))
    (cdir / f"{video}.json").write_text(json.dumps(
        {"video": video, "n_frames": n_frames, "n_boxes": int(len(data)), "orig_shape": [540, 960],
         "n_unreadable": 0, "n_truncated": 0, "spec_sha256": None}), encoding="utf-8")
    (cdir / "meta.json").write_text(json.dumps(
        {"backend": "gt-synthetic", "weights": None, "weights_sha256": None, "imgsz": 640, "conf": 0.05,
         "iou": 0.7, "classes": None, "max_det": 3000, "half": False, "device": "cpu",
         "mask_ignored_regions": False, "dataset": "synthetic", "part": "test", "agnostic_nms": False,
         "augment": False, "batch": 1, "gt_jitter": 0.0, "gt_dropout": 0.0, "gt_seed": 0,
         "gt_conf_range": None, "versions": {}}), encoding="utf-8")
    return cdir


def run_py(code: str, tmp: Path, backend: str = "np", block: bool = True) -> subprocess.CompletedProcess:
    src = (BLOCK if block else "") + f"import sys\nsys.path.insert(0, r{str(ROOT / 'src')!r})\n" + code
    env = {**os.environ, "BYTETRACK_BACKEND": backend}
    return subprocess.run([PY, "-c", src], cwd=tmp, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def cli_code(tmp: Path, cache_id: str, video: str, yaml_path: Path, tracker_name: str,
             skip_existing: bool = False) -> str:
    """Kich ban goi CLI `run_tracking.main()` THAT tren cache tam."""
    extra = ', "--skip-existing"' if skip_existing else ""
    return f'''
import json, pathlib, config, det_cache
det_cache.CACHE_ROOT = pathlib.Path({str(tmp / "det_cache")!r})
import run_tracking
config.PROCESSED_DIR = pathlib.Path({str(tmp / "cli_processed")!r})
run_tracking.config.PROCESSED_DIR = config.PROCESSED_DIR
sys.argv = ["run_tracking.py", "--cache-id", {cache_id!r}, "--videos", {video!r},
            "--split-name", "SYN-test", "--tracker-name", {tracker_name!r},
            "--tracker-yaml", {str(yaml_path)!r}, "--conf-min", "0.25",
            "--fps", "25", "--elapsed-mode", "steps"{extra}]
rc = run_tracking.main()
d = config.PROCESSED_DIR / "trackers" / "SYN-test" / {tracker_name!r}
txt = d / "data" / f"{video}.txt"
raw = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
print("@@JSON@@" + json.dumps({{
    "rc": rc, "n_lines": len(raw.splitlines()) if raw else 0,
    "cfg_thresh": (meta.get("tracker_cfg") or {{}}).get("new_track_thresh"),
    "tracker_backend": meta.get("tracker_backend"), "status": meta.get("status"),
    "n_ran": meta.get("n_videos_ran"), "n_reused": meta.get("n_videos_reused"),
    "fingerprint": meta.get("input_fingerprint"),
    "has_video_meta": (d / "video_meta" / f"{video}.json").exists(),
    "torch": "torch" in sys.modules, "ul": "ultralytics" in sys.modules}}))
'''


def stage_track_code(tmp: Path, cache_id: str, video: str, yaml_path: Path, out_name: str,
                     assoc_log: bool = False) -> str:
    """Kich ban goi `run_experiment.stage_track()` THAT tren cache tam."""
    return f'''
import json, config, det_cache, run_experiment as rex
config.INTERIM_DIR = {str(tmp / "det_cache")!r}
import pathlib
config.INTERIM_DIR = pathlib.Path({str(tmp)!r})
det_cache.CACHE_ROOT = pathlib.Path({str(tmp / "det_cache")!r})
config.PROCESSED_DIR = pathlib.Path({str(tmp / "processed")!r})
rex.config.PROCESSED_DIR = config.PROCESSED_DIR
exp = {{"name": "runnernp", "split_name": "SYN-test", "cache_id": {cache_id!r}, "classes": None,
        "max_det": 3000, "fps": 25, "elapsed_mode": "steps", "assoc_log": {assoc_log!r},
        "occlusion": {{}}}}
rows = [{{"cond": "c25", "tracker": "cv", "seed": None, "yaml": {str(yaml_path)!r},
          "set": {{}}, "tracker_type": "bytetrack",
          "conf_min": 0.25, "dir": {out_name!r}, "track_buffer_sec": None}}]
import io, re, contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    rex.stage_track(exp, rows, [{video!r}], skip_existing=True)
_out = _buf.getvalue()
sys.stderr.write(_out)
# runner in: "<n> video moi,  <m> tai su dung"
_m = re.findall(r"(\d+)\s+video moi,\s+(\d+)\s+tai su dung", _out)
n = int(_m[-1][0]) if _m else -1
n_reuse = int(_m[-1][1]) if _m else -1
d = config.PROCESSED_DIR / "trackers" / "SYN-test" / {out_name!r}
txt = (d / "data" / f"{video}.txt")
meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
print("@@JSON@@" + json.dumps({{
    "n_ran": n, "n_reuse": n_reuse,
    "n_lines": len(txt.read_text(encoding="utf-8").strip().splitlines()) if txt.exists() else 0,
    "tracker_cfg": meta.get("tracker_cfg"), "yaml_sha": meta.get("tracker_yaml_sha256_12"),
    "fingerprint": meta.get("input_fingerprint"), "backend": meta.get("code", {{}}).get("backend"),
    "n_resumed": len(meta.get("resumed", []))}}))
'''


def parse(r: subprocess.CompletedProcess, what: str) -> dict:
    if "@@JSON@@" not in r.stdout:
        raise RuntimeError(f"{what} that bai (exit {r.returncode}):\n{r.stdout[-2500:]}\n{r.stderr[-2500:]}")
    return json.loads(r.stdout.split("@@JSON@@", 1)[1].splitlines()[0])


def main() -> int:
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="runner_np_"))
    try:
        CACHE, VIDEO = "syn_runner_np", "SYN_0001"
        make_cache(tmp, CACHE, VIDEO)
        yml = tmp / "tracker.yaml"
        yml.write_text(YAML_TMPL.format(thresh=0.25), encoding="utf-8")

        # --- 1. CLI run_tracking.py chay duoc khi CHAN torch/ultralytics ---
        cli = f'''
import pathlib, config, det_cache
det_cache.CACHE_ROOT = pathlib.Path({str(tmp / "det_cache")!r})
import run_tracking
config.PROCESSED_DIR = pathlib.Path({str(tmp / "cli_processed")!r})
run_tracking.config.PROCESSED_DIR = config.PROCESSED_DIR
sys.argv = ["run_tracking.py", "--cache-id", {CACHE!r}, "--videos", {VIDEO!r},
            "--split-name", "SYN-test", "--tracker-name", "clinp",
            "--tracker-yaml", {str(yml)!r}, "--conf-min", "0.25",
            "--fps", "25", "--elapsed-mode", "steps"]
rc = run_tracking.main()
import json
print("@@JSON@@" + json.dumps({{"rc": rc, "torch": "torch" in sys.modules,
                               "ul": "ultralytics" in sys.modules}}))
'''
        r = run_py(cli, tmp)
        ok_cli = "@@JSON@@" in r.stdout
        d = parse(r, "CLI") if ok_cli else {}
        _check("CLI `run_tracking.py` chay duoc voi backend np khi CHAN torch va ultralytics "
               "(P1-A: resolver YAML truoc day luon goi check_yaml cua ultralytics)",
               ok_cli and d.get("rc") in (0, None) and not d.get("torch") and not d.get("ul"),
               (r.stdout[-900:] + r.stderr[-900:]) if not ok_cli else
               f"rc={d.get('rc')} torch={d.get('torch')} ultralytics={d.get('ul')}")

        # --- 2 + 3 + 4a. stage_track that, cau hinh 0.25 ---
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml, "p2-runnernp-c25-cv"), tmp)
        a = parse(r, "stage_track (0.25)")
        _check("`run_experiment.stage_track()` chay duoc voi backend np khi CHAN torch va ultralytics",
               a["n_ran"] == 1 and a["n_lines"] == 4,
               f"n_ran={a['n_ran']} n_lines={a['n_lines']} (ky vong 1 video / 4 dong)")
        _check("`tracker_cfg` trong run_meta.json KHONG rong tren backend np "
               "(P1-B: truoc day `vars(_Cfg)` = {} nen metadata rong)",
               isinstance(a["tracker_cfg"], dict) and a["tracker_cfg"].get("new_track_thresh") == 0.25,
               f"tracker_cfg = {a['tracker_cfg']}")

        # --- 4b. SUA CHINH FILE YAML -> phai chay lai va doi ket qua ---
        yml.write_text(YAML_TMPL.format(thresh=0.95), encoding="utf-8")
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml, "p2-runnernp-c25-cv"), tmp)
        b = parse(r, "stage_track (0.95)")
        _check("sua NOI DUNG cung file YAML (new_track_thresh 0.25 -> 0.95) -> tracking CHAY LAI va cho "
               "ket qua cua cau hinh MOI, khong giu 4 dong cu",
               b["n_ran"] == 1 and b["n_lines"] != a["n_lines"]
               and b["fingerprint"] != a["fingerprint"]
               and b["tracker_cfg"].get("new_track_thresh") == 0.95,
               f"n_ran={b['n_ran']} n_lines: {a['n_lines']} -> {b['n_lines']}; "
               f"fingerprint doi: {a['fingerprint'][:12]} -> {b['fingerprint'][:12]}")

        # --- 4c. khong doi gi -> tai su dung ---
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml, "p2-runnernp-c25-cv"), tmp)
        c = parse(r, "stage_track (lap lai)")
        _check("chay lai khi KHONG doi gi -> TAI SU DUNG (0 video chay lai), fingerprint giu nguyen",
               c["n_ran"] == 0 and c["fingerprint"] == b["fingerprint"] and c["n_resumed"] >= 1,
               f"n_ran={c['n_ran']} resumed={c['n_resumed']}")

        # --- 5. doi backend qua runner -> THUC SU chay lai ---
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml, "p2-runnernp-c25-cv"),
                   tmp, backend="ultralytics", block=False)
        e = parse(r, "stage_track (backend ultralytics)")
        _check("DOI BACKEND (np -> ultralytics) qua runner that -> tracking CHAY LAI (khong tai dung), "
               "fingerprint doi",
               e["n_ran"] == 1 and e["fingerprint"] != c["fingerprint"],
               f"n_ran={e['n_ran']} fingerprint: {c['fingerprint'][:12]} -> {e['fingerprint'][:12]}")
        _check("`tracker_cfg` cung KHONG rong tren backend ultralytics (mot cach lay cau hinh cho ca hai)",
               isinstance(e["tracker_cfg"], dict)
               and e["tracker_cfg"].get("new_track_thresh") == 0.95, f"{e['tracker_cfg']}")
        # cung backend, khong doi gi -> tai su dung
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml, "p2-runnernp-c25-cv"),
                   tmp, backend="ultralytics", block=False)
        f = parse(r, "stage_track (ultralytics lap lai)")
        _check("cung backend + cung dau vao -> TAI SU DUNG", f["n_ran"] == 0, f"n_ran={f['n_ran']}")

        # --- 6. assoc_log=True tren backend np, va KHONG doi output tracking ---
        yml2 = tmp / "tracker2.yaml"
        yml2.write_text(YAML_TMPL.format(thresh=0.25), encoding="utf-8")
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml2, "p2-runnernp-nolog-cv"), tmp)
        g = parse(r, "stage_track (khong log)")
        r = run_py(stage_track_code(tmp, CACHE, VIDEO, yml2, "p2-runnernp-log-cv", assoc_log=True), tmp)
        ok_log = "@@JSON@@" in r.stdout
        h = parse(r, "stage_track (co log)") if ok_log else {}
        logdir = tmp / "processed" / "trackers" / "SYN-test" / "p2-runnernp-log-cv" / "assoc_log"
        t1 = (tmp / "processed/trackers/SYN-test/p2-runnernp-nolog-cv/data" / f"{VIDEO}.txt")
        t2 = (tmp / "processed/trackers/SYN-test/p2-runnernp-log-cv/data" / f"{VIDEO}.txt")
        _check("assoc_log=True chay duoc tren backend np (P2-3: `issubclass` truoc day so voi lop nen "
               "ULTRALYTICS nen luon False), co ghi log, va output tracking KHONG doi",
               ok_log and h.get("n_lines") == g["n_lines"]
               and t1.read_text(encoding="utf-8") == t2.read_text(encoding="utf-8")
               and logdir.is_dir() and any(logdir.iterdir()),
               (r.stdout[-900:] + r.stderr[-900:]) if not ok_log else
               f"n_lines {g['n_lines']} vs {h.get('n_lines')}; file log: "
               f"{[p.name for p in logdir.iterdir()] if logdir.is_dir() else 'KHONG CO'}")
        # --- 7. F2: CLI --skip-existing dung chung chinh sach resume ---
        yml3 = tmp / "tracker_cli.yaml"
        yml3.write_text(YAML_TMPL.format(thresh=0.25), encoding="utf-8")
        a1 = parse(run_py(cli_code(tmp, CACHE, VIDEO, yml3, "clires"), tmp), "CLI 0.25")
        _check("F2/CLI: chay lan dau (new_track_thresh=0.25) -> 4 dong, co video_meta, "
               "status=complete",
               a1["n_lines"] == 4 and a1["has_video_meta"] and a1["status"] == "complete"
               and not a1["torch"] and not a1["ul"],
               f"n_lines={a1['n_lines']} video_meta={a1['has_video_meta']} status={a1['status']}")

        yml3.write_text(YAML_TMPL.format(thresh=0.95), encoding="utf-8")
        a2 = parse(run_py(cli_code(tmp, CACHE, VIDEO, yml3, "clires", skip_existing=True), tmp),
                   "CLI 0.95 --skip-existing")
        _check("F2/CLI: sua CUNG file YAML -> 0.95 roi chay voi --skip-existing -> CHAY LAI, ket qua "
               "cua cau hinh MOI (0 dong), KHONG giu 4 dong cu",
               a2["n_lines"] == 0 and a2["n_ran"] == 1 and a2["cfg_thresh"] == 0.95
               and a2["fingerprint"] != a1["fingerprint"],
               f"n_lines: {a1['n_lines']} -> {a2['n_lines']}; n_ran={a2['n_ran']}; "
               f"cfg_thresh={a2['cfg_thresh']}")

        a3 = parse(run_py(cli_code(tmp, CACHE, VIDEO, yml3, "clires", skip_existing=True), tmp),
                   "CLI khong doi gi")
        _check("F2/CLI: chay lai khi KHONG doi gi -> TAI SU DUNG (0 video chay lai)",
               a3["n_ran"] in (0, None) and a3["n_reused"] == 1
               and a3["fingerprint"] == a2["fingerprint"],
               f"n_ran={a3['n_ran']} n_reused={a3['n_reused']}")

        # thieu video_meta -> phai chay lai (khong mac nhien tuong thich)
        (tmp / "cli_processed" / "trackers" / "SYN-test" / "clires" / "video_meta"
         / f"{VIDEO}.json").unlink()
        a4 = parse(run_py(cli_code(tmp, CACHE, VIDEO, yml3, "clires", skip_existing=True), tmp),
                   "CLI thieu video_meta")
        _check("F2/CLI: thieu bang chung hoan tat (`video_meta`) -> CHAY LAI, khong tai su dung",
               a4["n_ran"] == 1 and a4["has_video_meta"], f"n_ran={a4['n_ran']}")

        # --- 8. F3: backend tracker duoc luu TUONG MINH ---
        _check("F3: `run_meta.json` luu `tracker_backend` tuong minh tren CA HAI duong chay "
               "(truoc day grep '\"backend\": \"np\"' luon tra 0)",
               a1["tracker_backend"] == "np" and b.get("tracker_backend", "np") in ("np", None),
               f"CLI tracker_backend={a1['tracker_backend']!r}")
    except AssertionError:
        failed += 1
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"  [FAIL] loi khong mong doi: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
