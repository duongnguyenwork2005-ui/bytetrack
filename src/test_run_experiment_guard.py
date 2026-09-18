"""
test_run_experiment_guard.py -- (sau review Codex, loi P1 "runner tai dung ket qua cu") Kiem tra co che
DAU VAN TAY cua `run_experiment.py`:

  1. Chay lan dau -> co `eval_meta.json`; chay lai khi khong doi gi -> TAI SU DUNG (khong goi evaluator).
  2. Doi FILE TRACKING sau khi da co metric -> buoc danh gia phai CHAY LAI va metric phai doi.
  3. Doi CAU HINH khi "tiep tuc" mot run -> tracking chay lai toan bo; `run_meta.json` khong mat
     provenance (co `resumed` khi chi tai su dung).
  4. `--force` -> chay lai moi buoc du dau van tay trung khit.
  5. Evaluator nem loi -> `stage_eval` dem that bai va `main()` tra exit code != 0.

Chay tren FIXTURE BDD100K (khong anh, khong GPU): split `BDDFX-val`, cache `fixture_bddfx_gt` (tao boi
smoke test; neu chua co thi test tu tao). Ten thi nghiem rieng `guard_bddfx` -> khong dung cham ket qua that.

    python src/test_run_experiment_guard.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FIX = ROOT / "data/fixtures/bdd100k_mini"
EXP_DIR = ROOT / "configs/phase2/experiments"
EXP = EXP_DIR / "guard_bddfx.yaml"
NAME = "guard_bddfx"
SPLIT = "BDDFX-val"
_RESULTS: list[tuple[str, bool]] = []

EXP_YAML = """# Thi nghiem RIENG cho test_run_experiment_guard.py (fixture BDD100K, khong anh, khong GPU).
name: {name}
dataset: bdd100k
split_name: BDDFX-val
part: val
manifest: results/phase2/manifests/fixture_bdd100k_final_val.json
cache_id: fixture_bddfx_gt
classes: [1, 2, 3, 5, 7]
max_det: 300
fps: 5
track_buffer_sec: 1.2
elapsed_mode: steps
conditions:
  c25: {{conf_min: {conf_min}}}
trackers:
  - {{name: cv,  yaml: bytetrack.yaml}}
  - {{name: ekf, yaml: configs/bytetrack_ekf_ctrv.yaml}}
baseline: cv
occlusion: {{tag: bddfx_val, fps: 5, context_s: 1.2, gaps: true, n_boot: 50}}
"""


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def write_exp(conf_min=0.25):
    EXP.write_text(EXP_YAML.format(name=NAME, conf_min=conf_min), encoding="utf-8")


def ensure_fixture() -> bool:
    """Tao fixture + GT MOTChallenge + cache gt-synthetic neu chua co (giong test_smoke_phase2)."""
    cdir = config.INTERIM_DIR / "det_cache" / "fixture_bddfx_gt"
    if (config.PROCESSED_DIR / SPLIT).is_dir() and cdir.is_dir() and any(cdir.glob("*.npz")):
        return True
    rc, out = run([PY, "src/mot_datasets/make_bdd100k_fixture.py"])
    if rc:
        print(out[-800:]); return False
    rc, out = run([PY, "src/check_bdd100k.py", "--part", "val", "--data-root", str(FIX), "--no-images", "--rebuild"])
    if rc:
        print(out[-800:]); return False
    sys.path.insert(0, str(ROOT / "src"))
    from mot_datasets.bdd100k_mot import BDD100KMOTDataset, export_motchallenge
    ds = BDD100KMOTDataset(FIX)
    export_motchallenge(ds.gt_dataframe("val"), "val", "vehicle", benchmark="BDDFX")
    rc, out = run([PY, "src/det_cache.py", "--dataset", "bdd100k", "--data-root", str(FIX), "--part", "val",
                   "--all", "--source", "gt", "--cache-id", "fixture_bddfx_gt"])
    if rc:
        print(out[-800:]); return False
    return True


def clean():
    for p in [config.RESULTS_DIR / "phase2" / NAME,
              *(config.PROCESSED_DIR / "trackers" / SPLIT).glob(f"p2-{NAME}-*"),
              *(config.RESULTS_DIR / "trackeval" / SPLIT).glob(f"p2-{NAME}-*")]:
        if p.exists():
            shutil.rmtree(p)


def tdir(tracker="cv"):
    return config.PROCESSED_DIR / "trackers" / SPLIT / f"p2-{NAME}-c25-{tracker}"


def edir(tracker="cv"):
    return config.RESULTS_DIR / "trackeval" / SPLIT / f"p2-{NAME}-c25-{tracker}"


def hota(tracker="cv") -> float:
    import pandas as pd
    return float(pd.read_csv(edir(tracker) / "combined_metrics.csv").iloc[0]["HOTA"])


def main() -> int:
    failed = 0
    try:
        if not ensure_fixture():
            print("  [SKIP] khong tao duoc fixture BDD100K"); return 0
        clean(); write_exp()

        # --- 1. chay lan dau + chay lai khong doi gi ---
        rc, out = run([PY, "src/run_experiment.py", "--exp", str(EXP)])
        _check("chay lan dau: exit 0 va co eval_meta.json", rc == 0 and (edir() / "eval_meta.json").exists(), out)
        h0 = hota()
        rc, out2 = run([PY, "src/run_experiment.py", "--exp", str(EXP)])
        _check("chay lai khong doi gi: TAI SU DUNG (khong goi evaluator)",
               rc == 0 and "TAI SU DUNG" in out2 and "CHAY LAI" not in out2.split("[occl")[0], out2)

        # --- 2. doi file tracking -> phai danh gia lai va metric doi ---
        f = tdir() / "data" / "b1c66a42-6f7d68ca.txt"
        orig = f.read_text(encoding="utf-8")
        f.write_text("\n".join(orig.splitlines()[:-3]) + "\n", encoding="utf-8")   # bo 3 dong cuoi
        rc, out3 = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "eval"])
        h1 = hota()
        _check("doi file tracking -> eval CHAY LAI (ly do 'tracking') va HOTA doi",
               rc == 0 and "CHAY LAI" in out3 and "tracking" in out3 and h1 != h0, out3)

        # --- 3. doi cau hinh khi tiep tuc -> tracking chay lai toan bo, provenance duoc giu ---
        meta_before = json.loads((tdir() / "run_meta.json").read_text(encoding="utf-8"))
        write_exp(conf_min=0.40)
        rc, out4 = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "track"])
        meta_after = json.loads((tdir() / "run_meta.json").read_text(encoding="utf-8"))
        _check("doi cau hinh khi resume -> tracking CHAY LAI toan bo, fingerprint moi",
               rc == 0 and "CHAY LAI" in out4
               and meta_after["input_fingerprint"] != meta_before["input_fingerprint"]
               and meta_after["conf_min"] == 0.40, out4)
        write_exp()                                   # tra lai cau hinh goc
        rc, out5 = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "track"])
        # LUU provenance TRUOC lan resume (review vong 2: truoc day test so file vua doc voi chinh no doc lai)
        meta_before_resume = json.loads((tdir() / "run_meta.json").read_text(encoding="utf-8"))
        n_resumed_before = len(meta_before_resume.get("resumed", []))
        rc2, out5b = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "track"])
        meta_res = json.loads((tdir() / "run_meta.json").read_text(encoding="utf-8"))
        _check("resume khong doi gi -> giu NGUYEN created/git_commit truoc resume + ghi them 'resumed'",
               rc2 == 0 and meta_res["resumed"][-1]["n_videos_ran"] == 0
               and meta_res["created"] == meta_before_resume["created"]
               and meta_res["git_commit"] == meta_before_resume["git_commit"]
               and len(meta_res.get("resumed", [])) == n_resumed_before + 1,
               f"created {meta_before_resume['created']} -> {meta_res['created']}; "
               f"resumed {n_resumed_before} -> {len(meta_res.get('resumed', []))}")

        # --- 3b. BACKEND nam trong dau van tay: giu nguyen -> tai dung; doi backend -> chay lai ---
        run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "eval"])   # cho metric on dinh
        rc_a, out_a = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "eval"])
        _check("fingerprint KHONG doi (cung backend) -> buoc danh gia TAI SU DUNG",
               rc_a == 0 and "TAI SU DUNG" in out_a
               and "CHAY LAI" not in out_a.split("[occl")[0], out_a)

        import os as _os
        env_np = {**_os.environ, "BYTETRACK_BACKEND": "np"}
        subprocess.run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "eval", "--dry-run"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env_np)
        fp_khac = subprocess.run(
            [PY, "-c", "import sys;sys.path.insert(0,'src');import run_fingerprint as fp,json;"
                       "print(json.dumps(fp.code_fingerprint()['backend'],sort_keys=True))"],
            cwd=ROOT, capture_output=True, text=True, env=env_np).stdout.strip()
        fp_cu = subprocess.run(
            [PY, "-c", "import sys;sys.path.insert(0,'src');import run_fingerprint as fp,json;"
                       "print(json.dumps(fp.code_fingerprint()['backend'],sort_keys=True))"],
            cwd=ROOT, capture_output=True, text=True).stdout.strip()
        _check("DOI BACKEND -> dau van tay ma doi (runner buoc phai chay lai, khong tai dung ket qua cu)",
               fp_cu and fp_khac and fp_cu != fp_khac,
               f"ultralytics: {fp_cu[:90]}\n         np         : {fp_khac[:90]}")

        # --- 4. --force chay lai moi buoc ---
        rc, out6 = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--force"])
        _check("--force: moi buoc chay lai (khong co dong TAI SU DUNG)",
               rc == 0 and "TAI SU DUNG" not in out6 and "--force" in out6, out6)

        # --- 5. evaluator loi -> dem that bai va exit code != 0 ---
        code = (
            "import sys; sys.path.insert(0, 'src')\n"
            "import run_trackeval, run_experiment\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('evaluator gia lap loi')\n"
            "run_trackeval.run = boom\n"
            "exp = run_experiment.load_exp(r'%s')\n"
            "rows = run_experiment.plan(exp)\n"
            "import json, pathlib\n"
            "videos = json.loads((pathlib.Path('results/phase2/manifests/fixture_bdd100k_final_val.json')"
            ").read_text(encoding='utf-8'))['videos']\n"
            "n = run_experiment.stage_eval(exp, rows, videos, force=True)\n"
            "print('n_fail =', n)\n"
            "sys.exit(1 if n else 0)\n" % str(EXP).replace("\\", "/")
        )
        rc, out7 = run([PY, "-c", code])
        _check("evaluator nem loi -> stage_eval dem that bai, exit code != 0",
               rc != 0 and "n_fail =" in out7 and "THAT BAI" in out7, out7)
    except AssertionError:
        failed += 1
    finally:
        clean()
        if EXP.exists():
            EXP.unlink()
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
