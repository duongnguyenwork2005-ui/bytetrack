"""Isolated config/callback checks on the exact supplied runner functions.

This is NOT the submitted 34-test suite and does NOT emulate training. The
callback fixture replaces torch.cuda availability and the clock only.
Usage: python check_runner_isolated.py --tree TREE --base BASE_PREIMAGES --out result.json
"""
import argparse
import ast
import hashlib
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml


def functions(path, names, globals_):
    source = ast.parse(path.read_text(encoding="utf-8"))
    wanted = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in wanted} == set(names)
    env = dict(globals_)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), env)
    return env


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    target = args.tree / "src/train_detector.py"
    tests = []

    def check(name, condition, detail=None):
        tests.append({"name": name, "pass": bool(condition), "detail": detail})
        print("PASS" if condition else "FAIL", name, "" if detail is None else detail)

    with tempfile.TemporaryDirectory(prefix="codex_review37_cfg_") as td:
        root = Path(td)
        ns = functions(target, ["LoiCauHinh", "sha256_file", "p_abs", "doc_cfg_pilot", "doc_cfg", "tham_so",
                                "kiem_pilot_chi_doi_epoch"], {
            "Path": Path, "hashlib": hashlib,
            "config": types.SimpleNamespace(PROJECT_ROOT=root), "KHOA_PILOT_DUOC_DOI": ("epochs",),
            "KHOA_CAM_AUTO": ("optimizer", "batch", "amp", "workers", "imgsz", "epochs", "lr0", "momentum", "weight_decay"),
        })
        lock = json.loads((args.tree / "handoff/logs/review37/01_khoa_cau_hinh_pilot_yolov8n.json").read_text())
        ts = lock["tham_so_khai_bao"]
        # Synthetic config fixture, constructed from locked parameter values.
        # Not presented as the omitted real train_bdd5.yaml file.
        fixture = {"ten": "fixture", "model": {}, "dataset": {},
                   "huan_luyen": {k: v for k, v in ts.items() if k not in ("model", "data", "project", "name", "exist_ok", "device")},
                   "smoke": {"epochs": 1, "patience": 1}}
        fixture["huan_luyen"]["epochs"] = 15
        base = root / "base.yaml"
        base.write_text(yaml.safe_dump(fixture), encoding="utf-8")
        sha = hashlib.sha256(base.read_bytes()).hexdigest()
        pilot = {"ke_thua": str(base), "ke_thua_sha256": sha, "pilot": {"epochs": 2}}

        def read(config):
            p = root / "pilot.yaml"
            p.write_text(yaml.safe_dump(config), encoding="utf-8")
            return ns["doc_cfg"](p)

        def rejected(config):
            try:
                read(config)
                return False
            except ns["LoiCauHinh"]:
                return True

        cfg = read(pilot)
        delta = ns["kiem_pilot_chi_doi_epoch"](cfg, root / "data.yaml", root / "runs", "test", "cpu")
        check("inheritance only changes epochs", delta == {"epochs": [1, 2]})
        check("bad inherited hash rejected", rejected({**pilot, "ke_thua_sha256": "0" * 64}))
        check("extra LR override rejected", rejected({**pilot, "pilot": {"epochs": 2, "lr0": 0.1}}))
        check("extra training block rejected", rejected({**pilot, "huan_luyen": {"batch": 32}}))
        check("empty pilot rejected", rejected({**pilot, "pilot": {}}))
        original = base.read_bytes()
        base.write_bytes(original.replace(b"\n", b"\r\n"))
        check("inherited text hash stable across LF/CRLF", read(pilot)["_ke_thua"]["sha256_noi_dung_lf"] == sha)
        base.write_bytes(original + b"# changed content\n")
        check("changed inherited contents rejected", rejected(pilot))

    class FakeModel:
        def __init__(self):
            self.callbacks = {}
        def add_callback(self, event, callback):
            self.callbacks[event] = callback
        def fire(self, event, trainer):
            self.callbacks[event](trainer)

    # No tensor operation or inference is exercised by this fixture.
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    callback_results = {}
    try:
        for label, path in (("base", args.base / "src/train_detector.py"), ("target", target)):
            now = [100.0]
            ns = functions(path, ["gan_callback", "json_safe"], {
                "math": math, "time": types.SimpleNamespace(time=lambda: now[0])})
            model = FakeModel()
            rec = {}
            ns["gan_callback"](model, rec)
            loader = types.SimpleNamespace(num_workers=0, dataset=list(range(16)))
            class Loader:
                num_workers = 0
                dataset = list(range(16))
                def __len__(self):
                    return 2
            tr = types.SimpleNamespace(amp=False, batch_size=8, accumulate=8,
                optimizer=types.SimpleNamespace(param_groups=[{"lr": .01, "params": [1]}]),
                scheduler=types.SimpleNamespace(), train_loader=Loader(), test_loader=loader,
                data={"nc": 5, "names": {0: "car"}}, args=types.SimpleNamespace(), save_dir="fixture",
                tloss={"box_loss": 1.0}, epoch=0, metrics={"metric": .1}, fitness=.1)
            model.fire("on_train_start", tr)
            for ep in (0, 1):
                tr.epoch = ep
                now[0] += 10.0
                model.fire("on_train_batch_end", tr)
                model.fire("on_train_epoch_end", tr)
                now[0] += 2.0
                tr.metrics = {"metric": .1 * (ep + 1)}
                model.fire("on_fit_epoch_end", tr)
            now[0] += 50.0
            tr.epoch = 2
            tr.metrics = {"metric": .999}
            model.fire("on_fit_epoch_end", tr)
            callback_results[label] = rec
        old, new = callback_results["base"], callback_results["target"]
        check("old callback overwrite reproduced", old["epoch"][-1]["metrics"] == {"metric": .999}
              and old["epoch"][-1]["val_s"] == 52.0, old["epoch"][-1])
        check("target preserves epoch metrics/time", new["epoch"][-1]["metrics"] == {"metric": .2}
              and new["epoch"][-1]["val_s"] == 2.0, new["epoch"][-1])
        check("target records final evaluation separately", new["danh_gia_cuoi_ultralytics"]["metrics"] == {"metric": .999})
        check("target callback no recorded error", not new.get("loi_callback"))
    finally:
        if saved is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved
    report = {"scope": "Exact functions selected by AST; synthetic YAML config and callback fixture, clock and torch.cuda mocked; no training.",
              "checks": tests, "callback_results": callback_results,
              "pass": sum(t["pass"] for t in tests), "fail": sum(not t["pass"] for t in tests)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("RESULT", report["pass"], "PASS", report["fail"], "FAIL")
    return int(bool(report["fail"]))


if __name__ == "__main__":
    raise SystemExit(main())
