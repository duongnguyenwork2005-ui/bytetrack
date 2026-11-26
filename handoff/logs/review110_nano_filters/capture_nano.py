"""Capture the exact R109 overlap-mode detections for offline filter replay."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copytree, disk_usage
from subprocess import run
from tempfile import TemporaryDirectory


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("r109", type=Path)
    cli.add_argument("images", type=Path)
    cli.add_argument("package", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    old = args.r109.resolve(strict=True)
    images = args.images.resolve(strict=True)
    package = args.package.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = read(old / "inputs.json")
    assert inputs["modes"] == ["v21combo", "overlap"]
    pair = {(r["dataset"], r["name"]): r for r in read(old / "pairs.json") if r["mode"] == "overlap"}
    bdd, ua = ([r for r in inputs["videos"] if r["dataset"] == ds] for ds in ("bdd", "ua"))
    assert len(bdd) == len(ua) == 40 and len(pair) == 80
    assert inputs["videos"] == read(images / "sample_manifest.json")["rows"]
    card_errors = Path("/sys/fs/ext4/mmcblk1p1/errors_count")
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    boot = boot_id.read_text().strip()
    assert card_errors.read_text().strip() == "0"
    environment = dict(os.environ, BYTETRACK_BACKEND="np", PYTHONDONTWRITEBYTECODE="1", R103_EARLY_CUTOFF="1")
    for key in ("R103_CADENCE", "R103_ROI_Y", "R105_MODE"):
        environment.pop(key, None)
    captures = []
    for item in (row for both in zip(bdd, ua) for row in both):
        ds, name = item["dataset"], item["name"]
        source = images / ds / name
        previous = old / "overlap/runs/full/pipeline" / ds / name / "B"
        compare = pair[ds, name]
        argv = read(previous / "command.json")
        assert argv[1] == "src/nano_live.py" and "--save-cache" not in argv
        user_run = output / "runs" / ds / name
        user_run.parent.mkdir(parents=True, exist_ok=True)
        cache = output / "cache" / name
        hashes = read(old / (ds + "_" + name + "_images.json"))
        assert len(hashes) == 150
        needed = sum(p.stat().st_size for p in source.iterdir() if p.is_file())
        assert needed < disk_usage("/dev/shm").free / 4
        with TemporaryDirectory(prefix="r110_nano_", dir="/dev/shm") as tmp:
            mounted = Path(tmp) / name
            copytree(source, mounted)
            assert {p.name: digest(p) for p in mounted.glob("*.jpg")} == hashes
            argv[argv.index("--source") + 1] = str(mounted)
            argv[argv.index("--out") + 1] = str(user_run)
            if "--regions" in argv:
                argv[argv.index("--regions") + 1] = str(mounted / "ignored_regions.txt")
            argv.extend(("--save-cache", name, "--cache-root", str(cache.parent)))
            with (user_run.parent / (name + "_capture.log")).open("w") as log:
                run(argv, cwd=package, env=environment, check=True, stdout=log, stderr=log)
        snap = read(user_run / "run_meta.json")
        assert snap["hello_con"]["engine_sha256"] == compare["engine_sha256"]
        assert snap["cau_hinh"]["conf_min"] == snap["cau_hinh"]["det_conf"] == .25
        assert not snap["hien_thi"]["bat"]
        mot = digest(user_run / "data" / (name + ".txt"))
        label = digest(user_run / "data_cls" / (name + ".csv"))
        data = cache / (name + ".npz")
        capture = {"dataset": ds, "name": name, "gt_first": item["first"], "gt_last": item["last"],
                   "cache_sha256": digest(data), "mot_sha256": mot, "cls_sha256": label,
                   "r109_mot_parity": mot == compare["mot_sha256"], "r109_cls_parity": label == compare["cls_sha256"]}
        captures.append(capture)
        (output / "captures.json").write_text(json.dumps(captures, indent=2) + "\n")
        assert boot_id.read_text().strip() == boot and card_errors.read_text().strip() == "0"
        print(len(captures), name, capture["r109_mot_parity"], capture["r109_cls_parity"], flush=True)
    print("CAPTURE_COMPLETE", len(captures), flush=True)


if __name__ == "__main__":
    main()
