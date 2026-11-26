"""Install the experiment in a separate copy of the R103 Nano package."""
from pathlib import Path
from shutil import copy2, copytree


ROOT = Path.home() / "nano_run"
BASE = ROOT / "r103_optimization/nano_transfer"
PACKAGE = ROOT / "r105_nano_fps/nano_transfer"


def replace(source, target, replacement):
    if source.count(target) != 1:
        raise ValueError("Expected exactly one matching integration point")
    return source.replace(target, replacement, 1)


def patch_stream(source):
    source = replace(source, '    n_gui, ly_do = 0, "het_nguon"\n',
                     '    from r105_cadence import make_scheduler\n'
                     '    mode = os.environ.get("R105_MODE", "off")\n'
                     '    adaptive = make_scheduler(mode, cadence, nguon.fps, nguon.orig_shape, regions)\n'
                     '    scheduler_ms = 0.0\n'
                     '    n_gui, ly_do = 0, "het_nguon"\n')
    source = replace(source, '            if img is None:\n                d = np.zeros((0, 6), np.float32)\n',
                     '            adaptive_skip = False\n'
                     '            if adaptive is not None:\n'
                     '                started = time.perf_counter()\n'
                     '                if img is None:\n'
                     '                    adaptive.reset()\n'
                     '                else:\n'
                     '                    adaptive_skip = adaptive.should_skip(idx, img)\n'
                     '                scheduler_ms += (time.perf_counter() - started) * 1000\n'
                     '            if img is None:\n                d = np.zeros((0, 6), np.float32)\n')
    source = replace(source, '            elif (idx - 1) % cadence:\n',
                     '            elif adaptive_skip or (idx - 1) % cadence:\n')
    source = replace(source, '                if len(d) >= a.max_det:\n',
                     '                if adaptive is not None:\n'
                     '                    started = time.perf_counter()\n'
                     '                    adaptive.observe(d)\n'
                     '                    scheduler_ms += (time.perf_counter() - started) * 1000\n'
                     '                if len(d) >= a.max_det:\n')
    source = replace(source, '           "n_byte_la": kenh.bo_doc.n_byte_la}\n',
                     '           "n_byte_la": kenh.bo_doc.n_byte_la,\n'
                     '           "adaptive": {"mode": mode, "enabled": adaptive is not None,\n'
                     '                        "counts": adaptive.counts if adaptive is not None else {},\n'
                     '                        "cpu_ms": scheduler_ms}}\n')
    compile(source, "nano_detect_stream.py", "exec")
    return source


def patch_live(source):
    return replace(source, '    a = ap.parse_args()\n',
                   '    a = ap.parse_args()\n'
                   '    if os.environ.get("R105_MODE", "off") != "off" and a.save_cache:\n'
                   '        raise ValueError("Adaptive scheduling cannot export a dense detection cache")\n')


def validate_package():
    expected = {"nano_detect_stream.py": patch_stream, "nano_live.py": patch_live}
    for source in (BASE / "src").glob("*.py"):
        destination = PACKAGE / "src" / source.name
        if source.name in expected:
            valid = destination.read_text() == expected[source.name](source.read_text())
        else:
            valid = destination.read_bytes() == source.read_bytes()
        if not valid:
            raise ValueError("Experiment package differs from intended copy: " + source.name)
    if (PACKAGE / "src/r105_cadence.py").read_bytes() != Path(__file__).with_name("cadence.py").read_bytes():
        raise ValueError("Stale adaptive scheduler in experiment package")


def main():
    # Validate every anchor and compile the patches before creating the package.
    import ast
    import hashlib
    import json
    sources = {name: patch((BASE / "src" / name).read_text()) for name, patch in
               (("nano_detect_stream.py", patch_stream), ("nano_live.py", patch_live))}
    for name, text in sources.items():
        ast.parse(text, filename=name)
    PACKAGE.mkdir(exist_ok=False)
    copytree(BASE / "src", PACKAGE / "src")
    for entry in BASE.iterdir():
        if entry.name != "src":
            (PACKAGE / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    copy2(Path(__file__).with_name("cadence.py"), PACKAGE / "src/r105_cadence.py")
    for name, text in sources.items():
        (PACKAGE / "src" / name).write_text(text)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {p.name: {"base": sha(p), "copy": sha(PACKAGE / "src" / p.name)}
                for p in (BASE / "src").iterdir() if p.is_file()}
    (PACKAGE.parent / "package_sources.json").write_text(json.dumps(manifest, indent=2))
    validate_package()
    print(PACKAGE, flush=True)


if __name__ == "__main__":
    main()
