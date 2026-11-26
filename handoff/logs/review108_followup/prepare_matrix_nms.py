"""Build an isolated Nano package with the tested R85 matrix NMS."""
from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("r85_detector", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    output = args.output.resolve()
    assert base != output and base not in output.parents
    relative = Path("src/nano_detector.py")
    assert not (base / relative).is_symlink()
    source = (base / relative).read_text()
    reference = args.r85_detector.read_text()
    start = reference.index("def nms_matrix(")
    end = reference.index("def decode_nms(", start)
    call = "i = nms_numpy(boxes, x[:, 4], iou_thres)[:max_det]"
    assert source.count("def decode_nms(") == source.count(call) == 1
    source = source.replace("def decode_nms(", reference[start:end] + "def decode_nms(", 1)
    patched = source.replace(call, call.replace("nms_numpy", "nms_matrix"), 1)
    compile(patched, str(relative), "exec")
    output.mkdir(parents=True, exist_ok=False)
    copytree(str(base / "src"), str(output / "src"), symlinks=True)
    for entry in base.iterdir():
        if entry.name != "src":
            (output / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (output / relative).write_text(patched)


if __name__ == "__main__":
    main()
