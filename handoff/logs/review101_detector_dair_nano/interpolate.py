"""Interpolate two checkpoints of the same YOLO architecture at alpha=0.5."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    base, tuned, output = map(Path, sys.argv[1:4])
    anchor = YOLO(str(base), task="detect").model.float().eval()
    candidate = YOLO(str(tuned), task="detect").model.float().eval()
    first, second = anchor.state_dict(), candidate.state_dict()
    assert anchor.names == candidate.names and first.keys() == second.keys()
    with torch.no_grad():
        for name in first:
            original, adapted = first[name], second[name]
            assert original.shape == adapted.shape and original.dtype == adapted.dtype
            if original.is_floating_point():
                original.lerp_(adapted, 0.5)
                assert torch.isfinite(original).all()
    anchor.load_state_dict(first, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": anchor, "train_args": {"task": "detect", "imgsz": 640}}, output)
    result = {"alpha": 0.5, "base_sha256": digest(base), "tuned_sha256": digest(tuned),
              "output_sha256": digest(output), "tensor_count": len(first)}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
