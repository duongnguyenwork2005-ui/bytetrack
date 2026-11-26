"""Prepare an isolated Nano package that decodes one JPEG ahead of inference."""
from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree


ITERATOR = '''    def __iter__(self):
        if self.paths is None:
            yield from self._iter_sync()
            return
        from concurrent.futures import ThreadPoolExecutor

        frames = self._iter_sync()
        with ThreadPoolExecutor(max_workers=1) as reader:
            pending = reader.submit(next, frames, None)
            while True:
                frame = pending.result()
                if frame is None:
                    return
                # Only submit the next read once the current frame leaves the reader.
                pending = reader.submit(next, frames, None)
                yield frame

    def _iter_sync(self):
'''


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    output = args.output.resolve()
    assert base != output and base not in output.parents
    relative = Path("src/nano_detect_stream.py")
    assert not (base / relative).is_symlink()
    source = (base / relative).read_text()
    signature = "    def __iter__(self):\n"
    assert source.count(signature) == 1 and "    def _iter_sync(" not in source
    patched = source.replace(signature, ITERATOR, 1)
    compile(patched, str(relative), "exec")
    output.mkdir(parents=True, exist_ok=False)
    copytree(str(base / "src"), str(output / "src"), symlinks=True)
    for entry in base.iterdir():
        if entry.name != "src":
            (output / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (output / relative).write_text(patched)


if __name__ == "__main__":
    main()
