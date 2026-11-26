"""Repeat live A/B/B/A on the dense clip with a shared tmpfs image source."""
from pathlib import Path
from shutil import copytree, disk_usage
from tempfile import TemporaryDirectory

import live_abba

video = "MVI_40855"
source = live_abba.DATA / video
tmpfs = Path("/dev/shm")
needed = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
assert needed < disk_usage(tmpfs).free / 4
assert live_abba.swap_mb() < 100
with TemporaryDirectory(prefix="r100_pf_", dir=tmpfs) as temporary:
    root = Path(temporary)
    copytree(source, root / video)
    live_abba.DATA = root
    live_abba.VIDEOS = (video,)
    live_abba.OUT = live_abba.OUT.parent / "live_ram"
    live_abba.main()
