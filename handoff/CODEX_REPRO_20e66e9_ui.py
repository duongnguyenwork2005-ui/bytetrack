"""Targeted reproductions for N25-F1/F2 on snapshot 20e66e9.

Run from the repository root:
    python handoff/CODEX_REPRO_20e66e9_ui.py --repo .

No Nano, TensorRT or display server is required. Rendering, display-thread
logic and launcher callbacks come from the reviewed source. HighGUI window
operations and Tk widgets are stubs; the picker is a real subprocess standing
in for Zenity. This is not a physical HDMI/GTK acceptance test.

Does not edit repository files. Cleans up its temporary picker process.
Exit 1 means at least one bug reproduced; exit 0 means these checks passed.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from types import SimpleNamespace


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def closed_preview():
    import cv2
    import numpy as np
    import nano_hien_thi as nh
    import render_mot_video as rmv

    class HighGuiStub:
        def __init__(self):
            self.imshow_calls = 0

        def __getattr__(self, name):
            return getattr(cv2, name)

        def namedWindow(self, *args):
            pass

        def resizeWindow(self, *args):
            pass

        def imshow(self, *args):
            self.imshow_calls += 1

        def waitKey(self, *args):
            return -1

        def getWindowProperty(self, *args):
            # The user closes the window after the first displayed frame.
            return -1

        def destroyAllWindows(self):
            pass

    with tempfile.TemporaryDirectory(prefix="codex_n25_preview_") as tmp:
        gui = HighGuiStub()
        view = nh.HienThi(gui, rmv, "fixture", an=False,
                         stop_file=str(Path(tmp) / "STOP"))
        image = np.zeros((96, 160, 3), np.uint8)
        view.them_anh(1, image)
        view.nhan_ket_qua(1, [])
        if not wait_until(lambda: view._cua_so_da_dong):
            view.dong("fixture_cleanup")
            raise RuntimeError("Fixture did not reach the closed-window state")
        # A frame already in flight can finish tracking after the close event.
        view.them_anh(2, image)
        view.nhan_ket_qua(2, [])
        stats = view.dong("dung_theo_yeu_cau")
        observed = {
            "imshow_calls": gui.imshow_calls,
            "reported_n_hien_thi": stats["dem"]["n_hien_thi"],
            "reported_last_frame": stats["frame_hien_thi_cuoi"],
            "fps_hien_thi": stats["fps_hien_thi"],
            "stop_reason": stats["ly_do_dung_tu_cua_so"],
        }
        ok = (gui.imshow_calls == 1
              and observed["reported_n_hien_thi"] == 1
              and observed["reported_last_frame"] == 1
              and observed["fps_hien_thi"] is None)
        return {"finding": "N25-F1", "pass": ok, "observed": observed}


def picker_close():
    if os.name != "posix":
        return {"finding": "N25-F2", "skip": "Run the subprocess fixture on POSIX"}
    import nano_launcher as nl

    class Value:
        def get(self):
            return tempfile.gettempdir()

    with tempfile.TemporaryDirectory(prefix="codex_n25_picker_") as tmp:
        root = Path(tmp)
        marker = root / "picker.pid"
        fake = root / "zenity"
        # /bin/sh safely invokes interpreters whose paths contain spaces.
        import shlex
        fake.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable)
                        + " " + shlex.quote(str(root / "picker.py")) + "\n")
        fake.chmod(0o755)
        (root / "picker.py").write_text(
            "import os,time\nfrom pathlib import Path\nPath("
            + repr(str(marker)) + ").write_text(str(os.getpid()))\ntime.sleep(60)\n")
        original_which = shutil.which
        nl.shutil.which = lambda name: str(fake) if name == "zenity" else original_which(name)
        pid = None
        callbacks = []
        destroyed = []
        try:
            App = nl._build_app()
            app = object.__new__(App)  # Tk-free harness for the real callbacks.
            app.v = {"source": Value(), "out_base": Value()}
            app._chon_dang_mo = False
            app._dang_dong = False
            app.cau_hinh = {}
            app.after = lambda ms, callback: callbacks.append(callback)
            app.destroy = lambda: destroyed.append(True)
            app.ctl = SimpleNamespace(dang_chay=lambda: False)
            app._chon("video")
            if not wait_until(marker.exists):
                raise RuntimeError("Picker fixture failed to start")
            pid = int(marker.read_text())
            app._dong()
            time.sleep(0.15)

            def is_alive():
                try:
                    os.kill(pid, 0)
                    return True
                except ProcessLookupError:
                    return False

            alive = is_alive()
            observed = {"launcher_destroyed": bool(destroyed),
                        "picker_alive_after_close": alive,
                        "scheduled_callbacks": len(callbacks)}
            return {"finding": "N25-F2", "pass": bool(destroyed) and not alive,
                    "observed": observed}
        finally:
            nl.shutil.which = original_which
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(0.1)  # Let the real subprocess.run waiter reap its child.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    results = [closed_preview(), picker_close()]
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 1 if any(r.get("pass") is False for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
