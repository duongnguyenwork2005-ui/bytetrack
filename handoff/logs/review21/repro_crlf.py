# -*- coding: utf-8 -*-
"""Tai hien R20-P2 (CODEX_REVIEW_45e086a): `make_patch()` cua make_review_delta lam mat byte CR.

Repo TAM rieng (khong dung repo that); moi lenh git qua `-c core.autocrlf=false -c core.eol=lf`, khong `git config`.
Goi CHINH ham `make_patch()` cua ban packager duoc chi dinh (mac dinh: src/make_review_delta.py hien tai).

    python handoff/logs/review21/repro_crlf.py [--packager duong/dan/make_review_delta.py]
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GIT = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf"]
CR = bytes([13])
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def g(cwd, *args, out_bytes=False):
    r = subprocess.run(GIT + list(args), cwd=cwd, capture_output=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout if out_bytes else r.stdout.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packager", default=str(ROOT / "src" / "make_review_delta.py"))
    a = ap.parse_args()
    pk = Path(a.packager).resolve()
    print(f"# repro R20-P2 - packager: {pk} sha256 {hashlib.sha256(pk.read_bytes()).hexdigest()[:16]}")
    tmp = Path(tempfile.mkdtemp(prefix="repro_crlf_"))
    try:
        repo = tmp / "r"
        repo.mkdir()
        g(repo, "init", "-q", "-b", "main")
        (repo / "src").mkdir()
        shutil.copy2(pk, repo / "src" / "make_review_delta.py")
        (repo / "a.txt").write_bytes(b"base\n")
        g(repo, "add", "-A"); g(repo, "commit", "-qm", "base")
        base = g(repo, "rev-parse", "HEAD").strip()
        noi_dung = b"dong1\r\ndong22\r\n"                   # 15 byte, hai dong CRLF, KHONG danh dau binary
        (repo / "crlf.txt").write_bytes(noi_dung)
        g(repo, "add", "-A"); g(repo, "commit", "-qm", "them crlf")
        target = g(repo, "rev-parse", "HEAD").strip()
        blob = g(repo, "show", f"{target}:crlf.txt", out_bytes=True)
        print(f"file target: {len(blob)} B, CR {blob.count(CR)}, sha256 {hashlib.sha256(blob).hexdigest()[:16]}")

        tho = g(repo, "diff", "--no-renames", "--binary", f"{base}..{target}", "--", "crlf.txt", out_bytes=True)
        spec = importlib.util.spec_from_file_location("mrd_repro", repo / "src" / "make_review_delta.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                       # ROOT cua module = repo tam
        p = mod.make_patch(base, target, ["crlf.txt"])
        pb = p if isinstance(p, bytes) else p.encode("utf-8")
        print(f"git diff tho      : {len(tho)} B, CR {tho.count(CR)}")
        print(f"make_patch()      : kieu {type(p).__name__}, {len(pb)} B, CR {pb.count(CR)}, "
              f"bytes == diff tho: {pb == tho}")

        w = tmp / "w"
        g(tmp, "clone", "-q", str(repo), str(w))
        g(w, "checkout", "-q", base)
        (w / "p.diff").write_bytes(pb)
        r = subprocess.run(GIT + ["apply", "p.diff"], cwd=w, capture_output=True, env=ENV)
        print(f"git apply exit    : {r.returncode} {r.stderr.decode('utf-8', 'replace').strip()}")
        dung = (w / "crlf.txt").read_bytes() if (w / "crlf.txt").exists() else b""
        ok = dung == blob
        print(f"file sau khi ap   : {len(dung)} B, CR {dung.count(CR)}, sha256 {hashlib.sha256(dung).hexdigest()[:16]}")
        print(f"KET QUA: {'KHOP target' if ok else 'KHONG KHOP target (tai hien loi R20-P2)'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
