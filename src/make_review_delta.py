"""
make_review_delta.py -- Dong goi handoff/review_delta_latest.zip: CHI nhung gi thay doi trong MOT vong sua,
tinh tu dung snapshot ma nguoi review (Codex) DA DOC o vong truoc.

Muc dich: sau moi vong sua khong phai gui lai toan bo repository. Goi day du
(`src/make_handoff.py` -> `handoff/phase2_review_latest.zip`) van duoc giu de luu tru tren may va ban giao
o MOC LON; cac vong sua thong thuong dung goi delta nay.

BASE KHONG TU DONG CHAY THEO HEAD. Base duoc ghi trong `handoff/review_base.json` va CHI doi khi co
xac nhan rang nguoi review da doc mot snapshot moi:

    python src/make_review_delta.py --set-base <commit> --round <n> --base-snapshot <duong/dan.zip>

GOI GOM (giu nguyen duong dan tuong doi tu goc repo trong thu muc `files/`):
    REVIEW_BRIEF.md                 bao cao ngan cua vong hien tai (<= ~500 tu), do nguoi viet, doc truoc tien
    MANIFEST.csv / MANIFEST.json    base/target commit, tung file them/sua/xoa + sha256 (base va target)
    patch/delta_<base>..<target>.diff   patch so voi DUNG snapshot Codex da doc (da loai file khong dong goi)
    files/<path>                    ban DAY DU cua file code/test/config duoc them hoac sua trong vong nay
    logs/<path>                     log kiem thu MOI sinh trong vong nay
    data/<path>                     CSV/JSON moi/doi can de xac minh ket luan
    data/large_inputs_hashes.csv    bang thong ke + sha256 cua dau vao LON khong dong goi (cache detection...)

REVIEW_BRIEF.md tren dia KHONG ghi cung base/target (viet xong -> commit -> moi dung goi, nen so ghi tay
luon lac hau mot commit). Dat `<!-- BASE_TARGET -->` o cho can, script dien gia tri THUC luc dong goi.
    DEPENDENCIES.md                 file phu thuoc KHONG dong goi + vi tri trong snapshot goc + sha256

KHONG GOM: cache detection (*.npz), ket qua cu khong doi, toan bo lich su log, dataset, weights, engine
TensorRT, video, goi ZIP khac, va FILE RIENG CUA NGUOI DUNG (`.gitignore`, `src/demo/**`, `live.bat`, `outputs/**`).

PATCH GIU NGUYEN BYTE (review 45e086a, R20-P2): patch la **bytes** doc thang tu stdout cua `git diff --binary`
den luc ghi ZIP -- khong di qua che do text (universal newlines tung bien CRLF thanh LF va CR le thanh xuong
dong). Viec kiem duong dan trong patch dung mot BAN XEM giai ma rieng; bytes goc khong bi bien doi.
Sau khi dung goi, ROUNDTRIP duoc kiem tu dong: ap patch cua ZIP len base trong index TAM cua mot clone tam
(`apply --cached`, khong qua worktree/chuyen doi dong) va so blob tung duong dan voi target.

    python src/make_review_delta.py                 # dung goi delta cho vong hien tai (+ kiem roundtrip)
    python src/make_review_delta.py --dry-run       # chi in ra se gom nhung gi
    python src/make_review_delta.py --kiem-roundtrip [goi.zip]    # chi kiem roundtrip mot goi da dung

File thuoc snapshot base nhung KHONG BAO GIO duoc ban giao (vd. NPZ bi loai co chu y) duoc khai bao khi doi base
va luon in trong DEPENDENCIES.md cua cac goi sau:

    python src/make_review_delta.py --set-base <commit> --round <n> --giu-khai-bao <duong/dan>...
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "handoff" / "review_delta_latest.zip"
BASE_FILE = ROOT / "handoff" / "review_base.json"
BRIEF = ROOT / "handoff" / "REVIEW_BRIEF.md"

#: File RIENG CUA NGUOI DUNG - khong bao gio dong goi, ke ca khi git dang theo doi.
USER_FILES = (".gitignore", "live.bat")
USER_PREFIXES = ("src/demo/", "outputs/")

#: Khong gom vao goi delta (nhung van liet ke trong MANIFEST voi ly do).
#: - goi ZIP khac: tranh long goi trong goi
#: - handoff/patches/*.diff: patch TOAN BO lich su moc 2 (>130k dong), goi delta da co patch rieng
#: - FILE_HASHES/INDEX/versions: mo ta GOI DAY DU, khong phai vong sua nay -> ghi hash trong DEPENDENCIES.md
SKIP_EXACT = ("handoff/FILE_HASHES.csv", "handoff/INDEX.md", "handoff/versions.json",
              "handoff/patches/README.md", "handoff/patches/commits.txt")
SKIP_PREFIXES = ("handoff/archive/", "handoff/phase2_review", "handoff/review_delta",
                 "data/raw/", "data/extracted/", "models/", ".venv/", "env/",
                 # Ban LUU cua ket qua danh gia TRUOC mot vong sua: noi dung y het snapshot ma nguoi
                 # review DA CO (base), nen gui lai chi lam phinh goi. Van commit cuc bo de doi chieu.
                 "results/phase2/_PRE_")
#: --support: file KHONG doi so voi base, gui kem trong `support/` (khong patch, khong tinh la thay doi)
SKIP_SUFFIXES = (".zip", ".npz", ".pt", ".pth", ".engine", ".mp4", ".avi", ".mkv", ".mov", ".parquet", ".diff")

#: Phan loai file trong goi -> thu muc con trong ZIP.
CODE_PREFIXES = ("src/", "configs/", "scripts/", "third_party/", "data/fixtures/")
LOG_PREFIXES = ("handoff/logs/",)
DATA_PREFIXES = ("results/", "data/processed/", "data/interim/")
DOC_EXACT = ("CODEX_HANDOFF.md", "PHASE2_REPORT.md", "README.md", "requirements.txt",
             "handoff/_review_section.md", "handoff/_test_section.md")


PLACEHOLDER = "<!-- BASE_TARGET -->"


def git(*args) -> str:
    """Dau ra git dang TEXT -- chi cho ten file, commit, log. KHONG dung cho noi dung patch/file."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace")


def git_bytes(*args) -> bytes:
    """Dau ra git dang BYTES nguyen ven (khong universal newlines) -- dung cho patch."""
    return subprocess.check_output(["git", *args], cwd=ROOT)


def brief_text(base_info: dict, target: str) -> str:
    """REVIEW_BRIEF.md voi base/target THUC tai luc dong goi (khop MANIFEST.json)."""
    base = base_info["base_commit"]
    khoi = (f"**Base** (snapshot nguoi review da doc): `{base[:7]}`"
            + (f" — goi day du `{base_info['base_snapshot']}`" if base_info.get("base_snapshot") else "")
            + f". **Target**: `{target[:7]}` — {git('log', '-1', '--format=%s', target).strip()}\n"
            + f"({len(git('log', '--format=%h', f'{base}..{target}').split())} commit trong khoang; "
            + "base/target chinh xac trong `MANIFEST.json`.)")
    s = BRIEF.read_text(encoding="utf-8")
    return s.replace(PLACEHOLDER, khoi) if PLACEHOLDER in s else khoi + "\n\n" + s


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_at(commit: str, path: str) -> bytes | None:
    """Noi dung file tai mot commit (None neu commit do khong co file)."""
    try:
        return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def is_user_file(p: str) -> bool:
    return p in USER_FILES or p.startswith(USER_PREFIXES)


def skip_reason(p: str) -> str:
    """Ly do KHONG dong goi file nay (chuoi rong = co dong goi)."""
    if is_user_file(p):
        return "file rieng cua nguoi dung"
    if p in SKIP_EXACT:
        return "artifact cua GOI DAY DU (hash ghi trong DEPENDENCIES.md)"
    if p.startswith(SKIP_PREFIXES):
        return "dataset/weights/goi ZIP khac"
    if p.endswith(SKIP_SUFFIXES):
        return "du lieu nhi phan hoac patch toan bo lich su"
    return ""


def zip_dir(p: str) -> str:
    """Thu muc con trong ZIP cho mot duong dan repo."""
    if p in DOC_EXACT:
        return "files"
    if p.startswith(LOG_PREFIXES):
        return "logs"
    if p.startswith(CODE_PREFIXES):
        return "files"
    if p.startswith(DATA_PREFIXES):
        return "data"
    return "files"


# --------------------------------------------------------------------------------------- base


def load_base() -> dict:
    if not BASE_FILE.exists():
        sys.exit(f"Thieu {BASE_FILE.relative_to(ROOT)} - tao bang --set-base <commit> --round <n>")
    return json.loads(BASE_FILE.read_text(encoding="utf-8"))


def set_base(commit: str, rnd: int, snapshot: str | None, mo_ta: str | None,
             giu_khai_bao: list[str] | None = None) -> None:
    full = git("rev-parse", commit).strip()
    cur = json.loads(BASE_FILE.read_text(encoding="utf-8")) if BASE_FILE.exists() else {}
    lich_su = cur.get("lich_su", [])
    if cur.get("base_commit"):
        lich_su.append({k: cur.get(k) for k in ("vong", "base_commit", "base_snapshot", "cap_nhat_luc")})
    # File thuoc snapshot base nhung KHONG BAO GIO duoc ban giao: khai bao cu duoc MANG SANG, khai bao moi
    # lay sha256/bytes tai CHINH base moi. Neu khong khai bao, doi base se xoa dau vet cua chung khoi goi sau.
    khai_bao = {k["path"]: k for k in cur.get("khai_bao_khong_ban_giao", [])}
    for path in giu_khai_bao or []:
        path = path.replace("\\", "/")
        b = blob_at(full, path)
        if b is None:
            sys.exit(f"DUNG: --giu-khai-bao {path} khong co trong base {full[:7]}")
        khai_bao[path] = {"path": path, "bytes": len(b), "sha256": sha256_bytes(b),
                          "ly_do": skip_reason(path) or "khai bao tay: khong dong goi",
                          "khai_bao_tu_base": full[:7], "vong": rnd}
    BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASE_FILE.write_text(json.dumps({
        "vong": rnd,
        "base_commit": full,
        "base_commit_short": full[:7],
        "base_snapshot": snapshot or cur.get("base_snapshot"),
        "base_mo_ta": mo_ta or "snapshot ma nguoi review da doc o vong truoc",
        "cap_nhat_luc": dt.datetime.now().isoformat(timespec="seconds"),
        "ghi_chu": ("Base KHONG tu dong chay theo HEAD. Chi cap nhat khi nguoi review xac nhan da doc "
                    "mot snapshot moi (chay lai voi --set-base)."),
        "khai_bao_khong_ban_giao": sorted(khai_bao.values(), key=lambda k: k["path"]),
        "lich_su": lich_su,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"da ghi base = {full[:7]} (vong {rnd}) -> {BASE_FILE.relative_to(ROOT)}"
          + (f"; khai bao khong ban giao: {len(khai_bao)} file" if khai_bao else ""))


# --------------------------------------------------------------------------------------- thu thap


def collect(base: str, target: str) -> tuple[list[dict], list[dict]]:
    """Tra ve (rows_gom, rows_bo_qua) cho MOI duong dan doi giua base va target.

    Moi row co HAI co che doc lap:
        trong_patch  duong dan nay co mat trong patch khong (tap THAY DOI LOGIC)
        trong_goi    file vat ly duoc nen vao ZIP o dau (tap PAYLOAD), "" neu khong nen

    Ba truong hop:
        - bi LOAI theo chinh sach (file rieng cua nguoi dung, du lieu lon...): khong patch, khong payload.
        - bi XOA hoac trung noi dung: KHONG co payload nhung VAN nam trong patch, nen ap patch van tai
          tao dung target.
        - con lai: ca patch lan payload.

    Dung `--no-renames`: doi ten duoc bieu dien thanh xoa (ten cu) + them (ten moi), nho do chinh sach
    loai tru va patch xu ly du CA HAI ten (truoc day `-M` chi tra ten moi -> ap patch xong file cu con lai).
    """
    gom, bo = [], []
    out = git("diff", "--name-status", "--no-renames", f"{base}..{target}")
    for ln in out.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        status, path = parts[0][0], parts[-1].replace("\\", "/")
        b_base = blob_at(base, path)
        b_tgt = blob_at(target, path)
        row = {"status": {"A": "them", "M": "sua", "D": "xoa"}.get(status, status),
               "path": path, "trong_patch": False, "trong_goi": "",
               "sha256_base": sha256_bytes(b_base) if b_base is not None else "",
               "sha256_target": sha256_bytes(b_tgt) if b_tgt is not None else "",
               "bytes": len(b_tgt) if b_tgt is not None else 0}

        why = skip_reason(path)
        if why:
            # LOAI TRIET DE: khong payload va cung KHONG vao patch. Truoc day file rieng bi
            # `git rm --cached` van lot noi dung vao patch vi moi row 'xoa' deu duoc dua vao.
            row["ly_do_khong_gom"] = why
            bo.append(row)
            continue

        row["trong_patch"] = True           # moi thay doi khong bi loai deu nam trong patch
        if status == "D":
            row["ly_do_khong_gom"] = "file da bi xoa (thao tac xoa nam trong patch)"
            bo.append(row)
            continue

        # Dedup CHI cho payload va CHI voi file khong phai code/config: neu dedup dung cham code thi
        # ap patch xong van du file, nhung nguoi review mat ban day du trong `files/` -> khong danh doi.
        if row["sha256_target"] and not path.startswith(CODE_PREFIXES) and path not in DOC_EXACT:
            truoc = next((r for r in gom if r["sha256_target"] == row["sha256_target"]), None)
            if truoc is not None:
                row["trung_noi_dung_voi"] = truoc["trong_goi"]
                row["ly_do_khong_gom"] = (f"trung noi dung voi {truoc['trong_goi']} (cung sha256); "
                                          "thay doi van nam trong patch")
                bo.append(row)
                continue

        row["trong_goi"] = f"{zip_dir(path)}/{path}"
        gom.append(row)
    return gom, bo


def _paths_trong_diff(patch: str) -> set[str]:
    """Moi duong dan xuat hien trong tung phan cua diff (khong chi ten thanh vien ZIP)."""
    out = set()
    # split("\n"), KHONG splitlines(): splitlines cat ca o CR le / \x0b / \x1c... ben trong noi dung file
    for ln in patch.split("\n"):
        ln = ln[:-1] if ln.endswith("\r") else ln
        if ln.startswith("diff --git "):
            for tok in ln[len("diff --git "):].split():
                out.add(tok[2:] if tok[:2] in ("a/", "b/") else tok)
        elif ln.startswith(("--- a/", "+++ b/")):
            out.add(ln[6:])
        elif ln.startswith(("rename from ", "rename to ")):
            out.add(ln.split(" ", 2)[2])
    out.discard("dev/null")
    return out


def make_patch(base: str, target: str, paths: list[str]) -> bytes:
    """Patch (BYTES nguyen ven tu stdout git) cho DUNG cac duong dan thuoc tap thay doi logic.

    Sau khi sinh, kiem tra lai TUNG duong dan ben trong diff (tren ban xem giai ma rieng, khong dong vao
    bytes goc): neu co bat ky duong dan nao thuoc dien bi loai (file rieng cua nguoi dung, du lieu lon...)
    thi dung han - khong dong goi.
    """
    if not paths:
        return b""
    # --binary: file nhi phan nho (anh man hinh .png) duoc ma hoa day du trong patch -> `git apply` ap duoc
    # tren cay base sach; khong co no, git chi ghi "Binary files differ" va patch KHONG ap duoc.
    # BYTES (R20-P2): che do text tung bien CRLF -> LF va CR le -> xuong dong (hong so dong cua hunk).
    # (vong 25) Windows gioi han dong lenh ~32.767 ky tu (WinError 206 khi delta co hang tram duong dan dai): chia
    # danh sach thanh LO, moi lo mot `git diff`, noi BYTES theo thu tu duong dan (git ghi tung file theo thu tu duong dan,
    # cac lo khong giao nhau) -> cung noi dung voi mot lan diff; --kiem-roundtrip van ap + so blob tung duong dan.
    patch = b"".join(git_bytes("diff", "--no-renames", "--binary", f"{base}..{target}", "--", *lo)
                     for lo in _lo_duong_dan(paths))
    lot = sorted(q for q in _paths_trong_diff(patch.decode("utf-8", "replace")) if skip_reason(q))
    if lot:
        sys.exit("DUNG: patch chua duong dan thuoc dien bi loai: " + ", ".join(lot[:10]))
    return patch


MAX_DONG_LENH = 20000        # tong ky tu duong dan moi lo (duoi xa gioi han 32.767 cua CreateProcess tren Windows)


def _lo_duong_dan(paths, gioi_han: int = MAX_DONG_LENH):
    """Chia duong dan (da sap xep) thanh cac lo co tong do dai <= gioi_han; moi duong dan nam trong DUNG mot lo."""
    lo, dai = [], 0
    for p in sorted(paths):
        if lo and dai + len(p) + 3 > gioi_han:
            yield lo
            lo, dai = [], 0
        lo.append(p)
        dai += len(p) + 3
    if lo:
        yield lo


def large_inputs_table() -> list[dict]:
    """Bang thong ke + sha256 cua dau vao LON giu tren may (khong dong goi)."""
    rows = []
    for cdir in sorted((ROOT / "data" / "interim" / "det_cache").glob("*")):
        if not cdir.is_dir():
            continue
        npz = sorted(cdir.glob("*.npz"))
        if not npz:
            continue
        # hash cua DANH SACH hash tung file -> mot con so dai dien cho ca cache, doi chieu duoc
        per = {p.name: sha256_file(p) for p in npz}
        rows.append({"loai": "det_cache", "ten": cdir.name, "n_file": len(npz),
                     "bytes": sum(p.stat().st_size for p in npz),
                     "sha256_tong_hop": sha256_bytes(json.dumps(per, sort_keys=True).encode()),
                     "vi_tri_tren_may": str((cdir).relative_to(ROOT)).replace("\\", "/")})
    for w in sorted((ROOT / "models").glob("*.pt")) if (ROOT / "models").is_dir() else []:
        rows.append({"loai": "weights", "ten": w.name, "n_file": 1, "bytes": w.stat().st_size,
                     "sha256_tong_hop": sha256_file(w), "vi_tri_tren_may": f"models/{w.name}"})
    return rows


def collect_support(base: str, target: str, paths: list[str]) -> list[dict]:
    """File HO TRO review (--support): KHONG doi giua base va target, gui kem de moi truong review co dung
    ban hien hanh (vd. test cua packager). Khong nam trong patch, khong tinh la thay doi cua vong nay.
    File co doi so voi base -> DUNG: no phai di theo duong thuong (patch + payload)."""
    rows = []
    for path in paths:
        path = path.replace("\\", "/")
        b_tgt = blob_at(target, path)
        if b_tgt is None:
            sys.exit(f"DUNG: --support {path} khong co trong target {target[:7]}")
        if skip_reason(path):
            sys.exit(f"DUNG: --support {path} thuoc dien bi loai ({skip_reason(path)})")
        b_base = blob_at(base, path)
        h_t, h_b = sha256_bytes(b_tgt), (sha256_bytes(b_base) if b_base is not None else "")
        if h_t != h_b:
            sys.exit(f"DUNG: --support {path} CO doi so voi base {base[:7]} -> no da/phai nam trong patch, "
                     "khong dung --support")
        rows.append({"status": "ho_tro", "path": path, "trong_patch": False, "trong_goi": f"support/{path}",
                     "ly_do_khong_gom": "", "bytes": len(b_tgt), "sha256_base": h_b, "sha256_target": h_t,
                     "ghi_chu": "KHONG doi so voi base (cung sha256); gui kem de ho tro review, KHONG phai "
                                "thay doi cua vong nay"})
    return rows


def dependencies_md(base_info: dict, bo: list[dict], target: str, ho_tro: list[dict] | None = None) -> str:
    """Liet ke file PHU THUOC khong dong goi + vi tri trong snapshot goc + sha256."""
    lines = [
        "# Phu thuoc KHONG nam trong goi delta", "",
        "Cac file duoi day can de hieu / chay lai nhung **khong** duoc dong goi trong vong nay.",
        "Cot *vi tri trong snapshot goc* la duong dan tuong doi tu goc repository — cung la duong dan",
        f"trong goi day du `{base_info.get('base_snapshot') or 'handoff/phase2_review_latest.zip'}` "
        f"(snapshot base `{base_info['base_commit_short']}`).", "",
        "**Khong gia dinh nguoi review con giu ban goc.** Neu thieu file nao trong bang nay, bao lai —",
        "se bo sung dung nhung file do trong goi delta ke tiep, khong gui lai ca repository.", "",
        "| file | vi tri trong snapshot goc | sha256 tai base | sha256 tai target | ly do khong gom |",
        "|:--|:--|:--|:--|:--|",
    ]
    for r in sorted(bo, key=lambda r: r["path"]):
        lines.append(f"| `{Path(r['path']).name}` | `{r['path']}` | `{(r['sha256_base'] or '-')[:12]}` | "
                     f"`{(r['sha256_target'] or '-')[:12]}` | {r['ly_do_khong_gom']} |")
    khai_bao = base_info.get("khai_bao_khong_ban_giao") or []
    if khai_bao:
        lines += [
            "", "## File thuoc snapshot base nhung CHUA TUNG duoc ban giao (khai bao giu nguyen qua cac lan doi base)", "",
            "Cac file nay nam trong cay base nhung da bi loai co chu y o goi truoc; nguoi review **khong** co chung.",
            "Khong dung chung de tai dung ca repository. sha256 tinh tai base luc khai bao.", "",
            "| file | bytes | sha256 | khai bao tu base | ly do |", "|:--|--:|:--|:--|:--|",
        ]
        for k in khai_bao:
            lines.append(f"| `{k['path']}` | {k['bytes']:,} | `{k['sha256']}` | `{k['khai_bao_tu_base']}` | {k['ly_do']} |")
    if ho_tro:
        lines += [
            "", "## File ho tro review gui kem (`support/`) -- KHONG phai thay doi cua vong nay", "",
            "Cac file nay **khong doi** giua base va target (sha256 base == target). Chung duoc gui kem de moi",
            "truong review co dung ban hien hanh (vd. test cua packager); khong nam trong patch.", "",
            "| file | vi tri trong ZIP | sha256 (base == target) |", "|:--|:--|:--|",
        ]
        for r in sorted(ho_tro, key=lambda r: r["path"]):
            lines.append(f"| `{r['path']}` | `{r['trong_goi']}` | `{r['sha256_target'][:12]}` |")
    lines += [
        "", "## Dau vao lon giu tren may (khong bao gio dong goi)", "",
        "Bang thong ke + hash trong `data/large_inputs_hashes.csv` cua goi nay. Du lieu goc "
        "(anh UA-DETRAC, weights, cache `.npz`) nam tren may tac gia; xem `handoff/large_inputs.md` "
        "trong goi day du de biet cach nhan dien va lenh dung chung.", "",
        f"Target commit: `{target[:7]}`. Moi hash trong goi tinh bang sha256.", "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- roundtrip


def _rmtree(p: Path) -> None:
    """Xoa thu muc tam ke ca object git chi-doc (Windows)."""
    def _chmod_lai(fn, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    shutil.rmtree(p, onerror=_chmod_lai)


def kiem_roundtrip(zip_path: Path = ZIP) -> int:
    """Ap patch TRONG ZIP len base va so blob tung duong dan voi target -- theo BYTE (blob id).

    Clone `--shared --no-checkout` vao thu muc tam; `GIT_INDEX_FILE` tam + `read-tree <base>` +
    `apply --cached` (chi index: khong worktree, khong chuyen doi dong/thuoc tinh). Repo that khong bi dong
    vao index/config/worktree. Duong dan xoa -> phai vang mat; duong dan bi loai khong nam trong patch.
    """
    with zipfile.ZipFile(zip_path) as z:
        mj = json.loads(z.read("MANIFEST.json"))
        ten_patch = [n for n in z.namelist() if n.startswith("patch/") and n.endswith(".diff")]
        patch = z.read(ten_patch[0]) if ten_patch else b""
    base, target = mj["base_commit"], mj["target_commit"]
    rows = [r for r in mj["file_trong_goi"] + mj["file_bo_qua"] if r.get("trong_patch")]
    print(f"[roundtrip] {zip_path.name}: base {base[:7]} -> target {target[:7]}, patch {len(patch):,} B, "
          f"{len(rows)} duong dan trong patch")
    tmp = Path(tempfile.mkdtemp(prefix="delta_roundtrip_"))
    loi = []
    try:
        clone = tmp / "r"
        cfg = ["-c", "core.autocrlf=false", "-c", "core.eol=lf"]
        subprocess.run(["git", *cfg, "clone", "-q", "--shared", "--no-checkout", str(ROOT), str(clone)],
                       check=True, capture_output=True)
        env = {**os.environ, "GIT_INDEX_FILE": str(tmp / "index_tam")}

        def g(*args, **kw):
            return subprocess.run(["git", *cfg, *args], cwd=clone, env=env, capture_output=True, **kw)

        r = g("read-tree", base)
        if r.returncode:
            raise RuntimeError(f"read-tree {base[:7]}: {r.stderr.decode('utf-8', 'replace')}")
        if patch:
            (tmp / "p.diff").write_bytes(patch)
            r = g("apply", "--cached", "--whitespace=nowarn", str(tmp / "p.diff"))
            if r.returncode:
                loi.append(f"git apply --cached exit {r.returncode}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
        idx = {}
        for rec in g("ls-files", "-s", "-z").stdout.split(b"\0"):
            if rec:
                meta, path = rec.split(b"\t", 1)
                idx[path.decode("utf-8")] = meta.split()[1].decode()
        for row in rows:
            p = row["path"]
            want = subprocess.run(["git", "rev-parse", "-q", "--verify", f"{target}:{p}"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip() or None
            got = idx.get(p)
            if want != got:
                loi.append(f"{p}: blob sau khi ap {got} != target {want}")
            elif got is not None and row.get("sha256_target"):
                b = subprocess.run(["git", "cat-file", "blob", got], cwd=clone, capture_output=True).stdout
                if sha256_bytes(b) != row["sha256_target"]:
                    loi.append(f"{p}: sha256 sau khi ap khac MANIFEST")
    finally:
        _rmtree(tmp)
    print(f"[roundtrip] {len(rows) - len(loi)}/{len(rows)} duong dan khop target theo blob/sha256"
          + ("" if not loi else "; LOI:"))
    for e in loi[:20]:
        print(f"    {e}")
    return 1 if loi else 0


# --------------------------------------------------------------------------------------- main


def build(base_info: dict, target: str, dry: bool, support: list[str] | None = None) -> int:
    base = base_info["base_commit"]
    gom, bo = collect(base, target)
    ho_tro = collect_support(base, target, support or [])
    if not gom and not bo:
        print(f"khong co thay doi nao giua {base[:7]} va {target[:7]}")
        return 1
    if not BRIEF.exists():
        sys.exit(f"Thieu {BRIEF.relative_to(ROOT)} - viet bao cao ngan cua vong nay truoc khi dong goi")

    patch_paths = [r["path"] for r in gom + bo if r.get("trong_patch")]
    patch = make_patch(base, target, patch_paths)
    patch_name = f"patch/delta_{base[:7]}..{target[:7]}.diff"
    large = large_inputs_table()

    tong = sum(r["bytes"] for r in gom)
    print(f"base {base[:7]} -> target {target[:7]}: gom {len(gom)} file ({tong/1e6:.2f} MB), "
          f"bo qua {len(bo)} file" + (f", ho tro review (khong doi) {len(ho_tro)} file" if ho_tro else ""))
    for r in sorted(gom, key=lambda r: -r["bytes"])[:8]:
        print(f"    {r['bytes']:>9,} B  {r['path']}")
    if dry:
        print("\n-- BO QUA --")
        for r in sorted(bo, key=lambda r: r["path"]):
            print(f"    {r['ly_do_khong_gom']:<52s} {r['path']}")
        return 0

    manifest_rows = sorted(gom + bo + ho_tro, key=lambda r: (r.get("trong_goi") or "zzz", r["path"]))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["status", "path", "trong_goi", "trong_patch",
                                        "ly_do_khong_gom", "trung_noi_dung_voi", "bytes",
                                        "sha256_base", "sha256_target"],
                       extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in manifest_rows:
        w.writerow(r)
    manifest_csv = buf.getvalue()

    manifest_json = json.dumps({
        "vong": base_info.get("vong"),
        "base_commit": base, "base_commit_short": base[:7],
        "base_snapshot": base_info.get("base_snapshot"),
        "base_mo_ta": base_info.get("base_mo_ta"),
        "target_commit": target, "target_commit_short": target[:7],
        "target_subject": git("log", "-1", "--format=%s", target).strip(),
        "tao_luc": dt.datetime.now().isoformat(timespec="seconds"),
        "commits": [ln for ln in git("log", "--format=%h %s", f"{base}..{target}").splitlines() if ln],
        "so_file_trong_goi": len(gom), "so_file_bo_qua": len(bo), "so_file_ho_tro": len(ho_tro),
        "so_duong_dan_trong_patch": sum(1 for r in gom + bo if r.get("trong_patch")),
        "thu_tu_doc": ["REVIEW_BRIEF.md", patch_name, "logs/", "data/", "DEPENDENCIES.md"],
        "file_trong_goi": gom, "file_bo_qua": bo, "file_ho_tro": ho_tro,
        "dau_vao_lon_giu_tren_may": large,
    }, ensure_ascii=False, indent=2) + "\n"

    lbuf = io.StringIO()
    lw = csv.DictWriter(lbuf, fieldnames=["loai", "ten", "n_file", "bytes", "sha256_tong_hop",
                                          "vi_tri_tren_may"], lineterminator="\n")
    lw.writeheader()
    for r in large:
        lw.writerow(r)

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("REVIEW_BRIEF.md", brief_text(base_info, target))
        z.writestr("MANIFEST.csv", manifest_csv)
        z.writestr("MANIFEST.json", manifest_json)
        z.writestr("DEPENDENCIES.md", dependencies_md(base_info, bo, target, ho_tro))
        z.writestr("data/large_inputs_hashes.csv", lbuf.getvalue())
        if patch:
            z.writestr(patch_name, patch)
        for r in gom + ho_tro:
            b = blob_at(target, r["path"])
            if b is None:                                   # khong nen xay ra (A/M luon co blob)
                continue
            z.writestr(r["trong_goi"], b)

    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None, "ZIP hong"
        names = set(z.namelist())
        thieu = [n for n in ("REVIEW_BRIEF.md", "MANIFEST.csv", "MANIFEST.json", "DEPENDENCIES.md")
                 if n not in names]
        xau = [n for n in names if is_user_file(n.split("/", 1)[-1]) or "buffer90" in n
               or n.endswith((".npz", ".pt", ".mp4"))]
        # ...va ca duong dan BEN TRONG patch (ten thanh vien ZIP khong noi gi ve noi dung .diff)
        if patch:
            xau += ["(trong patch) " + q for q in sorted(_paths_trong_diff(
                z.read(patch_name).decode("utf-8", "replace"))) if skip_reason(q)]
    print(f"\nZIP: {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB, {len(names)} file); testzip: OK")
    print(f"file bat buoc thieu: {thieu or 'khong'}")
    print(f"file CAM co trong ZIP: {len(xau)} {xau[:5]}")
    print(f"thu tu doc: REVIEW_BRIEF.md -> {patch_name} -> logs/ + data/ -> DEPENDENCIES.md")
    if thieu or xau:
        return 1
    return kiem_roundtrip(ZIP)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="ghi de base cho lan dung nay (KHONG luu lai)")
    ap.add_argument("--target", default="HEAD")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-base", metavar="COMMIT",
                    help="cap nhat snapshot ma nguoi review da doc (chi dung khi da co xac nhan)")
    ap.add_argument("--round", type=int, help="so vong review (di kem --set-base)")
    ap.add_argument("--base-snapshot", help="duong dan goi day du tuong ung (di kem --set-base)")
    ap.add_argument("--base-note", help="mo ta base (di kem --set-base)")
    ap.add_argument("--support", nargs="*", default=None, metavar="PATH",
                    help="file KHONG doi so voi base gui kem trong support/ de ho tro review "
                         "(khong vao patch, khong tinh la thay doi); file co doi -> tu choi")
    ap.add_argument("--giu-khai-bao", nargs="*", default=None, metavar="PATH",
                    help="(di kem --set-base) file thuoc base nhung KHONG ban giao; giu khai bao trong DEPENDENCIES.md")
    ap.add_argument("--kiem-roundtrip", nargs="?", const=str(ZIP), default=None, metavar="ZIP",
                    help="chi kiem roundtrip patch cua mot goi da dung (mac dinh review_delta_latest.zip)")
    a = ap.parse_args()

    if a.kiem_roundtrip:
        return kiem_roundtrip(Path(a.kiem_roundtrip))
    if a.set_base:
        if a.round is None:
            sys.exit("--set-base can di kem --round <n>")
        set_base(a.set_base, a.round, a.base_snapshot, a.base_note, a.giu_khai_bao)
        return 0
    if a.giu_khai_bao:
        sys.exit("--giu-khai-bao chi dung cung --set-base")

    info = load_base()
    if a.base:
        full = git("rev-parse", a.base).strip()
        info = {**info, "base_commit": full, "base_commit_short": full[:7]}
        print(f"(dung base ghi de {full[:7]}; {BASE_FILE.name} KHONG bi thay doi)")
    target = git("rev-parse", a.target).strip()
    return build(info, target, a.dry_run, a.support)


if __name__ == "__main__":
    raise SystemExit(main())
