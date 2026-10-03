"""
test_tiep_tuc_huan_luyen.py -- Kiem resume cua runner detector (vong 39). Ke hoach so sanh duoc khoa TRUOC khi chay
(handoff/logs/review39/01_ke_hoach_kiem_resume.md).

Phan biet:
  - CHAY THAT: train_detector.py qua subprocess, ultralytics that, CPU, dataset fixture tong hop (bo xuat that, hop
    dong v2), cau hinh = train_bdd5.yaml voi imgsz 160, workers 0, epochs 3.
  - SPY: train_detector.main() trong tien trinh, `ultralytics.YOLO` thay bang spy ghi lai moi lan khoi tao / train va
    CHAN training. Moi thu truoc ranh gioi (hop dong dataset, chon / kiem checkpoint, sao luu) chay that.

  U   don vi (khong train): nhat ky, kiem_tuong_thich, kiem_ung_vien (checkpoint tong hop), so lan tiep tuc, chep nguyen tu.
  R   (moi model, CHAY THAT):
      - C1 = luot lien tuc 3 epoch; C2 = lap lai C1 (doi chung tat dinh);
      - I = cung cau hinh, bi KILL ngay sau khi epoch 1 da luu; R = `--tiep-tuc` I toi du 3 epoch.
      K1 C1 == C2 tung bit. K2 I == C1 truoc diem ngat.
      K3 trang thai khoi phuc (runner tu kiem + doi chieu file).
      K4 R trung C1 o ni / LR / momentum / batch / mosaic thuc te theo epoch.
      K5 accumulate + vi tri buoc toi uu cua C1 va R khop mo phong tu ma ultralytics.
      K6 khac biet du kien (loss, trong so cuoi, thu tu du lieu) -- bao do lon, khong dung nguong.
      K7 nhat ky / provenance / sao luu. K8 trang thai cuoi + tu choi tiep tuc luot da xong.
      K9 I2 = ngat SAU epoch 2 (sau epoch tat mosaic) roi tiep tuc -> nhanh resume_training tu tat mosaic.
  S   (yolov8n, SPY) ca tu choi dau vao: phai exit != 0 va KHONG goi train; + ca chon ban sao khi last.pt hong,
      dry-run tiep tuc, Ctrl+C. S20 (vong 40): dong hong o GIUA nhat ky -> exit 2, khong goi train.
  N   (vong 40, R39-F1; yolov8n, CHAY THAT qua subprocess) nhat ky co phan duoi: sach / JSON ghi do khong newline / JSON
      ghi do co newline / JSON hop le thieu newline. Moi ca:
        0a dry-run khong doi file; 0b dry-run bao ke hoach;
        A  resume that (kill sau epoch 2): doc lai duoc, su kien hop le cu giu nguyen + su kien moi;
        B  ban sao nhat ky truoc sua trung tung byte + bien nhan;
        C  checkpoint / bang chung cu bao toan;
        D  preflight tiep theo chay duoc va resume tiep toi het; B2 ban sao truoc lan tiep tuc 2.
      N-giua: dong hong o GIUA -> dry-run va resume that exit 2, thu muc luot khong doi.
      `--chi-nhom-n --ma-nguon <thu muc src cu>`: chay rieng nhom N voi train_detector.py cua MA CU (chung minh test bat
      loi); phan doc/kiem trong tien trinh test van dung ham doc cua ma hien tai.
  U9/U10 (vong 40): phan_tich_duoi_nhat_ky / sua_duoi_nhat_ky (ham that, thu muc tam); U10b MOCK os.replace loi -> nhat
      ky cu con nguyen.

    python src/test_tiep_tuc_huan_luyen.py [--model yolov8n|yolo11n ...] [--bo-spy]
    python src/test_tiep_tuc_huan_luyen.py --chi-nhom-n [--ma-nguon DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import config  # noqa: E402

PY = sys.executable
KQ: list[tuple[str, bool]] = []
SKIP: list[tuple[str, str]] = []
THONG_TIN: dict = {}


def _check(ten, ok, ct=""):
    KQ.append((ten, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {ten}" + (f"  ({str(ct)[:600]})" if ct else ""), flush=True)
    return bool(ok)


def _skip(ten, ly_do):
    SKIP.append((ten, ly_do))
    print(f"  [SKIP] {ten}  ({ly_do}) -- CHUA KIEM, khong tinh PASS", flush=True)


def _chay(nhom, fn):
    print(f"\n== {nhom}", flush=True)
    try:
        fn()
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- SystemExit (vd argparse) khong duoc dung ca suite
        _check(f"{nhom}: loi khong mong doi {type(e).__name__}: {e}", False)
        traceback.print_exc()


def _yaml(p: Path, d: dict) -> Path:
    import yaml
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return p


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg_test(tmp: Path, ten: str, **sua) -> Path:
    import yaml
    c = yaml.safe_load((config.CONFIGS_DIR / "phase2" / "detector" / "train_bdd5.yaml").read_text(encoding="utf-8"))
    c["model"]["yolov8n"]["weights"] = str(config.MODELS_DIR / "yolov8n.pt")
    c["model"]["yolo11n"]["weights"] = str(config.MODELS_DIR / "candidates" / "yolo11n.pt")
    c["huan_luyen"].update({"imgsz": 160, "workers": 0, "epochs": 3})
    c["huan_luyen"].update(sua)
    return _yaml(tmp / f"cfg_{ten}.yaml", c)


def lenh(cfg: Path, model: str, ten: str, data: Path, tmp: Path, *them) -> list[str]:
    return [PY, "-u", str(SRC / "train_detector.py"), "--config", str(cfg), "--model", model, "--che-do", "day-du",
            "--xac-nhan-train-dai", "--data", str(data), "--ten", ten, "--project", str(tmp / "runs"),
            "--ket-qua-dir", str(tmp / "kq"), "--device", "cpu", *them]


def chay_that(argv: list[str], log: Path, timeout=1800) -> int:
    with open(log, "wb") as f:
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def doc_nk(run_dir: Path) -> list[dict]:
    p = run_dir / "nhat_ky_luot.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def chay_va_ngat(argv: list[str], run_dir: Path, log: Path, epoch_ngat=1, timeout=1200) -> dict:
    """Chay that, KILL ca cay tien trinh ngay khi nhat ky ghi epoch `epoch_ngat` da luu (mo phong mat dien / dong cua so)."""
    import psutil
    with open(log, "wb") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        try:
            while time.time() - t0 < timeout:
                if p.poll() is not None:
                    return {"bi_kill": False, "exit": p.returncode, "ly_do": "tien trinh tu thoat truoc khi ngat"}
                if any(s.get("loai") == "epoch_da_luu" and s.get("epoch") == epoch_ngat for s in doc_nk(run_dir)):
                    pr = psutil.Process(p.pid)
                    for c in pr.children(recursive=True):
                        c.kill()
                    pr.kill()
                    return {"bi_kill": True, "exit": p.wait(timeout=60), "sau_s": round(time.time() - t0, 1)}
                time.sleep(0.1)
            return {"bi_kill": False, "exit": None, "ly_do": "het thoi gian cho"}
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()


def doc_batch(run_dir: Path) -> list[dict]:
    return list(csv.DictReader(open(run_dir / "theo_batch.csv", newline="", encoding="utf-8")))


def tensor_ck(p: Path) -> dict:
    import torch
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    return {"epoch": ck["epoch"], "updates": ck["updates"], "best_fitness": ck["best_fitness"],
            "ema": {k: v for k, v in ck["ema"].state_dict().items()},
            "opt": {(i, k): v for i, st in ck["optimizer"]["state"].items() for k, v in st.items()},
            "pg": [{k: v for k, v in g.items()} for g in ck["optimizer"]["param_groups"]]}


def ck_giong_het(a: dict, b: dict) -> tuple[bool, str]:
    import torch
    loi = [k for k in ("epoch", "updates", "best_fitness") if a[k] != b[k]]
    if a["ema"].keys() != b["ema"].keys() or any(not torch.equal(a["ema"][k], b["ema"][k]) for k in a["ema"]):
        loi.append("ema")
    if a["opt"].keys() != b["opt"].keys() or any(not torch.equal(a["opt"][k], b["opt"][k]) if torch.is_tensor(a["opt"][k])
                                                 else a["opt"][k] != b["opt"][k] for k in a["opt"]):
        loi.append("optimizer_state")
    if a["pg"] != b["pg"]:
        loi.append("param_groups")
    return not loi, f"khac {loi}" if loi else "trung"


def lech_ema_max(a: dict, b: dict) -> float:
    return max(float((a["ema"][k].float() - b["ema"][k].float()).abs().max()) for k in a["ema"]
               if a["ema"][k].is_floating_point())


COT_SO = ["ni", "loss_tong", "box_loss", "cls_loss", "dfl_loss", "lr_weight", "lr_bn", "lr_bias", "momentum_weight",
          "momentum_bn", "momentum_bias", "accumulate", "ema_updates", "batch_size", "thu_tu_batch"]


def mo_phong(E: int, nb: int, nw: int, nbs: int, batch: int, tu_epoch: int = 0, updates_dau: int = 0
             ) -> tuple[dict, list[int]]:
    """Mo phong DUNG logic BaseTrainer._do_train 8.4.141: accumulate (warmup) va buoc toi uu khi ni - last_opt_step >=
    accumulate. Bat dau (lien tuc hoac sau resume): last_opt_step = -1, accumulate = round(nbs / batch).
    Tra ({ni: (accumulate, ema.updates sau batch)}, [ni co buoc toi uu])."""
    acc, last, upd, out, buoc = max(round(nbs / batch), 1), -1, updates_dau, {}, []
    for ep in range(tu_epoch, E):
        for i in range(nb):
            ni = i + nb * ep
            if ni < nw:
                acc = max(1, int(np.interp(ni, [0, nw], [1, nbs / batch]).round()))
            if ni - last >= acc:
                upd += 1
                last = ni
                buoc.append(ni)
            out[ni] = (acc, upd)
    return out, buoc


# =========================================================================== spy
class _SpyDung(Exception):
    pass


class SpyYOLO:
    goi: list = []
    names_coco: dict = {}
    hanh_dong = "chan"

    def __init__(self, w, *a, **k):
        SpyYOLO.goi.append(("khoi_tao", str(w)))
        self.names = dict(SpyYOLO.names_coco)

    def add_callback(self, su_kien, *a, **k):
        SpyYOLO.goi.append(("callback", su_kien))

    def train(self, **kw):
        SpyYOLO.goi.append(("train", {k: str(v) for k, v in kw.items()}))
        if SpyYOLO.hanh_dong == "ctrl_c":
            raise KeyboardInterrupt
        raise _SpyDung("spy: chan training that")


def chay_spy(argv: list[str], hanh_dong="chan") -> tuple[int, str, str, list]:
    import ultralytics
    import yaml

    import train_detector
    if not SpyYOLO.names_coco:
        SpyYOLO.names_coco = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml")
                                            .read_text(encoding="utf-8"))["names"]
    goc = ultralytics.YOLO
    SpyYOLO.goi, SpyYOLO.hanh_dong = [], hanh_dong
    ultralytics.YOLO = SpyYOLO
    o, e, argv_cu = io.StringIO(), io.StringIO(), sys.argv
    sys.argv = ["train_detector.py", *argv]
    try:
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            try:
                code = train_detector.main()
            except SystemExit as se:                  # vd argparse: tra ma + ghi lai, khong lam dung ca suite
                code = se.code if isinstance(se.code, int) else 1
                print(f"[SystemExit {se.code!r}]", file=sys.stderr)
    finally:
        sys.argv = argv_cu
        ultralytics.YOLO = goc
    return code, o.getvalue(), e.getvalue(), list(SpyYOLO.goi)


def anh_chup(d: Path) -> dict:
    return {str(p.relative_to(d)).replace("\\", "/"): sha(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def co_train(goi) -> bool:
    return any(g[0] == "train" for g in goi)


# =========================================================================== U
def nhom_u(tmp: Path):
    import torch
    import torch.nn as nn

    import tiep_tuc_huan_luyen as ttl
    d = tmp / "u_nk"
    d.mkdir()
    ttl.ghi_nhat_ky(d, "bat_dau", che_do_luot="moi", lan=0)
    ttl.ghi_nhat_ky(d, "epoch_da_luu", lan=0, epoch=1)
    with open(d / ttl.TEN_NHAT_KY, "a", encoding="utf-8") as f:
        f.write('{"loai": "epoch_da_luu", "epoch": 2, "la')          # dong cuoi ghi do (bi ngat luc ghi)
    sk, cb = ttl.doc_nhat_ky(d)
    _check("U1a nhat ky: dong CUOI ghi do -> bo qua + canh bao, 2 su kien hop le", len(sk) == 2 and cb, str(cb))
    with open(d / ttl.TEN_NHAT_KY, "a", encoding="utf-8") as f:
        f.write('\n{"loai": "ket_thuc"}\n')
    try:
        ttl.doc_nhat_ky(d)
        ok = False
    except ttl.LoiTiepTuc:
        ok = True
    _check("U1b nhat ky: dong hong o GIUA -> LoiTiepTuc (khong tin nhat ky)", ok)
    ts = {"epochs": 3, "lr0": 0.01, "data": str(tmp / "a" / "data.yaml"), "device": "0", "model": "x.pt", "batch": 8}
    ck = {"epochs": 3, "lr0": 0.01, "data": str(tmp / "a" / "." / "data.yaml"), "device": "0", "batch": 8}
    k1 = ttl.kiem_tuong_thich(ck, ts)
    k2 = ttl.kiem_tuong_thich(ck | {"lr0": 0.02, "epochs": 4, "device": "cpu", "data": str(tmp / "b.yaml")}, ts)
    _check("U2 kiem_tuong_thich: trung (duong dan chuan hoa, bo qua model) -> []; lech lr0/epochs/device/data -> 4 muc",
           k1 == [] and len(k2) == 4 and all(any(x in m for m in k2) for x in ("lr0", "epochs", "device", "data")),
           f"{k1} | {k2}")
    run = tmp / "u_run"
    (run / "weights").mkdir(parents=True)
    ema = nn.Sequential(nn.Conv2d(3, 4, 3))
    ema.names = {0: "car"}

    def tao(ten, **sua):
        c = {"epoch": 0, "updates": 3, "best_fitness": 0.5, "ema": ema,
             "optimizer": {"state": {0: {"momentum_buffer": torch.zeros(2)}}, "param_groups": [{"lr": 0.01}]},
             "scaler": {}, "train_args": {"save_dir": str(run), "name": "u_run", "epochs": 3}}
        c.update(sua)
        p = run / f"{ten}.pt"
        torch.save(c, str(p))
        return p
    hop_le = tao("hop_le")
    ban_ghi = {sha(hop_le): {"epoch": 1, "ema_updates": 3, "best_fitness": 0.5}}
    u = ttl.kiem_ung_vien(hop_le, ban_ghi, run, "u_run")
    _check("U3a checkpoint du trang thai, sha trong nhat ky, khop noi dung -> hop le", u["hop_le"], str(u["ly_do"]))
    la = run / "la.pt"
    la.write_bytes(b"khong phai pickle, neu bi torch.load se no")
    u = ttl.kiem_ung_vien(la, ban_ghi, run, "u_run")
    _check("U3b file co sha256 KHONG co trong nhat ky -> loai, KHONG torch.load (khong co loi nap)",
           not u["hop_le"] and "khong co trong nhat ky" in str(u["ly_do"]) and "khong nap duoc" not in str(u["ly_do"]),
           str(u["ly_do"]))
    for ten, sua, mau in (("strip", {"epoch": -1, "optimizer": None}, "epoch -1"), ("thieu_opt", {"optimizer": None},
                          "thieu trang thai optimizer"), ("thieu_ema", {"ema": None}, "thieu EMA"),
                          ("save_dir", {"train_args": {"save_dir": str(tmp), "name": "u_run"}}, "save_dir"),
                          ("name", {"train_args": {"save_dir": str(run), "name": "khac"}}, "name"),
                          ("lech_nk", {"updates": 99}, "lech nhat ky")):
        p = tao(ten, **sua)
        u = ttl.kiem_ung_vien(p, {sha(p): {"epoch": 1, "ema_updates": 3, "best_fitness": 0.5}}, run, "u_run")
        _check(f"U3 checkpoint '{ten}' -> loai voi ly do '{mau}'", not u["hop_le"] and mau in str(u["ly_do"]),
               str(u["ly_do"]))
    run2 = tmp / "u_lan"
    (run2 / ttl.THU_MUC_KP).mkdir(parents=True)
    ttl.ghi_nhat_ky(run2, "bat_dau", che_do_luot="moi", lan=0)
    ttl.ghi_nhat_ky(run2, "ket_thuc", lan=1, trang_thai="loi")
    (run2 / ttl.THU_MUC_KP / "nguon_tiep_tuc_2.pt").write_bytes(b"x")
    _check("U4 so lan tiep tuc = max(lan trong nhat ky, so trong ten nguon_tiep_tuc_N.pt) -> 2 (lan hong truoc bat_dau "
           "van giu so)", ttl.so_lan_tiep_tuc_da_co(run2) == 2, str(ttl.so_lan_tiep_tuc_da_co(run2)))
    s = ttl.chep_nguyen_tu(hop_le, run / "ban_chep.pt")
    _check("U5 chep nguyen tu: ban chep cung sha256, khong con file tam", s == sha(hop_le) == sha(run / "ban_chep.pt")
           and not list(run.glob("*.dang_chep")))
    import types
    from copy import copy

    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import v8_transforms
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import DEFAULT_CFG
    gia = types.SimpleNamespace(data={}, cache=None, use_obb=False, use_keypoints=False, use_segments=False)
    h1, h0 = copy(get_cfg(DEFAULT_CFG)), copy(get_cfg(DEFAULT_CFG))
    h1.mosaic, h0.mosaic = 1.0, 0.0
    gia.transforms = v8_transforms(gia, 160, h1)
    p1 = ttl.xac_suat_mosaic(gia)
    gia.transforms = v8_transforms(gia, 160, h0)
    p0 = ttl.xac_suat_mosaic(gia)
    _check("U6 doc mosaic: p cua Mosaic trong transforms that (v8_transforms) = 1,0 khi bat / 0,0 khi tat; YOLODataset "
           "KHONG co thuoc tinh 'mosaic' (cach doc getattr(ds, 'mosaic') cu luon ra False)",
           p1 == 1.0 and p0 == 0.0 and not hasattr(YOLODataset, "mosaic"), f"p bat {p1}, p tat {p0}")

    # U7 quy dao: lan 0 luu e1..e5; lan 1 tiep tu e3 (e4, e5 cua lan 0 thanh NHANH BI BO), luu e4..e6; lan 2 tiep tu e5
    def ev(lan, e, best, s, t):
        return {"loai": "epoch_da_luu", "lan": lan, "epoch": e, "best_ghi_epoch_nay": best, "best_sha256": s if best
                else None, "last_sha256": s, "ban_sao_sha256": s, "ban_sao": f"khoi_phuc/last_e{e:03d}.pt", "t": t}
    sk = [{"loai": "bat_dau", "lan": 0, "che_do_luot": "moi", "t": 0}]
    sk += [ev(0, e, b, f"a{e}", e) for e, b in ((1, True), (2, True), (3, False), (4, True), (5, False))]
    sk += [{"loai": "bat_dau", "lan": 1, "che_do_luot": "tiep_tuc", "t": 10, "tiep_tuc": {"chon": {"sha256": "a3"}}}]
    sk += [ev(1, e, b, f"b{e}", 10 + e) for e, b in ((4, False), (5, True), (6, False))]
    sk += [{"loai": "bat_dau", "lan": 2, "che_do_luot": "tiep_tuc", "t": 20, "tiep_tuc": {"chon": {"sha256": "b5"}}}]
    sk += [ev(2, 6, False, "c6", 26)]

    def lay(lan, e):
        return next(s for s in sk if s["loai"] == "epoch_da_luu" and s["lan"] == lan and s["epoch"] == e)
    q_c6 = [(s["lan"], s["epoch"]) for s in ttl.quy_dao(sk, lay(2, 6))]
    qd_b4 = ttl.quy_dao(sk, lay(1, 4))
    best_b4 = [s for s in qd_b4 if s["best_ghi_epoch_nay"]][-1]["best_sha256"]
    gay = [dict(s) for s in sk]
    next(s for s in gay if s["loai"] == "bat_dau" and s["lan"] == 1)["tiep_tuc"] = {"chon": {"sha256": "zz"}}
    try:
        ttl.quy_dao(gay, next(s for s in gay if s["loai"] == "epoch_da_luu" and s["lan"] == 1 and s["epoch"] == 4))
        gay_loi = False
    except ttl.LoiTiepTuc:
        gay_loi = True
    _check("U7a quy dao qua 2 lan tiep tuc: lan2 e6 -> [(0,1),(0,2),(0,3),(1,4),(1,5),(2,6)]; lan1 e4 -> best quy dao = a2 "
           "(KHONG phai a4 cua nhanh bi bo); nguon khong truy duoc -> LoiTiepTuc",
           q_c6 == [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6)] and [(s["lan"], s["epoch"]) for s in qd_b4]
           == [(0, 1), (0, 2), (0, 3), (1, 4)] and best_b4 == "a2" and gay_loi, f"{q_c6} | best {best_b4}")
    rb = tmp / "u_best"
    (rb / "weights").mkdir(parents=True)
    (rb / "khoi_phuc").mkdir()
    (rb / "khoi_phuc" / "last_e002.pt").write_bytes(b"best e2")
    s2 = hashlib.sha256(b"best e2").hexdigest()
    qd = [{"lan": 0, "epoch": 1, "best_ghi_epoch_nay": False},
          {"lan": 0, "epoch": 2, "best_ghi_epoch_nay": True, "best_sha256": s2, "ban_sao": "khoi_phuc/last_e002.pt",
           "ban_sao_sha256": s2}]
    (rb / "weights" / "best.pt").write_bytes(b"best e2")
    k_khop = ttl.ke_hoach_best(rb, qd)
    (rb / "weights" / "best.pt").write_bytes(b"best cua nhanh bi bo")
    k_mc = ttl.ke_hoach_best(rb, qd)
    (rb / "khoi_phuc" / "last_e002.pt").write_bytes(b"ban sao hong")
    try:
        ttl.ke_hoach_best(rb, qd)
        hong_loi = False
    except ttl.LoiTiepTuc:
        hong_loi = True
    _check("U7b ke hoach best.pt: khop quy dao -> khong lam gi; lech -> khoi phuc tu ban sao cung byte; ban sao hong -> "
           "LoiTiepTuc (khong doan)", k_khop["khop"] and k_khop["khoi_phuc_tu"] is None and k_mc["khop"] is False
           and k_mc["khoi_phuc_tu"] == "khoi_phuc/last_e002.pt" and hong_loi, f"{k_khop} | {k_mc}")
    # U8 cat results.csv (BYTE, giu CRLF cua ultralytics tren Windows) sau khi da luu ban goc
    rs = tmp / "u_csv"
    rs.mkdir()
    goc_csv = b"epoch,time,x\r\n1,10,0.5\r\n2,20,0.6\r\n3,30,0.7\r\n"
    (rs / "results.csv").write_bytes(goc_csv)
    (rs / "nguon.pt").write_bytes(b"ck")
    sl = ttl.sao_luu_truoc_tiep_tuc(rs, rs / "nguon.pt", 1, best=None, epoch_da_luu=2)
    _check("U8 results.csv: chi giu epoch <= epoch da luu (dong epoch 3 bi bo), giu nguyen byte/CRLF; ban goc luu o "
           "truoc_tiep_tuc_1", (rs / "results.csv").read_bytes() == b"epoch,time,x\r\n1,10,0.5\r\n2,20,0.6\r\n"
           and (rs / "khoi_phuc" / "truoc_tiep_tuc_1" / "results.csv").read_bytes() == goc_csv
           and sl["results_csv"] == {"dong_giu": 2, "dong_bo": 1}, str(sl["results_csv"]))

    # U9 / U10 (vong 40, R39-F1): phan duoi nhat ky -- ham THAT, thu muc tam
    pre = b'{"loai": "bat_dau", "lan": 0}\n{"loai": "epoch_da_luu", "lan": 0, "epoch": 1}\n'
    ca_duoi = [("sach", b"", "khong", pre), ("ghi_do_khong_newline", b'{"loai": "epoch_da_luu", "ep', "cat_duoi", pre),
               ("ghi_do_co_newline", b'{"loai": "epoch_da_luu", "ep\n', "cat_duoi", pre),
               ("hop_le_thieu_newline", b'{"loai": "ket_thuc", "lan": 0}', "them_newline",
                pre + b'{"loai": "ket_thuc", "lan": 0}\n'), ("byte_nul", b"\x00" * 8, "cat_duoi", pre)]
    u9 = {}
    for ten, hau, hd, _ in ca_duoi:
        kh = ttl.phan_tich_duoi_nhat_ky(pre + hau)
        u9[ten] = kh["hanh_dong"] == hd and (hd != "cat_duoi" or (kh["giu_bytes"], kh["bo_bytes"]) == (len(pre), len(hau)))
    dg = pre.split(b"\n")
    try:
        ttl.phan_tich_duoi_nhat_ky(dg[0] + b"\n" + b'{"loai": "ho\n' + b"\n".join(dg[1:]))
        u9["giua_tu_choi"] = False
    except ttl.LoiTiepTuc:
        u9["giua_tu_choi"] = True
    _check("U9 phan_tich_duoi_nhat_ky (chi doc): sach -> khong; JSON ghi do khong/co newline, byte NUL -> cat_duoi dung "
           "ranh gioi dong; JSON hop le thieu newline -> them_newline; dong hong o GIUA -> LoiTiepTuc", all(u9.values()),
           str(u9))
    u10 = {}
    for ten, hau, hd, mong in ca_duoi:
        r = tmp / f"u10_{ten}"
        (r / "kp").mkdir(parents=True)
        (r / ttl.TEN_NHAT_KY).write_bytes(pre + hau)
        truoc, _ = ttl.doc_nhat_ky(r)
        rec = ttl.sua_duoi_nhat_ky(r, r / "kp", ttl.phan_tich_duoi_nhat_ky(pre + hau))
        sau, cb = ttl.doc_nhat_ky(r)
        bn = json.loads((r / "kp" / "xu_ly_nhat_ky.json").read_text(encoding="utf-8"))
        u10[ten] = ((r / "kp" / ttl.TEN_NHAT_KY).read_bytes() == pre + hau and (r / ttl.TEN_NHAT_KY).read_bytes() == mong
                    and sau == truoc and not cb and bn == json.loads(json.dumps(rec)) and bn["hanh_dong"] == hd
                    and bn["sau_sha256"] == hashlib.sha256(mong).hexdigest() and not list(r.rglob("*.dang_ghi")))
    _check("U10 sua_duoi_nhat_ky: ban sao nguyen byte; nhat ky sau sua dung byte mong doi (cat dong ghi do / them newline / "
           "giu nguyen); su kien hop le trung tung su kien; bien nhan = gia tri tra ve; khong con file tam", all(u10.values()),
           str(u10))
    r = tmp / "u10b"
    (r / "kp").mkdir(parents=True)
    goc_u10b = pre + b'{"loai": "epoch_da_luu", "ep'
    (r / ttl.TEN_NHAT_KY).write_bytes(goc_u10b)
    os_replace_that = ttl.os.replace

    def replace_loi(src, dst):                 # MOCK: chi lan thay NHAT KY CUA LUOT bi "ngat"; ban sao o kp/ ghi that
        if Path(dst).resolve() == (r / ttl.TEN_NHAT_KY).resolve():
            raise OSError("mo phong bi ngat truoc os.replace")
        return os_replace_that(src, dst)
    ttl.os.replace = replace_loi
    try:
        ttl.sua_duoi_nhat_ky(r, r / "kp", None)
        u10b = False
    except ttl.LoiTiepTuc:
        u10b = True
    finally:
        ttl.os.replace = os_replace_that
    _check("U10b (MOCK os.replace loi khi thay nhat ky) -> LoiTiepTuc; nhat ky cu con NGUYEN BYTE, ban sao da co (khong "
           "de lai nhat ky do dang)", u10b and (r / ttl.TEN_NHAT_KY).read_bytes() == goc_u10b
           and (r / "kp" / ttl.TEN_NHAT_KY).read_bytes() == goc_u10b)


# =========================================================================== R
def nhom_r(tmp: Path, data: Path, model: str) -> dict:
    cfg = cfg_test(tmp, "e3")
    kq = {}
    for ten in ("C1", "C2"):
        t0 = time.time()
        rc = chay_that(lenh(cfg, model, f"{model}_{ten}", data, tmp), tmp / f"{model}_{ten}.log")
        kq[ten] = {"exit": rc, "s": round(time.time() - t0, 1)}
    _check(f"R0 {model}: C1, C2 (lien tuc 3 epoch, chay that CPU) exit 0", kq["C1"]["exit"] == 0 and kq["C2"]["exit"] == 0,
           str(kq))
    run_i = tmp / "runs" / f"{model}_I"
    ng = chay_va_ngat(lenh(cfg, model, f"{model}_I", data, tmp), run_i, tmp / f"{model}_I.log")
    nk_i = doc_nk(run_i)
    _check(f"R1 {model}: I bi KILL ngay sau khi epoch 1 da luu; nhat ky: bat_dau + epoch_da_luu 1, KHONG ket_thuc",
           ng["bi_kill"] and [s["loai"] for s in nk_i] == ["bat_dau", "epoch_da_luu"] and nk_i[1]["epoch"] == 1,
           f"{ng}, su kien {[s['loai'] for s in nk_i]}")
    kq["I"] = ng
    shutil.copytree(run_i, tmp / f"nguyen_ban_{model}_I")         # ban goc cua I truoc moi thao tac (spy, resume)
    run_i2 = tmp / "runs" / f"{model}_I2"                          # K9: ngat SAU epoch tat mosaic
    ng2 = chay_va_ngat(lenh(cfg, model, f"{model}_I2", data, tmp), run_i2, tmp / f"{model}_I2.log", epoch_ngat=2)
    nk2 = doc_nk(run_i2)
    _check(f"R1b {model}: I2 bi KILL ngay sau khi epoch 2 da luu; nhat ky: bat_dau + epoch_da_luu 1, 2, KHONG ket_thuc",
           ng2["bi_kill"] and [(s["loai"], s.get("epoch")) for s in nk2] == [("bat_dau", None), ("epoch_da_luu", 1),
                                                                           ("epoch_da_luu", 2)], f"{ng2}")
    kq["I2"] = ng2
    return {"cfg": cfg, "kq": kq}


def so_sanh(tmp: Path, model: str, cfg: Path):
    import yaml
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))["huan_luyen"]
    E, nbs, batch = int(c["epochs"]), int(c["nbs"]), int(c["batch"])
    run = {t: tmp / "runs" / f"{model}_{t}" for t in ("C1", "C2", "I")}
    b = {t: doc_batch(run[t]) for t in run}
    loai_bo = ("t", "lan")
    # K1
    k1 = [i for i, (x, y) in enumerate(zip(b["C1"], b["C2"])) if any(x[k] != y[k] for k in x if k not in loai_bo)]
    ck = {t: {e: tensor_ck(run[t] / "khoi_phuc" / f"last_e{e:03d}.pt") for e in range(1, E + 1)} for t in ("C1", "C2")}
    k1_ck = [e for e in range(1, E + 1) if not ck_giong_het(ck["C1"][e], ck["C2"][e])[0]]
    _check(f"K1 {model}: doi chung C1 == C2 tung bit ({len(b['C1'])} dong theo batch moi cot tru thoi gian; checkpoint "
           f"e001..e{E:03d}: EMA, optimizer, updates, epoch, best_fitness)", len(b["C1"]) == len(b["C2"]) and not k1
           and not k1_ck, f"dong khac {k1[:5]}, checkpoint khac {k1_ck}")
    # K2
    nb = sum(1 for x in b["C1"] if x["epoch"] == "1")
    i_e1 = [x for x in b["I"] if x["lan"] == "0" and x["epoch"] == "1"]
    k2 = [x["ni"] for x, y in zip(i_e1, b["C1"][:nb]) if any(x[k] != y[k] for k in x if k not in loai_bo)]
    ck_i1 = tensor_ck(tmp / f"nguyen_ban_{model}_I" / "khoi_phuc" / "last_e001.pt")
    _check(f"K2 {model}: I truoc diem ngat == C1 (epoch 1: {nb} dong theo batch; checkpoint e001 tensor trung)",
           len(i_e1) == nb and not k2 and ck_giong_het(ck_i1, ck["C1"][1])[0],
           f"{len(i_e1)} dong, khac {k2[:5]}, ck {ck_giong_het(ck_i1, ck['C1'][1])[1]}")
    # K3
    rr = json.loads((tmp / "kq" / f"{model}_I__tiep_tuc_1.json").read_text(encoding="utf-8"))
    kk = rr.get("kiem_khoi_phuc") or {}
    nk = doc_nk(run["I"])
    nk_i1 = next(s for s in nk if s["loai"] == "epoch_da_luu" and s["epoch"] == 1)
    bd = next(s for s in nk if s["loai"] == "bat_dau" and s.get("che_do_luot") == "tiep_tuc")
    _check(f"K3a {model}: runner tu kiem SAU resume: {len(kk)} muc deu dat (start_epoch, tong epoch, ema.updates, "
           f"best_fitness, scheduler, momentum, param_groups, EMA, model dang hoc = EMA, args, batch, EarlyStopping)",
           len(kk) >= 13 and all(v["dat"] for v in kk.values()), str({k: v["dat"] for k, v in kk.items()}))
    _check(f"K3b {model}: checkpoint nguon = e001 cua I (sha256 trong nhat ky), ban bao toan nguon_tiep_tuc_1.pt cung "
           f"sha; bat dau tiep tu epoch index 1 / tong 3",
           bd["tiep_tuc"]["chon"]["sha256"] == nk_i1["last_sha256"] == bd["tiep_tuc"]["sao_luu"]["nguon_sha256"]
           == sha(run["I"] / "khoi_phuc" / "nguon_tiep_tuc_1.pt") and bd["start_epoch"] == 1 and bd["tong_epoch"] == E,
           f"start {bd['start_epoch']}, tong {bd['tong_epoch']}")
    # K4
    r_rows = [x for x in b["I"] if x["lan"] == "1"]
    c_sau = [x for x in b["C1"] if int(x["epoch"]) >= 2]
    cot4 = ("epoch", "i", "ni", "lr_weight", "lr_bn", "lr_bias", "momentum_weight", "momentum_bn", "momentum_bias",
            "batch_size")
    k4 = [(x["ni"], k) for x, y in zip(r_rows, c_sau) for k in cot4 if x[k] != y[k]]
    lr_hang = all(len({x["lr_weight"] for x in r_rows if x["epoch"] == str(e)}) == 1 for e in range(2, E + 1))
    mos = {t: {s["epoch"]: s.get("mosaic_thuc_te") for s in doc_nk(run[t]) if s["loai"] == "epoch_da_luu"}
           for t in ("C1", "I")}
    mos_p = {t: {s["epoch"]: s.get("mosaic_p") for s in doc_nk(run[t]) if s["loai"] == "epoch_da_luu"}
             for t in ("C1", "I")}
    mos_mong = {e: e - 1 < E - int(c["close_mosaic"]) for e in range(1, E + 1)}
    _check(f"K4 {model}: R (epoch 2..{E}) trung C1 TUNG GIA TRI o epoch / ni / LR 3 nhom / momentum 3 nhom / batch "
           f"({len(r_rows)} dong); LR khong doi trong epoch (khong chay lai warmup); mosaic thuc te theo epoch C1 = "
           f"I/R = du doan close_mosaic {mos_mong}",
           len(r_rows) == len(c_sau) and not k4 and lr_hang and mos["C1"] == mos["I"] == mos_mong,
           f"lech {k4[:6]}; mosaic C1 {mos['C1']}, I/R {mos['I']}; p C1 {mos_p['C1']}, I/R {mos_p['I']}")
    # K5
    nw = round(min(float(c["warmup_epochs"]), max(E - 1, 0)) * nb)
    mp_c, buoc_c_tat_ca = mo_phong(E, nb, nw, nbs, batch)
    mp_r, buoc_r = mo_phong(E, nb, nw, nbs, batch, tu_epoch=1, updates_dau=nk_i1["ema_updates"])
    k5c = [x["ni"] for x in b["C1"] if (int(x["accumulate"]), int(x["ema_updates"])) != mp_c[int(x["ni"])]]
    k5r = [x["ni"] for x in r_rows if (int(x["accumulate"]), int(x["ema_updates"])) != mp_r[int(x["ni"])]]
    buoc_c = [ni for ni in buoc_c_tat_ca if ni >= nb]
    _check(f"K5a {model}: accumulate + ema.updates tung batch cua C1 VA cua R khop mo phong ma ultralytics (nb {nb}, "
           f"nw {nw}); buoc toi uu sau epoch 1: C1 {buoc_c} vs R {buoc_r}", not k5c and not k5r,
           f"lech C1 {k5c[:5]}, R {k5r[:5]}")
    acc_cuoi_wu = max(1, int(np.interp(nw - 1, [0, nw], [1, nbs / batch]).round()))
    nw_dai = 6311
    acc_dai = max(1, int(np.interp(nw_dai - 1, [0, nw_dai], [1, nbs / batch]).round()))
    _check(f"K5b {model}: du doan khac biet: accumulate sau warmup C1 = {acc_cuoi_wu} (nw {nw} < 15), sau resume R = "
           f"{round(nbs / batch)}; ranh gioi buoc lech. Luot dai (nb = nw = 6.311): accumulate cuoi warmup = {acc_dai} = "
           f"{round(nbs / batch)} -> trung", acc_cuoi_wu != round(nbs / batch) and buoc_c != buoc_r and acc_dai == 8,
           f"C1 acc {acc_cuoi_wu}, R acc {round(nbs / batch)}")
    # K6
    loss_khac = sum(1 for x, y in zip(r_rows, c_sau) if x["loss_tong"] != y["loss_tong"])
    thu_tu_khac = sum(1 for x, y in zip(r_rows, c_sau) if x["thu_tu_batch"] != y["thu_tu_batch"])
    ck_r = tensor_ck(run["I"] / "khoi_phuc" / f"last_e{E:03d}.pt")
    lech = lech_ema_max(ck_r, ck["C1"][E])
    fit = {t: next(s["fitness"] for s in doc_nk(run[t]) if s["loai"] == "epoch_da_luu" and s["epoch"] == E)
           for t in ("C1", "I")}
    THONG_TIN[model] = {"loss_khac": loss_khac, "thu_tu_khac": thu_tu_khac, "so_dong": len(r_rows),
                        "lech_ema_max_cuoi": lech, "fitness_cuoi": fit, "buoc_c": buoc_c, "buoc_r": buoc_r}
    _check(f"K6 {model}: KHAC BIET DU KIEN (khong bit-identical) duoc ghi: loss khac {loss_khac}/{len(r_rows)} dong, thu "
           f"tu du lieu khac {thu_tu_khac}/{len(r_rows)}, |EMA cuoi R - C1| max {lech:.3g}, fitness cuoi R {fit['I']} vs "
           f"C1 {fit['C1']}", loss_khac > 0 and lech > 0, "")
    # K7
    loai = [(s["loai"], s.get("lan"), s.get("epoch")) for s in nk]
    mong = [("bat_dau", 0, None), ("epoch_da_luu", 0, 1), ("bat_dau", 1, None)] + \
           [("epoch_da_luu", 1, e) for e in range(2, E + 1)] + [("ket_thuc", 1, None)]
    kp = run["I"] / "khoi_phuc" / "truoc_tiep_tuc_1"
    goc = tmp / f"nguyen_ban_{model}_I"
    luu_cu = {f: sha(kp / f.replace("/", "__")) == sha(goc / f) for f in ("args.yaml", "results.csv", "theo_batch.csv",
                                                                          "weights/last.pt", "weights/best.pt")}
    prov = all(k in bd for k in ("lenh", "ma_nguon_sha256", "moi_truong", "git_head", "t_iso", "dataset_van_tay",
                                 "cau_hinh")) and bd["tiep_tuc"]["epoch_da_luu"] == 1
    n_res = len(list(csv.DictReader(open(run["I"] / "results.csv", encoding="utf-8"))))
    co_best = [s for t in ("C1", "I") for s in doc_nk(run[t]) if s["loai"] == "epoch_da_luu" and s["best_ghi_epoch_nay"]]
    best_cung_byte = bool(co_best) and all(s["best_sha256"] == s["ban_sao_sha256"] for s in co_best)
    _check(f"K7 {model}: nhat ky {loai == mong} (lich su lan 0 giu nguyen); provenance tiep tuc du {prov}; file cu luu "
           f"truoc khi ghi tiep {luu_cu}; results.csv {n_res} epoch; quy dao {bd['tiep_tuc'].get('quy_dao')}; moi epoch "
           f"ghi best: best.pt cung byte ban sao last ({len(co_best)} epoch) {best_cung_byte}",
           loai == mong and prov and all(luu_cu.values()) and n_res == E and best_cung_byte
           and bd["tiep_tuc"].get("quy_dao") == [[0, 1]], f"{loai}")
    # K8
    fin = json.loads((tmp / "kq" / f"{model}_I__tiep_tuc_1.json").read_text(encoding="utf-8"))
    import torch
    last = torch.load(str(run["I"] / "weights" / "last.pt"), map_location="cpu", weights_only=False)
    ket = nk[-1]
    _check(f"K8 {model}: R ket thuc hoan_tat exit 0; last.pt cuoi da strip (epoch -1, optimizer None), train_args epochs "
           f"{E}; checkpoint nap lai dung 5 lop", ket.get("trang_thai") == "hoan_tat" and ket.get("exit_code") == 0
           and last["epoch"] == -1 and last["optimizer"] is None and int(last["train_args"]["epochs"]) == E
           and all(v["dung_5_lop"] for v in fin["checkpoint"].values()), f"{ket.get('trang_thai')}")


def so_sanh_k9(tmp: Path, model: str, cfg: Path, rc: int, log_r: Path):
    """K9: tiep tuc SAU epoch tat mosaic (start_epoch 2 > epochs - close_mosaic = 1) -> nhanh resume_training cua
    ultralytics tu tat mosaic (_close_dataloader_mosaic) truoc vong train; vong lap khong tat lai."""
    import yaml
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))["huan_luyen"]
    E, nbs, batch = int(c["epochs"]), int(c["nbs"]), int(c["batch"])
    run_c1, run_i2 = tmp / "runs" / f"{model}_C1", tmp / "runs" / f"{model}_I2"
    b_c1, b_i2 = doc_batch(run_c1), doc_batch(run_i2)
    loai_bo = ("t", "lan")
    truoc = [x for x in b_i2 if x["lan"] == "0"]
    k9a = [x["ni"] for x, y in zip(truoc, b_c1) if any(x[k] != y[k] for k in x if k not in loai_bo)]
    nb = sum(1 for x in b_c1 if x["epoch"] == "1")
    nw = round(min(float(c["warmup_epochs"]), max(E - 1, 0)) * nb)
    f_kq = tmp / "kq" / f"{model}_I2__tiep_tuc_1.json"
    kk = (json.loads(f_kq.read_text(encoding="utf-8")).get("kiem_khoi_phuc") or {}) if f_kq.is_file() else {}
    nk = doc_nk(run_i2)
    bd = next((s for s in nk if s["loai"] == "bat_dau" and s.get("che_do_luot") == "tiep_tuc"), {})
    e2 = next(s for s in nk if s["loai"] == "epoch_da_luu" and s["epoch"] == 2 and s["lan"] == 0)
    r_rows = [x for x in b_i2 if x["lan"] == "1"]
    c_e3 = [x for x in b_c1 if x["epoch"] == str(E)]
    cot4 = ("epoch", "i", "ni", "lr_weight", "lr_bn", "lr_bias", "momentum_weight", "momentum_bn", "momentum_bias",
            "batch_size")
    k9b = [(x["ni"], k) for x, y in zip(r_rows, c_e3) for k in cot4 if x[k] != y[k]]
    mp_r, buoc_r = mo_phong(E, nb, nw, nbs, batch, tu_epoch=2, updates_dau=e2["ema_updates"])
    k9c = [x["ni"] for x in r_rows if (int(x["accumulate"]), int(x["ema_updates"])) != mp_r[int(x["ni"])]]
    mos = {s["epoch"]: s.get("mosaic_p") for s in nk if s["loai"] == "epoch_da_luu"}
    log = log_r.read_text(encoding="utf-8", errors="replace") if log_r.is_file() else ""
    _check(f"K9 {model}: tiep tuc SAU epoch tat mosaic (I2 ngat sau epoch 2): exit 0; I2 epoch 1-2 trung C1 tung bit "
           f"({len(truoc)} dong); runner tu kiem {len(kk)} muc dat; start_epoch 2; epoch 3 ({len(r_rows)} dong) trung C1 o "
           f"ni/LR/momentum/batch; accumulate/updates khop mo phong (buoc {buoc_r}); mosaic p {{1: 1,0, 2: 0,0, 3: 0,0}}; "
           f"log tiep tuc co 'Closing dataloader mosaic' (nhanh resume_training)",
           rc == 0 and len(truoc) == 2 * nb and not k9a and len(kk) >= 13 and all(v["dat"] for v in kk.values())
           and bd.get("start_epoch") == 2 and len(r_rows) == len(c_e3) == nb and not k9b and not k9c
           and mos == {1: 1.0, 2: 0.0, 3: 0.0} and "Closing dataloader mosaic" in log,
           f"exit {rc}; lech truoc {k9a[:5]}; lech e3 {k9b[:5]}; mo phong {k9c[:5]}; mosaic {mos}; start "
           f"{bd.get('start_epoch')}")


# =========================================================================== N (vong 40, R39-F1) -- CHAY THAT
DUOI_N = [("sach", b"", "khong"),
          ("ghi_do_khong_newline", b'{"loai": "epoch_da_luu", "lan": 0, "epoch": 2, "last_sha', "cat_duoi"),
          ("ghi_do_co_newline", b'{"loai": "epoch_da_luu", "lan": 0, "epoch": 2, "last_sha\n', "cat_duoi"),
          ("hop_le_thieu_newline", b'{"loai": "ket_thuc", "lan": 0, "trang_thai": "bi_ngat", "t": 1.0}', "them_newline")]


def lenh_n(ma: Path, cfg: Path, ten: str, data: Path, tmp: Path, kq: str, *them, dry: bool = False) -> list[str]:
    che = ["--dry-run"] if dry else ["--che-do", "day-du", "--xac-nhan-train-dai"]
    return [PY, "-u", str(Path(ma) / "train_detector.py"), "--config", str(cfg), "--model", "yolov8n", *che, "--data",
            str(data), "--ten", ten, "--project", str(tmp / "runs"), "--ket-qua-dir", str(tmp / kq), "--device", "cpu",
            *them]


def chay_lay(argv: list[str], log: Path, timeout=1800) -> tuple[int, str]:
    r = subprocess.run(argv, capture_output=True, timeout=timeout)
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    log.write_text(out, encoding="utf-8")
    return r.returncode, out


def doc_khoan_dung(run_dir: Path) -> list[dict]:
    """CHI de theo doi tien do khi cho ngat (khong phai phep kiem): parse tung dong, bo dong khong parse duoc."""
    p = run_dir / "nhat_ky_luot.jsonl"
    out = []
    for x in (p.read_bytes().decode("utf-8", "replace").split("\n") if p.is_file() else []):
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def chay_va_ngat_khi(argv: list[str], run_dir: Path, log: Path, dk, timeout=1200) -> dict:
    """Nhu chay_va_ngat, dieu kien ngat dk(su kien doc khoan dung) -- van theo doi duoc khi ma cu lam hong nhat ky."""
    import psutil
    with open(log, "wb") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        try:
            while time.time() - t0 < timeout:
                if p.poll() is not None:
                    return {"bi_kill": False, "exit": p.returncode, "ly_do": "tien trinh tu thoat truoc khi ngat"}
                if dk(doc_khoan_dung(run_dir)):
                    pr = psutil.Process(p.pid)
                    for c in pr.children(recursive=True):
                        c.kill()
                    pr.kill()
                    return {"bi_kill": True, "exit": p.wait(timeout=60), "sau_s": round(time.time() - t0, 1)}
                time.sleep(0.1)
            return {"bi_kill": False, "exit": None, "ly_do": "het thoi gian cho"}
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()


def doc_hoac_loi(run_dir: Path) -> tuple[list[dict], list[str], str | None]:
    import tiep_tuc_huan_luyen as ttl
    try:
        sk, cb = ttl.doc_nhat_ky(run_dir)
        return sk, cb, None
    except ttl.LoiTiepTuc as e:
        return [], [], str(e)


def nhom_n(tmp: Path, data: Path, ma: Path):
    """R39-F1 qua DUONG RESUME THAT (train_detector.py cua `ma`, subprocess, CPU). Doc/kiem dung ham doc hien tai."""
    import tiep_tuc_huan_luyen as ttl
    cfg = cfg_test(tmp, "n_e3")
    ten = "yolov8n_NK"
    run, goc = tmp / "runs" / ten, tmp / "nguyen_ban_NK"
    p = run / ttl.TEN_NHAT_KY
    ng = chay_va_ngat_khi(lenh_n(ma, cfg, ten, data, tmp, "kq_n0"), run, tmp / "n0.log",
                          lambda sk: any(s.get("loai") == "epoch_da_luu" and s.get("epoch") == 1 for s in sk))
    _check(f"N0 luot goc {ten} bi KILL ngay sau epoch 1 (chay that, ma {ma}); nhat ky: bat_dau + epoch_da_luu 1",
           ng["bi_kill"] and [s["loai"] for s in doc_nk(run)] == ["bat_dau", "epoch_da_luu"], str(ng))
    shutil.copytree(run, goc)
    b_goc = (goc / ttl.TEN_NHAT_KY).read_bytes()
    file_cu = ("args.yaml", "results.csv", "theo_batch.csv", "weights/last.pt", "weights/best.pt")

    def khoi_phuc():
        shutil.rmtree(run)
        shutil.copytree(goc, run)

    def la_epoch(lan, ep):
        return lambda sk: any(s.get("loai") == "epoch_da_luu" and s.get("lan") == lan and s.get("epoch") == ep for s in sk)
    for ca, hau, hd in DUOI_N:
        khoi_phuc()
        p.write_bytes(b_goc + hau)
        b_truoc = p.read_bytes()
        hop_le_truoc, _ = ttl.doc_nhat_ky(run)          # dong cuoi ghi do (neu co) duoc bo qua + canh bao
        anh0 = anh_chup(run)
        rc0, out0 = chay_lay(lenh_n(ma, cfg, ten, data, tmp, f"kq_n_{ca}_0", "--tiep-tuc", dry=True), tmp / f"n_{ca}_0.log")
        _check(f"N-{ca}-0a dry-run --tiep-tuc: exit 0, KHONG doi file (anh chup sha256 thu muc luot trung)",
               rc0 == 0 and anh_chup(run) == anh0, f"exit {rc0}; {out0.strip()[-200:]}")
        _check(f"N-{ca}-0b dry-run bao ke hoach xu ly nhat ky '{hd}'",
               f'"hanh_dong": "{hd}"' in out0 and f"nhat ky: {hd}" in out0, out0.strip()[-160:])
        ng1 = chay_va_ngat_khi(lenh_n(ma, cfg, ten, data, tmp, f"kq_n_{ca}_1", "--tiep-tuc"), run, tmp / f"n_{ca}_1.log",
                               la_epoch(1, 2))
        sk1, cb1, loi1 = doc_hoac_loi(run)
        moi1 = [(s.get("loai"), s.get("lan"), s.get("epoch")) for s in sk1[len(hop_le_truoc):]]
        _check(f"N-{ca}-A resume THAT bi kill sau epoch 2: nhat ky DOC LAI duoc, khong canh bao; {len(hop_le_truoc)} su "
               f"kien hop le cu giu nguyen tung su kien + [bat_dau lan 1, epoch_da_luu 2]",
               ng1["bi_kill"] and loi1 is None and not cb1 and sk1[:len(hop_le_truoc)] == hop_le_truoc
               and moi1 == [("bat_dau", 1, None), ("epoch_da_luu", 1, 2)],
               f"kill {ng1.get('bi_kill')}; loi {loi1}; canh bao {cb1}; moi {moi1}")
        kp = run / "khoi_phuc" / "truoc_tiep_tuc_1"
        f_bn = kp / "xu_ly_nhat_ky.json"
        bn = json.loads(f_bn.read_text(encoding="utf-8")) if f_bn.is_file() else {}
        bd1 = next((s for s in sk1 if s.get("loai") == "bat_dau" and s.get("lan") == 1), {})
        ghi = ((bd1.get("tiep_tuc") or {}).get("sao_luu") or {}).get("nhat_ky")
        bo = len(hau) if hd == "cat_duoi" else 0
        _check(f"N-{ca}-B ban sao nhat ky TRUOC sua trung tung byte (sha256 {hashlib.sha256(b_truoc).hexdigest()[:12]}); "
               f"bien nhan xu_ly_nhat_ky.json = ban ghi trong bat_dau lan 1: '{hd}', bo {bo} byte",
               (kp / ttl.TEN_NHAT_KY).is_file() and (kp / ttl.TEN_NHAT_KY).read_bytes() == b_truoc
               and bn.get("hanh_dong") == hd and bn.get("sha256") == hashlib.sha256(b_truoc).hexdigest()
               and bn.get("bo_bytes") == bo and ghi == bn,
               f"ban sao {(kp / ttl.TEN_NHAT_KY).is_file()}; bien nhan {bn.get('hanh_dong')} bo {bn.get('bo_bytes')}; "
               f"trong bat_dau {ghi is not None}")
        e1 = goc / "khoi_phuc" / "last_e001.pt"
        cu = {f: (kp / f.replace("/", "__")).is_file() and sha(kp / f.replace("/", "__")) == sha(goc / f) for f in file_cu}
        _check(f"N-{ca}-C checkpoint / bang chung cu bao toan: last_e001.pt va nguon_tiep_tuc_1.pt = e001 goc; file cu "
               f"luu truoc khi ghi tiep trung goc",
               sha(run / "khoi_phuc" / "last_e001.pt") == sha(e1) == sha(run / "khoi_phuc" / "nguon_tiep_tuc_1.pt")
               and all(cu.values()), str(cu))
        rc2, out2 = chay_lay(lenh_n(ma, cfg, ten, data, tmp, f"kq_n_{ca}_2", "--tiep-tuc", dry=True), tmp / f"n_{ca}_2.log")
        b_truoc2 = p.read_bytes()
        rc3, out3 = chay_lay(lenh_n(ma, cfg, ten, data, tmp, f"kq_n_{ca}_3", "--tiep-tuc"), tmp / f"n_{ca}_3.log")
        sk3, cb3, loi3 = doc_hoac_loi(run)
        moi3 = [(s.get("loai"), s.get("lan"), s.get("epoch")) for s in sk3[len(hop_le_truoc):]]
        _check(f"N-{ca}-D preflight tiep theo exit 0 'tiep tuc tu epoch 3/3'; resume tiep (lan 2) exit 0 toi het; nhat ky "
               f"doc lai du: cu + lan 1 + lan 2 + ket_thuc hoan_tat",
               rc2 == 0 and "tiep tuc tu epoch 3/3" in out2 and rc3 == 0 and loi3 is None and not cb3
               and sk3[:len(hop_le_truoc)] == hop_le_truoc and bool(sk3) and sk3[-1].get("trang_thai") == "hoan_tat"
               and moi3 == [("bat_dau", 1, None), ("epoch_da_luu", 1, 2), ("bat_dau", 2, None), ("epoch_da_luu", 2, 3),
                            ("ket_thuc", 2, None)],
               f"preflight {rc2}; resume {rc3}; loi {loi3}; moi {moi3}; {(out2 if rc2 else out3).strip()[-160:]}")
        bs2 = run / "khoi_phuc" / "truoc_tiep_tuc_2" / ttl.TEN_NHAT_KY
        bn2 = run / "khoi_phuc" / "truoc_tiep_tuc_2" / "xu_ly_nhat_ky.json"
        _check(f"N-{ca}-B2 lan tiep tuc 2: ban sao nhat ky truoc lan 2 trung tung byte; bien nhan 'khong' (nhat ky sach sau "
               f"lan 1)", bs2.is_file() and bs2.read_bytes() == b_truoc2 and bn2.is_file()
               and json.loads(bn2.read_text(encoding="utf-8")).get("hanh_dong") == "khong",
               f"ban sao {bs2.is_file()}; bien nhan {bn2.is_file()}")
    khoi_phuc()
    dg = b_goc.split(b"\n")
    p.write_bytes(dg[0] + b"\n" + b'{"loai": "hong\n' + b"\n".join(dg[1:]))
    anh0 = anh_chup(run)
    rc0, out0 = chay_lay(lenh_n(ma, cfg, ten, data, tmp, "kq_n_giua_0", "--tiep-tuc", dry=True), tmp / "n_giua_0.log")
    rc1, out1 = chay_lay(lenh_n(ma, cfg, ten, data, tmp, "kq_n_giua_1", "--tiep-tuc"), tmp / "n_giua_1.log")
    _check("N-giua dong hong o GIUA nhat ky: dry-run va resume THAT deu exit 2 'hong o GIUA'; thu muc luot KHONG doi (khong "
           "sao luu, khong train)", rc0 == 2 and rc1 == 2 and "hong o GIUA" in out0 and "hong o GIUA" in out1
           and anh_chup(run) == anh0, f"exit {rc0}/{rc1}; {out1.strip()[-160:]}")


# =========================================================================== S
def nhom_s(tmp: Path, data: Path, cfg: Path):
    m = "yolov8n"
    run_i = tmp / "runs" / f"{m}_I"
    goc = tmp / f"nguyen_ban_{m}_I"

    def khoi_phuc_i():
        if run_i.exists():
            shutil.rmtree(run_i)
        shutil.copytree(goc, run_i)

    def argv(ten=f"{m}_I", *them, model=m, c=cfg, d=data):
        return lenh(c, model, ten, d, tmp, *them)[3:]             # bo [python, -u, train_detector.py]

    def tu_choi(ma, ten, argv_, mau, ma_mong=2):
        khoi_phuc_i()
        if ten:
            ten(run_i)
        truoc = anh_chup(run_i) if run_i.exists() else {}
        code, so, se, goi = chay_spy(argv_)
        sau = anh_chup(run_i) if run_i.exists() else {}
        return _check(f"{ma} -> exit {ma_mong}, KHONG goi train, thu muc luot khong doi; ly do chua '{mau}'",
                      code == ma_mong and not co_train(goi) and truoc == sau and mau in (se + so),
                      f"exit {code}; train {co_train(goi)}; {(se.strip() or so.strip())[-260:]}")
    run_moi = tmp / "runs" / "khong_ton_tai"
    _check("S1 --tiep-tuc luot KHONG ton tai -> exit 2, khong goi train",
           (lambda r: r[0] == 2 and not co_train(r[3]) and "khong ton tai" in r[2])(
               chay_spy(argv("khong_ton_tai", "--tiep-tuc"))) and not run_moi.exists())
    smoke_gia = tmp / "runs" / "khong_nhat_ky"
    (smoke_gia / "weights").mkdir(parents=True)
    shutil.copyfile(goc / "weights" / "last.pt", smoke_gia / "weights" / "last.pt")
    code, so, se, goi = chay_spy(argv("khong_nhat_ky", "--tiep-tuc"))
    _check("S2 thu muc luot KHONG co nhat ky (nhu luot smoke/pilot cu) -> exit 2, khong goi train, khong tao metadata",
           code == 2 and not co_train(goi) and "khong co nhat ky" in se
           and sorted(p.name for p in smoke_gia.rglob("*")) == ["last.pt", "weights"], se.strip()[-200:])
    run_c1 = tmp / "runs" / f"{m}_C1"
    truoc = anh_chup(run_c1)
    code, so, se, goi = chay_spy(argv(f"{m}_C1", "--tiep-tuc"))
    _check("S3 luot DA HOAN TAT (C1) -> exit 2 'HOAN TAT', khong goi train, khong doi file",
           code == 2 and not co_train(goi) and "HOAN TAT" in se and anh_chup(run_c1) == truoc, se.strip()[-200:])
    nk_c1 = run_c1 / "nhat_ky_luot.jsonl"
    b_nk = nk_c1.read_bytes()
    nk_c1.write_bytes(b"\n".join(x for x in b_nk.split(b"\n") if b'"ket_thuc"' not in x) + b"\n")
    try:
        code, so, se, goi = chay_spy(argv(f"{m}_C1", "--tiep-tuc"))
    finally:
        nk_c1.write_bytes(b_nk)
    _check("S3b luot da chay du epoch nhung bi ngat truoc ket_thuc (xoa dong ket_thuc) -> exit 2 (epoch cuoi da quyet dinh "
           "DUNG / du epoch), khong goi train", code == 2 and not co_train(goi) and ("DUNG" in se or "du 3/3" in se),
           se.strip()[-200:])

    def cat_last(r):
        p = r / "weights" / "last.pt"
        b = p.read_bytes()
        p.write_bytes(b[: len(b) // 2])
    khoi_phuc_i()
    cat_last(run_i)
    code, so, se, goi = chay_spy(argv(f"{m}_I", "--tiep-tuc"))
    tr = [g for g in goi if g[0] == "train"]
    rec = json.loads((tmp / "kq" / f"{m}_I__tiep_tuc_1.json").read_text(encoding="utf-8")) if (
        tmp / "kq" / f"{m}_I__tiep_tuc_1.json").is_file() else {}
    uv = {Path(u["duong_dan"]).name: u for u in rec.get("tiep_tuc", {}).get("ung_vien", [])}
    _check("S4 last.pt HONG (cat doi) nhung con ban sao e001 -> chon ban sao (hop le gan nhat); spy nhan resume = "
           "khoi_phuc/nguon_tiep_tuc_1.pt cung sha e001; last.pt bi loai co ly do",
           len(tr) == 1 and Path(tr[0][1]["resume"]) == run_i / "khoi_phuc" / "nguon_tiep_tuc_1.pt"
           and sha(run_i / "khoi_phuc" / "nguon_tiep_tuc_1.pt") == sha(goc / "khoi_phuc" / "last_e001.pt")
           and not uv.get("last.pt", {}).get("hop_le", True),
           f"train {tr[:1]}; last.pt {uv.get('last.pt', {}).get('ly_do')}")
    (tmp / "kq" / f"{m}_I__tiep_tuc_1.json").unlink(missing_ok=True)

    def hong_het(r):
        cat_last(r)
        (r / "khoi_phuc" / "last_e001.pt").unlink()
    tu_choi("S5 last.pt hong VA khong con ban sao", hong_het, argv(f"{m}_I", "--tiep-tuc"), "khong co checkpoint hop le")

    def la(r):
        shutil.copyfile(run_c1 / "khoi_phuc" / "last_e002.pt", r / "weights" / "last.pt")
        (r / "khoi_phuc" / "last_e001.pt").unlink()
    tu_choi("S6 last.pt la checkpoint cua LUOT KHAC (C1 e002), khong con ban sao", la, argv(f"{m}_I", "--tiep-tuc"),
            "khong co trong nhat ky")

    khoi_phuc_i()                           # S19: bi ngat SAU khi ultralytics ghi best.pt + dong results.csv epoch 2
    shutil.copyfile(run_c1 / "khoi_phuc" / "last_e002.pt", run_i / "weights" / "best.pt")   # nhung TRUOC nhat ky
    dong = (run_i / "results.csv").read_bytes().splitlines(keepends=True)
    (run_i / "results.csv").write_bytes(b"".join(dong) + b"2" + dong[-1][dong[-1].index(b","):])
    sha_mo_coi, csv_mo_coi = sha(run_i / "weights" / "best.pt"), (run_i / "results.csv").read_bytes()
    code, so, se, goi = chay_spy(argv(f"{m}_I", "--tiep-tuc"))
    e1 = next(s for s in doc_nk(goc) if s["loai"] == "epoch_da_luu" and s["epoch"] == 1)
    kp = run_i / "khoi_phuc" / "truoc_tiep_tuc_1"
    ep_csv = [r["epoch"].strip() for r in csv.DictReader(open(run_i / "results.csv", newline="", encoding="utf-8"))]
    _check("S19 best.pt + dong results.csv cua NHANH BI BO (ghi sau epoch 1, truoc nhat ky) -> TRUOC khi train: best.pt = "
           "best quy dao (e001, chep tu ban sao cung byte), ban cu luu o truoc_tiep_tuc_1; results.csv chi con epoch 1 "
           "(ban cu luu nguyen byte); train van duoc goi",
           co_train(goi) and sha(run_i / "weights" / "best.pt") == e1["best_sha256"] == sha(goc / "weights" / "best.pt")
           and sha(kp / "weights__best.pt") == sha_mo_coi and ep_csv == ["1"]
           and (kp / "results.csv").read_bytes() == csv_mo_coi,
           f"exit {code}; best {sha(run_i / 'weights' / 'best.pt')[:12]} vs e001 {e1['best_sha256'][:12]}; csv {ep_csv}")
    (tmp / "kq" / f"{m}_I__tiep_tuc_1.json").unlink(missing_ok=True)
    nhan = sorted((data.parent / "labels" / "train").glob("*.txt"))[0]
    b_nhan = nhan.read_bytes()
    nhan.write_bytes(b_nhan + b"0 0.5 0.5 0.1 0.1\n")
    try:
        tu_choi("S7a dataset bi sua (them 1 box vao file nhan) -> hop dong", None, argv(f"{m}_I", "--tiep-tuc"),
                "KHONG dat hop dong")
    finally:
        nhan.write_bytes(b_nhan)
    import test_pilot_chan_doan as tp
    data2 = tp.tao_dataset_fixture(tmp / "fx2")
    tu_choi("S7b dataset KHAC (ban xuat khac, hop dong dat) -> van tay", None,
            argv(f"{m}_I", "--tiep-tuc", d=data2), "van tay dataset khac")
    tu_choi("S8 cau hinh doi lr0 0,02", None, argv(f"{m}_I", "--tiep-tuc", c=cfg_test(tmp, "lr", lr0=0.02)), "lr0")
    tu_choi("S9 cau hinh doi epochs 3 -> 4 (resume khong cong them epoch)", None,
            argv(f"{m}_I", "--tiep-tuc", c=cfg_test(tmp, "e4", epochs=4)), "epochs")
    tu_choi("S10 --tiep-tuc thieu --xac-nhan-train-dai", None,
            [x for x in argv(f"{m}_I", "--tiep-tuc") if x != "--xac-nhan-train-dai"], "xac-nhan-train-dai", ma_mong=1)
    tu_choi("S11 --tiep-tuc voi --che-do smoke", None,
            [("smoke" if x == "day-du" else x) for x in argv(f"{m}_I", "--tiep-tuc")
             if x != "--xac-nhan-train-dai"], "chi dung voi --che-do day-du", ma_mong=1)
    tu_choi("S12 luot MOI trung ten thu muc da co (khong --tiep-tuc)", None, argv(f"{m}_I"), "da ton tai")
    tu_choi("S13 sai model (--model yolo11n cho luot yolov8n)", None, argv(f"{m}_I", "--tiep-tuc", model="yolo11n"),
            "checkpoint pretrained")

    def dung(r):
        p = r / "nhat_ky_luot.jsonl"
        dong = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        for s in dong:
            if s["loai"] == "epoch_da_luu":
                s["dung"] = True
        p.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in dong), encoding="utf-8")
    tu_choi("S14 nhat ky ghi epoch da luu kem quyet dinh DUNG", dung, argv(f"{m}_I", "--tiep-tuc"), "quyet dinh DUNG")

    def ul(r):
        p = r / "nhat_ky_luot.jsonl"
        dong = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        dong[0]["moi_truong"]["ultralytics"] = "8.0.0"
        p.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in dong), encoding="utf-8")
    tu_choi("S15 luot goc chay voi phien ban ultralytics khac", ul, argv(f"{m}_I", "--tiep-tuc"), "moi truong khac")
    tu_choi("S16 --tiep-tuc --batch 4 (khong doi batch khi tiep tuc)", None, argv(f"{m}_I", "--tiep-tuc", "--batch", "4"),
            "--batch", ma_mong=1)

    def hong_giua(r):                       # vong 40: dong hong o GIUA (khong phai duoi ghi do) -> van tu choi
        p = r / "nhat_ky_luot.jsonl"
        d = p.read_bytes().split(b"\n")
        p.write_bytes(d[0] + b"\n" + b'{"loai": "hong\n' + b"\n".join(d[1:]))
    tu_choi("S20 dong hong o GIUA nhat ky (vong 40)", hong_giua, argv(f"{m}_I", "--tiep-tuc"), "hong o GIUA")
    khoi_phuc_i()
    truoc = anh_chup(run_i)
    code, so, se, goi = chay_spy(["--dry-run", "--tiep-tuc", *[x for x in argv(f"{m}_I") if x not in
                                                               ("--che-do", "day-du", "--xac-nhan-train-dai")]])
    _check("S17 --dry-run --tiep-tuc: exit 0, in ke hoach (tiep tu epoch 2/3, checkpoint e001), KHONG goi train, KHONG "
           "ghi gi (anh chup thu muc luot trung)", code == 0 and not co_train(goi) and anh_chup(run_i) == truoc
           and "tiep tuc tu epoch 2/3" in so, so.strip()[-200:])
    code, so, se, goi = chay_spy(argv(f"{m}_ctrl_c"), hanh_dong="ctrl_c")
    f = tmp / "kq" / f"{m}_ctrl_c.json"
    rec = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    _check("S18 Ctrl+C trong luc train (spy nem KeyboardInterrupt) -> exit 130, ket qua do dang ghi loi 'bi ngat'",
           code == 130 and "KeyboardInterrupt" in str(rec.get("loi")), f"exit {code}, loi {rec.get('loi')}")
    khoi_phuc_i()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", nargs="+", default=["yolov8n", "yolo11n"])
    ap.add_argument("--bo-spy", action="store_true")
    ap.add_argument("--chi-nhom-n", action="store_true",
                    help="CHI chay nhom N (vd voi --ma-nguon <src ma cu> de chung minh test moi bat loi R39-F1)")
    ap.add_argument("--ma-nguon", default=str(SRC),
                    help="thu muc chua train_detector.py cho tien trinh con cua nhom N (mac dinh: src hien tai)")
    a = ap.parse_args()
    ma = Path(a.ma_nguon).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="test_tiep_tuc_"))
    try:
        import test_pilot_chan_doan as tp
        if a.chi_nhom_n:
            data = tp.tao_dataset_fixture(tmp / "fx")
            _chay(f"N. nhat ky R39-F1 qua resume THAT (train_detector.py o {ma})", lambda: nhom_n(tmp, data, ma))
            return tong_ket()
        _chay("U. don vi", lambda: nhom_u(tmp))
        data = tp.tao_dataset_fixture(tmp / "fx")
        ctx = {}
        for m in a.model:
            def r(m=m):
                ctx[m] = nhom_r(tmp, data, m)
            _chay(f"R. {m}: C1, C2 lien tuc + I bi kill (CHAY THAT, CPU)", r)
            if m == "yolov8n" and not a.bo_spy and m in ctx:
                _chay("S. yolov8n: ca tu choi dau vao + chon ban sao + dry-run + Ctrl+C (SPY)",
                      lambda: nhom_s(tmp, data, ctx["yolov8n"]["cfg"]))
            if m in ctx:
                def tiep(m=m):
                    t0 = time.time()
                    rc = chay_that(lenh(ctx[m]["cfg"], m, f"{m}_I", data, tmp, "--tiep-tuc"), tmp / f"{m}_R.log")
                    _check(f"R2 {m}: --tiep-tuc I (CHAY THAT) exit 0", rc == 0,
                           f"exit {rc}, {round(time.time() - t0, 1)} s; log {(tmp / f'{m}_R.log').read_text(encoding='utf-8', errors='replace')[-400:] if rc else ''}")
                    so_sanh(tmp, m, ctx[m]["cfg"])
                    rc2 = chay_that(lenh(ctx[m]["cfg"], m, f"{m}_I2", data, tmp, "--tiep-tuc"), tmp / f"{m}_R2.log")
                    so_sanh_k9(tmp, m, ctx[m]["cfg"], rc2, tmp / f"{m}_R2.log")
                _chay(f"K. {m}: tiep tuc + so sanh voi luot lien tuc", tiep)
        _chay(f"N. nhat ky R39-F1 qua resume THAT (train_detector.py o {ma})", lambda: nhom_n(tmp, data, ma))
        print("\nTHONG TIN (khac biet du kien, khong dung nguong):", json.dumps(THONG_TIN, ensure_ascii=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return tong_ket()


def tong_ket() -> int:
    n = sum(ok for _, ok in KQ)
    print(f"\nKET QUA: {n}/{len(KQ)} PASS" + (f", {len(SKIP)} SKIP (CHUA KIEM): {[s for s, _ in SKIP]}" if SKIP else ""))
    return 0 if KQ and n == len(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
