# -*- coding: utf-8 -*-
"""Vong 42: kiem DIEU KIEN THANH CONG cua MOT model (02_ke_hoach_luot_dai.md muc 7) -- CHI DOC.
    python kiem_thanh_cong.py <yolov8n|yolo11n> <thu muc giam sat cua lan cuoi>
Thoat 0 khi DAT moi dieu kien, 1 khi co dieu kien khong dat."""
import json
import re
import sys
from pathlib import Path

GOC = Path(r"D:\LABOTARY\bytetrack")
m, gs = sys.argv[1], GOC / sys.argv[2]
ten = f"train_bdd5_v1_{m}_day_du"
R = GOC / "models/finetune_phase2" / ten
tom = json.loads((gs / "tom_tat.json").read_text(encoding="utf-8"))
log = (gs / "log.txt").read_text(encoding="utf-8", errors="replace")
nk = [json.loads(x) for x in (R / "nhat_ky_luot.jsonl").read_bytes().split(b"\n") if x.strip()]
bd = [e for e in nk if e["loai"] == "bat_dau"][-1]
lan = bd["lan"]
kq_file = GOC / "results/phase2/detector_train" / (f"{ten}.json" if lan == 0 else f"{ten}__tiep_tuc_{lan}.json")
kq = json.loads(kq_file.read_text(encoding="utf-8"))
kt = [e for e in nk if e["loai"] == "ket_thuc" and e.get("lan") == lan]
ep = [e for e in nk if e["loai"] == "epoch_da_luu"]
so_epoch = sorted({e["epoch"] for e in ep})
dk = {
    "bo giam sat exit 0": tom.get("exit_code") == 0,
    f"runner in [DAT] day-du {m}": re.search(rf"\[DAT\] day-du {m}:", log) is not None,
    "ket qua: loi = None": kq.get("loi") is None,
    "ket qua: khong loi callback": not kq.get("loi_callback"),
    "ket qua: loss huu han (lan cuoi)": kq.get("loss_huu_han") is True,
    "last.pt + best.pt co trong ket qua": {"last", "best"} <= set(kq.get("checkpoint") or {}),
    "checkpoint dung 5 lop": all(c.get("dung_5_lop") for c in (kq.get("checkpoint") or {}).values()),
    "checkpoint: train_args trung cau hinh": all(not c.get("train_args_lech_cau_hinh") for c in (kq.get("checkpoint") or {}).values()),
    "nhat ky ket_thuc hoan_tat (lan cuoi)": bool(kt) and kt[-1].get("trang_thai") == "hoan_tat",
    "nhat ky kiem_sau_train_dat = true": bool(kt) and kt[-1].get("kiem_sau_train_dat") is True,
    "nhat ky exit_code 0": bool(kt) and kt[-1].get("exit_code") == 0,
    "du 15 epoch hoac dung som hop le": so_epoch == list(range(1, 16)) or (bool(ep) and ep[-1].get("dung") is True
                                                                        and so_epoch == list(range(1, so_epoch[-1] + 1))),
}
for k, v in dk.items():
    print(f"  [{'DAT' if v else 'KHONG DAT'}] {k}")
print(f"lan cuoi {lan}; ket qua {kq_file.name}; epoch da luu {so_epoch}; bo giam sat exit {tom.get('exit_code')}")
print("KET LUAN:", "DAT -- duoc bat dau model tiep theo" if all(dk.values()) else "KHONG DAT -- dung, bao nguoi dung")
sys.exit(0 if all(dk.values()) else 1)
