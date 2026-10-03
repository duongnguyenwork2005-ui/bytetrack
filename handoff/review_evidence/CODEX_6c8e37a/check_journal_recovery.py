"""Independent round-40 checks: real backup/repair/append/read operations in temp dirs.

Round-41 rerun: only the old three-column batch fixture was replaced with the actual 20-column schema.
No torch or training. Checkpoint files contain marker bytes and are ONLY copied;
this does not test checkpoint validity, optimizer state, or the runner CLI.
One os.replace failure is deliberately injected, then the next attempt is run.
"""
import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser(__doc__)
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
src = a.repo / "src/tiep_tuc_huan_luyen.py"
s = importlib.util.spec_from_file_location("ttl_review40", src)
t = importlib.util.module_from_spec(s)
s.loader.exec_module(t)
checks, cases = [], []


def check(name, ok):
    checks.append({"name": name, "pass": bool(ok)})


def sha(b):
    return hashlib.sha256(b).hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def fixture(root, name, raw):
    r = root / name
    (r / "weights").mkdir(parents=True)
    for name, content in [(t.TEN_NHAT_KY,raw),("weights/last.pt",b"checkpoint-marker-only"),
                          ("weights/best.pt",b"best-marker-only"),("args.yaml",b"epochs: 3\r\n"),
                          ("results.csv",b"epoch,time,x\r\n1,10,0.5\r\n"),
                          (t.TEN_BATCH_CSV,(",".join(t.COT_BATCH)+"\r\n"+",".join(["0","1","0","0","1.0","","","","","True","","","","","","","8","1","8",""])+"\r\n").encode())]:
        (r / name).write_bytes(content)
    return r


prefix = b'{"loai":"bat_dau","lan":0}\n{"loai":"epoch_da_luu","lan":0,"epoch":1}\n'
valid_tail = '{"loai":"ket_thuc","trang_thai":"bị ngắt"}'.encode("utf-8")
inputs = [
    ("clean", prefix, prefix, "khong"),
    ("partial_no_lf", prefix+b'{"loai":"epoch_da_luu","epoch":2', prefix, "cat_duoi"),
    ("partial_with_lf", prefix+b'{"loai":"epoch_da_luu","epoch":2\n', prefix, "cat_duoi"),
    ("valid_no_lf", prefix+valid_tail, prefix+valid_tail+b"\n", "them_newline"),
    ("nul_tail", prefix+b"\x00"*8, prefix, "cat_duoi"),
    ("partial_and_blank_lines", prefix+b'{"loai":\n \t\n\n', prefix, "cat_duoi"),
    ("crlf_valid_no_lf", prefix.replace(b"\n",b"\r\n")+valid_tail,
     prefix.replace(b"\n",b"\r\n")+valid_tail+b"\n", "them_newline"),
]
with tempfile.TemporaryDirectory(prefix="codex_review40_") as td:
    root = Path(td)
    for name, raw, expected, action in inputs:
        r = fixture(root,name,raw)
        p = r/t.TEN_NHAT_KY
        original = snapshot(r)
        before, warnings = t.doc_nhat_ky(r)
        plan = t.phan_tich_duoi_nhat_ky(raw)
        check(name+": analysis is read-only", snapshot(r) == original)
        saved = t.sao_luu_truoc_tiep_tuc(r,r/"weights/last.pt",1,epoch_da_luu=1,nhat_ky_duoi=plan)
        receipt = saved["nhat_ky"]
        backup = Path(saved["thu_muc_luu_cu"])
        after,w = t.doc_nhat_ky(r)
        check(name+": exact repaired bytes and old events", p.read_bytes() == expected and after == before and not w)
        check(name+": backup byte-for-byte", (backup/t.TEN_NHAT_KY).read_bytes() == raw and receipt["ban_sao_sha256"] == sha(raw))
        check(name+": receipt action/hashes/content", receipt["hanh_dong"] == action and receipt["sha256"] == sha(raw)
              and receipt["sau_sha256"] == sha(expected) and receipt == json.loads((backup/"xu_ly_nhat_ky.json").read_text()))
        check(name+": old checkpoints and evidence preserved", all((backup/rel.replace('/','__')).read_bytes() == (r/rel).read_bytes()
              for rel in ("weights/last.pt","weights/best.pt","args.yaml","results.csv",t.TEN_BATCH_CSV))
              and Path(saved["nguon"]).read_bytes() == b"checkpoint-marker-only")
        t.ghi_nhat_ky(r,"bat_dau",che_do_luot="tiep_tuc",lan=1)
        t.ghi_nhat_ky(r,"epoch_da_luu",epoch=2,lan=1)
        events, warn = t.doc_nhat_ky(r)
        check(name+": append/read roundtrip", len(events) == len(before)+2 and events[:len(before)] == before and not warn)
        raw2 = p.read_bytes()
        saved2 = t.sao_luu_truoc_tiep_tuc(r,r/"weights/last.pt",2,epoch_da_luu=2,nhat_ky_duoi=t.phan_tich_duoi_nhat_ky(raw2))
        t.ghi_nhat_ky(r,"bat_dau",che_do_luot="tiep_tuc",lan=2)
        t.ghi_nhat_ky(r,"epoch_da_luu",epoch=3,lan=2)
        events2, warn2 = t.doc_nhat_ky(r)
        check(name+": second prepare/append/read and first backup preserved", len(events2) == len(before)+4 and events2[:len(events)] == events
              and not warn2 and saved2["nhat_ky"]["hanh_dong"] == "khong"
              and (Path(saved2["thu_muc_luu_cu"])/t.TEN_NHAT_KY).read_bytes() == raw2
              and (backup/t.TEN_NHAT_KY).read_bytes() == raw)
        cases.append({"case":name,"action":action,"before_events":len(before),"after_two_appends_events":len(events2)})
    # Reject corruption in the middle without any repair/output from the analyzer.
    bad = b'{"loai":"bat_dau"}\n{"loai":\n{"loai":"epoch_da_luu","epoch":1}\n'
    r = fixture(root,"middle",bad)
    before = snapshot(r)
    rejected = []
    for fn in (lambda:t.doc_nhat_ky(r),lambda:t.phan_tich_duoi_nhat_ky(bad)):
        try: fn();rejected.append(False)
        except t.LoiTiepTuc: rejected.append(True)
    check("middle corruption rejected by both read paths without writing", all(rejected) and snapshot(r) == before)
    # Journal changed after the plan: guard rejects before modifying the journal or writing its backup.
    r = fixture(root,"stale_plan",prefix)
    plan = t.phan_tich_duoi_nhat_ky(prefix)
    t.ghi_nhat_ky(r,"note",lan=0)
    raw = (r/t.TEN_NHAT_KY).read_bytes()
    (r/"save").mkdir()
    try:t.sua_duoi_nhat_ky(r,r/"save",plan);reject=False
    except t.LoiTiepTuc:reject=True
    check("changed journal rejected; original and empty backup dir preserved", reject and (r/t.TEN_NHAT_KY).read_bytes() == raw and not list((r/"save").iterdir()))
    # Simulate interruption immediately before journal replacement, after its backup exists.
    raw = prefix+b'{"loai":'
    r = fixture(root,"replace_failure",raw)
    p = r/t.TEN_NHAT_KY
    real_replace = t.os.replace
    def fail_replace(source,dest):
        if Path(dest).resolve() == p.resolve(): raise OSError("injected before journal replace")
        return real_replace(source,dest)
    t.os.replace = fail_replace
    try:
        try:t.sao_luu_truoc_tiep_tuc(r,r/"weights/last.pt",1,epoch_da_luu=1,nhat_ky_duoi=t.phan_tich_duoi_nhat_ky(raw));reject=False
        except t.LoiTiepTuc:reject=True
    finally:t.os.replace = real_replace
    check("replace failure leaves old journal and backup exact", reject and p.read_bytes() == raw and
          (r/"khoi_phuc/truoc_tiep_tuc_1"/t.TEN_NHAT_KY).read_bytes() == raw)
    next_lan=t.so_lan_tiep_tuc_da_co(r)+1
    saved=t.sao_luu_truoc_tiep_tuc(r,r/"weights/last.pt",next_lan,epoch_da_luu=1,nhat_ky_duoi=t.phan_tich_duoi_nhat_ky(raw))
    t.ghi_nhat_ky(r,"bat_dau",lan=next_lan)
    t.ghi_nhat_ky(r,"epoch_da_luu",epoch=2,lan=next_lan)
    e,w=t.doc_nhat_ky(r)
    check("next attempt succeeds after injected replacement failure",next_lan == 2 and len(e)==4 and not w and
          (r/"khoi_phuc/truoc_tiep_tuc_1"/t.TEN_NHAT_KY).read_bytes()==raw)
    before=snapshot(r)
    try:t.sao_luu_truoc_tiep_tuc(r,r/"weights/last.pt",2);reject=False
    except t.LoiTiepTuc:reject=True
    check("same attempt number cannot overwrite backup",reject and snapshot(r)==before)

result={"source_sha256":sha(src.read_bytes()),"scope":__doc__,"cases":cases,"checks":checks,
        "pass":sum(c["pass"] for c in checks),"fail":sum(not c["pass"] for c in checks)}
a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(int(result["fail"]>0))
