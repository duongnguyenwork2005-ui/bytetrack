# Kiem thu cuc bo kiem_nhan_ung_vien.py tren du lieu tong hop (khong cham data/).
import json, shutil, subprocess, sys
from pathlib import Path

G = Path(__file__).resolve().parent / "env"
SCRIPT = Path(r"D:\LABOTARY\bytetrack\handoff\logs\review32\kiem_nhan_ung_vien.py")
PY = sys.executable
kq = []


def kiem(ten, dk, ct=""):
    kq.append(bool(dk))
    print("PASS" if dk else "FAIL", ten, ct)


def nhan(i, cat="car", crowd=False, x1=10.0):
    return {"id": f"{i:08d}", "category": cat, "attributes": {"occluded": False, "truncated": False, "crowd": crowd},
            "box2d": {"x1": x1, "y1": 20.0, "x2": x1 + 30.5, "y2": 60.25}}


def dung(sai=False):
    shutil.rmtree(G, ignore_errors=True)
    for split, vids in (("train", ["ta", "tb"]), ("val", ["va"])):
        for v in vids:
            (G / "anh" / split / v).mkdir(parents=True)
            for k in (1, 2, 3):
                (G / "anh" / split / v / f"{v}-{k:07d}.jpg").write_bytes(b"x")
            (G / "nhan" / split).mkdir(parents=True, exist_ok=True)
            frames = [{"name": f"{v}-{k:07d}.jpg", "videoName": v, "frameIndex": k - 1,
                       "labels": [nhan(1), nhan(2, "pedestrian", crowd=(k == 2), x1=100.0)] if k != 3 else []} for k in (1, 2, 3)]
            (G / "nhan" / split / f"{v}.json").write_text(json.dumps(frames), encoding="utf-8")
    # COCO val khop quy uoc w = x2-x1+1 (nhu scalabel)
    imgs, anns = [], []
    for k in (1, 2, 3):
        imgs.append({"id": k, "video_id": 1, "frame_id": k - 1, "file_name": f"va/va-{k:07d}.jpg"})
        if k != 3:
            for j, (cat, x1, crowd) in enumerate([(3, 10.0, False), (1, 100.0, k == 2)], 1):
                anns.append({"id": len(anns) + 1, "image_id": k, "category_id": cat, "instance_id": j, "scalabel_id": f"{j:08d}",
                             "iscrowd": int(crowd), "ignore": 0, "bbox": [x1, 20.0, 31.5, 41.25]})
    coco = {"categories": [{"id": 1, "name": "pedestrian"}, {"id": 3, "name": "car"}], "videos": [{"id": 1, "name": "va"}],
            "images": imgs, "annotations": anns}
    (G / "coco.json").write_text(json.dumps(coco), encoding="utf-8")
    if sai:
        (G / "anh/train/tc").mkdir()                                  # video co anh, khong co nhan
        (G / "anh/train/tc/tc-0000001.jpg").write_bytes(b"x")
        (G / "anh/train/ta/ta-0000003.jpg").unlink()                   # frame nhan khong co anh
        (G / "anh/train/tb/tb-0000009.jpg").write_bytes(b"x")         # anh khong co frame nhan
        fr = json.loads((G / "nhan/train/tb.json").read_text(encoding="utf-8"))
        fr[0]["labels"][0]["box2d"]["x2"] = 1.0                        # box dao
        fr[1]["frameIndex"] = 7                                        # frameIndex lech ten
        (G / "nhan/train/tb.json").write_text(json.dumps(fr), encoding="utf-8")
        (G / "nhan/train/hong.json").write_text("{khong phai json", encoding="utf-8")


def chay():
    r = subprocess.run([PY, str(SCRIPT), "--nhan", str(G / "nhan"), "--anh", str(G / "anh"), "--coco-val", str(G / "coco.json"),
                        "--ra", str(G / "ra.json")], capture_output=True, text=True)
    return r.returncode, json.loads((G / "ra.json").read_text(encoding="utf-8")) if (G / "ra.json").exists() else {}, r.stderr


dung()
rc, k, e = chay()
t, v, d = k.get("train", {}), k.get("val", {}), k.get("doi_chieu_val_coco", {})
kiem("1 sach -> exit 0, dem dung", rc == 0 and t["so_file_json"] == 2 and t["so_frame"] == 6 and t["so_nhan"] == 8
     and t["so_frame_khong_nhan"] == 2 and not any(t["loi_schema"].values()) and t["crowd_theo_lop"] == {"pedestrian": 2},
     f"rc={rc} {e[-300:]} {json.dumps(t)[:300]}")
kiem("1 val doi chieu COCO: ghep 4/4, box khop quy uoc +1, id nhat quan", d.get("so_cap_ghep_duoc") == 4
     and d["so_box_khop_tuyet_doi"] == {"w=x2-x1": 0, "w=x2-x1+1": 4} and d["so_track_id_anh_xa_bat_nhat_quan"] == 0
     and d["so_crowd_khac_iscrowd"] == 0 and not d["nhan_chi_co_o_scalabel_theo_lop"] and not d["annotation_chi_co_o_coco_theo_lop"],
     json.dumps(d)[:400])
dung(sai=True)
rc, k, e = chay()
t = k.get("train", {})
kiem("2 loi cai san -> exit 1 va dem dung tung loai", rc == 1 and t["video_co_anh_khong_co_nhan"] == ["tc"]
     and t["so_frame_nhan_khong_co_anh"] == 1 and t["so_anh_khong_co_frame_nhan"] == 1 and t["loi_schema"]["box_dao"] == 1
     and t["loi_schema"]["frameIndex_lech_ten"] == 1 and len(t["json_hong"]) == 1, json.dumps(t)[:500])
print("TONG", sum(kq), "/", len(kq), "PASS")
sys.exit(0 if all(kq) else 1)
