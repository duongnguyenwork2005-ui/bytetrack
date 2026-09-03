"""
config.py — Cau hinh tap trung cho toan bo pipeline UA-DETRAC.

Moi script khac deu import tu day de tranh hard-code duong dan / hang so.
Neu ban doi cho luu dataset, chi can sua o FILE NAY.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. DUONG DAN
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"        # chua file .zip tai ve (chua giai nen)
EXTRACTED_DIR = DATA_DIR / "extracted"  # anh + XML sau khi giai nen
INTERIM_DIR   = DATA_DIR / "interim"    # DataFrame chuan hoa (.parquet/.csv)
PROCESSED_DIR = DATA_DIR / "processed"  # format MOTChallenge (gt.txt, seqinfo.ini)
RESULTS_DIR   = PROJECT_ROOT / "results"

for _d in (RAW_DIR, EXTRACTED_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Ten thu muc chuan sau khi giai nen (theo dung ten goc cua UA-DETRAC)
IMAGES_SUBDIR      = "Insight-MVT_Annotation_Train"    # <video>/img00001.jpg ...
ANNOTATIONS_SUBDIR = "DETRAC-Train-Annotations-XML"    # <video>.xml

# ---------------------------------------------------------------------------
# 2. NGUON TAI (Google Drive)
# ---------------------------------------------------------------------------
# Key  -> (google drive file id, ten file zip luu vao data/raw/)
# LUU Y: id cua train-images de None vi ban chua cung cap link;
#        script download se bao loi ro rang thay vi doan bua mot id sai.
DRIVE_FILES = {
    "train_annotations_xml": {
        "file_id":  "12xJc8S0Z7lYaAadsi2CoSK3WqH2OkUBu",
        "filename": "DETRAC-Train-Annotations-XML.zip",
        "note":     "Annotation XML co occlusion / track id (~ vai chuc MB)",
    },
    "train_images": {
        "file_id":  None,      # <-- DIEN GOOGLE DRIVE FILE ID VAO DAY
        "filename": "DETRAC-Train-Images.zip",
        "note":     "Anh train Insight-MVT_Annotation_Train (~5.2 GB)",
    },
}

# Nguon tai thay the (KHONG phai Google Drive): mirror tren HuggingFace.
# Uu diem: tai truc tiep qua HTTPS, ho tro resume, khong dinh quota Drive.
# Da kiem chung (doc central directory cua zip tu xa bang HTTP Range):
#   - giu NGUYEN cau truc goc  MVI_20011/img00001.jpg
#   - co du 60/60 sequence train (83.791 anh) + 40 sequence test
#   - kem ca DETRAC-Train-Annotations-XML va DETRAC-MOT-toolkit
MIRROR_URLS = {
    "ua_detrac_orig": (
        "https://huggingface.co/datasets/ShantyCam/ua-detrac/resolve/main/"
        "ua-detrac-orig.zip"
    ),
}

# ---------------------------------------------------------------------------
# 3. THONG SO VIDEO UA-DETRAC
# ---------------------------------------------------------------------------
FPS      = 25       # UA-DETRAC quay o 25 fps -> 1 frame = 0.04 s
IMG_W    = 960      # kich thuoc anh chuan cua bo du lieu
IMG_H    = 540

# ---------------------------------------------------------------------------
# 4. NGUONG PHAN NHOM CHO PHAN TICH (dung o Giai doan 5)
# ---------------------------------------------------------------------------
# --- 4a. Occlusion ---
# Mot frame duoc coi la "bi che khuat" khi occlusion_ratio >= nguong nay.
OCCLUSION_MIN_RATIO = 0.10

# Cho phep "vá" cac lo hong ngan: neu 2 doan che khuat cach nhau <= so frame nay
# thi gop lam mot doan (tranh bi cat vun do nhieu annotation).
OCCLUSION_MERGE_GAP = 2

# Bo qua doan che khuat qua ngan (nhieu annotation, khong co y nghia thong ke).
OCCLUSION_MIN_LEN_FRAMES = 2

# Nguong coi la CHE KHUAT GAN NHU HOAN TOAN.
# UA-DETRAC van annotate xe ngay ca khi no bi che 100%, trong khi detector
# gan nhu chac chan KHONG phat hien duoc -> tracker buoc phai dua hoan toan
# vao motion model. Day la kich ban quan trong nhat de so sanh CV va CTRV.
OCCLUSION_FULL_RATIO = 0.90

# Phan nhom do dai doan che khuat theo giay (theo yeu cau de tai).
# Dinh dang: (ten_nhom, can_duoi_giay_bao_gom, can_tren_giay_khong_bao_gom)
OCCLUSION_BUCKETS = [
    ("short",  0.0, 0.5),   # < 0.5 s   (< 12.5 frame)
    ("medium", 0.5, 1.5),   # 0.5 - 1.5 s (12.5 - 37.5 frame)
    ("long",   1.5, float("inf")),  # > 1.5 s (> 37.5 frame)
]

# --- 4b. Curvature ---
# Track ngan hon nguong nay khong du diem de uoc luong do cong -> danh dau 'unknown'.
CURVATURE_MIN_TRACK_LEN = 15

# Cua so Savitzky-Golay de lam tron quy dao truoc khi lay dao ham.
SAVGOL_WINDOW = 11   # so frame (phai le)
SAVGOL_POLY   = 2    # bac da thuc

# Track duoc coi la "cong / re" neu TONG goc doi huong vuot nguong (do).
TURN_ANGLE_THRESHOLD_DEG = 20.0
# ... HOAC chi so thang (chord / arc-length) nho hon nguong nay.
STRAIGHTNESS_THRESHOLD   = 0.97

# ---------------------------------------------------------------------------
# 5. FORMAT MOTChallenge
# ---------------------------------------------------------------------------
# TrackEval (preset MOT17/MOT20) chi tinh diem cho class id = 1.
# UA-DETRAC co 4 loai xe (car/bus/van/others); ta gop tat ca ve class 1 de
# danh gia bai toan "theo vet phuong tien" khong phan biet loai,
# nhung van luu vehicle_type rieng trong DataFrame de phan tich sau.
MOT_CLASS_ID = 1
VEHICLE_TYPES = ["car", "bus", "van", "others"]


# ---------------------------------------------------------------------------
# 6. TIM THU MUC ANH (tu dong, khong phu thuoc cach zip long thu muc)
# ---------------------------------------------------------------------------
def find_images_root(base=None, min_seqs: int = 3):
    """Tim thu muc chua cac thu muc sequence dang MVI_xxxxx/img*.jpg.

    VI SAO CAN HAM NAY?
    Moi ban mirror cua UA-DETRAC long thu muc mot kieu khac nhau:
        Insight-MVT_Annotation_Train/MVI_20011/img00001.jpg     (ban goc)
        DETRAC-Images/DETRAC-Images/MVI_20011/img00001.jpg      (mirror HuggingFace)
        DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_.../ (mot so ban khac)
    Thay vi bat nguoi dung sap xep lai thu muc cho dung, ta tu di tim.

    Cach lam: duyet cay thu muc tu `base` xuong toi da 4 cap (KHONG chui vao
    trong thu muc MVI_*), dem xem thu muc nao co nhieu thu muc con MVI_* chua
    anh nhat. Tra ve thu muc do, hoac None neu chua tai bo anh.
    """
    import os

    base = Path(base) if base else EXTRACTED_DIR
    if not base.is_dir():
        return None

    best, best_n = None, 0
    # (thu_muc, do_sau) - duyet theo chieu rong
    queue = [(base, 0)]
    while queue:
        d, depth = queue.pop(0)
        try:
            entries = [e for e in os.scandir(d) if e.is_dir()]
        except OSError:
            continue

        seq_dirs = [e for e in entries if e.name.upper().startswith("MVI_")]
        # Chi tinh nhung thu muc MVI_* thuc su co anh ben trong.
        n_ok = 0
        for e in seq_dirs:
            try:
                if any(f.name.lower().startswith("img") and f.name.lower().endswith(".jpg")
                       for f in os.scandir(e.path)):
                    n_ok += 1
            except OSError:
                continue
            if n_ok >= min_seqs:
                break

        if n_ok >= min_seqs and n_ok > best_n:
            best, best_n = Path(d), n_ok

        if depth < 4:
            # khong chui vao thu muc sequence (se rat cham vi co hang nghin anh)
            queue.extend((Path(e.path), depth + 1)
                         for e in entries if not e.name.upper().startswith("MVI_"))

    return best


def list_train_videos():
    """Danh sach 60 video train, lay tu ten cac file annotation XML da giai nen."""
    ann = EXTRACTED_DIR / ANNOTATIONS_SUBDIR
    if not ann.is_dir():
        return []
    return sorted(p.stem for p in ann.glob("*.xml"))


# ---------------------------------------------------------------------------
# 7. GIAI DOAN 2 - BASELINE YOLOv8 + ByteTrack
# ---------------------------------------------------------------------------
# 5 video mau dung de chay pilot baseline, chon tu bang thong ke
# data/interim/video_selection_stats.csv de bao phu da dang:
#   - thoi tiet: sunny / night / cloudy / rainy
#   - muc do che khuat: gan nhu khong che -> che gan nhu toan bo track
#   - do cong quy dao: it khuc cua -> nhieu khuc cua gap
SAMPLE_VIDEOS = [
    "MVI_20011",  # sunny, occlusion vua phai (32% track), 68% track cong - dai dien "trung binh"
    "MVI_40962",  # night, GAN NHU KHONG bi che (1.1% track) - doi chung ca "de"
    "MVI_39811",  # night, TAT CA track deu bi che it nhat 1 lan (100%), 4 doan che >=90%
    "MVI_40204",  # cloudy, occlusion nang nhat trong tap (40.8% track, 36 doan che >=90%)
    "MVI_63521",  # rainy, do cong cao nhat (70% track cong, 32 khuc cua gap - sharp)
]

# COCO class id ung voi phuong tien (YOLOv8 pretrained tren COCO):
#   2 = car, 3 = motorcycle, 5 = bus, 7 = truck
# UA-DETRAC chi quay canh giao thong (car/bus/van/others), gan nhu khong co
# xe may lam chu the chinh, nhung van giu motorcycle de khong bo sot xe 2 banh
# lon (một số "others" trong UA-DETRAC la xe 3 banh / xe may).
YOLO_VEHICLE_CLASSES = [2, 3, 5, 7]

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Duong dan tuyet doi -> Ultralytics luon luu/doc trong models/, khong tai
# nham vao thu muc dang chay lenh (cwd) moi lan goi script tu noi khac.
# DA KHOA lam cau hinh CHINH THUC cho toan bo de tai (sau khi bao cao ket qua
# Giai doan 2 va duoc nguoi dung xac nhan). Muc tieu chinh cua khoa luan la so
# sanh MOTION MODEL (KF/EKF/UKF), khong phai detector - nen tu Giai doan 3 tro
# di, CHI motion model duoc thay doi; detector + tracker giu nguyen cau hinh nay
# de phep so sanh cong bang (cung mot bo detection input cho ca 3 phuong phap).
# KHONG doi cac gia tri nay o Giai doan 3-5 neu khong co ly do ro rang.
YOLO_DEFAULT_MODEL = str(MODELS_DIR / "yolov8n.pt")   # pretrained COCO, nhe nhat ho YOLOv8
YOLO_DEFAULT_CONF = 0.25            # nguong tin cay mac dinh cua Ultralytics
BYTETRACK_CFG = "bytetrack.yaml"    # file cau hinh mac dinh di kem ultralytics

# Mau xam trung tinh dung de "che" (mask) vung ignored_region truoc khi dua anh
# vao detector - giong mau padding mac dinh cua YOLO (114,114,114) nen khong tao
# ra canh gia (artifact) lam detector chu y nham.
IGNORE_MASK_COLOR_BGR = (114, 114, 114)
