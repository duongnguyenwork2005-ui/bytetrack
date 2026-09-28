import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, r"D:\LABOTARY\bytetrack\src")
import render_mot_video as rmv
SP = Path(sys.argv[1]); D = Path(r"D:\LABOTARY\bytetrack\handoff\logs\review23\nano_dai_v23\results_nano\dai_v23\luot01_cv")
boxes, lop = rmv.doc_mot(D / "data/MVI_39031.txt"), rmv.doc_lop(D / "data_cls/MVI_39031.csv")
for i in (1, 90, 262):
    goc = cv2.imread(str(SP / ("png_nano/goc_f%04d.png" % i))); kvn = cv2.imread(str(SP / ("png_nano/ky_vong_f%04d.png" % i)))
    kvp = goc.copy(); rmv.ve_frame(cv2, kvp, boxes.get(i, []), lop, i)
    lech = np.any(kvp != kvn, axis=2)
    chu = goc.copy(); hop = goc.copy()
    for tid, x, y, bw, bh, _c in boxes.get(i, []):
        cv2.rectangle(hop, (int(x), int(y)), (int(x + bw), int(y + bh)), rmv.mau_id(tid), 2)
    m_hop = np.any(hop != goc, axis=2)
    ve_n = np.any(kvn != goc, axis=2); ve_p = np.any(kvp != goc, axis=2)
    print("frame %d: diem lech %d | trong net khung chu nhat (PC) %d | ngoai -> chu (putText) %d | diem ve Nano %d, PC %d"
          % (i, lech.sum(), (lech & m_hop).sum(), (lech & ~m_hop).sum(), ve_n.sum(), ve_p.sum()))
    # vung so frame goc trai tren
    print("   vung so frame [0:30,0:80]: diem ve Nano %d, PC %d, lech %d" % (ve_n[:30, :80].sum(), ve_p[:30, :80].sum(), lech[:30, :80].sum()))
    if i == 90:
        a = np.concatenate([cv2.resize(im[0:30, 0:80], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST) for im in (kvn, kvp)], axis=0)
        cv2.imwrite(str(SP / "so_frame_nano_vs_pc.png"), a)
