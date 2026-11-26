# -*- coding: utf-8 -*-
"""
nano_hien_thi.py -- HIEN THI TRUC TIEP trong luc tracking tren Jetson Nano (vong 25). Chay TRONG tien trinh detector
(`nano_detect_stream.py`, Python 3.6 he thong, OpenCV 3.2 co GTK): day la tien trinh DUY NHAT dang giu anh da giai ma,
nen khong phai truyen anh qua pipe va khong giai ma lan hai.

GHEP DUNG FRAME: con luu anh GOC cua frame i (truoc khi gui FRAME i) vao vong dem theo frame_idx; cha xu ly xong frame i
thi gui lenh R(i, hang MOT/lop cua frame i) thay cho credit "+" (`nano_stream_proto`). `nhan_ket_qua(i, ...)` lay DUNG anh
co khoa i -> nen, bbox, ID, lop tren moi anh hien thi cung mot frame. Khong co anh cho khoa i -> dem `n_khong_ghep`
(khong ve anh khac thay the).

KHONG LAM CHAM TRACKING: luong chinh cua con chi dat (i, anh, ket qua) vao HANG DOI CO GIOI HAN (`hang_doi`, mac dinh 2);
hang day thi frame CU NHAT chua kip ve bi bo va dem `n_bo_hien_thi` (frame xem truoc bi bo). Vong lap detector/tracker
khong bao gio cho hien thi, duong tracking van xu ly DU frame theo cau hinh. Ve bang CHINH `render_mot_video.ve_frame`
(cung ham render hau ky).

DUNG: phim q / Q / Esc trong cua so, hoac dong cua so -> GHI stop-file (`--stop-file` cua `nano_live`, cung duong voi nut
Dung cua launcher) -> cha dung co kiem soat, ghi du phan da xu ly. Loi o luong hien thi -> ghi `loi`, tat hien thi,
tracking tiep tuc.

DEM (vong 26, review 20e66e9 N25-F1): `n_hien_thi`, moc `t_hien_thi`, `frame_hien_thi_cuoi` CHI tang khi frame THUC SU duoc
trinh len cua so: cua so chua bi dong (kiem TRUOC moi imshow) va `cv2.imshow` chay xong khong loi. Frame toi SAU khi cua so da
dong (vd frame dang bay luc nguoi dung bam x) -> KHONG ve, KHONG imshow (imshow se mo lai cua so), dem rieng
`n_khong_hien_thi_cua_so_dong`; tracking van xu ly va ghi MOT/lop cua frame do (hien thi khong cham vao du lieu tracker).
Bat bien: n_ghep_duoc = n_hien_thi + n_bo_hien_thi + n_khong_hien_thi_cua_so_dong (tru khi luong hien thi loi).
`fps_hien_thi` = (N - 1) / (moc hien thi cuoi - moc hien thi dau), N = n_hien_thi; N < 2 -> None (khong co FPS co nghia).

Che do `an=True` (kiem thu / may khong man hinh): moi buoc tru tao cua so (ghep, ve, luu mau, dem, gia lap phim q, gia lap
dong cua so). O che do nay KHONG co cua so: "trinh" la GIA LAP = da ve xong (`che_do` ghi ro trong thong ke).
Cu phap Python 3.6.
"""
import collections
import json
import os
import sys
import threading
import time

PHIM_DUNG = (ord("q"), ord("Q"), 27)


def _tk(v):
    v = sorted(x for x in v if x == x)
    if not v:
        return None
    n = len(v)
    return {"n": n, "trung_vi": v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0, "tb": sum(v) / n,
            "p90": v[min(n - 1, int(round(0.9 * (n - 1))))], "max": v[-1]}


class HienThi(object):
    def __init__(self, cv2, rmv, tieu_de, an=False, stop_file=None, json_path=None, luu_mau_dir=None, luu_mau_moi=0,
                 dung_o_frame=None, suc_chua_anh=8, nhan_hud="", kich_thuoc_cua_so=None, log=None, hang_doi=2,
                 dong_cua_so_o_frame=None):
        self.cv2, self.rmv, self.tieu_de, self.an = cv2, rmv, tieu_de, bool(an)
        self.stop_file, self.json_path = stop_file, json_path
        self.luu_mau_dir, self.luu_mau_moi = luu_mau_dir, int(luu_mau_moi or 0)
        self.dung_o_frame = dung_o_frame
        self.dong_cua_so_o_frame = dong_cua_so_o_frame      # CHI kiem thu (che do an): gia lap nguoi dung dong cua so
        self.suc_chua_anh = max(2, int(suc_chua_anh))
        self.nhan_hud = nhan_hud
        self.kich_thuoc_cua_so = kich_thuoc_cua_so
        self.log = log or (lambda m: sys.stderr.write("[hien_thi] %s\n" % m))
        self._anh = collections.OrderedDict()           # frame_idx -> anh goc (chi luong chinh cham vao)
        self._cv = threading.Condition()
        self.hang_doi = max(1, int(hang_doi))
        self._q = collections.deque()                   # (idx, anh, hang) cho ve -- toi da `hang_doi` phan tu
        self._dung_luong = False
        self._cua_so = False
        self._cua_so_da_dong = False
        self.dem = {"n_anh_nhan": 0, "n_ket_qua_nhan": 0, "n_ghep_duoc": 0, "n_khong_ghep": 0, "n_anh_bo_khong_ket_qua": 0,
                    "n_da_ve": 0, "n_hien_thi": 0, "n_bo_hien_thi": 0, "n_khong_hien_thi_cua_so_dong": 0, "n_mau_luu": 0}
        self.frame_bo = []                              # frame xem truoc bi bo (toi da 200 phan tu dau)
        self.frame_khong_ghep = []
        self.frame_khong_hien_thi = []                  # frame toi sau khi cua so da dong (toi da 200 phan tu dau)
        self.ve_ms, self.imshow_ms = [], []
        self.t_hien_thi = []                            # moc perf_counter moi lan hien thi
        self.frame_hien_thi_cuoi = None
        self.ly_do_dung = None
        self.loi = None
        if self.luu_mau_dir and self.luu_mau_moi > 0 and not os.path.isdir(self.luu_mau_dir):
            os.makedirs(self.luu_mau_dir)
        self._luong = threading.Thread(target=self._chay, name="hien_thi", daemon=True)
        self._luong.start()

    # ------------------------------------------------------------------ luong chinh cua con
    def them_anh(self, idx, img):
        """Luu anh GOC cua frame idx (goi TRUOC khi gui FRAME idx). Vong dem co gioi han."""
        if img is None or self.loi:
            return
        self._anh[int(idx)] = img
        self.dem["n_anh_nhan"] += 1
        while len(self._anh) > self.suc_chua_anh:
            self._anh.popitem(last=False)
            self.dem["n_anh_bo_khong_ket_qua"] += 1

    def nhan_ket_qua(self, idx, hang):
        """Ket qua tracker cua frame idx (tu lenh R). Ghep voi anh CUNG khoa; o cho con frame cu -> thay (dem bo)."""
        idx = int(idx)
        self.dem["n_ket_qua_nhan"] += 1
        for k in [k for k in self._anh if k < idx]:     # frame khong co ket qua (vd dang bay khi dung) -> bo anh
            del self._anh[k]
            self.dem["n_anh_bo_khong_ket_qua"] += 1
        img = self._anh.pop(idx, None)
        if img is None:
            self.dem["n_khong_ghep"] += 1
            if len(self.frame_khong_ghep) < 200:
                self.frame_khong_ghep.append(idx)
            return
        self.dem["n_ghep_duoc"] += 1
        if self.loi:
            return
        with self._cv:
            while len(self._q) >= self.hang_doi:        # hang day -> bo frame CU NHAT chua ve (khong chan tracking)
                bo = self._q.popleft()
                self.dem["n_bo_hien_thi"] += 1
                if len(self.frame_bo) < 200:
                    self.frame_bo.append(bo[0])
            self._q.append((idx, img, hang))
            self._cv.notify()

    # ------------------------------------------------------------------ luong hien thi
    def _lay(self, cho_s):
        with self._cv:
            if not self._q and not self._dung_luong:
                self._cv.wait(cho_s)
            return self._q.popleft() if self._q else None

    def _chay(self):
        try:
            while True:
                o = self._lay(0.03)
                if o is not None:
                    self._ve(*o)
                elif self._dung_luong:
                    break
                self._su_kien()
        except Exception as e:  # noqa: BLE001 -- hien thi hong KHONG duoc lam dung tracking
            self.loi = "%s: %s" % (type(e).__name__, e)
            self.log("LOI hien thi (tat hien thi, tracking tiep tuc): %s" % self.loi)

    def _khong_hien_thi(self, idx):
        """Frame toi SAU khi cua so da dong: khong ve, khong tinh la da hien thi (tracking van ghi MOT/lop cua no)."""
        self.dem["n_khong_hien_thi_cua_so_dong"] += 1
        if len(self.frame_khong_hien_thi) < 200:
            self.frame_khong_hien_thi.append(idx)

    def _ve(self, idx, img, hang):
        cv2 = self.cv2
        if self._cua_so_da_dong:
            self._khong_hien_thi(idx)
            return
        t0 = time.perf_counter()
        im = img.copy()
        box = [(int(t), x, y, w, h, c) for t, x, y, w, h, c, _k in hang]
        lop = {(idx, int(t)): int(k) for t, _x, _y, _w, _h, _c, k in hang}
        self.rmv.ve_frame(cv2, im, box, lop, idx)
        if self.luu_mau_dir and self.luu_mau_moi > 0 and idx % self.luu_mau_moi == 0:
            cv2.imwrite(os.path.join(self.luu_mau_dir, "hien_thi_f%05d.png" % idx), im)   # PNG: khong mat mat
            self.dem["n_mau_luu"] += 1
        self._hud(im, idx, len(hang))
        self.dem["n_da_ve"] += 1
        t1 = time.perf_counter()
        self.ve_ms.append((t1 - t0) * 1e3)
        # che do an: KHONG co cua so -> "trinh" la gia lap (ve xong); che do that: chi True khi imshow da chay len cua so mo
        trinh = True if self.an else self._trinh_len_cua_so(im)
        t2 = time.perf_counter()
        if not trinh:
            self._khong_hien_thi(idx)
            return
        if not self.an:
            self.imshow_ms.append((t2 - t1) * 1e3)
        self.t_hien_thi.append(t2)
        self.dem["n_hien_thi"] += 1
        self.frame_hien_thi_cuoi = idx
        if self.dung_o_frame and idx >= self.dung_o_frame and self.ly_do_dung is None:
            self.yeu_cau_dung("gia_lap_phim_q")
        if self.an and self.dong_cua_so_o_frame and idx >= self.dong_cua_so_o_frame and not self._cua_so_da_dong:
            self._cua_so_da_dong = True                    # gia lap nguoi dung dong cua so SAU khi frame nay da hien
            self.yeu_cau_dung("gia_lap_dong_cua_so")

    def _cua_so_con_mo(self):
        try:
            return self.cv2.getWindowProperty(self.tieu_de, self.cv2.WND_PROP_AUTOSIZE) >= 0
        except Exception:  # noqa: BLE001
            return False

    def _trinh_len_cua_so(self, im):
        """imshow len cua so dang MO. Cua so da tao nhung nguoi dung da dong (ke ca giua hai lan _su_kien) -> KHONG imshow
        (imshow se tao lai cua so), danh dau dong + yeu cau dung, tra False. True chi khi imshow chay xong khong loi."""
        cv2 = self.cv2
        if self._cua_so and not self._cua_so_con_mo():
            self._cua_so_da_dong = True
            self.yeu_cau_dung("dong_cua_so")
            return False
        if not self._cua_so:
            cv2.namedWindow(self.tieu_de, cv2.WINDOW_NORMAL)
            if self.kich_thuoc_cua_so:
                cv2.resizeWindow(self.tieu_de, int(self.kich_thuoc_cua_so[0]), int(self.kich_thuoc_cua_so[1]))
            self._cua_so = True
        cv2.imshow(self.tieu_de, im)
        return True

    def _hud(self, im, idx, n):
        """Dong trang thai o GOC DUOI (so frame 'f<i>' cua ve_frame o goc tren giu nguyen). Chu ASCII (font Hershey)."""
        cv2 = self.cv2
        fps = ""
        if len(self.t_hien_thi) >= 10:
            dt = self.t_hien_thi[-1] - self.t_hien_thi[-10]
            if dt > 0:
                fps = " | hien thi %.1f fps" % (9.0 / dt)
        chu = "%s | frame %d | %d doi tuong%s | q/Esc: dung" % (self.nhan_hud, idx, n, fps)
        h = im.shape[0]
        (tw, th), _ = cv2.getTextSize(chu, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(im, (0, h - th - 14), (tw + 12, h), (0, 0, 0), -1)
        cv2.putText(im, chu, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _su_kien(self):
        if self.an or not self._cua_so or self._cua_so_da_dong:
            return
        cv2 = self.cv2
        k = cv2.waitKey(1)
        if k != -1 and (k & 0xFF) in PHIM_DUNG:
            self.yeu_cau_dung("phim_q_esc")
        if not self._cua_so_con_mo():
            self._cua_so_da_dong = True        # khong imshow lai (imshow se mo lai cua so)
            self.yeu_cau_dung("dong_cua_so")

    def yeu_cau_dung(self, ly_do):
        """Ghi stop-file cua tien trinh cha (cung duong voi nut Dung cua launcher). Chi ghi lan dau."""
        if self.ly_do_dung is not None:
            return
        self.ly_do_dung = ly_do
        self.log("yeu cau dung tu cua so hien thi: %s" % ly_do)
        if self.stop_file:
            try:
                tmp = self.stop_file + ".tmp"
                with open(tmp, "w") as f:
                    f.write("hien_thi:%s %s\n" % (ly_do, time.strftime("%Y-%m-%dT%H:%M:%S")))
                os.replace(tmp, self.stop_file)
            except OSError as e:
                self.log("khong ghi duoc stop-file %s: %s" % (self.stop_file, e))

    # ------------------------------------------------------------------ ket thuc
    def dong(self, ly_do_ket_thuc=None, cho_s=10.0):
        """Ve not frame cuoi dang cho, dung luong, dong cua so, ghi JSON thong ke. Goi tu luong chinh cua con."""
        with self._cv:
            self._dung_luong = True
            self._cv.notify()
        self._luong.join(cho_s)
        con_song = self._luong.is_alive()
        if self._cua_so and not self.an:
            try:
                self.cv2.destroyAllWindows()
                for _ in range(5):
                    self.cv2.waitKey(1)
            except Exception:  # noqa: BLE001
                pass
        t = self.t_hien_thi
        # (N - 1) / khoang: N moc hien thi co N - 1 khoang giua frame. N < 2 -> None (mot frame khong co toc do hien thi)
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) >= 2 and t[-1] > t[0] else None
        tk = {"bat": True, "an": self.an,
              "che_do": ("an: KHONG co cua so, 'hien thi' la GIA LAP (da ve xong)" if self.an
                         else "that: hien thi = imshow thanh cong len cua so dang mo"),
              "tieu_de": self.tieu_de, "hang_doi": self.hang_doi, "dem": dict(self.dem),
              "frame_bo_hien_thi_dau": self.frame_bo, "frame_khong_ghep_dau": self.frame_khong_ghep,
              "frame_khong_hien_thi_cua_so_dong_dau": self.frame_khong_hien_thi,
              "frame_hien_thi_cuoi": self.frame_hien_thi_cuoi, "fps_hien_thi": fps,
              "khoang_hien_thi_s": (t[-1] - t[0]) if len(t) >= 2 else None,
              "ve_ms": _tk(self.ve_ms), "imshow_ms": _tk(self.imshow_ms),
              "ly_do_dung_tu_cua_so": self.ly_do_dung, "ly_do_ket_thuc_con": ly_do_ket_thuc, "loi": self.loi,
              "luong_con_song_sau_dong": con_song, "luu_mau_dir": self.luu_mau_dir, "luu_mau_moi": self.luu_mau_moi,
              "dinh_nghia": {
                  "n_hien_thi": "frame DA TRINH len cua so: cua so con mo (kiem truoc moi imshow) va cv2.imshow chay xong "
                                "khong loi; che do an: gia lap (da ve xong)",
                  "fps_hien_thi": "(N - 1) / (moc hien thi cuoi - moc hien thi dau), N = n_hien_thi, moc lay NGAY SAU imshow; "
                                  "N < 2 -> null (khong co FPS hien thi co nghia)",
                  "n_da_ve": "frame da ve xong (ve_frame + HUD) truoc khi trinh len cua so",
                  "n_khong_hien_thi_cua_so_dong": "frame toi SAU khi cua so da dong (vd dang bay luc bam x): khong ve, khong "
                                                  "imshow, KHONG tinh la da hien thi; tracking van ghi MOT/lop",
                  "n_bo_hien_thi": "frame XEM TRUOC bi bo vi hang doi hien thi day (bo frame cu nhat; tracking van xu ly du)",
                  "n_khong_ghep": "ket qua tracker khong tim thay anh cung frame_idx (khong ve anh khac thay the)",
                  "n_anh_bo_khong_ket_qua": "anh da luu nhung khong co ket qua (frame dang bay khi dung, vuot vong dem)",
                  "tam_dung": "khong co chuc nang tam dung"}}
        if self.json_path:
            try:
                tmp = self.json_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(tk, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.json_path)
            except OSError as e:
                self.log("khong ghi duoc %s: %s" % (self.json_path, e))
        return tk
