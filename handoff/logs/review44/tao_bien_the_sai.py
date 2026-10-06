# -*- coding: utf-8 -*-
"""Vong 44: sinh BIEN THE SAI cua entry point (CHI de kiem regression B5c, KHONG phai ma du an): thay cau hinh hieu luc
bang cach SUA TRUC TIEP hang dung chung QUY_TAC -- kieu sua "nhanh" ma review canh bao (ro ri cau hinh giua cac lan goi).
    python handoff/logs/review44/tao_bien_the_sai.py <entry point da sua> <file ra>
Thay dung MOT dong; neu chuoi neo khong xuat hien dung 1 lan -> exit 1, khong ghi."""
import sys
from pathlib import Path

NEO = "    qt = quy_tac_hieu_luc(a.conf_van_hanh)                  # R43-N1: MOT nguon nguong cho metadata / scorer / chan doan\n"
THAY = ("    QUY_TAC[\"van_hanh\"][\"conf\"] = float(a.conf_van_hanh)   # BIEN THE SAI: sua hang dung chung\n"
        "    qt = QUY_TAC\n")


def main() -> int:
    s = Path(sys.argv[1]).read_text(encoding="utf-8")
    if s.count(NEO) != 1:
        print(f"chuoi neo xuat hien {s.count(NEO)} lan (can 1) -- khong ghi")
        return 1
    Path(sys.argv[2]).write_text(s.replace(NEO, THAY), encoding="utf-8", newline="\n")
    print(f"da ghi bien the sai -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
