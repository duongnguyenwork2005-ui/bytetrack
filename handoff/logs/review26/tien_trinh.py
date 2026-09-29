# -*- coding: utf-8 -*-
"""Vong 26: dem tien trinh THAT theo executable + doi so (doc /proc; KHONG dung `pgrep -f` -- no co the dem chinh no hoac
shell chua mau tim, review 20e66e9). Bo qua zombie. Cu phap Python 3.6.

    python tien_trinh.py <chuoi_trong_ten_exe> [<chuoi phai co trong MOT doi so> ...]   -> in "<so> <pid> <pid> ..."
"""
import os
import sys


def main():
    ten, can = sys.argv[1], sys.argv[2:]
    ra = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            exe = os.path.basename(os.readlink("/proc/%s/exe" % d))
            argv = [a.decode("utf-8", "replace") for a in open("/proc/%s/cmdline" % d, "rb").read().split(b"\0") if a]
            st = open("/proc/%s/stat" % d).read().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if st == "Z":
            continue
        if ten in exe and all(any(c in a for a in argv[1:]) for c in can):
            ra.append(d)
    print(len(ra), " ".join(ra))


if __name__ == "__main__":
    main()
