# Tim tien trinh THAT theo executable + doi so (khong dung pgrep -f: tranh dem chinh shell/lenh tim). Python 3.6+.
import os, sys
mau = sys.argv[1:]          # cac chuoi phai CO trong mot doi so argv[1:] (vd duoi file script)
ra = []
for d in os.listdir("/proc"):
    if not d.isdigit() or int(d) == os.getpid():
        continue
    try:
        exe = os.readlink("/proc/%s/exe" % d)
        argv = open("/proc/%s/cmdline" % d, "rb").read().split(b"\0")
    except OSError:
        continue
    args = [a.decode("utf-8", "replace") for a in argv[1:] if a]
    if "python" in os.path.basename(exe) and any(all(m in a for m in mau) for a in args):
        ra.append("%s exe=%s argv=%s" % (d, exe, " ".join(args)[:160]))
print("tien trinh python khop %s: %d" % (mau, len(ra)))
for r in ra:
    print("  " + r)
