# Lan chay full suite THAT BAI (giu nguyen, khong sua log)

`tests_20260921-1209.log` + `tests_summary.csv`: full suite tren HEAD `2e96bc7` -> **PASS 30 FAIL 1**.

FAIL: `test_pc_launcher` exit 1 -- `UnicodeEncodeError: 'charmap' codec can't encode character 'ố'` khi in
chi tiet phep kiem 12 (thong bao L-F4 co chu "So" tieng Viet co dau) vi runner `run_all_tests.py` chay test voi
stdout cp1252 (khong dat PYTHONIOENCODING). Phep 1-12 PASS (16/16 in ra), phep 13 KHONG chay. Cac lan chay tay
truoc do dat `PYTHONIOENCODING=utf-8` nen khong lo loi nay.

Sua (commit sau): test_pc_launcher.py tu `reconfigure` stdout sang utf-8 (errors=replace) -- chi doi cach IN cua
test, khong doi ma san pham. Full suite chay lai tren commit da sua: xem `../tests_*.log`.

## Lan 2 -- `tests_20260921-1220.log` (HEAD `f685c67`): PASS 30 FAIL 1

FAIL: `test_pc_launcher` phep 5 "bat dau lan hai khi dang chay": ket qua `DangChay tt=DANG_DUNG` (mong DA_DUNG).
Nguyen nhan: RACE trong helper `_cho()` cua TEST -- dieu kien `not ctl.dang_chay()` duoc danh gia SAU `poll()`;
worker thoat giua hai lenh nen `_cho` tra ve snapshot cu (DANG_DUNG) ma khong poll lai. Lo duoi tai cua full
suite (cac lan chay tay 17/17 x4 khong gap). Phep 12 (UnicodeEncodeError lan 1) da PASS o lan nay.
Sua (commit sau): `_cho()` poll lai mot lan sau khi dieu kien thoa. Chi doi test, khong doi ma san pham.
