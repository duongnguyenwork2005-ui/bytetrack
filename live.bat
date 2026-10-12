@echo off
REM live.bat -- Xem TRUC TIEP tracking (KF+CV | EKF+CTRV | UKF+CTRV) tren mot video bat ky.
REM
REM   live MVI_39031                                  sequence UA-DETRAC (train/test)
REM   live C:\videos\xe.mp4                           file video bat ky
REM   live MVI_40991 --track-buffer 90 --start 1600 --end 1850 --speed 0.5
REM
REM Trong PowerShell (terminal VS Code) go:  .\live MVI_39031
REM Trong cmd:                               live MVI_39031
REM Phim trong cua so: SPACE dung/tiep, n tien 1 frame, s chup man hinh, +/- nhanh cham, q thoat.
setlocal
set "PY=C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe"
set "PYTHONWARNINGS=ignore"
cd /d "%~dp0"

if "%~1"=="" (
  echo Cach dung:
  echo   live MVI_39031
  echo   live C:\videos\xe.mp4
  echo   live MVI_40991 --track-buffer 90 --start 1600 --end 1850 --speed 0.5
  echo Tuy chon: --models cv ekf_ctrv   --conf 0.25   --trail 30   --no-lost   --save outputs\live\ten.mp4
  exit /b 1
)
if not exist "%PY%" (
  echo Khong thay python cua env khoaluan-mot tai %PY%
  echo Hay chay:  conda activate khoaluan-mot ^&^& python src\demo\live_view.py --video %1
  exit /b 1
)

set "ARG=%~1"
shift
if exist "%ARG%" (
  "%PY%" src\demo\live_view.py --file "%ARG%" %1 %2 %3 %4 %5 %6 %7 %8 %9
) else (
  "%PY%" src\demo\live_view.py --video %ARG% %1 %2 %3 %4 %5 %6 %7 %8 %9
)
endlocal
