#!/usr/bin/env bash
# Tai HELMET (OSF 4pwj8) TUNG FILE (zip ca project cua OSF bi cat 2 lan), co resume (-C -), kiem sha256 voi API OSF.
# Dich: F:/LABOTARY_data/goc_tai_ve/HELMET (archive tai lai duoc -> o roi, LUU_TRU muc 5).
set -uo pipefail
PY=C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
L=$(dirname "$0"); DICH=/f/LABOTARY_data/goc_tai_ve/HELMET
$PY -I -c "
import json,sys
for f in sys.argv[1:]:
    for x in json.load(open(f,encoding='utf-8'))['data']:
        a=x['attributes']
        if a['kind']=='file': print(a['name'], x['links']['download'], a['extra']['hashes']['sha256'], a['size'])
" "$L/osf_helmet_root_listing.json" "$L/osf_helmet_image_listing.json" | sort -k4 -n | while read -r ten url sha size; do
  f="$DICH/$ten"
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$size" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; then
    echo "[bo qua] $ten da du + khop sha256"; continue; fi
  echo "[$(date +%T)] tai $ten ($size byte)"
  for lan in 1 2 3 4 5; do
    curl -sS -L -C - --retry 3 --retry-delay 10 -o "$f" "$url" && break
    echo "  lan $lan loi, thu tiep"; sleep 15
  done
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" = "$sha" ]; then echo "  KHOP sha256 $ten"; else echo "  LECH sha256 $ten ($got != $sha)"; fi
done
echo "[$(date +%T)] xong"
