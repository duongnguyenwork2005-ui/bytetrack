"""Package every frozen dev image and label for real TensorRT validation."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data/yolo_mixed5_v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    assert not (HERE/'inputs.json').exists();target=HERE/'dev_images.tar';assert not target.exists()
    manifest=DATA/'manifest/anh.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='dev']
    assert len(rows)==7190 and len({r['anh_ra'] for r in rows})==7190
    total=sum((DATA/r['anh_ra']).stat().st_size for r in rows);assert total<4*1024**3
    images=[];hashes={str(p):sha(p) for p in (Path(__file__),HERE/'00_protocol.md',manifest,
          HERE/'capture.py',HERE/'score.py',HERE.parent/'review151_resolution_tradeoff/inputs.json',
          HERE.parent/'review150_nano_pruning/source_snapshot/nano_detector.py')}
    with tarfile.open(target,'x',dereference=True) as archive:
        for index,row in enumerate(rows):
            image=DATA/row['anh_ra'];label=DATA/row['nhan_ra']
            assert sha(image)==row['sha256_ra'] and sha(label)==row['sha256_nhan']
            hashes[str(image)]=row['sha256_ra'];hashes[str(label)]=row['sha256_nhan']
            boxes=[[float(x) for x in line.split()] for line in label.read_text().splitlines() if line]
            assert all(len(b)==5 and b[0] in range(5) and all(0<=v<=1 for v in b[1:]) for b in boxes)
            images.append({'index':index,'relative':row['anh_ra'],'sha256':row['sha256_ra'],
                  'shape':[int(row['h']),int(row['w'])],'video':row['video'],'GT_xywh_normalized':boxes})
            archive.add(image,arcname=row['anh_ra'],recursive=False)
            if (index+1)%1000==0:print('PACKAGE',index+1,'/7190',flush=True)
    dataset={'images':images,'classes':{0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'},
             'image_bytes':total,'archive_sha256':sha(target),'source':'all mixed5 dev manifest images'}
    write(HERE/'dataset.json',dataset);hashes[str(HERE/'dataset.json')]=sha(HERE/'dataset.json')
    hashes[str(target)]=dataset['archive_sha256']
    write(HERE/'inputs.json',{'sha256':hashes,'images':7190,'archive_bytes':target.stat().st_size,
                              'Nano_detection_AP_measured':False})
    print('RUNTIME DATASET LOCKED',7190,target.stat().st_size,flush=True)


if __name__=='__main__':main()
