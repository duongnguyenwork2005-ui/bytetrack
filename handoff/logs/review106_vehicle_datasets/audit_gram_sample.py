"""GT-only prefix audit; does not execute the downloaded devkit or read its results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
import zipfile

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_r106 as audit


def main():
    manifest = audit.read_json(HERE / 'gram_manifest.json')
    entry = manifest['annotation_kit']
    archive_path = HERE / entry['file']
    assert audit.sha(archive_path) == entry['sha256']
    # Selected by timeline, before this runner reads objects. One frame-0 XML
    # was inspected for schema earlier in the session; no scores were inspected.
    selected = [f'GRAM-RTMv4/Annotations/Urban1/xml/{frame}.xml' for frame in range(250)]
    listed = {row['name']: row for row in entry['manifest']}
    assert all(name in listed for name in selected)
    rows, duplicate_ids, source_hashes = [], [], {}
    classes, trajectories, occlusion = Counter(), Counter(), Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for frame, name in enumerate(selected):
            content = archive.read(name)  # ZipFile checks this member's CRC.
            assert len(content) == listed[name]['bytes']
            source_hashes[name] = hashlib.sha256(content).hexdigest()
            root = ET.fromstring(content)
            assert root.findtext('filename') == f'{frame}.jpg'
            assert (root.findtext('size/width'), root.findtext('size/height')) == ('600', '360')
            seen = set()
            for obj in root.findall('object'):
                ident = int(obj.findtext('ID'))
                if ident in seen:
                    duplicate_ids.append([frame, ident])
                seen.add(ident)
                box = [float(obj.findtext('bndbox/' + k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                x1, y1, x2, y2 = box
                assert 0 <= x1 < x2 <= 600 and 0 <= y1 < y2 <= 360
                classes[obj.findtext('class')] += 1
                trajectories[obj.findtext('trajectory')] += 1
                occlusion[obj.findtext('occluded', 'missing')] += 1
                rows.append([frame + 1, ident, x1, y1, x2-x1, y2-y1, 1, 1, -1])
    audit.sha(Path(__file__))
    result = dict(protocol='GT-only, first 250 consecutive Urban1 frames 0..249, selected by filename; no scores read',
                  schema_inspection='Frame-0 XML inspected before this runner for schema; no alternative prefix or tracking score inspected',
                  frames=250, fps_from_author=25, boxes=len(rows), ids=len({row[1] for row in rows}),
                  classes=dict(classes), trajectories=dict(trajectories), occluded_field=dict(occlusion),
                  duplicate_frame_ids=duplicate_ids, selected_member_sha256=source_hashes,
                  crc_scope='250 selected XML members only; full archive CRC not tested',
                  tracking='NOT_SCORED: no images or matching YOLO cache; no use of bundled baseline results',
                  coordinates='Source xyxy differences for GT image-angle audit only; official metric not adopted',
                  inputs=audit.INPUTS)
    if not duplicate_ids:
        gt = HERE / 'gram_Urban1_prefix250_gt.csv'
        np.savetxt(gt, rows, delimiter=',', fmt='%.10g')
        ids, present, groups, episodes = audit.labels_for_camera(gt, 'Urban1', 250, 25)
        result['angle_groups'] = {k: dict(boxes=int((v >= 0).sum()),
                                         episodes=sum(e['level'] == k for e in episodes)) for k, v in groups.items()}
        result['episodes'] = episodes
        result['derived_gt_sha256'] = audit.sha(gt)
    audit.write_json('gram_sample_gt_audit.json', result)
    print('GRAM GT-only:', result['frames'], 'frames;', result['boxes'], 'boxes;', result['ids'],
          'IDs;', len(duplicate_ids), 'duplicate frame/IDs;', result.get('angle_groups'), flush=True)


if __name__ == '__main__':
    main()
