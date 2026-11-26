# SVMOT Zenodo Release v1.0

## Overview

This release contains the **SVMOT** dataset, a multi-object tracking dataset for
traffic scenes with a high proportion of **stationary vehicles** and frequent
**object occlusions**.

The dataset is designed for benchmarking multi-object tracking methods in urban
traffic surveillance, particularly in scenarios involving long-term stationary
vehicles.

The dataset follows the standard MOTChallenge structure and extends it with:

- visibility annotations,
- pairwise occlusion annotations,
- a per-object `stopped` attribute.

This Zenodo record is intended as the archival release of the dataset for
research use, citation, and long-term access.

## Files Included In This Release

The Zenodo release contains the following primary files:

- `svmot_data_v1.0.zip` - full dataset archive
- `README_RELEASE_v1.0.md` - release documentation
- `train.txt` - recommended training split
- `test.txt` - recommended test split
- `by_scene.json` - mapping between scenes and sequences, plus split metadata
- `SHA256SUMS` - checksums for release files

The dataset license is specified in the Zenodo metadata for this record.

## Dataset Archive Structure

The main archive `svmot_data_v1.0.zip` preserves the dataset directory
structure. Its internal layout is expected to be:

```text
data/
  seq_01_evening/
    img/
    gt/
      gt.txt
    det/
      det.txt
    occlusion.json
    roi_mask.png
    seqinfo.ini
  seq_04_morning/
    ...
classes.txt
```

Each sequence follows the MOTChallenge layout with additional files for
occlusion relationships and ROI masks.

## Official Splits

The official benchmark partitions are defined by the accompanying split files,
not by physically separating the dataset into different archives.

- `train.txt` lists sequence names used for training
- `test.txt` lists sequence names used for testing
- `by_scene.json` documents scene membership and is provided to help avoid
  scene leakage between splits

Because multiple sequences may come from the same physical scene, recommended
splits should be constructed at the **scene level**, not by random assignment of
individual sequences.

This release defines a single official train/test split. No separate validation split is included.

## Recommended Usage

1. Download all files from this release.
2. Verify checksums using `SHA256SUMS`.
3. Extract `svmot_data_v1.0.zip`.
4. Use the provided split files for benchmarking and reporting results.
5. Refer to the companion repository for annotation and evaluation details.

## Integrity Verification

Example checksum verification workflow:

```bash
sha256sum -c SHA256SUMS
```

On Windows PowerShell, checksums can be computed with:

```powershell
Get-FileHash svmot_data_v1.0.zip -Algorithm SHA256
```

## Documentation

This release provides only the archival dataset files.

Complete documentation, annotation guidelines, data format description,
and evaluation instructions are maintained in the companion project repository
referenced from the Zenodo record.

The primary files included in this release are:

- `README_RELEASE_v1.0.md`
- `train.txt`
- `test.txt`
- `by_scene.json`

## License And Reuse

This dataset is distributed under the license selected in the Zenodo metadata
for this record.

Users must review the license terms before redistribution, adaptation, or use
in downstream products and services.

## Citation

If you use this dataset in published research, please cite the corresponding Zenodo record.

Citation metadata is provided in the Zenodo record metadata.

DOI: `10.5281/zenodo.19468203`

## Contact

For questions regarding the dataset, annotation format, or reuse conditions:

Mikhail Parshikov  
Moscow Institute of Physics and Technology (MIPT)
parshikov.mm@phystech.edu
