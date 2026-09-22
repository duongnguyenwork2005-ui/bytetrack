# CODEX review — `20207ec`

## Verdict: REQUEST CHANGES

This delta correctly addresses the most dangerous round-14 behavior on the successful path: a normal
cache replay writes only `review-*` outputs, completes CV/EKF smoke and full parity, and leaves all four
`pc-ref-*` trees byte-for-byte unchanged. The delta itself is reproducible: its SHA-256 is
`cefec5fdddb6f0548229218ae7b838d0c41f955409c64ae3179cf5525ac5644d`; all 43 target payloads verified;
the patch applied cleanly to `a5eec39` and reconstructed the target tree.

It is not ready to run on the Nano yet. Three preflight claims in `NANO_START_HERE.md` and
`nano_run_cache.py` are not true under corrupt or incomplete inputs. These are small, local fixes, but
they are safety gates for a device handoff and should be fixed before copying/running package v3.

No Nano evidence was produced in this round: all device-side tracker, TensorRT, and timing claims remain
unverified. The PC TensorRT surrogate tests do not substitute for a Nano run.

## Independent evidence

I built a temporary package-shaped tree with two valid three-frame caches and genuine PC references made
by the target's `run_tracking.py` using the NumPy backend. I then invoked the target's
`nano_run_cache.py` as a separate process, with Torch and Ultralytics blocked exactly as the script does.

| Scenario | Observed result | Assessment |
|---|---|---|
| Valid inputs, `--tag review` | exit 0; CV/EKF smoke and full parity pass; four `review-*` outputs created; PC reference hashes unchanged | Pass |
| Missing `NANO-full/pc-ref-ekf/data/MVI_39031.txt` | exit 2; no `badmissing-*` output | Pass |
| Invalid cache sidecar `MVI_39031.json` | exit 1 after entering tracking/parity; partial `badside-*` output exists | Fail |
| `lap` absent from the runtime | `nano_env_check.py` exits nonzero, but `nano_run_cache.py` continues into smoke; later exit 1 | Fail |
| Cache directories contain no NPZ and four reference directories contain no `.txt` | standalone `nano_env_check.py` prints `DAT (1-4)` and exits 0 | Fail |

`src/test_nano_parity.py` also passes independently, 20/20, under the NumPy backend.

## Findings

### N15-P1 — corrupt sidecar is not rejected before tracking

`nano_run_cache.py` checks only that `<video>.json` is a file. It parses `meta.json` and opens the NPZ,
but never parses the sidecar. With sidecar text `{ bad json`, step 0 reports inputs sufficient; the
tracker creates `NANO-smoke/badside-cv` before `open_cache()` raises `JSONDecodeError`. This contradicts
the documented contract: missing or invalid input must be exit 2 and must not run.

**Required fix:** parse every required sidecar in step 0 and append an input error on any parse/schema
failure. Do this before hash/output creation and add a regression test that asserts exit 2 and no
`<tag>-*` directory for malformed JSON.

### N15-P2 — failed environment check is recorded but not enforced

The call to `nano_env_check.py` stores only `0_env_check_exit`; a nonzero return code is never added to
`thieu`. When `lap` is unavailable, the stated environment check fails, yet `nano_run_cache.py` enters
the smoke loop and fails later in tracker code. This is neither the promised exit 2 nor the promised
pre-run failure boundary.

**Required fix:** make a nonzero environment-check return code an input failure in step 0, preserve its
tail in the JSON/log, and return exit 2 without attempting `run_tracking.py`. Add a regression using a
runtime or controlled helper with missing `lap`.

### N15-P3 — the standalone environment checker still accepts unusable assets

This was the original round-14 preflight defect and remains in `nano_env_check.py`. It iterates any NPZ
that happens to exist and treats any `pc-ref-*` directory as a valid reference even when it contains
zero data files. With both required NPZ files removed and each `pc-ref-*` data file removed, it prints
four references with `(0 file)` but finishes `DAT (1-4)`, exit 0. The guide tells the user that exit 0
means the cache/reference inputs are readable, so this is misleading even though the newer wrapper catches
some of it.

**Required fix:** share one explicit validation contract between `nano_env_check.py` and
`nano_run_cache.py`: require both named cache IDs, valid `meta.json`, the named NPZ with required arrays
and consistent counts/timeline, parseable sidecar, and exactly the four named readable reference files.
The standalone checker must return nonzero if any is absent or malformed. Add its negative regression.

## Acceptance conditions for the next delta

1. Add the three negative tests above, register them in `run_all_tests.py`, and show they fail on
   `20207ec` and pass after the fix.
2. Run the normal `nano_run_cache` integration fixture too: smoke/full CV/EKF parity must pass and the
   SHA-256 tree of `pc-ref-*` must be unchanged.
3. Rebuild package v3 (or v4) and verify the actual archive on two PC dependency stacks as before.
4. Keep all research locks, UA-DETRAC/BDD results, private files, and existing references unchanged.

After that review passes, connect the Nano using its current IP and run only milestone 1 first:
manifest verification, environment check, NumPy backend import, `nano_run_cache`, then parity. Do not
build TensorRT or report Nano speed until the cache replay parity passes on the actual device.
