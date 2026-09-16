"""Cache write regressions from review round 3; no GPU or Ultralytics required.

Uses actual cache IO and the actual main() entrypoint. CLI tests replace model
loading/inference with doubles and temporary checkpoint bytes, not real YOLO.
Run: python src/test_cache_provenance.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_cache as dc
from mot_datasets import FrameRef


def content_hash(data):
    return hashlib.sha256(data).hexdigest()


def files_under(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def runtime_modules(loader):
    ultra = types.ModuleType('ultralytics')
    ultra.__version__ = 'test-double'
    ultra.YOLO = loader
    torch = types.ModuleType('torch')
    torch.__version__ = 'test-double'
    torch.version = types.SimpleNamespace(cuda=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    return {'ultralytics': ultra, 'torch': torch}


class CacheProvenanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='cache_provenance_')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.caches = self.root / 'caches'
        self.caches.mkdir()
        cache_root = patch.object(dc, 'CACHE_ROOT', self.caches)
        cache_root.start()
        self.addCleanup(cache_root.stop)
        self.arrays = dict(frame_idx=np.array([1, 2], np.int32), n_per_frame=np.array([1, 0], np.int32),
                           unreadable=np.zeros(2, bool), truncated=np.zeros(2, bool),
                           time_s=np.array([0., .04]),
                           data=np.array([[10, 10, 30, 30, .8, 2]], np.float32),
                           orig_shape=np.array([480, 640], np.int32))
        self.synthetic = dc.DetectorSpec(backend='gt-synthetic', weights='unused.pt')
        self.known = dc.DetectorSpec(weights='A.pt', weights_sha256=content_hash(b'checkpoint A'))

    def test_old_hash_append_is_rejected_without_changes(self):
        directory = self.caches / 'old'
        directory.mkdir()
        # Exact hash definition from the preceding release (4889209).
        old_fields = {k: v for k, v in asdict(self.synthetic).items()
                      if k not in ('created', 'git_commit', 'part')}
        old_hash = content_hash(json.dumps(old_fields, sort_keys=True, default=str).encode())
        self.assertNotEqual(old_hash, self.synthetic.spec_sha256())
        metadata = dict(asdict(self.synthetic), spec_sha256=old_hash)
        (directory / 'meta.json').write_text(json.dumps(metadata), encoding='utf-8')
        dc.save_video(directory, 'v1', self.arrays, extra={'spec_sha256': old_hash})
        before = files_under(directory)
        with self.assertRaisesRegex(RuntimeError, 'spec_sha256'):
            dc.write_meta(directory, self.synthetic)
        self.assertEqual(files_under(directory), before)
        self.assertEqual(len(dc.open_cache('old', 'v1', .25, [2], 300)), 2)

    def test_missing_or_invalid_checkpoint_hash_rejects_create(self):
        for digest in (None, '', 'abc', 'g' * 64):
            with self.subTest(digest=digest):
                directory = self.caches / 'unhashed'
                with self.assertRaisesRegex(RuntimeError, 'weights_sha256'):
                    dc.write_meta(directory, replace(self.known, weights_sha256=digest))
                self.assertFalse(directory.exists())

    def test_unhashed_checkpoint_paths_cannot_share_existing_cache(self):
        directory = self.caches / 'legacy_yolo'
        directory.mkdir()
        a = replace(self.known, weights_sha256=None)
        (directory / 'meta.json').write_text(json.dumps(asdict(a)), encoding='utf-8')
        dc.save_video(directory, 'v1', self.arrays)
        before = files_under(directory)
        with self.assertRaisesRegex(RuntimeError, 'weights_sha256'):
            dc.write_meta(directory, replace(a, weights='B.pt'))
        self.assertEqual(files_under(directory), before)

    def test_same_checkpoint_hash_after_move_roundtrips(self):
        directory = self.caches / 'known'
        dc.write_meta(directory, self.known)
        before = files_under(directory)
        moved = replace(self.known, weights='elsewhere/A.pt')
        dc.write_meta(directory, moved)
        self.assertEqual(files_under(directory), before)
        dc.save_video(directory, 'v1', self.arrays, spec=moved)
        self.assertEqual(dc.sequence_spec_status(directory, 'v1')[0], 'ok')
        self.assertEqual(len(dc.open_cache('known', 'v1', .25, [2], 300)), 2)

    def test_different_known_checkpoint_hash_rejects_append(self):
        directory = self.caches / 'known'
        dc.write_meta(directory, self.known)
        dc.save_video(directory, 'v1', self.arrays, spec=self.known)
        before = files_under(directory)
        with self.assertRaisesRegex(RuntimeError, 'weights_sha256'):
            dc.write_meta(directory, replace(self.known, weights='B.pt', weights_sha256=content_hash(b'B')))
        self.assertEqual(files_under(directory), before)

    def test_synthetic_cache_does_not_require_checkpoint(self):
        directory = self.caches / 'synthetic'
        dc.write_meta(directory, self.synthetic)
        dc.save_video(directory, 'v1', self.arrays, spec=self.synthetic)
        self.assertEqual(dc.sequence_spec_status(directory, 'v1')[0], 'ok')
        self.assertEqual(len(dc.open_cache('synthetic', 'v1', .25, [2], 300)), 2)

    def test_legacy_cache_does_not_gain_retrospective_hash(self):
        directory = self.caches / 'legacy'
        directory.mkdir()
        (directory / 'meta.json').write_text(json.dumps(asdict(self.synthetic)), encoding='utf-8')
        before = files_under(directory)
        dc.write_meta(directory, self.synthetic)
        self.assertEqual(files_under(directory), before)

    def run_cli(self, loader, build=None, model_path=None):
        dataset = types.SimpleNamespace(name='fixture', orig_shape=(480, 640),
            frames=lambda *args: [FrameRef(1, None, 0.)], ignored_regions=lambda *args: [])
        argv = ['det_cache.py', '--dataset', 'detrac', '--source', 'yolo', '--cache-id', 'cli',
                '--videos', 'v1', '--model', str(model_path or self.root / 'alias.pt')]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, runtime_modules(loader)))
            stack.enter_context(patch.object(dc, 'get_dataset', return_value=dataset))
            stack.enter_context(patch.object(dc, 'git_commit', return_value='fixture'))
            stack.enter_context(patch.object(sys, 'argv', argv))
            inference = stack.enter_context(patch.object(dc, 'build_video_yolo',
                                                          side_effect=build or (lambda *args: self.arrays)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = dc.main()
        return result, inference.call_count

    def test_cli_resolves_downloaded_checkpoint_before_cache_write(self):
        checkpoint = self.root / 'downloaded' / 'actual.pt'
        expected = b'actual checkpoint bytes supplied by the loader'
        model = types.SimpleNamespace(ckpt_path=str(checkpoint))
        events = []

        def load(requested):
            self.assertFalse((self.caches / 'cli').exists())
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(expected)
            events.append('loaded')
            return model

        def build(loaded, frames, regions, spec, shape):
            self.assertIs(loaded, model)
            self.assertEqual(events, ['loaded'])
            metadata = dc.load_meta(self.caches / 'cli')
            self.assertEqual(metadata['weights_sha256'], content_hash(expected))
            self.assertEqual(Path(metadata['weights']), checkpoint.resolve())
            self.assertEqual(spec.weights_sha256, content_hash(expected))
            events.append('inference')
            return self.arrays

        result, count = self.run_cli(load, build)
        self.assertEqual((result, count), (0, 1))
        self.assertEqual(events, ['loaded', 'inference'])
        self.assertEqual(len(dc.open_cache('cli', 'v1', .25, [2], 300)), 2)

    def test_cli_hashes_resolved_checkpoint_not_input_alias_file(self):
        alias = self.root / 'alias.pt'; alias.write_bytes(b'not the loaded checkpoint')
        actual = self.root / 'actual.pt'; actual.write_bytes(b'loaded checkpoint')
        result, count = self.run_cli(lambda name: types.SimpleNamespace(ckpt_path=str(actual)), model_path=alias)
        self.assertEqual((result, count), (0, 1))
        self.assertEqual(dc.load_meta(self.caches / 'cli')['weights_sha256'], content_hash(actual.read_bytes()))

    def test_cli_no_resolved_checkpoint_leaves_cache_absent(self):
        for path in (None, str(self.root / 'missing.pt')):
            with self.subTest(path=path):
                with self.assertRaisesRegex(RuntimeError, 'checkpoint'):
                    self.run_cli(lambda name: types.SimpleNamespace(ckpt_path=path))
                self.assertFalse((self.caches / 'cli').exists())

    def test_fill_env_checks_declared_hash_against_actual_file(self):
        checkpoint = self.root / 'actual.pt'; checkpoint.write_bytes(b'checkpoint changed')
        spec = replace(self.known, weights=str(checkpoint))
        with patch.dict(sys.modules, runtime_modules(None)), patch.object(dc, 'git_commit', return_value='fixture'):
            with self.assertRaisesRegex(RuntimeError, 'weights_sha256'):
                spec.fill_env()

    def test_fill_env_rejects_unavailable_checkpoint(self):
        spec = replace(self.known, weights=str(self.root / 'missing.pt'), weights_sha256=None)
        with patch.dict(sys.modules, runtime_modules(None)), patch.object(dc, 'git_commit', return_value='fixture'):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint'):
                spec.fill_env()


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CacheProvenanceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed_cases = {getattr(test, 'test_case', test).id()
                    for test, _ in result.failures + result.errors}
    passed = result.testsRun - len(failed_cases) - len(result.skipped)
    print(f'KET QUA: {passed}/{result.testsRun} test cases PASS; '
          f'failing_cases={len(failed_cases)}, assertion_failures={len(result.failures)}, '
          f'errors={len(result.errors)}, skipped={len(result.skipped)}')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
