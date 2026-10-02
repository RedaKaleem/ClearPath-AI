import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate_model as evaluation
from evaluation_protocol import load_metadata, protocol_issues, slice_metrics, presentation_wording


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root / 'metadata.csv'
        with self.metadata.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['split', 'image', 'group_id', 'lighting'])
            for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
                for kind in ['images', 'labels']:
                    (self.root / folder / kind).mkdir(parents=True)
                for name, label in [('positive', '0 0.5 0.5 0.2 0.2'), ('negative', '')]:
                    (self.root / folder / 'images' / f'{name}.jpg').write_bytes(f'{split}/{name}'.encode())
                    (self.root / folder / 'labels' / f'{name}.txt').write_text(label)
                    writer.writerow([split, f'{name}.jpg', f'{split}-{name}', 'night' if name == 'negative' else 'day'])
        self.names = {0: 'ambulance', 1: 'siren'}
        (self.root / 'data.yaml').write_text('names: [ambulance, siren]\n')

    def audit(self):
        return evaluation.audit_dataset(self.root, self.names)

    def test_independent_mixed_dataset_passes(self):
        _, records, overlaps = self.audit()
        metadata, groups = load_metadata(self.metadata, records)
        self.assertEqual(protocol_issues(records, 'test', {0}, overlaps, metadata, groups), [])

    def test_group_leakage_is_detected(self):
        self.metadata.write_text(self.metadata.read_text().replace('test-positive', 'train-positive'))
        _, records, overlaps = self.audit()
        metadata, groups = load_metadata(self.metadata, records)
        self.assertEqual(groups, {'train-positive': ['test', 'train']})
        self.assertIn('Original scene/video groups cross splits.',
                      protocol_issues(records, 'test', {0}, overlaps, metadata, groups))

    def test_incomplete_duplicate_and_unknown_metadata_rejected(self):
        original = self.metadata.read_text()
        _, records, _ = self.audit()
        for contents in ['\n'.join(original.splitlines()[:-1]),
                         original + original.splitlines()[1] + '\n',
                         original.replace('test,positive.jpg', 'invalid,positive.jpg')]:
            with self.subTest(contents=contents):
                self.metadata.write_text(contents)
                with self.assertRaises(ValueError):
                    load_metadata(self.metadata, records)

    def test_exact_duplicates_and_missing_labels(self):
        (self.root / 'test/images/positive.jpg').write_bytes(b'train/positive')
        self.assertEqual(len(self.audit()[2]), 1)
        (self.root / 'test/labels/negative.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing annotation'):
            self.audit()

    def test_target_specific_controls_and_undefined_metrics(self):
        _, records, _ = self.audit()
        records['test'][0]['classes'] = [1]  # siren-only is negative for ambulance
        issues = protocol_issues(records, 'test', {0}, [], {}, {})
        self.assertEqual(issues, [])
        issues = protocol_issues(records, 'test', {0, 1}, [], {}, {})
        self.assertIn('no target-negative', issues[0])
        self.assertIsNone(evaluation.binary_metrics(0, 0, 0, 2)['recall'])

    def test_slices_preserve_support_and_missing_denominators(self):
        result = slice_metrics([dict(outcome='FN', lighting='night'), dict(outcome='FP', lighting='day')], evaluation.binary_metrics)
        self.assertEqual(result['lighting']['night']['metrics']['recall'], 0)
        self.assertIsNone(result['lighting']['night']['metrics']['false_positive_rate'])
        self.assertEqual(result['lighting']['day']['negative_images'], 1)
        self.assertEqual(result['source']['unknown']['images'], 2)

    def test_wording_uses_actual_split_and_does_not_invent_checkpoint(self):
        wording = presentation_wording('val', 2, .5, 'app', {0})
        self.assertIn('2 annotated val images', wording)
        self.assertNotIn('pretrained', wording)
        self.assertNotIn('lacks negative', wording)

    def test_audit_only_needs_no_checkpoint(self):
        argv = ['evaluate_model.py', '--data', str(self.root / 'data.yaml'), '--metadata', str(self.metadata), '--audit-only', '--strict-protocol']
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()) as output:
            evaluation.main()
        self.assertTrue(json.loads(output.getvalue())['protocol']['checks_passed'])

    def test_strict_failure_precedes_checkpoint_loading(self):
        argv = ['evaluate_model.py', '--data', str(self.root / 'data.yaml'), '--strict-protocol']
        with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            evaluation.main()
        self.assertEqual(raised.exception.code, 2)

    def test_inference_pipeline_with_stub_and_target_filter(self):
        # Synthetic model outputs test plumbing only; never represented as measured model quality.
        import numpy as np
        class FakeModel:
            names = {0: 'ambulance', 1: 'siren'}
            def predict(self, image, **kwargs):
                ids = [0] if Path(image).stem == 'positive' else [1]
                return [SimpleNamespace(names=self.names, boxes=SimpleNamespace(
                    cls=SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: ids))))]
            def val(self, **kwargs):
                return SimpleNamespace(box=SimpleNamespace(mp=1., mr=1., f1=np.array([1.]),
                    map50=1., map=1., ap_class_index=[0], p=[1.], r=[1.], ap50=[1.], ap=[1.]))
        model = self.root / 'fake.pt'
        model.write_bytes(b'synthetic-test-only')
        argv = ['evaluate_model.py', '--data', str(self.root / 'data.yaml'), '--model', str(model),
                '--metadata', str(self.metadata), '--output', str(self.root / 'results'),
                '--mode', 'detector', '--target-ids', '0', '--strict-protocol']
        with patch('sys.argv', argv), patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda _: FakeModel(), __version__='test-stub')}), contextlib.redirect_stdout(io.StringIO()):
            evaluation.main()
        run = next((self.root / 'results').iterdir())
        metrics = json.loads((run / 'metrics.json').read_text())
        self.assertEqual(metrics['image_level_metrics']['tn'], 1)
        self.assertEqual(metrics['image_level_metrics']['tp'], 1)
        self.assertTrue(metrics['protocol']['checks_passed'])
        self.assertEqual(len((run / 'failure_cases.csv').read_text().splitlines()), 1)
        self.assertEqual(metrics['image_alert_slices']['lighting']['night']['negative_images'], 1)
        self.assertFalse(Path(json.loads((run / 'test_manifest.json').read_text())[0]['image']).is_absolute())


if __name__ == '__main__':
    unittest.main()
