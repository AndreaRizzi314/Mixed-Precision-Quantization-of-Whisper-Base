import unittest

from whisper_precision.prepare import workspace_path
from whisper_precision.verify import check_scores
from whisper_precision.core import corpus_wer
from whisper_precision.statistics import paired_ci


class EvidenceTests(unittest.TestCase):
    def test_audit_rejects_changed_reference_and_stored_errors(self):
        clips = [{"id": "a", "text": "ONE TWO"}]
        predictions = [{"id": "a", "reference": "ONE TWO", "hypothesis": "ONE"}]
        row = {**corpus_wer([("ONE TWO", "ONE")]), "predictions": predictions}
        check_scores(row, clips)
        with self.assertRaisesRegex(ValueError, "Stored errors"):
            check_scores({**row, "errors": 0}, clips)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            check_scores(row, [{"id": "a", "text": "ONE THREE"}])

    def test_bootstrap_rejects_unpaired_predictions(self):
        base = {"predictions": [{"id": "a", "reference": "ONE", "hypothesis": "ONE"}]}
        candidate = {"predictions": [{"id": "b", "reference": "ONE", "hypothesis": "ONE"}]}
        with self.assertRaisesRegex(RuntimeError, "do not align"):
            paired_ci(base, candidate, [{"id": "a", "speaker_id": 1}], 0, 100)

    def test_bootstrap_requires_matching_reference_text(self):
        base = {"predictions": [{"id": "a", "reference": "ONE", "hypothesis": "ONE"}]}
        candidate = {"predictions": [{"id": "a", "reference": "TWO", "hypothesis": "ONE"}]}
        with self.assertRaisesRegex(RuntimeError, "do not align"):
            paired_ci(base, candidate, [{"id": "a", "speaker_id": 1}], 0, 100)

    def test_input_restore_rejects_paths_outside_workspace(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            workspace_path("../outside.flac")
