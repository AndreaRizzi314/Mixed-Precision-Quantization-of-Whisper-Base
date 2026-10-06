import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from whisper_precision.io import checksum, read_json
from whisper_precision.search import (ablations, assert_disjoint, candidate_id,
    beam_proposals, choose_finalists, evaluation_key, immutable, passes, save, shortlist,
    specification, subset_result, cached_features, atomic_replace, file_digest)


class SearchTests(unittest.TestCase):
    def test_search_cannot_restart_after_unfrozen_test_predictions_exist(self):
        from whisper_precision.cli import assert_tests_unread
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assert_tests_unread(root)
            cache = root / "results" / "test_a" / "model.json"
            save(cache, {"predictions": [{"id": "consumed"}]})
            with self.assertRaisesRegex(RuntimeError, "Fresh test predictions already exist"):
                assert_tests_unread(root)

    def test_low_disk_space_stops_search_with_candidate_pending(self):
        from whisper_precision.cli import Study
        with tempfile.TemporaryDirectory() as directory:
            study = Study.__new__(Study)
            study.root = Path(directory)
            study.catalog = {"candidate": {"status": "pending"}}
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, "Insufficient model-export disk space"):
                    study.run("candidate")
            self.assertEqual(study.catalog["candidate"]["status"], "pending")

    def test_atomic_save_retries_transient_locks_without_masking_permanent_failure(self):
        with patch("whisper_precision.search.os.replace", side_effect=[PermissionError(), None]) as replace:
            with patch("whisper_precision.search.time.sleep"):
                atomic_replace(Path("source"), Path("target"))
            self.assertEqual(replace.call_count, 2)
        with patch("whisper_precision.search.os.replace", side_effect=PermissionError()) as replace:
            with patch("whisper_precision.search.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_replace(Path("source"), Path("target"))
            self.assertEqual(replace.call_count, 8)

    def test_feature_cache_preserves_tensors_and_invalidates_audio_or_revision(self):
        import torch
        inputs = {"input_features": torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
                  "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.int64)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, clip = {"model_revision": "one"}, {"audio_sha256": "audio_one"}
            with patch("whisper_precision.inference._audio_features", return_value=inputs) as process:
                first = cached_features(config, root, None, clip)
                second = cached_features(config, root, None, clip)
                self.assertEqual(process.call_count, 1)
                for key in inputs:
                    self.assertTrue(torch.equal(inputs[key], first[key]))
                    self.assertEqual(inputs[key].dtype, second[key].dtype)
                second["input_features"].zero_()
                self.assertTrue(torch.equal(cached_features(config, root, None, clip)["input_features"], inputs["input_features"]))
                cached_features(config, root, None, {"audio_sha256": "audio_two"})
                cached_features({"model_revision": "two"}, root, None, clip)
                self.assertEqual(process.call_count, 3)

    def test_manifest_disjointness_detects_duplicate_and_overlap(self):
        a = {"clips": [{"id": "a"}, {"id": "b"}]}
        assert_disjoint(a, {"clips": [{"id": "c"}]})
        with self.assertRaises(ValueError):
            assert_disjoint(a, {"clips": [{"id": "b"}]})
        with self.assertRaises(ValueError):
            assert_disjoint({"clips": [{"id": "a"}, {"id": "a"}]})

    def test_relative_wer_limit_uses_exact_edits(self):
        fp = {"errors": 91, "reference_words": 1852}
        self.assertTrue(passes({"errors": 95, "reference_words": 1852}, fp))
        self.assertFalse(passes({"errors": 96, "reference_words": 1852}, fp))
        with self.assertRaises(ValueError):
            passes({"errors": 95, "reference_words": 1900}, fp)

    def test_candidate_generation_is_order_independent_and_deduplicated(self):
        operations = {f"node{i}": {"component": f"group{i//2}"} for i in range(20)}
        candidates = ablations(operations)
        self.assertEqual(len(candidates), 20)
        self.assertEqual(len({candidate_id(spec) for _, spec in candidates}), 20)
        reversed_ops = dict(reversed(list(operations.items())))
        self.assertEqual([candidate_id(s) for _, s in candidates],
                         [candidate_id(s) for _, s in ablations(reversed_ops)])
        self.assertEqual(candidate_id(specification({"b": 8, "a": 4})),
                         candidate_id(specification({"a": 4, "b": 8})))

    def test_immutable_selection_and_atomic_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            immutable(path, {"label": "one"})
            immutable(path, {"label": "one"})
            with self.assertRaises(RuntimeError):
                immutable(path, {"label": "two"})
            self.assertEqual(read_json(path), {"label": "one"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_frozen_selection_rejects_changed_search_records(self):
        from whisper_precision.cli import verify_selection
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"purpose": "test"}
            records = {"development": {"clips": [{"id": "dev"}]},
                       "screen": {"clips": [{"id": "dev"}]},
                       "test_a": {"clips": [{"id": "a"}]},
                       "test_b": {"clips": [{"id": "b"}]}}
            reservations = {"config_sha256": checksum(config)}
            for name, value in records.items():
                save(root / f"{name}_manifest.json", value)
                reservations[f"{name}_sha256"] = checksum(value)
            save(root / "reservations.json", reservations)
            choice = {"config_sha256": checksum(config)}
            for name in ("reservations", "candidates", "screen_results", "development_results", "search_complete", "preflight", "activation_scores", "audio_integrity"):
                value = reservations if name == "reservations" else {"label": "unchanged"}
                save(root / f"{name}.json", value)
                choice[f"{name}_sha256"] = checksum(value)
            source = root / "evaluation.py"
            source.write_text("original", encoding="utf-8")
            choice["algorithm_sources_sha256"] = {str(source): file_digest(source)}
            save(root / "selection.json", choice)
            verify_selection(config, root)
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Frozen evaluation source changed"):
                verify_selection(config, root)
            source.write_text("original", encoding="utf-8")
            save(root / "candidates.json", {"label": "changed"})
            with self.assertRaisesRegex(RuntimeError, "Frozen candidates changed"):
                verify_selection(config, root)

    def test_resume_key_changes_with_audio_model_or_decode(self):
        config = {"model_revision": "revision", "max_new_tokens": 128}
        manifest = {"clips": [{"id": "a", "text": "reference", "audio_sha256": "first"}]}
        key = evaluation_key(config, manifest, "model")
        self.assertEqual(key, evaluation_key(config, manifest, "model"))
        self.assertNotEqual(key, evaluation_key(config, manifest, "other_model"))
        self.assertNotEqual(key, evaluation_key({**config, "max_new_tokens": 256}, manifest, "model"))
        changed = {"clips": [{"id": "a", "text": "reference", "audio_sha256": "second"}]}
        self.assertNotEqual(key, evaluation_key(config, changed, "model"))

    def test_resume_key_changes_when_evaluation_sources_change(self):
        config = {"model_revision": "revision", "max_new_tokens": 128}
        manifest = {"clips": [{"id": "a", "text": "reference"}]}
        before = evaluation_key(config, manifest, "model")
        with patch("whisper_precision.search.file_digest", return_value="changed source"):
            self.assertNotEqual(before, evaluation_key(config, manifest, "model"))

    def test_subset_cache_preserves_order_and_rejects_reference_drift(self):
        predictions = [{"id": "a", "reference": "ONE TWO", "hypothesis": "ONE"},
                       {"id": "b", "reference": "THREE", "hypothesis": "THREE"}]
        row = {"predictions": predictions, "serialized_bytes": 10}
        result = subset_result(row, [{"id": "b", "text": "THREE"}, {"id": "a", "text": "ONE TWO"}])
        self.assertEqual([x["id"] for x in result["predictions"]], ["b", "a"])
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["reference_words"], 3)
        with self.assertRaises(RuntimeError):
            subset_result(row, [{"id": "a", "text": "CHANGED"}])

    def test_selection_prefers_smallest_with_development_margin(self):
        fp = {"errors": 100, "reference_words": 1000}
        rows = {"small_fragile": {"errors": 104, "reference_words": 1000, "wer": .104, "serialized_bytes": 50},
                "middle_robust": {"errors": 102, "reference_words": 1000, "wer": .102, "serialized_bytes": 60},
                "large_robust": {"errors": 100, "reference_words": 1000, "wer": .100, "serialized_bytes": 70}}
        labels, rule = choose_finalists(rows, fp)
        self.assertEqual(labels, ["middle_robust", "large_robust"])
        self.assertEqual(rule, "within_2%_dev")

    def test_frontier_shortlist_keeps_accuracy_and_size_extremes(self):
        rows = {"small": {"wer": .2, "serialized_bytes": 50},
                "good": {"wer": .1, "serialized_bytes": 60},
                "dominated": {"wer": .2, "serialized_bytes": 70}}
        self.assertEqual(shortlist(rows, 2, .1), ["small", "good"])

    def test_runner_is_reserved_when_only_one_candidate_has_two_percent_margin(self):
        fp = {"errors": 100, "reference_words": 1000}
        rows = {"robust": {"errors": 102, "reference_words": 1000, "wer": .102, "serialized_bytes": 60},
                "fallback": {"errors": 104, "reference_words": 1000, "wer": .104, "serialized_bytes": 50},
                "bad": {"errors": 120, "reference_words": 1000, "wer": .12, "serialized_bytes": 40}}
        labels, rule = choose_finalists(rows, fp)
        self.assertEqual(labels, ["robust", "fallback"])
        self.assertIn("runner_within_5%_dev", rule)

    def test_beam_is_deterministic_and_avoids_known_allocations(self):
        operations = {f"node{i}": {"component": f"group{i//2}"} for i in range(20)}
        catalog, rows = {}, {}
        for index, (_, spec) in enumerate(ablations(operations)):
            label = candidate_id(spec)
            catalog[label] = {"spec": spec}
            count = sum(b == 4 for b in spec["assignments"].values())
            rows[label] = {"wer": .05 + count*.0001 + (index%3)*.0002,
                           "serialized_bytes": int((76-count)*1e6)}
        left = beam_proposals(catalog, rows, operations, set(catalog), .05)
        right = beam_proposals(catalog, rows, operations, set(catalog), .05)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 6)
        self.assertTrue(all(candidate_id(s) not in catalog for _,s in left))


if __name__ == "__main__":
    unittest.main()
