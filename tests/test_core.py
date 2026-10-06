import unittest
from whisper_precision.core import corpus_wer, normalize, word_errors
from whisper_precision.data import duration_band
from whisper_precision.graphs import component_for_op
from whisper_precision.statistics import paired_ci


class CoreTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize("  Don't, STOP!  "), "DONT STOP")
        self.assertEqual(normalize("Caf\u00e9 42"), "CAF\u00c9 42")

    def test_word_errors(self):
        self.assertEqual(word_errors("A B C", "A X C"), (1, 3))
        self.assertEqual(word_errors("A B", "A B C"), (1, 2))
        self.assertEqual(word_errors("A B C", "A C"), (1, 3))
        self.assertEqual(corpus_wer([("A B", "A X"), ("C D", "C D")])["wer"], 0.25)

    def test_duration_bins(self):
        self.assertEqual(duration_band(9.99), "short")
        self.assertEqual(duration_band(10), "medium")
        self.assertEqual(duration_band(20), "long")
        self.assertEqual(duration_band(30), "long")
        self.assertIsNone(duration_band(30.01))

    def test_exported_graph_component_names(self):
        self.assertEqual(component_for_op("encoder", "MatMul",
                         "__module.layers.0.self_attn.q_proj/aten::linear/MatMul"),
                         "encoder_attention_qkv")
        self.assertEqual(component_for_op("decoder", "MatMul",
                         "__module.model.decoder.layers.2.encoder_attn.out_proj/aten::linear/MatMul"),
                         "decoder_cross_output")
        self.assertEqual(component_for_op("decoder", "Gather",
                         "__module.model.decoder.embed_tokens/aten::embedding/Gather"),
                         "decoder_token_embedding")

    def test_cluster_bootstrap_identical_predictions(self):
        predictions = {"predictions": [
            {"id": "a", "reference": "ONE TWO", "hypothesis": "ONE TWO"},
            {"id": "b", "reference": "THREE FOUR", "hypothesis": "THREE"},
        ]}
        clips = [{"id": "a", "speaker_id": 1}, {"id": "b", "speaker_id": 2}]
        self.assertEqual(paired_ci(predictions, predictions, clips, 1, 100), [0, 0])
