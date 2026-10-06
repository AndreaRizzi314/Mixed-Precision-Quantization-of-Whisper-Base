import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from whisper_precision.quantization import effective_bits, fake_quantize, parameter_name


class QuantizationTests(unittest.TestCase):
    def test_saved_packed_bit_types_are_read_from_weight_inputs(self):
        import openvino as ov
        from openvino import opset13 as ops
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_node = ops.parameter([1, 8], ov.Type.f32)
            results, rows = [], []
            for bit in (2, 3, 4, 8):
                packed = ops.constant(ov.Tensor(getattr(ov.Type, f"u{bit}"), [8, 8]))
                restored = ops.multiply(ops.convert(packed, ov.Type.f32), ops.constant(0.25, ov.Type.f32))
                product = ops.matmul(input_node, restored, False, False)
                product.set_friendly_name(f"weight_{bit}")
                results.append(product)
                rows.append({"name": f"weight_{bit}", "graph": "test"})
            # Exact type matching must not confuse int32 with packed int3.
            wide = ops.matmul(input_node, ops.convert(ops.constant(ov.Tensor(ov.Type.i32, [8, 8])), ov.Type.f32), False, False)
            wide.set_friendly_name("wide_integer")
            results.append(wide)
            rows.append({"name": "wide_integer", "graph": "test"})
            ov.save_model(ov.Model(results, [input_node]), root / "packed.xml", compress_to_fp16=False)
            with patch("whisper_precision.quantization.GRAPH_FILES", {"test": "packed.xml"}):
                self.assertEqual(effective_bits(root, rows), {**{f"weight_{b}": b for b in (2, 3, 4, 8)}, "wide_integer": None})

    def test_graph_names_map_to_original_whisper_weights(self):
        self.assertEqual(
            parameter_name({"graph": "encoder", "name": "__module.layers.2.fc1/MatMul"}),
            "model.encoder.layers.2.fc1.weight")
        self.assertEqual(
            parameter_name({"graph": "decoder", "name": "__module.model.decoder.layers.4.encoder_attn.q_proj/MatMul"}),
            "model.decoder.layers.4.encoder_attn.q_proj.weight")

    def test_fake_quantizer_preserves_shape_and_finite_values(self):
        weights = torch.tensor([[-2.0, -0.5, 0.0, 0.7, 1.9]], dtype=torch.float32)
        for bits in (2, 3, 4, 8):
            restored = fake_quantize(weights, bits, group_size=4)
            self.assertEqual(restored.shape, weights.shape)
            self.assertTrue(torch.isfinite(restored).all())
