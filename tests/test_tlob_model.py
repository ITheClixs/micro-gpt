import os
import tempfile
import unittest

import torch

from src.quantlab.tlob_model import (
    EMAWeightTracker,
    TLOBQConfig,
    TLOBQModel,
    load_tlob_q_artifact,
    save_tlob_q_artifact,
)


def _smoke_config():
    return TLOBQConfig(
        feature_keys=tuple(f"f{i}" for i in range(8)),
        sequence_length=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_expansion=2,
        dropout=0.1,
        horizons=(1, 5),
        head_volatility=True,
        head_spread=True,
        seed=42,
    )


class TestTLOBQModel(unittest.TestCase):
    def test_forward_shapes(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        x = torch.randn(4, config.sequence_length, len(config.feature_keys))
        outputs = model(x)
        for horizon in config.horizons:
            self.assertEqual(outputs[horizon]["direction"].shape, (4, 3))
            self.assertEqual(outputs[horizon]["future_vol"].shape, (4, 1))
            self.assertEqual(outputs[horizon]["future_spread"].shape, (4, 1))

    def test_param_count_within_budget(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        count = sum(p.numel() for p in model.parameters())
        self.assertGreater(count, 1_000)
        self.assertLess(count, 1_000_000)

    def test_mc_dropout_produces_variance(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        model.eval()
        model.set_mc_active(True)
        x = torch.randn(4, config.sequence_length, len(config.feature_keys))
        sample_a = model(x)[1]["direction"]
        sample_b = model(x)[1]["direction"]
        self.assertFalse(torch.allclose(sample_a, sample_b))

    def test_invalid_layer_count_raises(self):
        config = TLOBQConfig(
            feature_keys=("a",),
            sequence_length=4,
            d_model=8,
            n_heads=2,
            n_layers=0,
            ffn_expansion=2,
            dropout=0.0,
            horizons=(1,),
        )
        with self.assertRaises(ValueError):
            TLOBQModel(config)

    def test_ema_tracker_moves_toward_target(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        ema = EMAWeightTracker(model, decay=0.5)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)
        shadow_state = ema.shadow_state()
        current_state = model.state_dict()
        diverged = False
        for name in shadow_state:
            if name in current_state and not torch.allclose(shadow_state[name], current_state[name]):
                diverged = True
                break
        self.assertTrue(diverged)

    def test_save_load_round_trip(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        ema = EMAWeightTracker(model, decay=0.5)
        ema.update(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact.pt")
            save_tlob_q_artifact(
                path,
                model=model,
                ema=ema,
                config=config,
                standardizer={"mean": [0.0], "scale": [1.0]},
                metrics={"loss": 0.5},
                meta={"seed": 42},
            )
            loaded = load_tlob_q_artifact(path)
            self.assertEqual(loaded["config"].seed, 42)
            self.assertIn("ema_state_dict", loaded)
            self.assertEqual(loaded["metrics"]["loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
