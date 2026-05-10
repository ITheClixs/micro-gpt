import unittest

import torch

from src.quantlab.tlob_modules import BilinearNormalization, MCDropout, RMSNorm


class TestTlobBaseModules(unittest.TestCase):
    def test_rmsnorm_preserves_shape(self):
        norm = RMSNorm(dim=16)
        x = torch.randn(2, 4, 16)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_bin_centers_per_channel(self):
        bin_module = BilinearNormalization(num_features=4, num_timesteps=8)
        torch.manual_seed(0)
        x = torch.randn(3, 8, 4) * 3 + 7
        bin_module.train()
        y = bin_module(x)
        mean = y.mean(dim=(0, 1))
        std = y.std(dim=(0, 1), unbiased=False)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=0.5))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=0.5))

    def test_mc_dropout_active_in_eval(self):
        module = MCDropout(p=0.5)
        module.eval()
        x = torch.ones(8, 16)
        y = module(x)
        self.assertFalse(torch.allclose(y, x))

    def test_mc_dropout_zero_when_disabled(self):
        module = MCDropout(p=0.5, mc_active=False)
        module.eval()
        x = torch.ones(8, 16)
        y = module(x)
        self.assertTrue(torch.allclose(y, x))


if __name__ == "__main__":
    unittest.main()
