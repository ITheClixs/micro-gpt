import unittest

import torch
import torch.nn.functional as F

from src.algorithms.adapters import LoRALinear, trainable_parameter_count
from src.algorithms.alignment import dpo_loss, group_relative_advantages
from src.algorithms.optimizers import lion_step, muon_step, orthogonalize_newton_schulz


class CuttingEdgeAlgorithmPrimitiveTest(unittest.TestCase):
    def test_dpo_loss_rewards_chosen_log_probability_margin(self):
        policy_chosen = torch.tensor([-0.2, -0.4])
        policy_rejected = torch.tensor([-1.3, -1.6])
        reference_chosen = torch.tensor([-0.9, -0.7])
        reference_rejected = torch.tensor([-1.0, -1.1])
        beta = 0.2

        loss, diagnostics = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=beta,
        )
        expected_logits = beta * (
            (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        )
        expected_losses = -F.logsigmoid(expected_logits)

        self.assertEqual(tuple(loss.shape), ())
        torch.testing.assert_close(diagnostics["logits"], expected_logits)
        torch.testing.assert_close(diagnostics["losses"], expected_losses)
        torch.testing.assert_close(loss, expected_losses.mean())
        self.assertGreater(float(diagnostics["preference_accuracy"]), 0.5)
        self.assertTrue(torch.all(diagnostics["rewards_chosen"] > diagnostics["rewards_rejected"]))

    def test_dpo_loss_includes_reference_margin_when_policy_margin_is_positive(self):
        policy_chosen = torch.tensor([-0.2])
        policy_rejected = torch.tensor([-0.7])
        reference_chosen = torch.tensor([-0.1])
        reference_rejected = torch.tensor([-1.5])
        beta = 1.0

        loss, diagnostics = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=beta,
        )
        expected_logits = beta * (
            (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        )
        expected_losses = -F.logsigmoid(expected_logits)

        self.assertLess(float(diagnostics["logits"]), 0.0)
        torch.testing.assert_close(diagnostics["logits"], expected_logits)
        torch.testing.assert_close(diagnostics["losses"], expected_losses)
        torch.testing.assert_close(loss, expected_losses.mean())

    def test_group_relative_advantages_zero_center_each_reward_group(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 12.0, 14.0])

        advantages = group_relative_advantages(rewards, group_size=3)

        grouped = advantages.view(2, 3)
        torch.testing.assert_close(grouped.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(grouped.std(dim=1, unbiased=False), torch.ones(2))

    def test_lora_linear_starts_equivalent_to_base_and_freezes_base_weights(self):
        torch.manual_seed(5)
        base = torch.nn.Linear(4, 3, dtype=torch.float64)
        layer = LoRALinear(base, rank=2, alpha=4.0)
        x = torch.randn(6, 4, dtype=torch.float64)

        torch.testing.assert_close(layer(x), base(x))

        self.assertEqual(layer.lora_a.dtype, base.weight.dtype)
        self.assertEqual(layer.lora_a.device, base.weight.device)
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertFalse(layer.base.bias.requires_grad)
        self.assertTrue(layer.lora_a.requires_grad)
        self.assertTrue(layer.lora_b.requires_grad)
        self.assertEqual(trainable_parameter_count(layer), 2 * (4 + 3))

    def test_lion_step_uses_sign_of_momentum_mixed_gradient(self):
        parameter = torch.tensor([1.0, -1.0])
        gradient = torch.tensor([-0.25, 0.5])
        momentum = torch.tensor([1.0, -1.0])

        updated, next_momentum = lion_step(parameter, gradient, momentum, lr=0.1, betas=(0.9, 0.99))

        torch.testing.assert_close(updated, torch.tensor([0.9, -0.9]))
        torch.testing.assert_close(next_momentum, torch.tensor([0.9875, -0.9850]))

    def test_muon_step_returns_finite_matrix_and_orthogonalized_update(self):
        parameter = torch.randn(4, 3)
        gradient = torch.randn(4, 3)
        momentum = torch.randn(4, 3)

        orthogonalized = orthogonalize_newton_schulz(gradient, steps=6)
        lr = 0.05
        momentum_factor = 0.9
        steps = 6
        updated, next_momentum, update = muon_step(
            parameter,
            gradient,
            momentum,
            lr=lr,
            momentum=momentum_factor,
            steps=steps,
        )
        expected_momentum = momentum_factor * momentum + gradient
        expected_update = orthogonalize_newton_schulz(expected_momentum, steps=steps)

        self.assertEqual(tuple(updated.shape), (4, 3))
        self.assertEqual(tuple(next_momentum.shape), (4, 3))
        self.assertTrue(torch.isfinite(updated).all())
        self.assertTrue(torch.isfinite(update).all())
        torch.testing.assert_close(next_momentum, expected_momentum)
        torch.testing.assert_close(update, expected_update)
        torch.testing.assert_close(updated, parameter - lr * expected_update)
        torch.testing.assert_close(update.T @ update, torch.eye(3), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(orthogonalized.T @ orthogonalized, torch.eye(3), atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
