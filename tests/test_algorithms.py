import unittest

import torch

from src.algorithms import backprop, cnn, optimizers, rl, rnn


class AlgorithmPrimitiveTest(unittest.TestCase):
    def test_manual_mlp_gradients_match_autograd(self):
        sample = backprop.gradient_check(seed=7)

        self.assertLess(sample["max_abs_error"], 1e-5)
        self.assertEqual(sample["layer_count"], 2)

    def test_manual_conv2d_matches_torch_conv2d(self):
        image = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
        kernel = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])

        actual = cnn.conv2d_manual(image, kernel)
        expected = torch.nn.functional.conv2d(image, kernel)

        torch.testing.assert_close(actual, expected)

    def test_rnn_cell_returns_hidden_state_with_expected_shape(self):
        cell = rnn.VanillaRNNCell(input_size=3, hidden_size=5)
        x_t = torch.randn(2, 3)
        h_t = torch.zeros(2, 5)

        next_h = cell(x_t, h_t)

        self.assertEqual(tuple(next_h.shape), (2, 5))
        self.assertTrue(torch.all(next_h <= 1.0))
        self.assertTrue(torch.all(next_h >= -1.0))

    def test_discounted_returns_are_computed_backwards(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])

        returns = rl.discounted_returns(rewards, gamma=0.9)

        torch.testing.assert_close(returns, torch.tensor([2.71, 1.9, 1.0]))

    def test_ppo_clipped_objective_matches_surrogate_minimum(self):
        log_probs = torch.log(torch.tensor([1.2, 0.7]))
        old_log_probs = torch.log(torch.tensor([1.0, 1.0]))
        advantages = torch.tensor([1.0, -1.0])

        objective = rl.ppo_clipped_objective(
            log_probs,
            old_log_probs,
            advantages,
            clip_epsilon=0.2,
        )

        self.assertAlmostEqual(float(objective), -0.2, places=6)

    def test_adamw_step_updates_parameters_and_decouples_weight_decay(self):
        param = torch.tensor([1.0, -1.0])
        grad = torch.tensor([0.5, -0.25])
        state = optimizers.AdamWState.zeros_like(param)

        updated, state = optimizers.adamw_step(
            param,
            grad,
            state,
            lr=0.1,
            weight_decay=0.1,
        )

        self.assertEqual(state.step, 1)
        self.assertLess(float(updated[0]), 1.0)
        self.assertGreater(float(updated[1]), -1.0)


if __name__ == "__main__":
    unittest.main()
