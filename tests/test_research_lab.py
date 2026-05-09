import unittest

from src.research_lab import demo_data


class ResearchLabDataTest(unittest.TestCase):
    def test_backprop_demo_payload_has_loss_curve_and_gradients(self):
        payload = demo_data.backprop_payload()

        self.assertIn("loss_curve", payload)
        self.assertIn("gradient_norms", payload)
        self.assertGreater(len(payload["loss_curve"]), 1)

    def test_micro_gpt_payload_has_attention_and_token_probabilities(self):
        payload = demo_data.micro_gpt_payload()

        self.assertIn("attention", payload)
        self.assertIn("token_probabilities", payload)
        self.assertEqual(len(payload["attention"]), len(payload["attention"][0]))

    def test_rnn_payload_has_gradient_value_for_each_hidden_state(self):
        payload = demo_data.rnn_payload()

        self.assertEqual(len(payload["hidden_states"]), len(payload["gradient_norms"]))


if __name__ == "__main__":
    unittest.main()
