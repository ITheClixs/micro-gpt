import unittest
from contextlib import redirect_stdout
import io

import torch

from src.finetune_model import print_sample_summary


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        self.last_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return "generated"


class FakeModel:
    device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


class FineTuneModelTest(unittest.TestCase):
    def test_sample_summary_supports_custom_columns(self):
        tokenizer = FakeTokenizer()

        with redirect_stdout(io.StringIO()):
            print_sample_summary(
                FakeModel(),
                tokenizer,
                [{"text": "source article", "summary": "target summary"}],
                max_length=16,
                max_target_length=4,
                article_column="text",
                summary_column="summary",
            )

        self.assertIn("source article", tokenizer.last_text)


if __name__ == "__main__":
    unittest.main()
