import tempfile
import unittest
from pathlib import Path

import torch

from src.micro_gpt.config import MicroGPTConfig, load_config
from src.micro_gpt.checkpoint import load_micro_gpt_checkpoint
from src.micro_gpt.checkpoint import save_micro_gpt_checkpoint
from src.micro_gpt.data import BPETokenizer, CharTokenizer, make_lm_batch
from src.micro_gpt.model import MicroGPT
from src.micro_gpt.train import resolve_training_text, run_dry_training, run_training


class MicroGPTTest(unittest.TestCase):
    def test_config_rejects_invalid_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "n_embd"):
            MicroGPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=3, n_embd=10)

    def test_config_loads_from_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(
                '{"vocab_size": 32, "block_size": 8, "n_layer": 1, '
                '"n_head": 2, "n_embd": 16, "dropout": 0.0}',
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.vocab_size, 32)
        self.assertEqual(config.n_head, 2)

    def test_model_forward_returns_logits_and_loss(self):
        config = MicroGPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        model = MicroGPT(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids, labels=labels)

        self.assertEqual(tuple(output.logits.shape), (2, 8, config.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_generation_respects_context_and_length(self):
        config = MicroGPTConfig(
            vocab_size=16,
            block_size=4,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        model = MicroGPT(config)
        prompt = torch.tensor([[1, 2, 3, 4]])

        generated = model.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=5)

        self.assertEqual(tuple(generated.shape), (1, 7))

    def test_char_tokenizer_round_trips_text(self):
        tokenizer = CharTokenizer.from_text("banana")

        encoded = tokenizer.encode("ban")
        decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, "ban")

    def test_bpe_tokenizer_round_trips_text(self):
        tokenizer = BPETokenizer.from_text("banana bread", target_vocab_size=16)

        encoded = tokenizer.encode("banana")
        decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, "banana")

    def test_make_lm_batch_returns_shifted_targets(self):
        tokens = torch.arange(12)

        x, y = make_lm_batch(tokens, block_size=4, batch_size=2, seed=3)

        self.assertEqual(tuple(x.shape), (2, 4))
        torch.testing.assert_close(y[:, :-1], x[:, 1:])

    def test_dry_training_returns_metrics_without_checkpointing(self):
        config = MicroGPTConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            batch_size=2,
            max_steps=2,
        )

        metrics = run_dry_training(config, text="micro gpt research lab")

        self.assertEqual(metrics["steps"], 2)
        self.assertGreater(metrics["parameter_count"], 0)
        self.assertIn("loss", metrics)

    def test_training_text_can_be_loaded_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "corpus.txt"
            text_path.write_text("file based micro gpt corpus", encoding="utf-8")

            text = resolve_training_text(text_file=text_path)

        self.assertEqual(text, "file based micro gpt corpus")

    def test_training_text_file_rejects_empty_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "empty.txt"
            text_path.write_text("   ", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                resolve_training_text(text_file=text_path)

    def test_training_writes_checkpoint_and_metrics(self):
        config = MicroGPTConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            batch_size=2,
            max_steps=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            metrics_path = Path(temp_dir) / "metrics.json"

            metrics = run_training(
                config,
                text="micro gpt can now train locally",
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path,
                run_name="unit-test",
            )
            checkpoint = load_micro_gpt_checkpoint(checkpoint_path)
            metrics_file_exists = metrics_path.exists()

        self.assertEqual(metrics["steps"], 2)
        self.assertFalse(metrics["dry_run"])
        self.assertEqual(metrics["run_name"], "unit-test")
        self.assertTrue(metrics_file_exists)
        self.assertEqual(checkpoint["metadata"]["run_name"], "unit-test")
        self.assertIn("model", checkpoint)

    def test_bpe_training_reports_tokenizer_kind(self):
        config = MicroGPTConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            batch_size=2,
            max_steps=2,
            tokenizer_kind="bpe",
            tokenizer_vocab_size=24,
        )

        metrics = run_dry_training(config, text="micro gpt bpe corpus")

        self.assertEqual(metrics["tokenizer_kind"], "bpe")
        self.assertGreaterEqual(metrics["tokenizer_vocab_size"], 2)

    def test_bpe_checkpoint_round_trip_preserves_tokenizer_metadata(self):
        config = MicroGPTConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            batch_size=1,
            max_steps=1,
            tokenizer_kind="bpe",
            tokenizer_vocab_size=16,
        )
        model = MicroGPT(config)
        tokenizer = BPETokenizer.from_text("micro gpt checkpoint", target_vocab_size=16)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "bpe.pt"
            save_micro_gpt_checkpoint(checkpoint_path, model, config, tokenizer, metadata={"source": "unit-test"})
            payload = load_micro_gpt_checkpoint(checkpoint_path)

        self.assertEqual(payload["tokenizer"].kind, "bpe")
        self.assertEqual(payload["metadata"]["source"], "unit-test")


if __name__ == "__main__":
    unittest.main()
