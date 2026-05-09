import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.micro_gpt import cli
from src.micro_gpt.config import MicroGPTConfig, save_config
from src.micro_gpt.generate import main as generate_main
import torch


class MicroGPTCLITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_path = Path(self.temp_dir.name) / "tiny.json"
        save_config(
            MicroGPTConfig(
                vocab_size=8,
                block_size=4,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                batch_size=1,
                max_steps=1,
                learning_rate=1e-3,
                seed=7,
            ),
            self.config_path,
        )

    def run_cli(self, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exit_code = cli.main(list(args))
        self.assertEqual(exit_code, 0)
        return stream.getvalue()

    def test_inspect_prints_config_and_architecture_json(self):
        output = self.run_cli("inspect", "--config", str(self.config_path))

        payload = json.loads(output)

        self.assertEqual(payload["config"]["block_size"], 4)
        self.assertGreater(payload["parameter_count"], 0)
        self.assertGreater(payload["estimated_size_mb"], 0)
        self.assertEqual(
            payload["architecture"],
            {
                "decoder_only": True,
                "causal_attention": True,
                "rope": True,
                "rms_norm": True,
                "swiglu": True,
                "weight_tying": True,
            },
        )

    def test_smoke_runs_without_saving_checkpoint(self):
        checkpoint_path = Path(self.temp_dir.name) / "smoke.pt"

        output = self.run_cli(
            "smoke",
            "--config",
            str(self.config_path),
            "--text",
            "hello micro gpt",
            "--max-new-tokens",
            "2",
        )
        payload = json.loads(output)

        self.assertFalse(checkpoint_path.exists())
        self.assertEqual(payload["steps"], 1)
        self.assertIn("generated_text", payload)
        self.assertGreaterEqual(len(payload["generated_text"]), len(payload["prompt"]))
        self.assertFalse(payload["checkpoint_saved"])

    def test_smoke_accepts_text_shorter_than_block_size(self):
        output = self.run_cli(
            "smoke",
            "--config",
            str(self.config_path),
            "--text",
            "hi",
            "--max-new-tokens",
            "1",
        )
        payload = json.loads(output)

        self.assertEqual(payload["steps"], 1)
        self.assertTrue(payload["generated_text"])

    def test_smoke_can_save_checkpoint_and_generate_can_load_it(self):
        checkpoint_path = Path(self.temp_dir.name) / "smoke.pt"

        smoke_output = self.run_cli(
            "smoke",
            "--config",
            str(self.config_path),
            "--text",
            "terminal checkpoint text",
            "--max-new-tokens",
            "2",
            "--save-checkpoint",
            str(checkpoint_path),
        )
        smoke_payload = json.loads(smoke_output)
        generate_output = self.run_cli(
            "generate",
            "--config",
            str(self.config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "term",
            "--max-new-tokens",
            "2",
        )

        self.assertTrue(checkpoint_path.exists())
        self.assertTrue(smoke_payload["checkpoint_saved"])
        self.assertGreaterEqual(len(generate_output.strip()), len("term"))

    def test_legacy_generate_entrypoint_loads_cli_checkpoint_metadata(self):
        checkpoint_path = Path(self.temp_dir.name) / "smoke.pt"
        self.run_cli(
            "smoke",
            "--config",
            str(self.config_path),
            "--text",
            "terminal checkpoint text",
            "--max-new-tokens",
            "1",
            "--save-checkpoint",
            str(checkpoint_path),
        )

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            generate_main(
                [
                    "--config",
                    str(self.config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "term",
                    "--max-new-tokens",
                    "1",
                ]
            )

        self.assertGreaterEqual(len(stream.getvalue().strip()), len("term"))

    def test_legacy_generate_rejects_empty_prompt(self):
        checkpoint_path = Path(self.temp_dir.name) / "smoke.pt"
        self.run_cli(
            "smoke",
            "--config",
            str(self.config_path),
            "--text",
            "terminal checkpoint text",
            "--max-new-tokens",
            "1",
            "--save-checkpoint",
            str(checkpoint_path),
        )

        with self.assertRaisesRegex(SystemExit, "prompt"):
            generate_main(
                [
                    "--config",
                    str(self.config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "",
                    "--max-new-tokens",
                    "1",
                ]
            )

    def test_legacy_generate_rejects_negative_new_token_count(self):
        checkpoint_path = Path(self.temp_dir.name) / "smoke.pt"
        self.run_cli(
            "smoke",
            "--config",
            str(self.config_path),
            "--text",
            "terminal checkpoint text",
            "--max-new-tokens",
            "1",
            "--save-checkpoint",
            str(checkpoint_path),
        )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                generate_main(
                    [
                        "--config",
                        str(self.config_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--prompt",
                        "term",
                        "--max-new-tokens",
                        "-1",
                    ]
                )

    def test_generate_random_init_uses_prompt_tokenizer(self):
        output = self.run_cli(
            "generate",
            "--config",
            str(self.config_path),
            "--prompt",
            "abc",
            "--max-new-tokens",
            "2",
            "--random-init",
        )

        self.assertGreaterEqual(len(output.strip()), len("abc"))

    def test_random_init_generation_decodes_each_requested_token(self):
        output = self.run_cli(
            "generate",
            "--config",
            str(self.config_path),
            "--prompt",
            "abc",
            "--max-new-tokens",
            "3",
            "--random-init",
        )

        self.assertEqual(len(output.strip()), len("abc") + 3)

    def test_cli_masks_logits_outside_terminal_tokenizer_vocab(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 100.0]])

        masked = cli.mask_logits_to_vocab(logits, vocab_size=3)

        self.assertTrue(torch.isneginf(masked[0, 3]))
        torch.testing.assert_close(masked[0, :3], logits[0, :3])

    def test_generate_rejects_empty_prompt(self):
        with self.assertRaisesRegex(SystemExit, "prompt"):
            cli.generate_text(
                self.config_path,
                "",
                max_new_tokens=1,
                random_init=True,
            )

    def test_negative_new_token_count_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(
                    [
                        "generate",
                        "--config",
                        str(self.config_path),
                        "--prompt",
                        "abc",
                        "--max-new-tokens",
                        "-1",
                        "--random-init",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
