import importlib
import sys
import types
import unittest


class MainCliTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.original_modules = {
            name: sys.modules.get(name)
            for name in ("main", "src.finetune_model")
        }

        def fake_finetune_model(**kwargs):
            self.calls.append(kwargs)

        sys.modules["src.finetune_model"] = types.SimpleNamespace(
            finetune_model=fake_finetune_model
        )
        sys.modules.pop("main", None)
        self.main = importlib.import_module("main")

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_build_data_files_requires_both_local_splits(self):
        with self.assertRaisesRegex(ValueError, "both"):
            self.main.build_data_files("train.jsonl", None)

    def test_build_data_files_returns_none_without_local_files(self):
        self.assertIsNone(self.main.build_data_files(None, None))

    def test_main_routes_local_json_files_to_training(self):
        self.main.main(
            [
                "--train-file",
                "train.jsonl",
                "--validation-file",
                "validation.jsonl",
                "--article-column",
                "text",
                "--summary-column",
                "summary",
                "--model-checkpoint",
                "sshleifer/tiny-gpt2",
                "--train-size",
                "4",
                "--eval-size",
                "2",
            ]
        )

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            self.calls[0]["data_files"],
            {"train": "train.jsonl", "validation": "validation.jsonl"},
        )
        self.assertEqual(self.calls[0]["dataset_name"], "json")
        self.assertIsNone(self.calls[0]["dataset_config"])
        self.assertEqual(self.calls[0]["article_column"], "text")
        self.assertEqual(self.calls[0]["summary_column"], "summary")

    def test_main_drops_default_dataset_config_for_explicit_local_loader(self):
        self.main.main(
            [
                "--dataset-name",
                "csv",
                "--train-file",
                "train.csv",
                "--validation-file",
                "validation.csv",
            ]
        )

        self.assertEqual(self.calls[0]["dataset_name"], "csv")
        self.assertIsNone(self.calls[0]["dataset_config"])


if __name__ == "__main__":
    unittest.main()
