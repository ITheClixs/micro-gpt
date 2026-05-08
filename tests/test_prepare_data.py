import importlib
import sys
import types
import unittest


class FakeSplit:
    def __init__(self, column_names):
        self.column_names = column_names


class FakeDatasetDict(dict):
    pass


class PrepareDataTest(unittest.TestCase):
    def setUp(self):
        self.load_dataset_calls = []
        self.original_modules = {
            name: sys.modules.get(name)
            for name in ("datasets", "requests", "src.prepare_data", "transformers")
        }

        def fake_load_dataset(*args, **kwargs):
            self.load_dataset_calls.append((args, kwargs))
            return FakeDatasetDict(
                {
                    "train": FakeSplit(["text", "summary"]),
                    "validation": FakeSplit(["text", "summary"]),
                }
            )

        fake_datasets = types.SimpleNamespace(
            DatasetDict=FakeDatasetDict,
            load_dataset=fake_load_dataset,
        )
        fake_transformers = types.SimpleNamespace(AutoTokenizer=object)
        fake_requests = types.SimpleNamespace(RequestException=Exception)
        sys.modules["datasets"] = fake_datasets
        sys.modules["transformers"] = fake_transformers
        sys.modules["requests"] = fake_requests
        sys.modules.pop("src.prepare_data", None)
        self.prepare_data = importlib.import_module("src.prepare_data")

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_load_summarization_dataset_passes_data_files_for_local_json(self):
        dataset = self.prepare_data.load_summarization_dataset(
            dataset_name="json",
            dataset_config=None,
            data_files={"train": "train.jsonl", "validation": "validation.jsonl"},
        )

        self.assertIsInstance(dataset, FakeDatasetDict)
        self.assertEqual(self.load_dataset_calls[0][0], ("json",))
        self.assertEqual(
            self.load_dataset_calls[0][1]["data_files"],
            {"train": "train.jsonl", "validation": "validation.jsonl"},
        )

    def test_load_summarization_dataset_keeps_named_dataset_config(self):
        self.prepare_data.load_summarization_dataset(
            dataset_name="cnn_dailymail",
            dataset_config="3.0.0",
            data_files=None,
        )

        self.assertEqual(self.load_dataset_calls[0][0], ("cnn_dailymail", "3.0.0"))
        self.assertNotIn("data_files", self.load_dataset_calls[0][1])

    def test_validate_summarization_columns_accepts_custom_columns(self):
        dataset = FakeDatasetDict(
            {
                "train": FakeSplit(["text", "summary"]),
                "validation": FakeSplit(["text", "summary"]),
            }
        )

        self.prepare_data.validate_summarization_columns(
            dataset,
            article_column="text",
            summary_column="summary",
        )

    def test_validate_summarization_columns_rejects_missing_columns(self):
        dataset = FakeDatasetDict(
            {
                "train": FakeSplit(["text"]),
                "validation": FakeSplit(["text", "summary"]),
            }
        )

        with self.assertRaisesRegex(ValueError, "summary"):
            self.prepare_data.validate_summarization_columns(
                dataset,
                article_column="text",
                summary_column="summary",
            )


if __name__ == "__main__":
    unittest.main()
