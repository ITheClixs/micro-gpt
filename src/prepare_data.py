from requests import RequestException
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


DEFAULT_DATASET_NAME = "cnn_dailymail"
DEFAULT_DATASET_CONFIG = "3.0.0"
PROMPT_PREFIX = "Summarize the following article:\n\n"
SUMMARY_PREFIX = "\n\nSummary:\n"


def build_prompt(article):
    return f"{PROMPT_PREFIX}{article}{SUMMARY_PREFIX}"


def load_summarization_dataset(
    dataset_name=DEFAULT_DATASET_NAME,
    dataset_config=DEFAULT_DATASET_CONFIG,
    data_files=None,
):
    load_kwargs = {}
    if data_files:
        load_kwargs["data_files"] = data_files

    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **load_kwargs)
    return load_dataset(dataset_name, **load_kwargs)


def validate_summarization_columns(dataset, article_column="article", summary_column="highlights"):
    required_columns = {article_column, summary_column}
    for split_name in ("train", "validation"):
        if split_name not in dataset:
            raise ValueError(f"Dataset must include a '{split_name}' split.")

        split_columns = set(dataset[split_name].column_names)
        missing_columns = sorted(required_columns - split_columns)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(
                f"Dataset split '{split_name}' is missing required column(s): {missing}"
            )


def load_tokenizer(model_checkpoint, local_files_only):
    try:
        return AutoTokenizer.from_pretrained(
            model_checkpoint,
            local_files_only=local_files_only,
        )
    except (OSError, RequestException) as exc:
        if local_files_only:
            raise
        print(f"Falling back to cached tokenizer for {model_checkpoint}: {exc}")
        return AutoTokenizer.from_pretrained(
            model_checkpoint,
            local_files_only=True,
        )


def prepare_data(
    model_checkpoint="gpt2",
    dataset_name=DEFAULT_DATASET_NAME,
    dataset_config=DEFAULT_DATASET_CONFIG,
    data_files=None,
    article_column="article",
    summary_column="highlights",
    max_length=1024,
    max_target_length=128,
    train_size=100,
    eval_size=10,
    local_files_only=False,
    return_raw=False,
):
    print("Loading dataset...")
    dataset = load_summarization_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        data_files=data_files,
    )
    validate_summarization_columns(
        dataset,
        article_column=article_column,
        summary_column=summary_column,
    )

    print(f"Loading tokenizer for {model_checkpoint}...")
    tokenizer = load_tokenizer(model_checkpoint, local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = min(train_size, len(dataset["train"]))
    validation_examples = min(eval_size, len(dataset["validation"]))
    dataset_subset = DatasetDict(
        {
            "train": dataset["train"].select(range(train_examples)),
            "validation": dataset["validation"].select(range(validation_examples)),
        }
    )

    def preprocess_function(examples):
        input_ids_batch = []
        attention_masks_batch = []
        labels_batch = []
        prompt_prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
        summary_prefix_ids = tokenizer(SUMMARY_PREFIX, add_special_tokens=False)["input_ids"]

        for article, highlights in zip(examples[article_column], examples[summary_column]):
            summary_ids = tokenizer(
                highlights,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, max_target_length - 1),
            )["input_ids"] + [tokenizer.eos_token_id]
            article_token_budget = max_length - len(prompt_prefix_ids) - len(summary_prefix_ids) - len(summary_ids)
            article_ids = tokenizer(
                article,
                add_special_tokens=False,
                truncation=True,
                max_length=max(0, article_token_budget),
            )["input_ids"]

            prompt_ids = prompt_prefix_ids + article_ids + summary_prefix_ids
            combined_ids = (prompt_ids + summary_ids)[:max_length]
            prompt_token_count = min(len(prompt_ids), len(combined_ids))
            labels = ([-100] * prompt_token_count + combined_ids[prompt_token_count:])[:max_length]
            attention_mask = [1] * len(combined_ids)

            padding_length = max_length - len(combined_ids)
            if padding_length > 0:
                combined_ids = combined_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                labels = labels + ([-100] * padding_length)

            input_ids_batch.append(combined_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
        }

    print("Preprocessing dataset subset...")
    tokenized_datasets = dataset_subset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_subset["train"].column_names,
    )

    print("Dataset preparation complete.")
    if return_raw:
        return tokenized_datasets, tokenizer, dataset_subset
    return tokenized_datasets, tokenizer


if __name__ == "__main__":
    tokenized_datasets, tokenizer = prepare_data(local_files_only=True)
    print(tokenized_datasets)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
