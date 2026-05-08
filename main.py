import argparse


DEFAULT_DATASET_NAME = "cnn_dailymail"
DEFAULT_DATASET_CONFIG = "3.0.0"


def build_data_files(train_file, validation_file):
    if not train_file and not validation_file:
        return None
    if not train_file or not validation_file:
        raise ValueError("Provide both --train-file and --validation-file for local data.")
    return {"train": train_file, "validation": validation_file}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT-2 style causal language model for summarization."
    )
    parser.add_argument("--model-checkpoint", default="gpt2")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-file", help="Local JSON/JSONL training file.")
    parser.add_argument("--validation-file", help="Local JSON/JSONL validation file.")
    parser.add_argument("--article-column", default="article")
    parser.add_argument("--summary-column", default="highlights")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument(
        "--model-output-dir",
        default="./fine_tuned_gpt2_summarizer",
    )
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=1)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached Hugging Face models/tokenizers.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        data_files = build_data_files(args.train_file, args.validation_file)
    except ValueError as exc:
        build_parser().error(str(exc))

    dataset_name = args.dataset_name
    dataset_config = args.dataset_config

    if data_files:
        if dataset_name == DEFAULT_DATASET_NAME:
            dataset_name = "json"
        if dataset_config == DEFAULT_DATASET_CONFIG:
            dataset_config = None

    from src.finetune_model import finetune_model

    finetune_model(
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        model_output_dir=args.model_output_dir,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        data_files=data_files,
        article_column=args.article_column,
        summary_column=args.summary_column,
        train_size=args.train_size,
        eval_size=args.eval_size,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        num_train_epochs=args.num_train_epochs,
        local_files_only=args.local_files_only or None,
    )


if __name__ == "__main__":
    main()
