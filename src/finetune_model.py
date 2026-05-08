import os

import torch
from requests import RequestException
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

from .prepare_data import build_prompt, prepare_data


def print_sample_summary(model, tokenizer, raw_validation_dataset, max_length, max_target_length):
    sample = raw_validation_dataset[0]
    prompt = build_prompt(sample["article"])
    prompt_max_length = max(1, max_length - max_target_length)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max_length,
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_target_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_summary = tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print("\nSample generated summary:")
    print(generated_summary or "(empty)")
    print("\nReference summary:")
    print(sample["highlights"])


def load_model(model_checkpoint, local_files_only):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            local_files_only=local_files_only,
        )
    except (OSError, RequestException) as exc:
        if local_files_only:
            raise
        print(f"Falling back to cached model for {model_checkpoint}: {exc}")
        return AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            local_files_only=True,
        )


def finetune_model(
    model_checkpoint="gpt2",
    output_dir="./results",
    model_output_dir="./fine_tuned_gpt2_summarizer",
    dataset_name="cnn_dailymail",
    dataset_config="3.0.0",
    data_files=None,
    article_column="article",
    summary_column="highlights",
    train_size=100,
    eval_size=10,
    max_length=1024,
    max_target_length=128,
    num_train_epochs=1,
    local_files_only=None,
):
    if local_files_only is None:
        local_files_only = (
            os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        )

    tokenized_datasets, tokenizer, raw_datasets = prepare_data(
        model_checkpoint=model_checkpoint,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        data_files=data_files,
        article_column=article_column,
        summary_column=summary_column,
        train_size=train_size,
        eval_size=eval_size,
        max_length=max_length,
        max_target_length=max_target_length,
        local_files_only=local_files_only,
        return_raw=True,
    )

    model = load_model(model_checkpoint, local_files_only)
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
    )

    print("Starting model fine-tuning...")
    trainer.train()

    print_sample_summary(
        trainer.model,
        tokenizer,
        raw_datasets["validation"],
        max_length=max_length,
        max_target_length=max_target_length,
    )

    print(f"Saving fine-tuned model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    finetune_model()
