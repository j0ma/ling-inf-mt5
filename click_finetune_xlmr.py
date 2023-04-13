import click
import datasets as ds
import evaluate
import numpy as np
import pudb
from datasets import Dataset
from rich.pretty import pprint
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          default_data_collator)

accuracy = evaluate.load("accuracy")
micro_f1 = evaluate.load("f1")


@click.command()
@click.option(
    "--flores-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to binarized Flores corpus",
)
@click.option(
    "--ntrex-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to binarized NTREX corpus",
)
@click.option(
    "--model-name",
    type=str,
    default="xlm-roberta-base",
    help="Name or path of the pre-trained model to use",
)
@click.option(
    "--label-column",
    type=str,
    default="language",
    help="Name of the label column in the dataset",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of the text column in the dataset",
)
@click.option(
    "--finetune-langs",
    type=lambda x: str(x).split(","),
    default="",
    help="Comma-separated list of languages to fine tune on",
)
@click.option(
    "--evaluate-langs",
    type=lambda x: str(x).split(","),
    default="",
    help="Comma-separated list of languages to fine tune on",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./results",
    help="Directory to store the output files",
)
@click.option(
    "--max-steps", type=int, default=0, help="Maximum number of training steps"
)
@click.option(
    "--num-train-epochs", type=int, default=0, help="Number of epochs to train"
)
@click.option("--batch-size", type=int, default=8, help="Training batch size")
@click.option(
    "--learning-rate", type=float, default=5e-5, help="Learning rate for training"
)
@click.option(
    "--max-length-tokens", type=int, default=128, help="Maximum number of tokens"
)
@click.option("--num-gpus", type=int, default=0, help="Number of GPUs")
@click.option("--save-total-limit", type=int, default=1)
@click.option("--save-steps", help="Save every `save-steps` steps.", default=100)
@click.option("--eval-steps", help="Eval every `eval-steps` steps.", default=100)
@click.option("--warmup-steps", default=100)
@click.option("--logging-steps", default=50)
def train_xlmr(
    flores_path,
    ntrex_path,
    model_name,
    label_column,
    text_column,
    finetune_langs,
    evaluate_langs,
    output_dir,
    max_steps,
    num_train_epochs,
    batch_size,
    learning_rate,
    max_length_tokens,
    save_total_limit,
    save_steps,
    eval_steps,
    warmup_steps,
    logging_steps,
    num_gpus,
):
    assert max_steps ^ num_train_epochs

    if num_gpus < 1:
        import torch

        num_gpus = torch.cuda.device_count()
        click.echo(
            f"--num-gpus not set! Found {num_gpus} GPUs.",
            file=click.get_text_stream("stderr"),
        )
    else:
        click.echo(
            f"--num-gpus set to {num_gpus} GPUs.", file=click.get_text_stream("stderr")
        )

    click.echo(f"Learning rate: {learning_rate}")

    # Load the XLM-R tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=100
    )

    flores = ds.load_from_disk(flores_path)
    ntrex = ds.load_from_disk(ntrex_path)

    # Define new dataset based on label column and finetuning langs
    data_for_finetune = ds.concatenate_datasets(
        [ntrex[lang] for lang in finetune_langs]
    )

    if "text" not in data_for_finetune.column_names:
        data_for_finetune = data_for_finetune.rename_column(text_column, "text")

    if "label" not in data_for_finetune.column_names:
        data_for_finetune = data_for_finetune.rename_column(label_column, "label")

    # Define test data
    data_for_test = ds.concatenate_datasets([flores[lang] for lang in evaluate_langs])

    if "text" not in data_for_test.column_names:
        data_for_test = data_for_test.rename_column(text_column, "text")

    if "label" not in data_for_test.column_names:
        data_for_test = data_for_test.rename_column(label_column, "label")

    # Create label -> integer id mapping
    label_to_id = {
        label: i for i, label in enumerate(set(finetune_langs + evaluate_langs))
    }

    def preprocess_function(examples):
        inputs = examples["text"]
        labels = [label_to_id[label] for label in examples["label"]]

        model_inputs = tokenizer(
            inputs, padding=True, truncation=True, max_length=max_length_tokens
        )
        model_inputs["label"] = labels

        return model_inputs

    data_for_finetune = data_for_finetune.map(preprocess_function, batched=True)
    data_for_test = data_for_test.map(preprocess_function, batched=True)

    # Define the training arguments

    if max_steps:
        how_long_to_train_args = {
            "evaluation_strategy": "steps",
            "max_steps": max_steps,
        }
    else:
        how_long_to_train_args = {
            "evaluation_strategy": "epoch",
            "num_train_epochs": num_train_epochs,
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=save_total_limit,
        learning_rate=learning_rate,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        overwrite_output_dir=True,
        **how_long_to_train_args,
    )

    pprint(training_args)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels),
            "micro_f1": micro_f1.compute(
                predictions=predictions, references=labels, accuracy="micro"
            ),
        }

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_for_finetune,
        eval_dataset=data_for_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()


if __name__ == "__main__":
    train_xlmr()
