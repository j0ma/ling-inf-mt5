import click
import datasets as ds
import evaluate
import numpy as np
import pudb
from datasets import Dataset
from rich.pretty import pprint
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

import pandas as pd
from sklearn.metrics import confusion_matrix

f1 = evaluate.load("f1")


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
    "--test-langs",
    type=lambda x: str(x).split(","),
    default="",
    help="Comma-separated list of languages to test on",
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
    test_langs,
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

    flores = ds.load_from_disk(flores_path)
    ntrex = ds.load_from_disk(ntrex_path)

    # Create language -> integer id mapping
    all_langs_in_common = {lang for lang in flores} & {lang for lang in ntrex}
    language_to_id = {language: i for i, language in enumerate(all_langs_in_common)}
    id_to_language = {i: language for language, i in language_to_id.items()}


    # Define new dataset based on label column and finetuning langs
    data_for_finetune = ds.concatenate_datasets(
        [ntrex[lang] for lang in finetune_langs]
    )

    if "text" not in data_for_finetune.column_names:
        data_for_finetune = data_for_finetune.rename_column(text_column, "text")

    if "label" not in data_for_finetune.column_names:
        data_for_finetune = data_for_finetune.rename_column(label_column, "label")

    # Define test data
    data_for_test = ds.concatenate_datasets([flores[lang] for lang in test_langs])

    if "text" not in data_for_test.column_names:
        data_for_test = data_for_test.rename_column(text_column, "text")

    if "label" not in data_for_test.column_names:
        data_for_test = data_for_test.rename_column(label_column, "label")

    # Load the XLM-R tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=max(language_to_id.values())
    )

    pprint("Classifier:")
    pprint(model.classifier)

    def preprocess_function(examples):
        inputs = examples["text"]
        labels = [language_to_id[label] for label in examples["label"]]

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
            "max_steps": max_steps,
        }
    else:
        how_long_to_train_args = {
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
        evaluation_strategy="no"
        if eval_steps < 0
        else {0: "epoch"}.get(eval_steps, "steps"),
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        logging_strategy="no"
        if logging_steps < 0
        else {0: "epoch"}.get(logging_steps, "steps"),
        overwrite_output_dir=True,
        **how_long_to_train_args,
    )

    pprint("Training args:")
    pprint(training_args)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    _langs_set = set(finetune_langs) | set(test_langs)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        metrics = {
            "macro_f1": f1.compute(
                predictions=predictions, references=labels, average="macro"
            )["f1"],
            "micro_f1": f1.compute(
                predictions=predictions, references=labels, average="micro"
            )["f1"],
        }

        # Per-class F1s
        class_f1s = f1.compute(
            predictions=predictions, references=labels, average=None
        )["f1"]
        class_ids = sorted(set(predictions) | set(labels))
        f1s_per_class = {
            id_to_language[class_id]: f1_score
            for class_id, f1_score in zip(class_ids, class_f1s)
        }

        # Score each test lang
        for lang in test_langs:
            metrics[f"f1_{lang}"] = f1s_per_class.get(lang, 0)

        # Confusion matrix
        labels_human_readable = [id_to_language[i] for i in labels]
        predictions_human_readable = [id_to_language[i] for i in predictions]

        for hum_readable in labels_human_readable, predictions_human_readable:
            _langs_set.update(hum_readable)

        _langs = sorted(_langs_set)

        remove_zero_rows = lambda df: df.loc[~(df==0).all(axis=1)]
        remove_zero_cols = lambda df: df.loc[:, (df != 0).any(axis=0)]

        cm = confusion_matrix(
            y_true=labels_human_readable,
            y_pred=predictions_human_readable,
            labels=_langs,
        )
        confusion_df = pd.DataFrame(cm)
        confusion_df.columns = _langs
        confusion_df.index = _langs
        confusion_df = remove_zero_rows(confusion_df)
        confusion_df = remove_zero_cols(confusion_df)

        print("\n\n")
        pprint(confusion_df)
        print("\n\n")

        return metrics

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

    # Evaluate the model before finetuning
    pprint("Pre-finetune eval")
    metrics_pre_finetune = trainer.evaluate()
    pprint(metrics_pre_finetune)

    # Finetune the model
    trainer.train()

    # Evaluate the model after finetuning
    pprint("Post-finetune eval")
    metrics_post_finetune = trainer.evaluate()
    pprint(metrics_post_finetune)


if __name__ == "__main__":
    train_xlmr()
