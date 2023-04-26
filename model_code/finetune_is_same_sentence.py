import functools as ft
import itertools as it
import os
from pathlib import Path

import pudb
import click
import datasets as ds
import evaluate
import numpy as np
import pandas as pd

from rich.pretty import pprint
from rich.progress import track, Progress
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_callback import PrinterCallback

MBERT = "bert-base-multilingual-cased"
MAX_LENGTH_TOKENS = 128


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_flores_ntrex(flores_path, ntrex_path):
    flores = ds.load_from_disk(flores_path)
    ntrex = ds.load_from_disk(ntrex_path)

    # Create language -> integer id mapping
    all_langs_in_common = {lang for lang in flores} & {lang for lang in ntrex}
    language_to_id = {language: i for i, language in enumerate(all_langs_in_common)}
    id_to_language = {i: language for language, i in language_to_id.items()}

    return flores, ntrex, language_to_id, id_to_language


def create_pairs_data(dataset, lang_pairs, n_pos, n_neg, num_rows):
    indices = [ix for ix in range(num_rows)]

    if n_pos > 0:
        positive_indices = np.random.randint(low=0, high=num_rows, size=n_pos)
    else:
        positive_indices = indices
        n_pos = len(indices)

    potential_negative_pairs = [
        (a, b) for a, b in it.combinations(indices, 2) if a != b
    ]

    if n_neg < 0:
        n_neg = n_pos

    if n_neg > 0:
        negative_indices = np.random.choice(
            range(len(potential_negative_pairs)), size=n_neg
        )
    else:
        negative_indices = list(range(len(potential_negative_pairs)))

    negative_indices = [potential_negative_pairs[idx] for idx in negative_indices]
    negative_as = [a for (a, b) in negative_indices]
    negative_bs = [b for (a, b) in negative_indices]

    N_lang_pairs = len(lang_pairs)
    positive_examples = []
    negative_examples = []

    for ix, pair in enumerate(lang_pairs, start=1):
        lang1, lang2 = pair.split("-")

        lang1_pos = dataset[lang1][positive_indices]["text"]
        lang2_pos = dataset[lang2][positive_indices]["text"]
        lang1_neg = dataset[lang1][negative_as]["text"]
        lang2_neg = dataset[lang2][negative_bs]["text"]

        _positive_examples = [
            {
                "language1": lang1,
                "language2": lang2,
                "sentence1": s1,
                "sentence2": s2,
                "label": "yes",
            }
            for s1, s2 in zip(lang1_pos, lang2_pos)
        ]
        positive_examples.extend(_positive_examples)

        _negative_examples = [
            {
                "language1": lang1,
                "language2": lang2,
                "sentence1": s1,
                "sentence2": s2,
                "label": "no",
            }
            for s1, s2 in zip(lang1_neg, lang2_neg)
        ]
        negative_examples.extend(_negative_examples)

    sentence_pairs = ds.Dataset.from_pandas(
        pd.DataFrame.from_records(positive_examples + negative_examples)
    )

    return sentence_pairs


def sentence_pair_preprocess_function(
    examples,
    label_to_id,
    tokenizer,
    max_length_tokens=128,
    sentence1_col="sentence1",
    sentence2_col="sentence2",
    label_col="label",
):
    first_sents = examples[sentence1_col]
    second_sents = examples[sentence2_col]

    labels = [label_to_id[label] for label in examples[label_col]]

    model_inputs = tokenizer(
        first_sents,
        second_sents,
        padding=True,
        truncation=True,
        max_length=max_length_tokens,
    )
    model_inputs["tokens"] = [
        tokenizer.convert_ids_to_tokens(ids) for ids in model_inputs["input_ids"]
    ]
    model_inputs["label"] = labels

    return model_inputs


def load_data_for_sentence_pair_clf(
    flores_path, ntrex_path, lang_pairs, n_pos, n_neg, intermediate_functions=None
):
    if intermediate_functions is None:
        intermediate_functions = []

    # STEP 1: LOAD RAW DATA & CREATE MAPPINGS
    flores, ntrex, language_to_id, id_to_language = load_flores_ntrex(
        flores_path, ntrex_path
    )

    # STEP 2: CREATE PAIRS
    data_for_finetune = create_pairs_data(
        ntrex, lang_pairs, n_pos, n_neg, num_rows=ntrex["eng"].num_rows
    )
    data_for_test = create_pairs_data(
        flores, lang_pairs, n_pos, n_neg, num_rows=flores["eng"].num_rows
    )

    # STEP 3: APPLY OPTIONAL INTERMEDIATE FUNCTIONS

    for func in intermediate_functions:
        try:
            print(f"Executing intermediate func: {func.__name__}")
        except ValueError:
            print(f"Executing intermediate func: {func.func.__name__}")
        data_for_finetune = func(data_for_finetune)
        data_for_test = func(data_for_test)

    return (
        data_for_finetune,
        data_for_test,
        flores,
        ntrex,
        language_to_id,
        id_to_language,
    )


def sentence_pair_experiment(
    flores_path,
    ntrex_path,
    model_name,
    lang_pairs,
    max_length_tokens,
    output_dir,
    batch_size,
    learning_rate,
    max_steps,
    compute_metrics,
    n_pos=-1,
    n_neg=-1,
    label_to_id=None,
    save_total_limit=1,
    save_steps=0,
    eval_steps=0,
    warmup_steps=0,
    logging_steps=0,
    num_train_epochs=None,
    data_loading_functions=None,
    verbose=False,
    should_resume_from_checkpoint=False,
    *args,
    **kwargs,
):
    if not data_loading_functions:
        data_loading_functions = []

    _load_data_for_sentence_pair_clf = ft.partial(
        load_data_for_sentence_pair_clf,
        flores_path=flores_path,
        ntrex_path=ntrex_path,
        lang_pairs=lang_pairs,
        n_pos=n_pos,
        n_neg=n_neg,
    )

    (
        data_ft,
        data_test,
        flores,
        ntrex,
        language_to_id,
        id_to_language,
    ) = _load_data_for_sentence_pair_clf(intermediate_functions=data_loading_functions)

    if not label_to_id:
        id_to_label = dict(
            enumerate(sorted(set(data_ft["label"]) | set(data_test["label"])))
        )

    label_to_id = invert_dict(id_to_label)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clf_preprocess_function = ft.partial(
        sentence_pair_preprocess_function,
        label_to_id=label_to_id,
        tokenizer=tokenizer,
        max_length_tokens=max_length_tokens,
    )
    data_ft = data_ft.map(clf_preprocess_function, batched=True)
    data_test = data_test.map(clf_preprocess_function, batched=True)

    label_set = set(data_ft["label"]) | set(data_test["label"])

    # Load the tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_set)
    )

    training_args = TrainingArguments(
        optim="adamw_torch",
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
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        ddp_find_unused_parameters=False,
    )

    if verbose:
        pprint(training_args)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = ft.partial(compute_metrics, id_to_label=id_to_label)

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_ft,
        eval_dataset=data_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # We don't want trainer to print stuff!
    trainer.remove_callback(PrinterCallback)

    metrics_before_train = trainer.evaluate()
    if should_resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    metrics_after_train = trainer.evaluate()

    return {
        "metrics_before_train": metrics_before_train,
        "metrics_after_train": metrics_after_train,
        "finetune_data": data_ft,
        "test_data": data_test,
    }


def compute_same_sentence_metrics(eval_pred, id_to_label, verbose=True):
    f1 = evaluate.load("f1")
    predictions, labels = eval_pred

    if verbose:
        print("predictions")
        print(predictions)

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
    class_f1s = f1.compute(predictions=predictions, references=labels, average=None)[
        "f1"
    ]

    class_ids = sorted(set(predictions) | set(labels))

    f1s_per_class = {
        id_to_label[class_id]: f1_score
        for class_id, f1_score in zip(class_ids, class_f1s)
    }
    metrics.update(
        {
            f"f1_{class_label}": f1_score
            for class_label, f1_score in f1s_per_class.items()
        }
    )

    # Confusion matrix
    labels_human_readable = [id_to_label[i] for i in labels]
    predictions_human_readable = [id_to_label[i] for i in predictions]

    _labels_set = set()

    for hum_readable in labels_human_readable, predictions_human_readable:
        _labels_set.update(hum_readable)

    _labels = sorted(_labels_set)

    cm = sk_confusion_matrix(
        y_true=labels_human_readable,
        y_pred=predictions_human_readable,
        labels=_labels,
    )
    confusion_df = pd.DataFrame(cm)
    confusion_df.columns = _labels
    confusion_df.index = _labels

    if verbose:
        print("\n\n")
        pprint(confusion_df)
        print("\n\n")

    metrics["confusion_matrix"] = confusion_df

    return metrics


def get_metrics_df(results):
    pairs = [pair for pair in results]
    confusion_matrices = {
        split: {
            pair: results[pair][f"metrics_{split}_train"]["eval_confusion_matrix"]
            for pair in pairs
        }
        for split in ["before", "after"]
    }

    metric_dfs = {}

    for split, cms in confusion_matrices.items():
        false_positives = []
        true_positives = []
        false_negatives = []
        true_negatives = []
        _pairs = []

        for pair, cm in cms.items():
            _pairs.append(pair)
            false_positives.append(cm.loc["no", "yes"])
            false_negatives.append(cm.loc["yes", "no"])
            true_positives.append(cm.loc["yes", "yes"])
            true_negatives.append(cm.loc["no", "no"])

        metric_df = pd.DataFrame(
            {
                "language_pair": _pairs,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
            }
        )
        metric_df["total_correct"] = metric_df.true_positives + metric_df.true_negatives
        metric_df["total_incorrect"] = (
            metric_df.false_positives + metric_df.false_negatives
        )
        metric_dfs[split] = metric_df

    metric_df_before = metric_dfs["before"]
    metric_df_after = metric_dfs["after"]
    metric_df_before_after = pd.merge(
        left=metric_df_before,
        right=metric_df_after,
        left_on="language_pair",
        right_on="language_pair",
        suffixes=["_before", "_after"],
    )

    return metric_df_before_after


@click.command()
@click.option("--flores-path", type=str, default=os.environ.get("FLORES_PATH"))
@click.option("--ntrex-path", type=str, default=os.environ.get("NTREX_PATH"))
@click.option(
    "--language",
    type=str,
    required=True,
    help="Language to fix as the first part of all the pairs",
)
@click.option("--model-name", type=str, default=MBERT)
@click.option("--output-dir", type=click.Path(dir_okay=True), required=True)
@click.option(
    "--max-steps", type=int, default=0, help="Maximum number of training steps"
)
@click.option(
    "--num-train-epochs", type=int, default=0, help="Number of epochs to train"
)
@click.option("--batch-size", type=int, default=160, help="Training batch size")
@click.option(
    "--learning-rate", type=float, default=5e-5, help="Learning rate for training"
)
@click.option(
    "--max-length-tokens", type=int, default=128, help="Maximum number of tokens"
)
@click.option("--save-total-limit", type=int, default=4)
@click.option("--save-steps", help="Save every `save-steps` steps.", default=400)
@click.option("--eval-steps", help="Eval every `eval-steps` steps.", default=-1)
@click.option("--warmup-steps", default=0)
@click.option("--logging-steps", default=0)
@click.option("--debug", is_flag=True)
@click.option("--n-pos", type=int, default=-1, help="Number of positive examples")
@click.option("--n-neg", type=int, default=-1, help="Number of negative examples")
@click.option("--should-resume-from-checkpoint", is_flag=True)
def main(
    flores_path,
    ntrex_path,
    language,
    model_name,
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
    debug,
    n_pos,
    n_neg,
    should_resume_from_checkpoint,
):
    assert max_steps or num_train_epochs, (
        f"Either max_steps or num_train_epochs must be specified!"
        f"Got max_steps={max_steps} "
        f"num_train_epochs={num_train_epochs}"
    )

    # Disable datasets progress bar
    ds.utils.logging.disable_progress_bar()

    _compute_same_sentence_metrics = ft.partial(
        compute_same_sentence_metrics, verbose=False
    )
    same_sentence_experiment = ft.partial(
        sentence_pair_experiment,
        flores_path=flores_path,
        ntrex_path=ntrex_path,
        model_name=model_name,
        max_length_tokens=max_length_tokens,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        data_loading_functions=None,
        compute_metrics=_compute_same_sentence_metrics,
        verbose=False,
        n_pos=n_pos,
        n_neg=n_neg,
        debug=debug,
        should_resume_from_checkpoint=should_resume_from_checkpoint,
    )

    _, __, language_to_id, ___ = load_flores_ntrex(flores_path, ntrex_path)
    del _, __, ___

    if debug:
        pprint("Computing all available language pairs...")
    all_available_lang_pairs = [
        f"{a}-{b}" for a, b in it.product(language_to_id, language_to_id) if a != b
    ]

    if debug:
        pprint("Filtering language pairs...")
    filtered_lang_pairs = sorted(
        [p for p in all_available_lang_pairs if p.startswith(f"{language}-")]
    )

    if debug:
        pprint("Running experiments!")

    all_same_sentence_exp_results = {}

    with Progress() as prog:
        lang_pair_progress = prog.add_task(
            "[red]Running experiment on pairs...", total=len(filtered_lang_pairs)
        )
        for pair in filtered_lang_pairs:
            prog.console.print(f"Pair: {pair}")
            all_same_sentence_exp_results[pair] = same_sentence_experiment(
                lang_pairs=[pair]
            )
            prog.update(lang_pair_progress, advance=1)

    metric_df_before_after = get_metrics_df(all_same_sentence_exp_results)

    metric_df_before_after.to_csv(
        Path(output_dir) / "metrics_before_after.csv", index=False
    )


if __name__ == "__main__":
    main()
