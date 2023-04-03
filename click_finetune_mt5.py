import click
import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
import pudb


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
    default="google/mt5-small",
    help="Name or path of the pre-trained model to use",
)
@click.option(
    "--source-lang",
    type=str,
    default="src",
    help="Name of the source language in the dataset",
)
@click.option(
    "--target-lang",
    type=str,
    default="tgt",
    help="Name of the target language in the dataset",
)
@click.option(
    "--finetune-langs",
    type=lambda x: str(x).split(","),
    default="",
    help="Comma-separated list of languages to fine tune on",
)
@click.option(
    "--prefix",
    type=str,
    default="translate: ",
    help="Prefix to add to source sentences for translation",
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
@click.option("--batch-size", type=int, default=1, help="Training batch size")
@click.option(
    "--learning-rate", type=float, default=3e-4, help="Learning rate for training"
)
@click.option(
    "--max-length-tokens", type=int, default=128, help="Maximum number of tokens"
)
@click.option("--save-total-limit", type=int, default=5)
@click.option("--learning-rate", type=float, default=5e-5)
@click.option("--save-steps", help="Save every `save-steps` steps.", default=100)
@click.option("--eval-steps", help="Eval every `eval-steps` steps.", default=100)
@click.option("--warmup-steps", default=100)
@click.option("--logging-steps", default=50)
@click.option("--predict-with-generate", is_flag=True)
def train_mt5(
    flores_path,
    ntrex_path,
    model_name,
    source_lang,
    target_lang,
    finetune_langs,
    prefix,
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
    predict_with_generate,
):

    assert max_steps ^ num_train_epochs

    # Load the MT5 tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    import datasets as ds

    flores = ds.load_from_disk(flores_path)
    ntrex = ds.load_from_disk(ntrex_path)

    # Define new dataset based on source and target lang as well as finetuning langs
    source_data_finetune = ds.concatenate_datasets(
        [ntrex[lang] for lang in finetune_langs]
    )  # .rename_column("text", "source")
    target_data_finetune = ds.concatenate_datasets(
        [ntrex[target_lang] for lang in finetune_langs]
    )  # .rename_column("text", "target")

    data_for_finetune = ds.Dataset.from_dict(
        {
            "source": source_data_finetune["text"],
            "target": target_data_finetune["text"],
            "source_language": source_data_finetune["language"],
            "target_language": target_data_finetune["language"],
        }
    )

    # Define test data
    source_data_test = flores[source_lang]
    target_data_test = flores[target_lang]

    data_for_test = ds.Dataset.from_dict(
        {
            "source": source_data_test["text"],
            "target": target_data_test["text"],
            "source_language": source_data_test["language"],
            "target_language": target_data_test["language"],
        }
    )

    def preprocess_function(examples):

        prefixes = [
            f"Translate from {srclang} to {tgtlang}: "
            for srclang, tgtlang in zip(
                examples["source_language"], examples["target_language"]
            )
        ]

        inputs = [f"{pf}{ex}" for pf, ex in zip(prefixes, examples["source"])]
        targets = examples["target"]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length_tokens, truncation=True
        )
        model_inputs["input_tokens"] = [
            tokenizer.convert_ids_to_tokens(ids) for ids in model_inputs["input_ids"]
        ]
        model_inputs["output_tokens"] = [
            tokenizer.convert_ids_to_tokens(ids) for ids in model_inputs["labels"]
        ]

        return model_inputs

    data_for_finetune = data_for_finetune.map(preprocess_function, batched=True)
    data_for_test = data_for_test.map(preprocess_function, batched=True)

    # SacreBLEU
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):

        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    # Define the training arguments
    if max_steps:
        how_long_to_train_args = {
            "evaluation_strategy": "steps",
            "max_steps": max_steps
        }
    else:
        how_long_to_train_args = {
            "evaluation_strategy": "epoch",
            "num_train_epochs": num_train_epochs
        }


    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=save_total_limit,
        learning_rate=learning_rate,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        overwrite_output_dir=True,
        predict_with_generate=predict_with_generate,
        eval_accumulation_steps=30,
        **how_long_to_train_args,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    # Define the trainer
    trainer = Seq2SeqTrainer(
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
    train_mt5()
