import click
from datasets import Dataset
from transformers import (DataCollatorForSeq2Seq, MT5ForConditionalGeneration,
                          MT5Tokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainingArguments,
                          default_data_collator)


@click.command()
@click.option(
    "--source-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to source file",
)
@click.option(
    "--target-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to target file",
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
    "--max-steps", type=int, default=10, help="Maximum number of training steps"
)
@click.option("--batch-size", type=int, default=1, help="Training batch size")
@click.option(
    "--learning-rate", type=float, default=3e-4, help="Learning rate for training"
)
@click.option(
    "--max-length-tokens", type=int, default=128, help="Maximum number of tokens"
)
def train_mt5(
    source_path,
    target_path,
    model_name,
    source_lang,
    target_lang,
    prefix,
    output_dir,
    max_steps,
    batch_size,
    learning_rate,
    max_length_tokens
):
    # Load the MT5 tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # Define a function to load the data from the source and target files
    def load_data(source_file, target_file):
        with open(source_file, "r", encoding="utf-8") as f:
            source_lines = [line.strip() for line in f]
        with open(target_file, "r", encoding="utf-8") as f:
            target_lines = [line.strip() for line in f]

        # Create a dataset with the source and target sentences
        dataset = Dataset.from_dict({source_lang: source_lines, target_lang: target_lines})

        return dataset

    def preprocess_function(examples):
        inputs = [f"{prefix}{ex}" for ex in examples[source_lang]]
        targets = examples[target_lang]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length_tokens, truncation=True
        )

        return model_inputs

    data = (
        load_data(source_path, target_path)
        .train_test_split(test_size=0.15)
        .map(preprocess_function, batched=True)
    )
    train_data = data["train"]
    test_data = data["test"]

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        save_total_limit=1,
        learning_rate=learning_rate,
        save_steps=10,
        eval_steps=10,
        warmup_steps=0,
        logging_steps=1,
        overwrite_output_dir=True,
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()


if __name__ == "__main__":
    train_mt5()
