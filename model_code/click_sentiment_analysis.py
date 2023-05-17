import click
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import datasets as ds

MAX_LENGTH_TOKENS=128

@click.command()
@click.option("--dataset-path", type=click.Path(exists=True))
@click.option(
    "--model-name",
    default="nlptown/bert-base-multilingual-uncased-sentiment",
    help="Name of the pre-trained model to use",
)
@click.option("--num-epochs", default=2, help="Number of epochs to train the model for")
@click.option("--batch-size", default=16, help="Batch size for training")
def main(dataset_path, model_name, num_epochs, batch_size):

    # Initialize the tokenizer and the model
    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=MAX_LENGTH_TOKENS)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id))

    # Load the dataset and preprocess it
    dataset_dict: ds.DatasetDict = ds.load_from_disk(dataset_path)
    new_dataset_dict = ds.DatasetDict()
    for lang, _ds in dataset_dict.items():
        # split to train and test sets with 20% in test
        ds_with_split = _ds.train_test_split(test_size=0.2).rename_column("label", "labels")
        new_dataset_dict[lang] = ds_with_split

    dataset_dict = new_dataset_dict

    # Create the training and validation datasets
    train_dataset = ds.concatenate_datasets([dataset_dict[lang]["train"] for lang in dataset_dict])
    test_dataset = ds.concatenate_datasets([dataset_dict[lang]["test"] for lang in dataset_dict])

    def preprocess_function(examples):
        inputs = examples["text"]
        labels = [label_to_id[label] for label in examples["labels"]]

        model_inputs = tokenizer(
            inputs, padding=True, truncation=True, max_length=MAX_LENGTH_TOKENS
        )
        model_inputs["labels"] = labels

        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Define a training function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    import pudb; pudb.set_trace()
    trainer.train()
    metrics = trainer.evaluate()


if __name__ == "__main__":
    main()
