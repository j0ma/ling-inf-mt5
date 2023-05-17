import click
import datasets as ds
import evaluate
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MAX_LENGTH_TOKENS = 128

f1 = evaluate.load("f1")


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
    id_to_label = {v: k for k, v in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding=True, truncation=True, max_length=MAX_LENGTH_TOKENS
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_to_id)
    )

    # Load the dataset and preprocess it
    dataset_dict: ds.DatasetDict = ds.load_from_disk(dataset_path)

    # Create the training and validation datasets
    train_dataset = dataset_dict['train']
    test_dataset = dataset_dict['test']

    def preprocess_function(examples):
        inputs = examples["text"]
        labels = [label_to_id[label] for label in examples["label"]]
        del examples['label']

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
            f"f1_{id_to_label[class_id]}": f1_score
            for class_id, f1_score in zip(class_ids, class_f1s)
        }
        metrics.update(f1s_per_class)

        return metrics

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./sentiment_results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_dir="./sentiment_logs",
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

    trainer.train()
    metrics = trainer.evaluate()


if __name__ == "__main__":
    main()
