import sys

from datasets import Dataset
from transformers import (DataCollatorForSeq2Seq, MT5ForConditionalGeneration,
                          MT5Tokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainingArguments,
                          default_data_collator)


# Define a function to load the data from the source and target files
def load_data(source_file, target_file):
    with open(source_file, "r", encoding="utf-8") as f:
        source_lines = [line.strip() for line in f]
    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = [line.strip() for line in f]

    # Create a dataset with the source and target sentences
    dataset = Dataset.from_dict({"src": source_lines, "tgt": target_lines})

    return dataset


# Load the MT5 tokenizer and model
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

source_lang = "src"
target_lang = "tgt"
prefix = "translate: "


def preprocess_function(examples):
    inputs = [f"{prefix}{ex}" for ex in examples["src"]]
    targets = examples["tgt"]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=128, truncation=True
    )

    return model_inputs


src_path, tgt_path = sys.argv[1:3]
data = (
    load_data(src_path, tgt_path)
    .train_test_split(test_size=0.15)
    .map(preprocess_function, batched=True)
)
train_data = data["train"]
test_data = data["test"]

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    # num_train_epochs=1,
    max_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    save_total_limit=1,
    learning_rate=3e-4,
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
