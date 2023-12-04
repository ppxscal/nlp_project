from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM, TrainingArguments,Trainer, DataCollatorForLanguageModeling)
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch

# Load the dataset
dataset = load_dataset("bigscience/P3", "cos_e_v1.11_aligned_with_common_sense")

# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["inputs_pretokenized"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create a data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
)

prompt_tuning_config = PromptTuningConfig(
    task_type=TaskType.QUESTION_ANS,
    prompt_init=PromptTuningInit.from_string(""),
    n_tokens=20,
    tokenizer=tokenizer,
)

model = get_peft_model(
    model=model,
    config=prompt_tuning_config,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    optimizers=(torch.optim.Adam(model.get_prompt_params(), lr=2e-5), None),
)

# Train the model
trainer.train()