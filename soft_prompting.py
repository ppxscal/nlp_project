from datasets import load_dataset
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments

# List of dataset splits
splits = ["cos_e_v1.11_aligned_with_common_sense", 
        #   "cos_e_v1.11_description_question_option_id", 
        #   "cos_e_v1.11_description_question_option_text",
        #   "cos_e_v1.11_explain_why_human", 
        #   "cos_e_v1.11_generate_explanation_given_text", 
        #   "cos_e_v1.11_i_think", 
        #   "cos_e_v1.11_question_description_option_id", 
        #   "cos_e_v1.11_question_description_option_text", 
        #   "cos_e_v1.11_question_option_description_id", 
        #   "cos_e_v1.11_question_option_description_text", 
        #   "cos_e_v1.11_rationale"
          ]


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt_tuning_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=10
)

peft_model = get_peft_model(model, prompt_tuning_config)

# split_data = {}
# for split in splits:
#     split_data[split] = load_dataset("bigscience/P3", split)
    # print(f"First few examples from {split}:")
    # print(split_data[split].keys())
    # print(split_data[split]['train'][:5])
    # print(split_data[split]['validation'][:5])

# data = split_data['cos_e_v1.11_aligned_with_common_sense']


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

data = load_dataset("bigscience/P3", 'cos_e_v1.11_aligned_with_common_sense')

# Check if the dataset is loaded correctly
if 'train' not in data or len(data['train']) == 0:
    raise ValueError("Training dataset is empty or not loaded correctly")

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data['train'],
)

trainer.train()

trainer.evaluate()

