# !pip install -qqq -U transformers datasets accelerate peft trl bitsandbytes --progress-bar off
# !pip install -qqq flash-attn deepspeed

import torch
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# 1. CONFIG
# Model
base_model = "unsloth/gemma-1.1-2b-it"
new_model = "physiology-8k-gemma-2b"

max_seq_length = 4096

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# 2. BASE MODEL
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model)

# 3. DATASET
dataset = load_dataset("sardukar/physiology-mcqa-8k", split="train")


def format_prompt(row):
    prompt = row["prompt"][1:]
    chosen = row["chosen"][1:]
    rejected = row["rejected"][1:]
    row["prompt"] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    row["chosen"] = tokenizer.apply_chat_template(chosen, tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(rejected, tokenize=False)
    return row


dataset = dataset.map(format_prompt)
dataset = dataset.train_test_split(test_size=0.2)

# 4. TRAINER
orpo_trainer = ORPOTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    tokenizer = tokenizer,
    peft_config=peft_config,
    args = ORPOConfig(
        max_length = max_seq_length,
        max_prompt_length = max_seq_length//2,
        max_completion_length = max_seq_length//2,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        save_steps=200,
        beta = 0.1,
        logging_steps = 1,
        optim = "adamw_torch",
        lr_scheduler_type = "linear",
        num_train_epochs = 3,
        output_dir = "outputs",
        remove_unused_columns=False,
        fp16=True,
        bf16=False
    )
)

orpo_trainer.train(resume_from_checkpoint=False)
orpo_trainer.save_model(new_model + "-qlora")
