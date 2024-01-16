#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: train_data.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Sunday January 14 21:52:00 2024
@desc: train-validate data for LLMs fine-tuning.
@run: python3 train_data.py
"""

import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the path to your dataset
train_jsonl_path = './data/train_split.jsonl'
validation_jsonl_path = './data/validation_split.jsonl'

# Load the tokenizer and model from Hugging Face Model Hub
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')

# Set padding token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and configure the model
model = AutoModelForCausalLM.from_pretrained('mistralai/Mixtral-8x7B-v0.1')
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Load and preprocess the datasets
def tokenize_function(examples):
    return tokenizer(examples['question'], padding='max_length', truncation=True)

# Load the datasets
datasets = load_dataset('json', data_files={'train': train_jsonl_path, 'validation': validation_jsonl_path})

# Tokenize all datasets
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Specify a data collator for padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir='./mixtral_finetuned',   # Directory to save the fine-tuned model
    overwrite_output_dir=True,          # Overwrite the content of the output directory
    num_train_epochs=3,                 # Number of training epochs
    per_device_train_batch_size=1,      # Batch size per device (adjust based on GPU memory)
    save_strategy="epoch",              # Save strategy
    save_total_limit=2,                 # Only last 2 checkpoints are saved
    evaluation_strategy="epoch",        # Evaluate at the end of each epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./mixtral_finetuned')