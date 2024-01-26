#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: train_data.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Sunday January 14 21:52:00 2024
@updated: Friday January 26 5:24:00 2024
@desc: train-validate-test data for LLMs fine-tuning.
@run: python3 train_data.py
"""

import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a model with specified configuration.')
parser.add_argument('--model', type=str, required=True, choices=['tinymistral', 'tinyllama'],
                    help='Model identifier to use for training (e.g., tinymistral, tinyllama)')

# Print the help message
args = parser.parse_args()

# Define mapping from command line argument to model name
model_names = {
    'tinymistral': 'Felladrin/TinyMistral-248M-SFT-v4',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
}
model_name = model_names[args.model]

# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the path to your dataset
train_jsonl_path = './data/train_split.jsonl'
validation_jsonl_path = './data/validation_split.jsonl'
test_jsonl_path = './data/test_split.jsonl'

# Load the tokenizer and model from Hugging Face Model Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and configure the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Load and preprocess the datasets
def tokenize_function(examples):
    return tokenizer(examples['question'], padding='max_length', truncation=True)

# Load the datasets
datasets = load_dataset('json', data_files={
    'train': train_jsonl_path,
    'validation': validation_jsonl_path,
    'test': test_jsonl_path
})

# Tokenize all datasets
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Specify a data collator for padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=f'./{args.model}_finetuned',  # Directory to save the fine-tuned model
    overwrite_output_dir=True,               # Overwrite the content of the output directory
    num_train_epochs=3,                      # Number of training epochs
    per_device_train_batch_size=1,           # Batch size per device (adjust based on GPU memory)
    save_strategy="epoch",                   # Save strategy
    save_total_limit=2,                      # Only last 2 checkpoints are saved
    evaluation_strategy="epoch"              # Evaluate at the end of each epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    # No need to pass the test set to Trainer since we are not using it for training
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(f'./{args.model}_finetuned')

# After training, we can evaluate the model on the test set
eval_results = trainer.evaluate(tokenized_datasets['test'])
print(f"Test set evaluation results for {model_name}:", eval_results)