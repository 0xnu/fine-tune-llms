#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: split_data.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Sunday January 14 21:05:00 2024
@updated: Friday January 26 5:15:00 2024
@desc: train-validation-test splits for LLMs fine-tuning.
@run: python3 split_data.py
"""

import json
from sklearn.model_selection import train_test_split

# Define the path to the dataset and where to store the splits
data_folder = 'data'
original_data_jsonl_path = f'{data_folder}/train.jsonl'
train_jsonl_path = f'{data_folder}/train_split.jsonl'
validation_jsonl_path = f'{data_folder}/validation_split.jsonl'
test_jsonl_path = f'{data_folder}/test_split.jsonl'

# Read the data from the original train.jsonl file
with open(original_data_jsonl_path, 'r', encoding='utf-8') as jsonl_file:
    lines = jsonl_file.readlines()
    data = [json.loads(line) for line in lines]

# First, split to separate the 15% test data
train_and_validation_data, test_data = train_test_split(data, test_size=0.15, random_state=42)

# Further, split the remaining data into training and validation sets
# Doing so splits the remaining data into 70% train and 15% validation
train_data_size = 0.7 / (0.7 + 0.15)
train_data, validation_data = train_test_split(train_and_validation_data, test_size=1-train_data_size, random_state=42)

# Write the new training data to train_split.jsonl
with open(train_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in train_data:
        jsonl_file.write(json.dumps(entry) + '\n')

# Write the validation data to validation_split.jsonl
with open(validation_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in validation_data:
        jsonl_file.write(json.dumps(entry) + '\n')

# Write the test data to test_split.jsonl
with open(test_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in test_data:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"Train/validation/test split completed. Training data: {len(train_data)} entries.")
print(f"Validation data: {len(validation_data)} entries.")
print(f"Test data: {len(test_data)} entries.")