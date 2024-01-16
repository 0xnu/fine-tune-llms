## Fine-Tuning LLMs

A template or starting point for fine-tuning large language models (LLMs) using Hugging Face's `transformers` library. You can customise the [scripts](./script/) to fit the specifics of your project, dataset, and any modifications you may require during the setup or training process.

### Prerequisites

- Python 3.6+
- PyTorch
- Transformers library: `transformers>=4.0.0`
- Datasets library: `datasets>=1.0.0`
- Accelerate library (for distributed training): `accelerate>=0.20.1`

### Installation

To install the required libraries, run the following commands:

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following

```python
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m pip install --upgrade pip
deactivate
```

### Split Data Script

The key components of the [split_dataset.py](./script/split_data.py) script, which handles splitting the `train.jsonl` file into a new training set and a validation set, are described below:

1. **Importing Libraries**: The script uses the `json` library to handle JSON data and `train_test_split` from `scikit-learn` to facilitate the splitting of the dataset.

2. **Defining File Paths**: The script sets up file paths for the original training dataset (`train.jsonl`) as well as the paths where we will save the new training and validation datasets (`train_split.jsonl` and `validation_split.jsonl`, respectively).

3. **Loading Data**: It reads the original `train.jsonl` file and deserialises each line (a JSON entry) from a JSON string to a Python dictionary—collecting all entries into a list.

4. **Splitting Data**: The loaded data is split into new training and validation sets using the `train_test_split` function. The split ratio is 80% for training and 20% for validation, providing a random seed for reproducibility.

5. **Writing New Data to Files**: Writes the new training and validation datasets to their respective `.jsonl` files. It serialised each entry back to a JSON string, with each line written to the corresponding file.

6. **Logging Information**: The script prints out the number of entries in the new training and validation datasets to provide confirmation and a quick overview of the split.

The script encapsulates the data preparation step, which often precedes training machine learning models, ensuring you have separate datasets for training and validating the performance of your models. It's necessary to prevent overfitting and evaluate your model's generalisation capabilities.

### Usage

To start the training process, run the training script with the following command:

```bash
python3 ./script/split_data.py
```

### Training Script

The key components of the training script are:

1. **Importing Libraries**: The script imports necessary Python libraries, such as `json`, `torch`, `datasets`, and classes from `transformers` for model loading, data tokenisation, model training, etc.

2. **GPU Availability Check**: The script checks if a GPU is available for training, which can significantly speed up the process.

3. **Dataset Paths**: The script defines file paths for the training and validation datasets; both are in JSONL format and located inside the `data` directory.

4. **Tokenizer Loading and Configuration**: The tokeniser associated with the model, [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), is loaded from the Hugging Face Model Hub. Additionally, the script sets the padding token for the tokeniser if it is not defined.

5. **Model Loading and Configuration**: The pre-trained language model is loaded, with its embeddings resized to accommodate any new tokens added by the tokeniser.

6. **Data Loading and Preprocessing**: The training and validation datasets are loaded and processed using the tokeniser. A custom tokenise function is applied to the datasets to convert raw text into the format expected by the model.

7. **Data Collator Definition**: A `DataCollatorForLanguageModeling` is created to handle dynamic padding of input sequences during training.

8. **Training Arguments Configuration**: `TrainingArguments` are defined to specify the output directory, number of epochs, batch size, saving strategy, evaluation strategy, and more.

9. **Trainer Initialization**: The script initialises a `Trainer` instance with the model, training arguments, data collator, and training and evaluation datasets.

10. **Model Training**: The `train` method of the `Trainer` instance fine-tunes the model on the training dataset whilst periodically evaluating its performance on the validation dataset.

11. **Saving the Fine-Tuned Model**: After training, the script saves the fine-tuned model to the output directory.

The [train_data.py](./script/train_data.py) script is a starting point! Feel free to revise it for your use case.

### Usage

To start the training process, run the training script with the following command:

```bash
python3 ./script/train_data.py
```

## Troubleshooting

If you encounter any issues during installation or training, ensure that:

- You installed the dependencies correctly.
- The `train.jsonl` file is appropriately formatted and accessible.
- The training script has the correct path to the `train.jsonl` file.
- The tokenizer is appropriately configured with a pad token.

For any error messages, please refer to the error-specific tips provided in the logs and address them accordingly.

### Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

### Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/moatsystems/imdb_scrapy/tags).

### License

This project is licensed under the [MIT License](LICENSE) - see the file for details.

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu).