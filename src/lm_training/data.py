import os
from typing import Optional, Tuple

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast


def tokenize_wrapper(tokenizer):
    def tokenize(element, min_length=0, max_length=128):
        input_ids = [
            item
            for item in tokenizer(element["text"])["input_ids"]
            if max_length > len(item) > min_length
        ]
        return {"input_ids": input_ids}

    return tokenize


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    data_dir: str,
    train_size: Optional[int] = None,
    dev_size: Optional[int] = None,
    test_size: Optional[int] = None,
    train_file: str = 'train.txt',
    dev_file: str = 'dev.txt',
    test_file: str = 'test.txt',
    eval_file: Optional[str] = 'eval.txt',
) -> DatasetDict:
    raw_train = load_dataset("text", data_files=os.path.join(data_dir, train_file))[
        "train"
    ]
    raw_dev = load_dataset("text", data_files=os.path.join(data_dir, dev_file))[
        "train"
    ]
    raw_test = load_dataset("text", data_files=os.path.join(data_dir, test_file))[
        "train"
    ]
    if eval_file is not None:
        raw_eval = load_dataset("text", data_files=os.path.join(data_dir, eval_file))[
            "train"
        ]
    else:
        raw_eval = None

    if train_size is not None:
        raw_train = raw_train.shuffle().select(range(train_size))
    if dev_size is not None:
        raw_dev = raw_dev.shuffle().select(range(dev_size))
    if test_size is not None:
        raw_test = raw_test.shuffle().select(range(test_size))

    dataset_dict = {
        "train": raw_train,
        "valid": raw_dev,
        "test": raw_test,
    }

    if raw_eval is not None:
        dataset_dict["eval"] = raw_eval

    raw_datasets = DatasetDict(dataset_dict)

    tokenized_datasets = raw_datasets.map(
        tokenize_wrapper(tokenizer),
        batched=True,
    )

    return tokenized_datasets
