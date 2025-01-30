import os
from typing import Optional, Tuple

from datasets import DatasetDict
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
    dsd: DatasetDict,
    train_size: Optional[int] = None,
    dev_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> DatasetDict:

    for split, size in zip(['train', 'dev', 'test'], [train_size, dev_size, test_size]):
        if size is not None:
            dsd[split] = dsd[split].filter(lambda e,i: i < size, with_indices=True)

    return (dsd
            .shuffle()
            .map(
                tokenize_wrapper(tokenizer),
                batched=True,
            )
        )
