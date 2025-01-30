# coding: utf-8
import datasets
from tokenizer import create_tokenizer
from transformers import PreTrainedTokenizerFast
from data import tokenize_wrapper

dsd = datasets.DatasetDict.load_from_disk("../../resources/corpora/nltk/dataset")

from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

special_tokens = ["<unk>", "<pad>", "<mask>", "<bos>"]
tokenizer_trainer = WordLevelTrainer(
        min_frequency=min_freq, special_tokens=special_tokens
    )
base_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
base_tokenizer.pre_tokenizer = WhitespaceSplit()
base_tokenizer.train_from_iterator(dsd['train']['text'], trainer=tokenizer_trainer)

tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        mask_token="<mask>",
        pad_token="<pad>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
dsd = dsd.shuffle().map(tokenize_wrapper(tokenizer), batched=True)
