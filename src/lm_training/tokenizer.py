import json

from transformers import PreTrainedTokenizer, BertTokenizer
from collections import Counter


def create_tokenizer(
    corpus: str, 
    unk_token: str = '<unk>', 
    pad_token: str = '<pad>', 
    mask_token: str = '<mask>',
    min_freq: int = 1,
):
    vocab = create_vocab(corpus, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, min_freq=min_freq)

    tokenizer = create_tf_tokenizer_from_vocab(vocab, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token)
    
    return tokenizer


def create_vocab(
    corpus: str, 
    unk_token: str = '<unk>', 
    pad_token: str = '<pad>', 
    mask_token: str = '<mask>',
    min_freq: int = 1,
):
    with open(corpus) as f:
        train = f.read().split('\n')

    token_freqs = Counter()

    for sen in train:
        for w in sen.split():
            token_freqs[w] += 1
            
    vocab = {unk_token: 0, pad_token: 1, mask_token: 2}
    
    for w, freq in token_freqs.most_common():
        if freq >= min_freq:
            vocab[w] = len(vocab)

    return vocab


class CustomTokenizer(PreTrainedTokenizer):
    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def save_vocabulary(self, *args, **kwargs):
        return BertTokenizer.save_vocabulary(self, *args, **kwargs)
    
    def _tokenize(self, sen: str):
        return sen.split(" ")
    
    def _convert_token_to_id(self, w: str):
        return self.vocab.get(w, self.vocab[self.unk_token])


def create_tf_tokenizer_from_vocab(
    vocab, 
    unk_token: str = '<unk>', 
    pad_token: str = '<pad>',
    mask_token: str = '<mask>',
):
    tokenizer = CustomTokenizer()

    tokenizer.added_tokens_encoder = vocab
    tokenizer.added_tokens_decoder = {idx: w for w, idx in vocab.items()}
    tokenizer.vocab = tokenizer.added_tokens_encoder
    tokenizer.ids_to_tokens = tokenizer.added_tokens_decoder
    
    tokenizer.unk_token = unk_token
    tokenizer.pad_token = pad_token
    tokenizer.mask_token = mask_token

    return tokenizer
    
    
def load_pretrained_tokenizer(path: str):
    with open(path) as f:
        vocab = json.load(f)

    tokenizer = create_tf_tokenizer_from_vocab(vocab)
    
    return tokenizer
