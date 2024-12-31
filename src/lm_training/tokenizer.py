from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizer, BertTokenizer
from transformers import PreTrainedTokenizerFast


def create_tokenizer(
    corpus: str,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
    bos_token: str = "<bos>",
    min_freq: int = 1,
    add_bos_token: bool = False,
):
    special_tokens = [unk_token, pad_token, mask_token, bos_token]
    tokenizer_trainer = WordLevelTrainer(
        min_frequency=min_freq, special_tokens=special_tokens
    )

    base_tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
    base_tokenizer.pre_tokenizer = WhitespaceSplit()

    base_tokenizer.train([corpus], trainer=tokenizer_trainer)

    print("Tokenizer size:", len(base_tokenizer.get_vocab()))

    if add_bos_token:
        base_tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A",
            special_tokens=[(bos_token, base_tokenizer.token_to_id(bos_token))],
        )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        mask_token=mask_token,
        pad_token=pad_token,
        bos_token=bos_token,
        unk_token=unk_token,
        additional_special_tokens=special_tokens,
    )

    return tokenizer


class CustomTokenizer(PreTrainedTokenizer):
    """Legacy tokenizer with custom whitespace tokenization"""

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
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
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
