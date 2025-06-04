from transformers import AutoTokenizer, AutoModel

def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_id)

def load_model_tokenizer(args):
    tokenizer = load_tokenizer(args)
    model = AutoModel.from_pretrained(args.model_id)

    return model, tokenizer