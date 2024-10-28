import argparse

from transformers import AutoModelForMaskedLM, AutoTokenizer
from minicons import scorer


def eval_mlm(eval_corpus_path: str, model_name: str, device: str):
    ilm_model = scorer.MaskedLMScorer(model_name, device)

    with open(eval_corpus_path) as f:
        eval_corpus = f.read().strip().split('\n')
    
    scores = ilm_model.sequence_score(eval_corpus, reduction = lambda x: x)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", help="Path to evaluation corpus", required=True)
    parser.add_argument("-m", "--model_name", help="Path saved LM", required=True)
    parser.add_argument("-d", "--device", help="Device", default='cuda')
    args = vars(parser.parse_args())

    model = AutoModelForMaskedLM.from_pretrained(args['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    print(tokenizer)
    # scores = eval_mlm(args['corpus'],  args['model_name'], args['device'])
    # print(scores[:2])
