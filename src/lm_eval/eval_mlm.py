import argparse

from transformers import AutoModelForMaskedLM
from minicons import scorer


def compute_mlm_scores(eval_corpus_path: str, model_name: str, device: str):
    mlm_model = scorer.MaskedLMScorer(model_name, device)

    with open(eval_corpus_path) as f:
        eval_corpus = f.read().strip().split("\n")

    scores = mlm_model.sequence_score(eval_corpus, reduction=lambda x: x)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--corpus", help="Path to evaluation corpus", required=True
    )
    parser.add_argument("-m", "--model_name", help="Path saved LM", required=True)
    parser.add_argument("-d", "--device", help="Device", default="cuda")
    args = vars(parser.parse_args())

    scores = compute_mlm_scores(args["corpus"], args["model_name"], args["device"])
    print(scores[:2])
