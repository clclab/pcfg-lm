import argparse
import os

from pcfg import PCFGConfig, PCFG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--grammar_file", help="Path to NLTK grammar", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to write corpora to", required=True
    )
    parser.add_argument(
        "--min_length", help="Minimal sentence length", default=3, type=int
    )
    parser.add_argument(
        "--max_length", help="Maximal sentence length", default=29, type=int
    )
    parser.add_argument(
        "--corpus_size", help="Total corpus size", required=True, type=int
    )
    parser.add_argument("--start_symbol", help="Start symbol in PCFG", default="ROOT_0")
    parser.add_argument(
        "--split_ratio",
        help="Train/test/dev split. Defaults to 1.0/0.0/0.0 to only generate train corpus",
        default="1.0/0.0/0.0",
    )
    parser.add_argument(
        "--store_trees", help="Store trees of sampled strings", action="store_true"
    )
    args = vars(parser.parse_args())

    split_ratio = tuple(map(float, args["split_ratio"].split("/")))

    config = PCFGConfig(
        min_length=args["min_length"],
        max_length=args["max_length"],
        max_depth=args["max_length"],
        corpus_size=args["corpus_size"],
        grammar_file=args["grammar_file"],
        start=args["start_symbol"],
        allow_duplicates=True,
        split_ratio=split_ratio,
        verbose=True,
        store_trees=args["store_trees"],
    )

    lm_language = PCFG(config)

    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    with open(os.path.join(args["output"], "train.txt"), "w") as f:
        f.write("\n".join(lm_language.train_corpus))

    if split_ratio[1] > 0.0:
        with open(os.path.join(args["output"], "dev.txt"), "w") as f:
            f.write("\n".join(lm_language.dev_corpus))

    if split_ratio[2] > 0.0:
        with open(os.path.join(args["output"], "test.txt"), "w") as f:
            f.write("\n".join(lm_language.test_corpus))
