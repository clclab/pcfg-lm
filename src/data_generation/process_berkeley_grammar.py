"""
Processes the Petrov .grammar and .lexicon files and writes it to an NLTK format.
"""

import argparse
from collections import defaultdict


def read_grammar_rules(path, threshold=1e-7):
    with open(f"{path}.grammar") as f:
        lines = [l.strip() for l in f]

    # skip root and A -> A recursions
    # ROOT_0 is start terminal!
    non_rec_rules = [
        l
        for l in lines
        if not (l.split()[0] in l.split()[1:] and len(l.split()) == 4)
        and float(l.split()[-1]) > threshold
    ]

    return non_rec_rules


def read_lexicon_rules(path, threshold=1e-7):
    with open(f"{path}.lexicon") as f:
        raw_lexicon_rules = [l.strip().replace("NaN", "0.0") for l in f]

    lexicon_rules = []

    for line in raw_lexicon_rules:
        nt = line.split()[0]
        word = line.split()[1].replace("'", "<apostrophe>")
        probs = eval(" ".join(line.split()[2:]))

        for idx, prob in enumerate(probs):
            if prob > threshold:
                new_line = f"{nt}_{idx} -> '{word}' {prob}"
                lexicon_rules.append(new_line)

    return lexicon_rules


def normalize_rules(all_rules):
    prob_sums = defaultdict(float)

    for rule in all_rules:
        prob_sums[rule.split()[0]] += float(rule.split()[-1])

    normalized_rules = [normalize_rule(rule, prob_sums) for rule in all_rules]

    return normalized_rules


def normalize_rule(rule, prob_sums):
    split_rule = rule.split()
    nt = split_rule[0]
    total_sum = prob_sums[nt]
    new_prob = float(split_rule[-1]) / total_sum
    new_prob = f"{new_prob:.16f}"

    return " ".join([nt] + split_rule[1:-1] + [new_prob])


def format_rules(all_rules):
    replacements = [
        ("``_", "TICK_"),
        ("._", "DOT_"),
        ("#_", "HASH_"),
        ("-LRB-_", "LRB_"),
        ("-RRB-_", "RRB_"),
        ("''_", "APOSTROPHE_"),
        ("$_", "DOLLAR_"),
        (":_", "COLON_"),
        (",_", "COMMA_"),
        ("@", "AT"),
        ("PRT|ADVP", "PRTADVP"),
        ("†", "<cross>"),
        ("ā", "a"),
    ]

    def format_rule(rule):
        for s1, s2 in replacements:
            rule = rule.replace(s1, s2)

        split_rule = rule.split()
        split_rule[-1] = "[" + split_rule[-1] + "]"
        return " ".join(split_rule)

    formatted_rules = [format_rule(rule) for rule in normalized_rules]

    return formatted_rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to Petrov grammar", required=True)
    parser.add_argument("-o", "--out", help="Path write nltk grammar to", required=True)
    parser.add_argument(
        "-t",
        "--threshold",
        help="Rule probability threshold (default: 1e-7)",
        default=1e-7,
        type=float,
    )
    args = vars(parser.parse_args())

    threshold = args["threshold"]
    grammar_rules = read_grammar_rules(args["path"], threshold=threshold)
    lexicon_rules = read_lexicon_rules(args["path"], threshold=threshold)

    all_rules = grammar_rules + lexicon_rules

    normalized_rules = normalize_rules(all_rules)

    formatted_rules = format_rules(normalized_rules)

    with open(f"{args['out']}_pcfg.txt", "w") as f:
        f.write("\n".join(formatted_rules))

    print(
        f"{len(grammar_rules)}+{len(lexicon_rules)}={len(formatted_rules)} NLTK rules"
    )
    print("Processed grammar and saved to", args["out"])
