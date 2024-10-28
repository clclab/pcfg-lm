import argparse
from collections import defaultdict


def process_earleyx_grammar(grammar_file, output_file, start_nt: str = "S_0"):
    with open(grammar_file) as f:
        lines = f.read().strip().split("\n")
    
    base_rules = defaultdict(float)
    
    for line in lines:
        line_items = line.split(' ')
        lhs = line_items[0]
        rhs = line_items[2:-1]
        if len(rhs) == 1 and rhs[0].startswith("'") and rhs[0].endswith("'"):
            rhs = f"_{rhs[0][1:-1]}"
        else:
            rhs = " ".join(rhs)
        prob = line_items[-1][1:-1]
        
        base_rules[f"{lhs}->[{rhs}]"] += float(prob)
        
    earleyx_lines = [f"{key} : {prob:.12f}" for key, prob in base_rules.items()]
    earleyx_lines.append(f"ROOT->[{start_nt}] : 1.0")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(earleyx_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grammar_file", help="Path to NLTK grammar", required=True)
    parser.add_argument("-o", "--output_file", help="Path write earleyx grammar to", required=True)
    parser.add_argument("-s", "--start_nt", help="Start symbol (default: S_0)", default="S_0")
    args = vars(parser.parse_args())

    process_earleyx_grammar(
        args['grammar_file'], 
        args['output_file'], 
        start_nt=args['start_nt'],
    )
