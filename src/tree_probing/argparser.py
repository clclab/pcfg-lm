from argparse import ArgumentParser
import argparse
from collections import defaultdict
from typing import *
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)

ConfigDict = Dict[str, Dict[str, Any]]

ARG_TYPES = ["tokenizer", "model", "trainer", "data", "activations", "experiments", "results"]


def create_config_dict() -> ConfigDict:
    """Parse cmd args of the form `--type.name value`

    Returns
    -------
    config_dict : ConfigDict
        Dictionary mapping each arg type to their config values.
    """
    parser = _create_arg_parser()

    args, unk_args = parser.parse_known_args()

    config_dict = {arg_type: {} for arg_type in ARG_TYPES}

    for arg, value in vars(args).items():
        arg_type, arg_name = arg.split(".")

        assert arg_type in ARG_TYPES, f"unknown arg type '{arg_type}'"

        config_dict[arg_type][arg_name] = value

    _add_unk_args(config_dict, unk_args)

    return config_dict


def _create_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # MODEL
    # parser.add_argument("--model.model_type", required=True, choices=['deberta', 'gpt2', 'babyberta'])
    # parser.add_argument("--model.model_file", type=Path, required=True)

    # DATA
    parser.add_argument("--data.data_dir", type=Path, required=True)
    parser.add_argument('--data.output_dir', type=Path, required=True)
    # parser.add_argument("--data.train_file", type=Path, default="train.txt")
    # parser.add_argument("--data.dev_file", type=Path, default="dev.txt")
    # parser.add_argument("--data.test_file", type=Path, default="test.txt")
    # parser.add_argument("--data.eval_file", type=Path, default="eval.txt")
    parser.add_argument("--data.train_size", type=int)
    parser.add_argument("--data.dev_size", type=int)
    parser.add_argument("--data.test_size", type=int)
    parser.add_argument("--data.sampling", action=argparse.BooleanOptionalAction)
    parser.add_argument("--data.sampling_size", type=int, default=0)
    parser.add_argument("--data.rel_toks", type=Path, default="data/train_rel_toks.txt")
    parser.add_argument("--data.pos_tags", type=Path, default="data/train_POS_labels.txt")
    parser.add_argument("--data.gold_trees", type=Path, default="data/gold_trees_cleaned.txt")
    parser.add_argument("--data.sentences", type=Path, default="data/train_sentences.txt")
    parser.add_argument("--data.generate_test_data", action=argparse.BooleanOptionalAction)

    # EXTRACT ACTIVATIONS
    parser.add_argument("--activations.output_dir", type=Path, default='data',
                        help='Directory where the activations are stored.')
    parser.add_argument('--activations.dtype', default='float32', choices=["float16", "float32"], 
                        help="Data type of the activations")
    parser.add_argument('--activations.mode', default='', choices=['', 'avg', 'max', 'concat'],
                        help="How the activations are combined. Use in combination with LCA experiments.") 

    # EXPERIMENTS
    parser.add_argument("--experiments.type", default='chunking', choices=['chunking', 'lca', 'lca_tree', 'shared_levels', 'unary'])
    parser.add_argument("--experiments.checkpoint_path", type=Path, default='models/')
    parser.add_argument("--experiments.control_task", action=argparse.BooleanOptionalAction)
    parser.add_argument("--experiments.version", default='normal', choices=['lexical', 'normal', 'pos'])
    parser.add_argument('--experiments.top_k', type=float, default=0.2)

    # RESULTS
    parser.add_argument("--results.confusion_matrix", action=argparse.BooleanOptionalAction)

    # TRAINER
    parser.add_argument("--trainer.batch_size", type=int, default=128)
    parser.add_argument("--trainer.num_workers", type=int, default=8)
    parser.add_argument("--trainer.epochs", type=int, default=10)
    parser.add_argument("--trainer.lr", type=float, default=1e-3)
    parser.add_argument("--trainer.device", type=str, default=None)

    return parser


def _add_unk_args(config_dict: ConfigDict, unk_args: List[str]) -> None:
    """Add args that are not part of the arg parser arguments"""
    prev_arg_type = None
    prev_arg_name = None

    for arg in unk_args:
        if arg.startswith("--"):
            arg_type, arg_name = arg[2:].split(".")

            assert arg_type in ARG_TYPES, f"unknown arg type '{arg_type}'"

            config_dict[arg_type][arg_name] = True

            prev_arg_type = arg_type
            prev_arg_name = arg_name
        else:
            assert prev_arg_type is not None, "arg input not well-formed"

            try:
                config_dict[prev_arg_type][prev_arg_name] = eval(arg)
            except NameError:
                config_dict[prev_arg_type][prev_arg_name] = arg

            prev_arg_type = None
            prev_arg_name = None