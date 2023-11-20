from argparse import ArgumentParser
from collections import defaultdict
from typing import *

ConfigDict = Dict[str, Dict[str, Any]]

ARG_TYPES = ["tokenizer", "model", "trainer", "data"]


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
    parser.add_argument("--model.model_type", required=True)
    parser.add_argument("--model.is_mlm", action="store_true")
    parser.add_argument("--model.num_hidden_layers", type=int, default=8)
    parser.add_argument("--model.intermediate_size", type=int, default=256)
    parser.add_argument("--model.hidden_size", type=int, default=256)
    parser.add_argument("--model.num_attention_heads", type=int, default=8)

    # TOKENIZER
    parser.add_argument("--tokenizer.path", required=True)

    # DATA
    parser.add_argument("--data.data_dir", required=True)
    parser.add_argument("--data.train_file", default="train.txt")
    parser.add_argument("--data.dev_file", default="dev.txt")
    parser.add_argument("--data.test_file", default="test.txt")
    parser.add_argument("--data.eval_file", default="eval.txt")
    parser.add_argument("--data.train_size", type=int)
    parser.add_argument("--data.dev_size", type=int)
    parser.add_argument("--data.test_size", type=int)

    # TRAINER
    parser.add_argument("--trainer.output_dir", required=True)
    parser.add_argument("--trainer.per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--trainer.per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--trainer.evaluation_strategy", default="steps")
    parser.add_argument("--trainer.eval_steps", type=int, default=100)
    parser.add_argument("--trainer.logging_steps", type=int, default=100)
    parser.add_argument("--trainer.save_steps", type=int, default=1000)
    parser.add_argument("--trainer.gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--trainer.max_grad_norm", type=int, default=0.5)
    parser.add_argument("--trainer.num_train_epochs", type=int, default=1)
    parser.add_argument("--trainer.weight_decay", type=float, default=0.1)
    parser.add_argument("--trainer.lr_scheduler_type", default="cosine")
    parser.add_argument("--trainer.learning_rate", type=float, default=5e-4)
    parser.add_argument("--trainer.push_to_hub", action="store_true")
    parser.add_argument("--trainer.report_to", default="none")

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
