from transformers import DataCollatorForLanguageModeling
import wandb
import os
import shutil
import torch
import datasets
from pprint import pprint
from argparser import create_config_dict
from data import load_data
from model import initialize_model
from tokenizer import create_tokenizer, load_pretrained_tokenizer
from trainer import initialize_trainer


if __name__ == "__main__":
    config_dict = create_config_dict()
    pprint(config_dict)

    is_mlm = config_dict["model"]["is_mlm"]

    shutil.rmtree(config_dict["trainer"]["output_dir"], ignore_errors=True)

    dsd = datasets.DatasetDict.load_from_disk(config_dict["data"]["data_dir"])

    if config_dict["tokenizer"].get("path") is not None:
        tokenizer = load_pretrained_tokenizer(config_dict["tokenizer"]["path"])
    else:
        tokenizer = create_tokenizer(
            dsd['train'],
            min_freq=config_dict["tokenizer"]["min_freq"],
            add_bos_token=not (is_mlm),
        )

    dsd = load_data(tokenizer, dsd)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=is_mlm)

    model = initialize_model(
        tokenizer.vocab_size,
        **config_dict["model"],
    )

    print(model)
    print("#params", sum(param.numel() for param in model.parameters()))
    print(len(tokenizer), "tokens")

    if config_dict["trainer"]["report_to"] == "wandb":
        os.environ["WANDB_PROJECT"] = "pcfg-lm"

    if ("hub_model_id" in config_dict["trainer"]) and (config_dict["trainer"]["hub_model_id"] == "none"):
        del config_dict["trainer"]["hub_model_id"]

    push_to_hub = "hub_model_id" in config_dict["trainer"]
    if "hub_token" in config_dict["trainer"]:
        with open(config_dict["trainer"]["hub_token"]) as f:
            config_dict["trainer"]["hub_token"] = f.read().strip()

    trainer = initialize_trainer(
        model,
        tokenizer,
        data_collator,
        dsd,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        auto_find_batch_size=False,
        do_eval=True,
        push_to_hub=push_to_hub,
        **config_dict["trainer"],
    )

    trainer.train()
    trainer._save_checkpoint(trainer.model, None)
