from transformers import DataCollatorForLanguageModeling
import wandb
import os
import torch
from pprint import pprint
from argparser import create_config_dict
from data import load_data
from model import initialize_model
from tokenizer import create_tokenizer, load_pretrained_tokenizer
from trainer import initialize_trainer


if __name__ == "__main__":
    config_dict = create_config_dict()
    pprint(config_dict)

    if config_dict['tokenizer'].get('path') is not None:
        tokenizer = load_pretrained_tokenizer(config_dict['tokenizer']['path'])
    else:
        train_corpus_path = os.path.join(config_dict['data']['data_dir'], config_dict['data']['train_file'])
        tokenizer = create_tokenizer(train_corpus_path, min_freq=config_dict['tokenizer']['min_freq'])

    datasets = load_data(tokenizer, **config_dict['data'])

    is_mlm = config_dict['model']['is_mlm']

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=is_mlm)

    model = initialize_model(
        tokenizer, 
        **config_dict['model'],
    )

    print(model)
    print('#params', sum(param.numel() for param in model.parameters()))

    lr = 5e-4

    if config_dict['trainer']['report_to'] == 'wandb':
        os.environ["WANDB_PROJECT"] = "pcfg-lm"

    trainer = initialize_trainer(
        model, 
        tokenizer, 
        data_collator, 
        datasets, 
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        auto_find_batch_size=False,
        do_eval=True,
        #report_to="wandb",
        **config_dict['trainer'],
    )
    
    trainer.train()
    trainer._save_checkpoint(trainer.model, None)
