from transformers import DataCollatorForLanguageModeling
import wandb
import os
import torch
from pprint import pprint
from argparser import create_config_dict
from data import load_data
from model import initialize_model
from tokenizer import load_pretrained_tokenizer
from trainer import initialize_trainer


def main(config_dict):
    tokenizer = load_pretrained_tokenizer(**config_dict['tokenizer'])

    datasets = load_data(tokenizer, **config_dict['data'])

    is_mlm = config_dict['model']['is_mlm']

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=is_mlm)

    model = initialize_model(
        tokenizer, 
        **config_dict['model'],
    )

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
        **config_dict['trainer'],
    )
    
    trainer.train()
    trainer._save_checkpoint(trainer.model, None)

    print(trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    config_dict = create_config_dict()
    pprint(config_dict)

    output_dir = config_dict['trainer']['output_dir']

    for corpus_size in [7_500_000]: #[10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 7_500_000]:
        print("CORPUS SIZE", corpus_size)
        config_dict['data']['train_size'] = corpus_size
        config_dict['trainer']['num_train_epochs'] = min(7_500_000 / corpus_size, 50)
        config_dict['trainer']['output_dir'] = os.path.join(output_dir, f"{corpus_size}_size")

        main(config_dict)

