from pathlib import Path
from argparser import create_config_dict
from data import ExperimentManager
from utils import format_predictions, idx_labels2text, get_pos_test_set, get_gold_trees_test_set
from model import DiagModule

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import pickle
from tqdm import tqdm
import numpy as np
from pprint import pprint
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main():
    config_dict = create_config_dict()
    pprint(config_dict)

    if config_dict['trainer']['device'] is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info('Running on GPU.')

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info('Running on M1 GPU.')

        else:
            device = torch.device("cpu")
            logger.info('Running on CPU.')
    else:
        device = torch.device(config_dict['trainer']['device'])

    # config_dict = set_experiment_config(config_dict)
    # OGmodel, _ = load_model_tokenizer(config_dict)
    # OGmodel.to(device).eval()

    # Initiate experiments
    CurrentExperiment = ExperimentManager(config_dict)

    val_final = []
    test_final = []

    for layer_idx, states in tqdm(CurrentExperiment.activations.items()):
        logging.info(f"Training layer {layer_idx}...")
        save_name = f"{CurrentExperiment.base_name}_{layer_idx}"

        # with torch.no_grad():
        #     states = OGmodel.cls.predictions.transform(states.to(device))

        logging.info("Splitting data in train, dev and test sets.")
        X_train, y_train, X_dev, y_dev, X_test, y_test = CurrentExperiment.create_train_dev_test_split(layer_idx)

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                    shuffle=True,
                                    persistent_workers=True,
                                    batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'])
                                    # multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
        
        devset = torch.utils.data.TensorDataset(X_dev, y_dev)
        dev_loader = torch.utils.data.DataLoader(dataset= devset, 
                                    shuffle=False,
                                    persistent_workers=True,
                                    batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'])
                                    # multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
        
        testset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset= testset, 
                                    shuffle=False,
                                    persistent_workers=True,
                                    batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'])
                                    # multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

        logging.info("Started training...")
        config_dict['experiments']['checkpoint_path'].mkdir(parents=True, exist_ok=True)
        trainer = pl.Trainer(default_root_dir=os.path.join(config_dict['experiments']['checkpoint_path'], save_name),                          
                                accelerator='mps' if device == 'mps' else 'cpu', 
                                devices=1,                                            
                                max_epochs=config_dict['trainer']['epochs'],                                                                    
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")])

        pl.seed_everything(42)
        model = DiagModule(model_hparams={"num_inp":X_train.shape[-1], "num_units":len(CurrentExperiment.label_vocab)}, optimizer_hparams={"lr": config_dict['trainer']['lr']})
        trainer.fit(model, train_loader, dev_loader)

        model = DiagModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Test best model on test set
        test_result = idx_labels2text(trainer.test(model, test_loader, verbose=False), CurrentExperiment.label_vocab)
        test_final.append(test_result)
        result = {"test": test_result}

        CurrentExperiment.results_file.write(f'Layer {layer_idx} \n {result}\n')

        # save confusion matrix
        if config_dict['results']['confusion_matrix']:
            np.save(f'{config_dict["data"]["output_dir"]}/{CurrentExperiment.name}/confusion_matrix_{layer_idx}.npy', model.final_confusion_matrix)
        
        # save predictions
        if CurrentExperiment.name in ['lca_tree', 'shared_levels', 'unary']:
            predictions = format_predictions(model.predictions, CurrentExperiment.idx2class, CurrentExperiment.rel_toks_test)
            with open(f'{config_dict["data"]["output_dir"]}/{CurrentExperiment.name}/predictions_{CurrentExperiment.name}.txt', 'wb') as f:
                f.write(predictions.encode('utf-8'))
            
            # write also function to write POS tags to file
            if layer_idx == 0:
                # test_indices = set([int(tok.split('_')[0]) for tok in CurrentExperiment.rel_toks_test])
                # check for de eerste die midden in de zin begint + als het wel de eerste is
                get_pos_test_set(config_dict, CurrentExperiment)
                get_gold_trees_test_set(config_dict, CurrentExperiment)

    CurrentExperiment.results_file.close()

    # write val_final and test_final to seperate pickle files
    with open(CurrentExperiment.val_results_file, 'wb') as f:
        pickle.dump(val_final, f)
    with open(CurrentExperiment.test_results_file, 'wb') as f:
        pickle.dump(test_final, f)
    print("Label vocab", CurrentExperiment.label_vocab)


if __name__ == "__main__":
    """
    """
    main()