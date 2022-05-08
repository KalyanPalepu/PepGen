"""
Script for hyperparameter search for training
"""

import torch
import optuna
from model import PepGenGPT
from data import PartnerPeptideDataModule
import os
import pytorch_lightning as pl
import argparse
import pickle

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# random name and submit script modified from https://github.com/AllanSCosta/InterDocker/blob/main/utils.py

import randomname
def random_name():
    return randomname.get_name(
        adj=('speed', 'emotions', 'temperature', 'weather', 'character', 'algorithms', 'geometry', 'complexity', 'physics', 'shape', 'taste', 'colors', 'size', 'appearance'),
        noun=('astronomy', 'set_theory', 'military_navy', 'infrastructure', 'chemistry', 'physics', 'algorithms', 'geometry', 'coding', 'architecture', 'metals', 'apex_predators')
    )

def submit_script(script_path, base_path, config):
    model_id = config['model_id']
    worskpace_dir = os.path.join(base_path, 'scripts')
    os.makedirs(worskpace_dir, exist_ok=True)
    script = os.path.join(worskpace_dir, f'{model_id}.sh')
    with open(script, 'w') as file:
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH --gres=gpu:volta:1\n'
        preamble += f'#SBATCH -o {os.path.join(worskpace_dir, model_id)}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task=10\n'
        preamble += f'#SBATCH --job-name={model_id}\n\n'
        preamble += f'module load anaconda/2021b\n'
        file.write(preamble)
        config = [(key, value) for key, value in config.items() if (key != 'submit')]
        config_strings = []
        for key, value in config:
            if type(value) == bool and value:
                config_strings.append(f'--{key}')
            elif type(value) != bool:
                config_strings.append(f'--{key} {str(value)}')

        config_string = ' '.join(config_strings)
        file.write(f'python -u {script_path} {config_string}')
        file.close()
    os.system(f'LLsub {script}')
    print(f'submitted {model_id}!')

def model_init(trial, alphabet, esm_token_embedding, max_seq_length, esm_dim):
    # hidden dim 1024, num heads 8, num layers 12, dim feedforward 2048, bs 32 fits in memory
    config = {
        "alphabet": alphabet,
        "lr": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        "hidden_dim": trial.suggest_int('hidden_size', 32, 1024, step=32), # dim used in transformer encoder
        "num_heads": trial.suggest_int('num_heads', 4, 8, step=4), 
        "num_layers": trial.suggest_int('num_layers', 2, 8, step=2),
        "dim_feedforward": trial.suggest_int('dim_feedforward', 64, 2048, step=64), # dim used in feedforward in transformer encoder
        "layer_norm_eps": trial.suggest_float("layer_norm_eps", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "norm_first": True,
        "learnable_positional_embedding": trial.suggest_categorical("learnable_positional_embedding", [True, False]),
        # "use_esm_token_embedding": trial.suggest_categorical("use_esm_token_embedding", [True, False]),
        "use_esm_token_embedding": False,
        "esm_token_embedding": esm_token_embedding,
        "positional_embed_peptide_only": trial.suggest_categorical("positional_embed_peptide_only", [True, False]),
        "esm_dim": esm_dim,
        "max_seq_length": max_seq_length
    }
    return PepGenGPT(config), config

def objective(trial, args, model_id):
    datamodule = PartnerPeptideDataModule(args.regions_csv_path, 
                                        args.partner_embeddings_path, 
                                        random_seed=args.random_seed, 
                                        batch_size=args.batch_size, 
                                        max_seq_length=args.max_seq_length,
                                        max_peptide_length=args.max_peptide_length,
                                        use_interface=args.use_interface)

    model, config = model_init(trial, 
                               alphabet=datamodule.alphabet, 
                               esm_token_embedding=esm_token_embedding, 
                               max_seq_length=args.max_seq_length,
                               esm_dim=args.esm_dim)
    print(config)
    
    callbacks = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))

    trainer = pl.Trainer(logger=True,
                         max_epochs=args.max_epochs,
                         gpus=args.n_gpus,
                         precision=args.precision,
                         enable_checkpointing=True,
                         callbacks=callbacks,
                         default_root_dir=os.path.join("optuna_models", model_id, str(trial.number)),
                         detect_anomaly=(not args.no_anomaly_detection))
    
    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=datamodule)  
    torch.cuda.empty_cache()
    
    print("    Trial {} validation loss: {}".format(trial.number, trainer.callback_metrics["val_loss"].item()))
    print("    Trial {} validation accuracy: {}".format(trial.number, trainer.validate(dataloaders=datamodule.val_dataloader())))

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PepGen')
    parser.add_argument('--regions_csv_path', dest='regions_csv_path', default="pepgen_dataset.csv")
    parser.add_argument('--partner_embeddings_path', dest='partner_embeddings_path', default="embeddings.pkl")
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--model_id', dest='model_id', default=None)
    parser.add_argument('--esm_token_embedding_path', dest='esm_token_embedding_path', default="esm_token_embedding.pt")
    parser.add_argument('--max_seq_length', dest='max_seq_length', type=int, default=1024)
    parser.add_argument('--max_peptide_length', dest='max_peptide_length', type=int, default=50)
    parser.add_argument('--esm_dim', dest='esm_dim', type=int, default=768)
    parser.add_argument('--use_interface', action='store_true')
    parser.add_argument('--n_gpus', dest='n_gpus', type=int, default=-1)    
    parser.add_argument('--precision', dest='precision', type=int, default=16)  
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--no_anomaly_detection', action='store_true')

    parser.add_argument('--optuna', dest='optuna', action="store_true")
    parser.add_argument('--optuna_trials', dest='optuna_trials', type=int, default=50)
    parser.add_argument('--prune_optuna', action='store_true')

    parser.add_argument('--lr', dest="lr", default=1e-3, type=float)
    parser.add_argument('--hidden_dim', dest="hidden_dim", type=int, default=32)
    parser.add_argument('--num_heads', dest="num_heads", type=int, default=4)
    parser.add_argument('--num_layers', dest="num_layers", type=int, default=2)
    parser.add_argument('--dim_feedforward', dest="dim_feedforward", type=int, default=64)
    parser.add_argument('--layer_norm_eps', dest="layer_norm_eps", type=float, default=1e-5)
    parser.add_argument('--dropout', dest="dropout", type=float, default=1e-1)
    parser.add_argument('--use_esm_token_embedding', dest='use_esm_token_embedding', action='store_true')
    parser.add_argument('--positional_embed_peptide_only', dest='positional_embed_peptide_only', action='store_true')
    parser.add_argument('--learnable_positional_embedding', dest='learnable_positional_embedding', action='store_true')
    parser.add_argument('--fast_dev_run', dest='fast_dev_run', action='store_true')

    parser.add_argument('--submit', dest='submit', action='store_true')


    args, _ = parser.parse_known_args()
    config = vars(args)


    if args.model_id == None:
        config['model_id'] = random_name()
    else:
        config['model_id'] = args.model_id

    if args.submit:
        submit_script(os.path.realpath(__file__), os.getcwd(), config)
        exit()
    
    print(f"Training {config['model_id']}!")

    esm_token_embedding = torch.load(args.esm_token_embedding_path)

    if args.use_interface:
        config['esm_dim'] += 1
        print("ESM Dim:", config['esm_dim'])

    if args.optuna:
        if args.prune_optuna:
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = None
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(lambda trial: objective(trial, args, config['model_id']),
                         n_trials=args.optuna_trials)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        datamodule = PartnerPeptideDataModule(args.regions_csv_path, 
                                        args.partner_embeddings_path, 
                                        random_seed=args.random_seed, 
                                        batch_size=args.batch_size, 
                                        max_seq_length=args.max_seq_length,
                                        max_peptide_length=args.max_peptide_length,
                                        use_interface=args.use_interface)
        
        config["norm_first"] = True
        config["alphabet"] = datamodule.alphabet
        config["esm_token_embedding"] = esm_token_embedding

        if args.fast_dev_run:
            fast_dev_run = 5
        else:
            fast_dev_run = False

        root_dir = os.path.join("manual_models", config['model_id'])
        os.makedirs(root_dir, exist_ok=True)

        with open(os.path.join(root_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

        if args.early_stopping:
            callbacks = [EarlyStopping(monitor="val_loss", mode="min")]
        else:
            callbacks = []
        trainer = pl.Trainer(logger=True,
                            max_epochs=args.max_epochs,
                            precision=args.precision,
                            gpus=args.n_gpus,
                            enable_checkpointing=True,
                            callbacks=callbacks,
                            default_root_dir=root_dir,
                            fast_dev_run=args.fast_dev_run,
                            detect_anomaly=(not args.no_anomaly_detection),
                            log_every_n_steps=5,
                            check_val_every_n_epoch=1
                            )


        model = PepGenGPT(config)
        trainer.logger.log_hyperparams(config)
        trainer.fit(model, datamodule=datamodule)
