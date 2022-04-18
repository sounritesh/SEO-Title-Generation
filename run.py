import src.data.dataset as dataset
import src.utils.engine as engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import wandb

import transformers

from torch.optim import Adam, lr_scheduler, SGD

import optuna
import random
from tqdm.notebook import tqdm

import os

from argparse import ArgumentParser

parser = ArgumentParser(description="Fine tune GPT-2 model on the dataset and evaluate results.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for all sampling purposes")
parser.add_argument("--data_path", type=str, help="Path to training file")
parser.add_argument("--gpt_path", default="gpt2", type=str, help="Path to base gpt model")

parser.add_argument("--lr", type=float, default=1e-4, help="Specifies the learning rate for optimizer")

parser.add_argument("--preprocess", action="store_true", help="To apply preprocessing step")
parser.add_argument("--tune", action="store_true", help="To tune model by trying different hyperparams")

parser.add_argument("--output_dir", type=str, help="Path to output directory for saving model checkpoints")

parser.add_argument("--max_len", type=int, default=128, help="Specifies the maximum length of input sequence")

parser.add_argument("--epochs", type=int, default=15, help="Specifies the number of training epochs")

parser.add_argument("--train_batch_size", type=int, default=64, help="Specifies the training batch size")
parser.add_argument("--val_batch_size", type=int, default=256, help="Specifies the validation and testing batch size")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def run(params, save_model=True):
    wandb.init(
        project="seo-title-generation", 
        entity="now-and-me",
        config=params
    )

    df = pd.read_csv(args.data_path).sample(frac=1).reset_index(drop=True)

    df_train = df.sample(frac=0.8)
    df_rest = df.drop(df_train.index)    
    df_val = df_rest.sample(frac=0.4)
    df_test = df_rest.drop(df_val.index)

    print(f"Stratification Split, \ntrain: {len(df_train)} \nval: {len(df_val)} \ntest: {len(df_test)}")

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        params['gpt_path'], 
        do_lower_case=True,
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>', 
        pad_token='<|pad|>'
    )

    train_dataset = dataset.TitleDataset(
        df_train.text.values, 
        df_train.title.values, 
        tokenizer, 
        args.max_len,
        args.preprocess
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size
    )

    valid_dataset = dataset.ToxicityDatasetBERT(
        df_val.text.values, 
        df_val.title.values, 
        tokenizer, 
        args.max_len,
        args.preprocess
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.val_batch_size
    )

    test_dataset = dataset.ToxicityDatasetBERT(
        df_test.text.values, 
        df_test.title.values, 
        tokenizer, 
        args.max_len,
        args.preprocess
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.val_batch_size
    )

    device = torch.device("cuda")
    model = transformers.GPT2LMHeadModel.from_pretrained(params['gpt_path'])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    wandb.watch(model, log="all", log_freq=10, idx=None, log_graph=(True))

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)
    optimizer = SGD(optimizer_parameters, lr=params['lr'])

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    early_stopping_iter = 3
    early_stopping_counter = 0

    best_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device)
        val_loss = engine.eval_fn(valid_data_loader, model, device)
        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'{epoch}_model.bin'))
        else:
            early_stopping_counter += 1

        if early_stopping_iter < early_stopping_counter:
            break
        
        scheduler.step()
        print(f"EPOCH[{epoch+1}]: train loss: {train_loss}, val loss: {val_loss}")
        wandb.log({
            "train loss": train_loss,
            "val loss": val_loss
        })

    test_loss = engine.eval_fn(test_data_loader, model, device)

    wandb.summary['test_loss'] = test_loss
    wandb.finish()

    print(f"TEST RESULTS Loss: {test_loss}")

    return best_loss

def objective(trial):
    params = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'gpt_path': args.gpt_path,
    }
    return run(params, False)

def main():
    if args.tune:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        trial_ = study.best_trial
        print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")

        score = run(trial_.params, True)
        print(score)
    else:
        params = {
            'lr': args.lr,
            'bert_path': args.gpt_path,
        }

        run(params)

if __name__ == "__main__":
    main()
