from ast import mod
from src.config.config import device
import src.data.dataset as dataset
from src.data.prepare_data import load_data_from_mongo
from src.utils.transform import sample_seq

import torch
import pandas as pd
import numpy as np
import os

import transformers

import random
from tqdm.notebook import tqdm

from argparse import ArgumentParser

parser = ArgumentParser(description="Fine tune GPT-2 model on the dataset and evaluate results.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for all sampling purposes")
parser.add_argument("--gpt_path", default="gpt2", type=str, help="Path to base gpt model")
parser.add_argument("--model_path", type=str, help="Path to fine-tuned model")

parser.add_argument("--n_samples", type=int, default=-1)
parser.add_argument("--temp", type=int, default=1)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--top_p", type=int, default=0.8)
parser.add_argument("--length", type=int, default=30)

parser.add_argument("--preprocess", action="store_true", help="To apply preprocessing step")
parser.add_argument("--output_dir", type=str, help="Path to output directory for saving model checkpoints")
parser.add_argument("--max_len", type=int, default=128, help="Specifies the maximum length of input sequence")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def run(params):
    if args.n_samples == -1:
        df = load_data_from_mongo().sample(frac=1).reset_index(drop=True)
    else:
        df = load_data_from_mongo().sample(n=args.n_samples).reset_index(drop=True)

    df.to_csv(os.path.join(args.output_dir, "thoughts.csv"), index=False)


    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        params['gpt_path'],
        do_lower_case=True,
        # bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<|pad|>'
    )

    predict_dataset = dataset.ThoughtDataset(
        df.body.values,
        tokenizer,
        args.max_len,
        args.preprocess
    )

    model = transformers.GPT2LMHeadModel.from_pretrained(params['gpt_path'])
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    f =  open(os.path.join(args.output_dir, "titles.txt"), "w")

    with torch.no_grad():
        for i, d in tqdm(enumerate(predict_dataset), total=len(predict_dataset), position=0, leave=True):
            ids = d.to(device)
            sampled_seq = sample_seq(model, ids, args.length, device, args.temp, args.top_k, args.top_p)

            title = tokenizer.decode(sampled_seq.squeeze(), skip_special_tokens=True)

            f.write("\n{}".format(title))

    f.close()




def main():
    params = {
        'gpt_path': args.gpt_path,
    }

    run(params)

if __name__ == "__main__":
    main()
