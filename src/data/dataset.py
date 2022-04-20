from calendar import day_abbr
from torch.utils.data import Dataset
import re
import torch

class TitleDataset(Dataset):
    def __init__(self, texts, titles, tokenizer, max_len, preprocess) -> None:
        super().__init__()
        self.texts = texts
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess = preprocess
        self.extra_length = 5
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.titles)

    @staticmethod
    def clean_text(text):
        text = str(text)
        text = re.sub(r'[0-9"]', '', text) # number
        text = re.sub(r'#[\S]+\b', '', text) # hash
        text = re.sub(r'@[\S]+\b', '', text) # mention
        text = re.sub(r'https?\S+', '', text) # link
        text = re.sub(r'\s+', ' ', text) # multiple white spaces
        
        return text

    def pad_truncate(self, name):
        name_length = len(name) - self.extra_length
        if name_length < self.max_len:
            difference = self.max_len - name_length
            result = name + [self.eos_id] * difference
        elif name_length > self.max_len:
            result = name[:self.max_len + (self.extra_length-1)]+[self.eos_id] 
        else:
            result = name
        return result

    def __getitem__(self, idx):
        text = self.texts[idx]
        title = self.titles[idx]

        if self.preprocess:
            text = self.clean_text(text)
            title = self.clean_text(title)

        seo_post = text + " TL;DR: " + title + self.eos

        tokenized = self.tokenizer.encode(
            seo_post,
            # return_tensors = 'pt'
        )

        padded = torch.tensor(self.pad_truncate(tokenized))

        # labels = self.tokenizer.encode(
        #     '<|startoftext|>' + title + '<|endoftext|>',
        #     truncation=True, 
        #     padding='max_length',
        #     max_length=self.max_len,
        #     return_tensors = 'pt'
        # )

        return padded

