from calendar import day_abbr
from torch.utils.data import Dataset
import re

class TitleDataset(Dataset):
    def __init__(self, texts, titles, tokenizer, max_len, preprocess) -> None:
        super().__init__()
        self.texts = texts
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess = preprocess

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

    def __getitem__(self, idx):
        text = self.texts[idx]
        title = self.titles[idx]

        if self.preprocess:
            text = self.clean_text(text)
            title = self.clean_text(title)

        inputs = self.tokenizer.encode(
            '<|startoftext|>' + text + " TL;DR: " + title + '<|endoftext|>',
            truncation=True, 
            padding=True,
            max_length=self.max_len,
            return_tensors = 'pt'
        )

        return inputs

