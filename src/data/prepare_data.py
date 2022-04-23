import pandas as pd

from src.data.load_data import load_data_from_mongo

def prepare_dataset():
    df = load_data_from_mongo()

    df.dropna(inplace=True)
    df["len"] = df.body.apply(lambda x: len(x.split()))
    df = df[df["len"]>20][["body", "len"]]

    print("Dataset has {} samples".format(len(df)))

    return df