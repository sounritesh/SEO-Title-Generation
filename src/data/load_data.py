from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd

from src.config.secrets import secrets

def load_data_from_mongo():
    client = MongoClient(secrets["client"])

    thought_list = []
    thoughts = client.forum.thoughts
    for thought in tqdm(thoughts.find(no_cursor_timeout=True), total=thoughts.count_documents({})):
        thought_list.append(thought)
    print("{} thoughts retrieved from client.".format(len(thought_list)))
    df_thoughts = pd.DataFrame(thought_list)[["created_at", "author", "body", "blocked"]]

    return df_thoughts