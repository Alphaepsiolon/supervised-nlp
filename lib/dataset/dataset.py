import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class ClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 label_mapping: dict,
                 mode:str='train',
                 model_id:str = 'albert-base-v2'):
        self.mode = mode
        self.df = df
        self.model_id = model_id
        self.label_mapping = label_mapping
        # preprocess df to include numeric labels
        self.get_label_mapping()
        self.text,self.labels = self.preprocess()

        # obtain labels and encodings
        self.tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
        self.encodings = self.tokenize()
    

    def tokenize(self):
        train_text_encoded = self.tokenizer(self.text,padding=True,truncation=True, return_tensors='pt')
        return train_text_encoded
    
    def get_label_mapping(self) -> None:
        unique_labels = self.df['label'].unique().tolist()
        label_mapping = self.label_mapping

        # create labels maps
        numeric_list = [*map(label_mapping.get, self.df['label'].tolist())]
        self.df['num_label'] = numeric_list

    def preprocess(self) -> None:
        self.unique_labels = self.df['num_label'].unique().tolist()
        text = self.df['text'].tolist()
        labels = self.df['num_label'].tolist()
        return text, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

    @property
    def get_unique_labels(self):
        return self.unique_labels
                