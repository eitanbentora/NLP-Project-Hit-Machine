import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
from torch import nn
from transformers import TrainingArguments
from transformers import Trainer
from pathlib import Path
import torch

import re

import tensorflow as tf
from preprocessors import fill_in_the_blank
import sys


def get_parmas(mode):
    if mode == 'lyrics':
        kw = {'epochs' : 3,
              'batch_size' : 3,
              'max_length_input' : 256,
              'max_length_output' : 256,
              'model_name' : 't5-base'}
        
    elif mode == 'chords':
        kw = {'epochs' : 2,
              'batch_size' : 16,
              'max_length_input' : 256,
              'max_length_output' : 32,
              'model_name' : 't5-small'}
    return kw

class FillingModel(nn.Module):

    def __init__(self):
        super(FillingModel, self).__init__()
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        labels = labels.squeeze(1)  # TODO maybe remove squeeze
        pred = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return pred


def tokenize_fn(examples, max_length_input, max_length_output):
    tokenized_examples = tokenizer(
        examples["input_ids"], truncation=True, max_length=max_length_input, padding="max_length")
    labels = examples['labels']
    tok_labels = tokenizer(
        labels, truncation=True, max_length=max_length_output, padding="max_length")['input_ids']
    tokenized_examples['labels'] = tok_labels
    return tokenized_examples


if __name__ == "__main__":
    version = sys.argv[1]
    mode = sys.argv[2]
    assert mode == 'chords' or mode == 'lyrics'
    print('version:', version)
    print('mode:', mode)
    
    kw = get_parmas(mode)
    
    labeled_df = pd.read_pickle(f'/home/student/Desktop/Project/Data/T5Data/LabeledData/labeled_{mode}_data_{version}.pkl')
    
    datasets = DatasetDict({'train': Dataset.from_pandas(labeled_df)})
    del labeled_df
    model_name = kw['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length_input, max_length_output = kw['max_length_input'], kw['max_length_output']
    tokenized_datasets = datasets.map(lambda x: tokenize_fn(x, max_length_input, max_length_output), remove_columns=datasets["train"].column_names)
    tokenized_datasets.set_format('torch')

    model = FillingModel()
    OUT_PATH = Path("results")
    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=kw['batch_size'],
                             save_strategy='no', do_train=True, num_train_epochs=kw['epochs'], report_to='none')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ", device)
    model.to(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
    )
    
    trainer.train()
    
    model.model.save_pretrained(f'/home/student/Desktop/Project/Data/T5Data/Models/model_{mode}_{version}')



