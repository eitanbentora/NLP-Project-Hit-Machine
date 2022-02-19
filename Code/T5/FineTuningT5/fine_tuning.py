import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
from torch import nn
from transformers import TrainingArguments
from transformers import Trainer
from pathlib import Path
import torch


class FillingModel(nn.Module):

    def __init__(self):
        super(FillingModel, self).__init__()
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        labels = labels.squeeze(1)  # TODO maybe remove squeeze
        pred = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return pred


def tokenize_fn(examples):
    tokenized_examples = tokenizer(
        examples["input_ids"], truncation=True, max_length=max_length, padding="max_length")
    labels = examples['labels']
    tok_labels = tokenizer(
        labels, truncation=True, max_length=max_length, padding="max_length")['input_ids']
    tokenized_examples['labels'] = tok_labels
    return tokenized_examples


for ind in [0, 1, 2, 3, 4]:
    print(ind)

    labeled_df = pd.read_pickle(f'/home/vmadmin/StudentData/Generative/TrainingDatasets/train_set_{ind}.pkl')
    datasets = DatasetDict({'train': Dataset.from_pandas(labeled_df)})
    del labeled_df
    model_name = 't5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 128

    tokenized_datasets = datasets.map(tokenize_fn, remove_columns=datasets["train"].column_names)
    tokenized_datasets.set_format('torch')

    model = FillingModel()
    OUT_PATH = Path("results")
    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=8,
                             save_strategy='no',
                             do_train=True, num_train_epochs=5, report_to='none')
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
    )
    trainer.train()
    model.model.save_pretrained(f'/home/vmadmin/StudentData/Generative/Models/model_{ind}.hfm')
