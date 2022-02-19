#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
from torch import nn
from transformers import TrainingArguments
from transformers import Trainer
from pathlib import Path
import torch

import tensorflow as tf
from preprocessors import fill_in_the_blank


# In[41]:


def replace_X_with_extra_id(text):
    i = 1
    new_text = []
    for word in text.split():
        if word == 'X':
            new_text.append(f'<extra_id_{i}>')
            i +=1
            if i > 100:
                break
        else:
            new_text.append(word)
    return ' '.join(new_text)


def _fill_in_the_blank(texts, label='fill: '):
    dataset = tf.data.Dataset.from_tensor_slices({'text': texts})
    dataset = fill_in_the_blank(dataset, label=label)
    input_ids_list= []
    labels_list = []
    
    for example in dataset:
        inputs = replace_X_with_extra_id(str(example['inputs'].numpy().decode("utf-8") ))
        labels = replace_X_with_extra_id(str(example['targets'].numpy().decode("utf-8") ))
        input_ids_list.append(inputs)
        labels_list.append(labels)
    return pd.DataFrame({'input_ids': input_ids_list, 'labels': labels_list})
        
        
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
df = pd.read_pickle('/home/student/Desktop/Project/Data/chords_en.pkl').head(20)
df = df.rename({'chords&lyrics': 'text'}, axis=1)


texts = df['text'].values.tolist()

labeled_df = _fill_in_the_blank(texts, label='fill song: ')
# labeled_df.to_pickle(f'/home/vmadmin/StudentData/Generative/TrainingDatasets/train_set_{ind}.pkl')


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


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




# In[16]:


labeled_df = pd.read_pickle('/home/student/Desktop/Project/Data/chords_en.pkl').head(100)[['chords&lyrics']].rename({'chords&lyrics': 'input_ids'}, axis=1)
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
model.model.save_pretrained('/home/student/Desktop/Project/Data/Models/model_basic')

