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
import ast

version = 'v2'


def replace_X_with_extra_id(text):
    i = 1
    new_text = []
    for word in text.split():
        if word == 'X':
            new_text.append(f'<extra_id_{i}>')
            i += 1
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


df = pd.read_pickle(f'/home/student/Desktop/Project/Data/T5Data/PreparedText/prepared_data_{version}.pkl')
texts = df['text'].values.tolist()
genres = df['genres'].values.tolist()

labeled_df = _fill_in_the_blank(texts, label='')

df['genres'] = df['genres'].apply(lambda x: str(x))

df['genres'] = df['genres'].apply(lambda x: '[' +', '.join(sorted(ast.literal_eval(x))) + ']')


labeled_df['input_ids'] = 'fill song ' + df['genres'] + ': ' + labeled_df['input_ids']
labeled_df = labeled_df.dropna()
labeled_df.to_pickle(f'/home/student/Desktop/Project/Data/T5Data/LabeledData/labeled_data_{version}.pkl')

