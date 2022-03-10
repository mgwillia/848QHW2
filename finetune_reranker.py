# -*- coding: utf-8 -*-
"""finetune_reranker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lm9u7dFVMSgt2kquUwGHL1qdS4CVpNC4
"""

#!pip install transformers
#!pip install wget

from qbdata import QantaDatabase, WikiLookup
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
import os
import json
import random

from typing import List, Union
#!pip install transformers datasets
from transformers import DataCollatorWithPadding, BertTokenizer
import datasets
import pickle
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from transformers import TrainingArguments
from transformers import Trainer
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW, optimizer, lr_scheduler
from transformers import BertForSequenceClassification, pipeline, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification

#data_path = "/content/drive/MyDrive/848-hw-main/data"
#model_path = "/content/drive/MyDrive/848-hw-main/hw2/models"
#wiki_path = "/content/drive/MyDrive/848-hw-main/data/wiki_lookup.2018.json"

data_path = "../data"
model_path = "models/reranker-finetuned-small"
wiki_path = "../data/wiki_lookup.2018.json"

#from google.colab import drive
#drive.mount('/content/drive')

qanta_db_train = QantaDatabase(data_path+'/small.guesstrain.json')
qanta_db_dev = QantaDatabase(data_path+'/small.guessdev.json')

#qanta_db_train = QantaDatabase(data_path+'/qanta.train.2018.json')
#qanta_db_dev = QantaDatabase(data_path+'/qanta.dev.2018.json')
wiki_data = WikiLookup(wiki_path)

GUESSER_TRAIN_FOLD = 'guesstrain'
BUZZER_TRAIN_FOLD = 'buzztrain'
TRAIN_FOLDS = {GUESSER_TRAIN_FOLD, BUZZER_TRAIN_FOLD}

# Guesser and buzzers produce reports on these for cross validation
GUESSER_DEV_FOLD = 'guessdev'
BUZZER_DEV_FOLD = 'buzzdev'
DEV_FOLDS = {GUESSER_DEV_FOLD, BUZZER_DEV_FOLD}

# System-wide cross validation and testing
GUESSER_TEST_FOLD = 'guesstest'
BUZZER_TEST_FOLD = 'buzztest'



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



training_data = qanta_db_train
validation_data = qanta_db_dev

model_identifier = 'amberoad/bert-multilingual-passage-reranking-msmarco'
max_model_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)

def preprocess_function(data):
  return tokenizer(data["questions"], data["content"], truncation=True)

questions = [x.text for x in training_data.guess_train_questions]
answers = [x.page for x in training_data.guess_train_questions]
randints = [random.randrange(0,3) for i in range(len(answers))] #list of len(answers) random numbers from [0,1,2] 
print(randints)

passages = [wiki_data[x.page]['text'].replace(x.page.replace('_',' '), ' ') if randints[i]<2 else wiki_data.get_random_passage_masked() for i,x in enumerate(training_data.guess_train_questions)]
labels = [1 if randints[i]<2 else 0 for i in randints]

#print(labels[:5])

for i in range(5):
  print(questions[i])
  print(passages[i])
  print(labels[i])

#replaced page title in text with space
#with prob 1/3 take random negative samples from wiki_data


data_train = datasets.Dataset.from_dict({'questions':questions,  'labels':labels, 'content':passages})

print(data_train)

questions = [x.text for x in validation_data.guess_dev_questions]
answers = [x.page for x in validation_data.guess_dev_questions]
randints = [random.randrange(0,3) for i in range(len(answers))] #list of len(answers) random numbers from [0,1,2] 
print(randints)

passages = [wiki_data[x.page]['text'].replace(x.page.replace('_',' '), ' ') if randints[i]<2 else wiki_data.get_random_passage_masked() for i,x in enumerate(validation_data.guess_dev_questions)]
labels = [1 if randints[i]<2 else 0 for i in randints]

print(labels[:5])

#for i in range(5):
#  print(questions[i])
#  print(passages[i])
#  print(labels[i])

#replaced page title in text with space
#with prob 1/3 take random negative samples from wiki_data


data_dev = datasets.Dataset.from_dict({'questions':questions,  'labels':labels, 'content':passages})
print(data_dev)

data = datasets.DatasetDict({'train': data_train, 'validation': data_dev})
print(data)

raw_train_dataset = data["train"]
print(raw_train_dataset[0])

tokenized_dataset = data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokenized_dataset)




try:
  tokenized_dataset = tokenized_dataset.remove_columns(["questions", "content"])
except:
  pass
tokenized_dataset.set_format("torch")
tokenized_dataset["train"].column_names

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
#!pip install accelerate

import torch
from tqdm.auto import tqdm


from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.save_pretrained(model_path)




