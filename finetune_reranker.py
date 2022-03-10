import random

import datasets
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          get_scheduler)

from qbdata import QantaDatabase, WikiLookup

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_prepped_reranker_dataset_split(guess_questions, wiki_data):
    train_questions = []
    train_passages = []
    train_labels = []
    for guess_question in guess_questions:
        train_questions.append(guess_question.text)
        if random.randrange(0,3) < 2:
            train_passages.append(wiki_data[guess_question.page]['text'].replace(guess_question.page.replace('_',' '), ' '))
            train_labels.append(1)
        else:
            train_passages.append(wiki_data.get_random_passage_masked())
            train_labels.append(0)

    return datasets.Dataset.from_dict({'questions':train_questions, 'labels':train_labels, 'content':train_passages})


def get_prepped_reranker_data():
    data_path = "data"
    wiki_path = "data/wiki_lookup.2018.json"

    qanta_db_train = QantaDatabase(data_path+'/small.guesstrain.json') # QantaDatabase(data_path+'/qanta.train.2018.json')
    qanta_db_dev = QantaDatabase(data_path+'/small.guessdev.json') # QantaDatabase(data_path+'/qanta.dev.2018.json')
    wiki_data = WikiLookup(wiki_path)

    data_train = get_prepped_reranker_dataset_split(qanta_db_train.guess_train_questions, wiki_data)
    data_dev = get_prepped_reranker_dataset_split(qanta_db_dev.guess_dev_questions, wiki_data)
    data = datasets.DatasetDict({'train': data_train, 'validation': data_dev})
    return data

if __name__ == "__main__":
    NUM_EPOCHS = 3
    MODEL_PATH = "models/reranker-finetuned-small"


    ### PREP MODEL ###
    model_identifier = 'amberoad/bert-multilingual-passage-reranking-msmarco'
    max_model_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    ### PREP DATA ###
    def preprocess_function(data):
        return tokenizer(data["questions"], data["content"], truncation=True)

    prepped_reranker_data = get_prepped_reranker_data()
    tokenized_dataset = prepped_reranker_data.map(preprocess_function, batched=True)

    try:
        tokenized_dataset = tokenized_dataset.remove_columns(["questions", "content"])
    except:
        print('WARNING: removing columns has failed!')
    
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator)


    ### PREP TRAIN ###
    accelerator = Accelerator()
    train_dl, eval_dl, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

    num_training_steps = NUM_EPOCHS * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))


    ### TRAIN ###
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        model.save_pretrained(f'{MODEL_PATH}_{epoch}')
    model.save_pretrained(MODEL_PATH)
