import datasets
import random
import torch

from accelerate import Accelerator
from qbdata import QantaDatabase, WikiLookup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess_function(data):
    return tokenizer(data["questions"], data["content"], truncation=True)


if __name__ == "__main__":
    data_path = "../data"
    model_path = "models/reranker-finetuned-small"
    wiki_path = "../data/wiki_lookup.2018.json"

    qanta_db_train = QantaDatabase(data_path+'/small.guesstrain.json') # QantaDatabase(data_path+'/qanta.train.2018.json')
    qanta_db_dev = QantaDatabase(data_path+'/small.guessdev.json') # QantaDatabase(data_path+'/qanta.dev.2018.json')
    wiki_data = WikiLookup(wiki_path)

    training_data = qanta_db_train
    validation_data = qanta_db_dev

    model_identifier = 'amberoad/bert-multilingual-passage-reranking-msmarco'
    max_model_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)

    questions = [x.text for x in training_data.guess_train_questions]
    answers = [x.page for x in training_data.guess_train_questions]
    randints = [random.randrange(0,3) for i in range(len(answers))]

    passages = [wiki_data[x.page]['text'].replace(x.page.replace('_',' '), ' ') if randints[i]<2 else wiki_data.get_random_passage_masked() for i,x in enumerate(training_data.guess_train_questions)]
    labels = [1 if randints[i]<2 else 0 for i in randints]

    data_train = datasets.Dataset.from_dict({'questions':questions,  'labels':labels, 'content':passages})

    questions = [x.text for x in validation_data.guess_dev_questions]
    answers = [x.page for x in validation_data.guess_dev_questions]
    randints = [random.randrange(0,3) for i in range(len(answers))]

    passages = [wiki_data[x.page]['text'].replace(x.page.replace('_',' '), ' ') if randints[i]<2 else wiki_data.get_random_passage_masked() for i,x in enumerate(validation_data.guess_dev_questions)]
    labels = [1 if randints[i]<2 else 0 for i in randints]


    data_dev = datasets.Dataset.from_dict({'questions':questions,  'labels':labels, 'content':passages})

    data = datasets.DatasetDict({'train': data_train, 'validation': data_dev})

    tokenized_dataset = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    try:
        tokenized_dataset = tokenized_dataset.remove_columns(["questions", "content"])
    except:
        pass
    tokenized_dataset.set_format("torch")
    tokenized_dataset["train"].column_names

    train_dataloader = DataLoader(
        tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator
    )

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
        model.save_pretrained(f'{model_path}_{epoch}')
    model.save_pretrained(model_path)
