import os
import torch
from qbdata import WikiLookup, QantaDatabase


class GuessTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: QantaDatabase, question_tokenizer, context_tokenizer, wiki_lookup, limit: int=-1):
        super(GuessTrainDataset, self).__init__()

        questions = [x.text for x in data.guess_train_questions]
        answers = [x.page for x in data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        self.questions = []
        self.answer_pages = []

        for doc, page in zip(questions, answers):
            self.questions.append(question_tokenizer(doc, return_tensors="pt")["input_ids"])
            self.answer_pages.append(context_tokenizer(wiki_lookup[page]['text'], return_tensors="pt")["input_ids"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        
        
        
        out = {'image': img, 'target': target, 'universal_label': universal_label}

        return out
