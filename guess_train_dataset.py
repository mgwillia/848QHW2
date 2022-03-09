import os
import torch
from qbdata import WikiLookup, QantaDatabase


class GuessTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: QantaDatabase, question_tokenizer, context_tokenizer, wiki_lookup, is_pretrain=False, limit: int=-1):
        super(GuessTrainDataset, self).__init__()

        questions = [x.text for x in data.guess_train_questions]
        answers = [x.page for x in data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        self.questions = []
        self.answer_pages = []
        self.answers = []

        for question, page in zip(questions, answers):
            if is_pretrain:
                self.questions.append(question_tokenizer(question.sentences()[0], return_tensors="pt")["input_ids"])
                self.questions.append(question_tokenizer(question.sentences()[-1], return_tensors="pt")["input_ids"])
                self.answer_pages.append(page)
                self.answer_pages.append(page)
                self.answers.append(context_tokenizer(wiki_lookup[page]['text'], return_tensors="pt")["input_ids"])
                self.answers.append(context_tokenizer(wiki_lookup[page]['text'], return_tensors="pt")["input_ids"])
            else:
                self.questions.append(question_tokenizer(question.sentences()[0], return_tensors="pt")["input_ids"])
                self.answer_pages.append(page)
                self.answers.append(context_tokenizer(wiki_lookup[page]['text'], return_tensors="pt")["input_ids"])


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        
        out = {'question': self.questions[index], 'answer_text': self.answers[index], 'answer_page': self.answer_pages[index]}

        return out
