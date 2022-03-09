import pickle
import torch
import random
import os
from qbdata import QantaDatabase


class GuessTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: QantaDatabase, tokenizer, wiki_lookup, dataset_name, limit: int=-1):
        super(GuessTrainDataset, self).__init__()

        questions = [x.sentences for x in data.guess_train_questions]
        answers = [x.page for x in data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        if os.path.isfile(f'data/{dataset_name}_questions.pkl'):
            print('Loading already-prepared dataset') 
            with open(f"data/{dataset_name}_questions.pkl", "rb") as fp:
                self.questions = pickle.load(fp)
            with open(f"data/{dataset_name}_pages.pkl", "rb") as fp:
                self.answer_pages = pickle.load(fp)
            with open(f"data/{dataset_name}_answers.pkl", "rb") as fp:
                self.answers = pickle.load(fp)
        else:

            self.questions = []
            self.answer_pages = []
            self.answers = []
            answers_buffer = []
            print(f'Preparing dataset with {len(questions)} entries')
            for i, (question, page) in enumerate(zip(questions, answers)):
                answers_buffer.append(wiki_lookup[page]['text'])

                if len(answers_buffer) == 100:
                    answer_toks_batch = tokenizer(answers_buffer, return_tensors="pt", max_length=512, truncation=True, padding=True)["input_ids"]
                    self.answers.extend(answer_toks_batch)
                    answers_buffer = []

                
                question_toks = tokenizer(question, return_tensors="pt", max_length=512, truncation=True, padding=True)["input_ids"]
                #answer_tok = tokenizer(wiki_lookup[page]['text'], return_tensors="pt", max_length=512, truncation=True, padding=True)["input_ids"]

                self.questions.append(question_toks)
                self.answer_pages.append(page)
                #self.answers.append(answer_tok)

                if i % 1000 == 0:
                    print(f'Tokenized {i} questions and answers', flush=True)
            if len(answers_buffer) > 0:
                answer_toks_batch = tokenizer(answers_buffer, return_tensors="pt", max_length=512, truncation=True, padding=True)["input_ids"]
                self.answers.extend(answer_toks_batch)

            with open(f"data/{dataset_name}_questions.pkl", "wb") as fp:
                pickle.dump(self.questions, fp)        
            with open(f"data/{dataset_name}_pages.pkl", "wb") as fp:
                pickle.dump(self.answer_pages, fp)        
            with open(f"data/{dataset_name}_answers.pkl", "wb") as fp:
                pickle.dump(self.answers, fp)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        
        out = {'question': self.questions[index][random.randint(0, len(self.questions[index]))], 'answer_text': self.answers[index], 'answer_page': self.answer_pages[index]}

        return out
