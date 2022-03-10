from typing import List, Union, Tuple, Optional, Dict
from collections import defaultdict
from base_models import BaseGuesser, BaseReRanker, BaseRetriever, BaseAnswerExtractor
from qbdata import WikiLookup, QantaDatabase, Question
from guess_train_dataset import GuessTrainDataset

from datasets import Dataset, load_dataset

import faiss
import torch
import numpy as np
import tqdm
import argparse
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import DPRQuestionEncoder, BertTokenizerFast, DPRContextEncoder
from transformers import BertForSequenceClassification, pipeline, TrainingArguments, Trainer, default_data_collator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback, get_cosine_with_hard_restarts_schedule_with_warmup, AdamW, \
    DataCollatorWithPadding

# Change this based on the GPU you use on your machine
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BiEncoderNllLoss(torch.nn.Module):

    def __init__(self):
        super(BiEncoderNllLoss, self).__init__()

    def forward(
        self,
        q_vectors: torch.Tensor,
        ctx_vectors: torch.Tensor,
        positive_idx_per_question: list,
    ) -> Tuple[torch.Tensor, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, correct_predictions_count


def get_guesser_scheduler(optimizer, warmup_steps, total_training_steps, steps_shift=0, last_epoch=-1):

    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def make_dictionary(data: List[Question]):
    data_dict = {"text": [], "page": [], "first_sentence": [], "last_sentence": [], "answer": [], "category": []}
    for question in data:
        data_dict["text"].append(question.text)
        data_dict["page"].append(question.page)
        data_dict["first_sentence"].append(question.first_sentence)
        data_dict["last_sentence"].append(question.sentences[-1])
        data_dict["answer"].append(question.answer)
        data_dict["category"].append(question.category)
    return data_dict


class Guesser(BaseGuesser):
    """You can implement your own Bert based Guesser here"""
    def __init__(self) -> None:
        self.tokenizer = None
        self.question_model = None
        self.context_model = None
        self.wiki_lookup = None
        self.index = None

    def load(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

        self.context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

        self.wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')

    def finetune(self, training_data: QantaDatabase, dev_data: QantaDatabase, limit: int=-1):
        NUM_EPOCHS = 10 ### NOTE: this is for small datasets, maybe 40 for larger ones?
        BASE_BATCH_SIZE = 128
        BATCH_SIZE = 4
        LR_SCALE_FACTOR = BATCH_SIZE / BASE_BATCH_SIZE

        self.load()

        ### FIRST, PREP THE DATA ###
        train_dataset = GuessTrainDataset(training_data, self.tokenizer, self.wiki_lookup, 'train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True, shuffle=True)
        dev_dataset = GuessTrainDataset(dev_data, self.tokenizer, self.wiki_lookup, 'dev')
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset, num_workers=4, batch_size=BATCH_SIZE*2, pin_memory=True, drop_last=False, shuffle=False)

        ### THEN, TRAIN THE ENCODERS ###
        question_optim = torch.optim.Adam(self.question_model.parameters(), lr=1e-5*LR_SCALE_FACTOR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        context_optim = torch.optim.Adam(self.context_model.parameters(), lr=1e-5*LR_SCALE_FACTOR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        question_scheduler = get_guesser_scheduler(question_optim, 100, NUM_EPOCHS * (len(train_dataset) // BATCH_SIZE))
        context_scheduler = get_guesser_scheduler(context_optim, 100, NUM_EPOCHS * (len(train_dataset) // BATCH_SIZE))
        loss_fn = BiEncoderNllLoss()

        print('Ready to finetune', flush=True)
        self.question_model.train()
        self.context_model.train()
        losses = []
        for epoch_num in range(NUM_EPOCHS):
            for batch_num, batch in enumerate(train_dataloader):
                questions, answers = batch['question'].to(device), batch['answer_text'].to(device)
                question_embeddings = self.question_model(questions).pooler_output
                context_embeddings = self.context_model(answers).pooler_output
                batch_loss, _ = loss_fn(question_embeddings, context_embeddings, list(range(question_embeddings.shape[0])))

                question_optim.zero_grad()
                context_optim.zero_grad()
                batch_loss.backward()
                question_optim.step()
                context_optim.step()
                question_scheduler.step()
                context_scheduler.step()

                losses.append(batch_loss.item())

                if batch_num % 250 == 249:
                    print(f'Batch Num: {batch_num}, Loss: {np.array(losses).mean()}', flush=True)
                    losses = []

            torch.save(self.question_model, f'models/guesser_question_encoder_{epoch_num}.pth.tar')
            torch.save(self.context_model, f'models/guesser_context_encoder_{epoch_num}.pth.tar')
        torch.save(self.question_model, f'models/guesser_question_encoder.pth.tar')
        torch.save(self.context_model, f'models/guesser_context_encoder.pth.tar')

    def train(self, training_data: QantaDatabase, limit: int=-1):
        ### BUILD THE FAISS INDEX OF CONTEXT VECTORS ###
        BATCH_SIZE = 256
        DIMENSION = 768 ### TODO: double check embed length
        if os.path.isfile('models/guesser_question_encoder.pth.tar'):
            self.question_model = torch.load('models/guesser_question_encoder.pth.tar', map_location=device).to(device)
        else:
            self.question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
        if os.path.isfile('models/guesser_context_encoder.pth.tar'):
            self.context_model = torch.load('models/guesser_context_encoder.pth.tar', map_location=device)
        else:
            self.context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)


        train_dataset = GuessTrainDataset(training_data, self.tokenizer, self.wiki_lookup, 'train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, pin_memory=True, drop_last=False, shuffle=False)

        self.question_model.eval()
        self.context_model.eval()
        context_embeddings = torch.zeros((len(train_dataset), DIMENSION))
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                answers = batch['answer_text'].to(device)
                context_embeddings[i*BATCH_SIZE:min(len(train_dataset), (i+1)*BATCH_SIZE)] = self.context_model(answers).pooler_output

        self.index = faiss.IndexFlatL2(DIMENSION)
        print(self.index.is_trained)
        self.index.add(context_embeddings)
        print(self.index.ntotal)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        """
        Given the text of questions, generate guesses (a tuple of page id and score) for each one.

        Keyword arguments:
        questions -- Raw text of questions in a list
        max_n_guesses -- How many top guesses to return
        """ 
        DIMENSION = 768 ### TODO: double check embed length
        question_embeddings = torch.zeros((len(questions), DIMENSION))
        with torch.no_grad():
            for i, question in enumerate(questions):
                question_embeddings[i] = self.question_model(self.tokenizer(question, return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"].to(device)).pooler_output

        neighbor_scores, neighbor_indices = self.index.search(question_embeddings, max_n_guesses)     #  returns neighbor embeddings, indices of neighbor embeddings
        #print(neighbor_indices[:5])                   # neighbors of the 5 first queries
        #print(neighbor_indices[-5:])                  # neighbors of the 5 last queries
        guesses = []
        for i in range(len(questions)):
            guess = []
            for j in range(max_n_guesses):
                guess.append((neighbor_indices[i][j], neighbor_scores[i][j]))
            guesses.append(guess)
        return guesses

    def confusion_matrix(self, evaluation_data: QantaDatabase, limit=-1) -> Dict[str, Dict[str, int]]:
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param evaluation_data: Database of questions and answers
        :param limit: How many evaluation questions to use
        """

        questions = [x.text for x in evaluation_data.guess_dev_questions]
        answers = [x.page for x in evaluation_data.guess_dev_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        print("Eval on %i question" % len(questions))
            
        d = defaultdict(dict)
        data_index = 0
        guesses = [x[0][0] for x in self.guess(questions, max_n_guesses=1)]
        for gg, yy in zip(guesses, answers):
            d[yy][gg] = d[yy].get(gg, 0) + 1
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index,
                                                      len(guesses)))
        return d


class ReRanker(BaseReRanker):
    """A Bert based Reranker that consumes a reference passage and a question as input text and predicts the similarity score:
        likelihood for the passage to contain the answer to the question.

    Task: Load any pretrained BERT-based and finetune on QuizBowl (or external) examples to enable this model to predict scores
        for each reference text for an input question, and use that score to rerank the reference texts.

    Hint: Try to create good negative samples for this binary classification / score regression task.

    Documentation Links:

        Pretrained Tokenizers:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained

        BERT for Sequence Classification:
            https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.BertForSequenceClassification

        SequenceClassifierOutput:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput

        Fine Tuning BERT for Seq Classification:
            https://huggingface.co/docs/transformers/master/en/custom_datasets#sequence-classification-with-imdb-reviews

        Passage Reranking:
            https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def load(self, model_identifier: str, finetuned: str = "", max_model_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)

        if finetuned == "":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_identifier, num_labels=2).to(device)
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                finetuned, num_labels=2).to(device)


    def get_best_document(self, question: str, ref_texts: List[str]) -> int:
        """Selects the best reference text from a list of reference text for each question."""


        with torch.no_grad():
            n_ref_texts = len(ref_texts)
            inputs_A = [question] * n_ref_texts
            inputs_B = ref_texts

            model_inputs = self.tokenizer(
                inputs_A, inputs_B, return_token_type_ids=True, padding=True, truncation=True,
                return_tensors='pt').to(device)

            model_outputs = self.model(**model_inputs)
            logits = model_outputs.logits[:, 1]  # Label 1 means they are similar

            return torch.argmax(logits, dim=-1)


class Retriever:
    """The component that indexes the documents and retrieves the top document from an index for an input open-domain question.

    It uses two systems:
     - Guesser that fetches top K documents for an input question, and
     - ReRanker that then reranks these top K documents by comparing each of them with the question to produce a similarity score."""

    def __init__(self, guesser: BaseGuesser, reranker: BaseReRanker, wiki_lookup: Union[str, WikiLookup],
                 max_n_guesses=10) -> None:
        if isinstance(wiki_lookup, str):
            self.wiki_lookup = WikiLookup(wiki_lookup)
        else:
            self.wiki_lookup = wiki_lookup
        self.guesser = guesser
        self.reranker = reranker
        self.max_n_guesses = max_n_guesses

    def retrieve_answer_document(self, question: str, disable_reranking=False) -> str:
        """Returns the best guessed page that contains the answer to the question."""
        guesses = self.guesser.guess([question], max_n_guesses=self.max_n_guesses)[0]

        if disable_reranking:
            _, best_page = max((score, page) for page, score in guesses)
            return best_page

        ref_texts = []
        for page, score in guesses:
            doc = self.wiki_lookup[page]['text']
            ref_texts.append(doc)

        best_doc_id = self.reranker.get_best_document(question, ref_texts)
        return guesses[best_doc_id][0]


class AnswerExtractor:
    """Load a huggingface model of type transformers.AutoModelForQuestionAnswering and finetune it for QuizBowl questions.

    Documentation Links:

        Extractive QA:
            https://huggingface.co/docs/transformers/v4.16.2/en/task_summary#extractive-question-answering

        QA Pipeline:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline

        QAModelOutput:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput

        Finetuning Answer Extraction:
            https://huggingface.co/docs/transformers/master/en/custom_datasets#question-answering-with-squad
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')

    def load(self, model_identifier: str, saved_model_path: str = None, max_model_length: int = 512):

        # You don't need to re-train the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, max_model_length=max_model_length)

        # Finetune this model for QuizBowl questions
        if saved_model_path is None:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                model_identifier).to(device)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(saved_model_path).to(device)

    def gen_train_data(self, data):
        pages = data['page']
        answers = data['answer']
        first_sentences = data['first_sentence']
        last_sentences = data['last_sentence']

        all_questions, references, splits = [], [], []
        length = len(pages)
        for i in range(length):
            page = pages[i]
            answer = answers[i]
            first_sentence = first_sentences[i]
            last_sentence = last_sentences[i]
            ref_text = self.wiki_lookup[page]['text']
            if ref_text.lower().find(answer) != -1:
                answer_dict = {'answer_start': [ref_text.lower().find(answer)], 'text': [answer]}
                all_questions.append(first_sentence)
                references.append(ref_text)
                splits.append(answer_dict)
                all_questions.append(last_sentence)
                references.append(ref_text)
                splits.append(answer_dict)
            ret_data = {'questions': all_questions, 'ref_texts': references, 'answers': splits}
            return ret_data

    def preprocess(self, data):
        inputs = self.tokenizer(
            data['questions'],
            data['ref_texts'],
            return_token_type_ids=True, padding=True, truncation=True, add_special_tokens=True,
            return_offsets_mapping=True,
        )

        start_positions = []
        end_positions = []
        offset_mapping = inputs.pop("offset_mapping")
        answers = data['answers']

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def train(self, evaluating_dataset=None):
        """Fill this method with code that finetunes Answer Extraction task on QuizBowl examples.
        Feel free to change and modify the signature of the method to suit your needs."""

        NUM_EPOCHS = 10

        # train_questions = QantaDatabase('data/qanta.train.2018.json').train_questions
        # eval_questions = QantaDatabase('data/qanta.dev.2018.json').dev_questions
        # train_dataset = Dataset.from_dict(self.gen_train_data(make_dictionary(train_questions))).shuffle(seed=42)
        # eval_dataset = Dataset.from_dict(self.gen_train_data(make_dictionary(eval_questions))).shuffle(seed=42)

        # modified from Huggingface QA fine tuning tutorial

        def preprocess_function(examples):
            questions = [q.strip() for q in examples["text"]]
            first_sentences = [q.strip() for q in examples["first_sentence"]]
            wiki_texts = list(map(lambda x: self.wiki_lookup[x]["text"], examples["page"]))
            inputs = self.tokenizer(
                [*questions, *first_sentences],
                [*wiki_texts, *wiki_texts],
                truncation=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            answers = [*examples["answer"], *examples["answer"]]
            # print(len(answers))
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                answer = answers[i]
                start_char = 0
                end_char = len(answer)
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        print("Dataset gen complete!")

        training_dataset = load_dataset("json", data_files={"train": 'data/qanta.train.2018.json',
                                                            "eval": 'data/qanta.dev.2018.json'}
                                        , field="questions")

        tokenized_train_dataset = training_dataset.map(preprocess_function, batched=True,
                                                       remove_columns=training_dataset['train'].column_names)
        print("Dataset preprocessing complete!")
        tokenized_train_dataset.shuffle(seed=0)
        data_collator = default_data_collator

        training_args = TrainingArguments(output_dir="models/extractor",
                                          save_total_limit=1,
                                          load_best_model_at_end=True,
                                          per_device_train_batch_size=4,
                                          per_device_eval_batch_size=4,
                                          num_train_epochs=NUM_EPOCHS,
                                          metric_for_best_model="eval_loss",
                                          fp16=True,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          #   save_strategy="no",
                                          #   save_total_limit=6,
                                          dataloader_num_workers=2,
                                          logging_steps=72,
                                          )

        num_training_steps = NUM_EPOCHS * len(tokenized_train_dataset["train"])
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=1 // 7 * num_training_steps // 5,
            num_training_steps=num_training_steps // 5,
            num_cycles=5,
        )

        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=tokenized_train_dataset["train"],
                          eval_dataset=tokenized_train_dataset["eval"],
                          data_collator=data_collator,
                          tokenizer=self.tokenizer,
                          optimizers=(optimizer, lr_scheduler),
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
                          )
        trainer.train()
        self.model.save_pretrained('models/extractor')

    def extract_answer(self, question: Union[str, List[str]], ref_text: Union[str, List[str]]) -> List[str]:
        """Takes a (batch of) questions and reference texts and returns an answer text from the
        reference which is answer to the input question.
        """
        with torch.no_grad():
            model_inputs = self.tokenizer(
                question, ref_text, return_tensors='pt', truncation=True, padding=True,
                return_token_type_ids=True, add_special_tokens=True).to(device)
            outputs = self.model(**model_inputs)
            input_tokens = model_inputs['input_ids']
            start_index = torch.argmax(outputs.start_logits, dim=-1)
            end_index = torch.argmax(outputs.end_logits, dim=-1)

            answer_ids = [tokens[s:e] for tokens, s, e in zip(input_tokens, start_index, end_index)]

            return self.tokenizer.batch_decode(answer_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", default="data/qanta.train.2018.json", type=str)
    parser.add_argument("--dev_data", default="data/qanta.dev.2018.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--show_confusion_matrix", default=True, type=bool)
    parser.add_argument("--train_guesser", action="store_true")
    parser.add_argument("--train_extractor", action="store_true")

    flags = parser.parse_args()

    print("Loading %s" % flags.train_data)
    guesstrain = QantaDatabase(flags.train_data)
    guessdev = QantaDatabase(flags.dev_data)
    
    if flags.train_guesser:
        guesser = Guesser()
        guesser.finetune(guesstrain, guessdev, limit=flags.limit)

        if flags.show_confusion_matrix:
            confusion = guesser.confusion_matrix(guessdev, limit=-1)
            print("Errors:\n=================================================")
            for ii in confusion:
                for jj in confusion[ii]:
                    if ii != jj:
                        print("%i\t%s\t%s\t" % (confusion[ii][jj], ii, jj))
    elif flags.train_extractor:
        extractor = AnswerExtractor()
        extractor.load('csarron/bert-base-uncased-squad-v1')
        extractor.train()
