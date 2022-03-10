from typing import List, Union

from datasets import Dataset, load_dataset

from base_models import BaseGuesser, BaseReRanker, BaseRetriever, BaseAnswerExtractor
from qbdata import WikiLookup, QantaDatabase, Question

import torch
from transformers import BertForSequenceClassification, pipeline, TrainingArguments, Trainer, default_data_collator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback, get_cosine_with_hard_restarts_schedule_with_warmup, AdamW, \
    DataCollatorWithPadding

# Change this based on the GPU you use on your machine
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
    pass


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
                                          num_train_epochs=1,
                                          metric_for_best_model="eval_loss",
                                          fp16=True,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          #   save_strategy="no",
                                          #   save_total_limit=6,
                                          dataloader_num_workers=2,
                                          logging_steps=72,
                                          )

        num_training_steps = 1 * len(tokenized_train_dataset["train"])
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
