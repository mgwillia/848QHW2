from typing import List, Union, Tuple
from base_models import BaseGuesser, BaseReRanker, BaseRetriever, BaseAnswerExtractor
from qbdata import WikiLookup, QantaDatabase
from guess_train_dataset import GuessTrainDataset

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer

# Change this based on the GPU you use on your machine
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
        b, n = q_vectors.shape
        scores = torch.bmm(q_vectors.view(b, 1, n), ctx_vectors.view(b, n, 1)).squeeze()
        print(scores.shape) # should be b, b
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


class Guesser(BaseGuesser):
    """You can implement your own Bert based Guesser here"""
    def __init__(self) -> None:
        self.question_tokenizer = None
        self.question_model = None
        self.context_tokenizer = None
        self.context_model = None
        self.wiki_lookup = None

    def load(self):
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

        self.wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')

    def train(self, training_data: QantaDatabase, limit: int=-1):

        ### FIRST, PREP THE DATA ###
        train_dataset = GuessTrainDataset(training_data, self.question_tokenizer, self.context_tokenizer, self.wiki_lookup)

        ### THEN, TRAIN THE ENCODERS ###
        NUM_EPOCHS = 100 ### NOTE: this is for small datasets, maybe 40 for larger ones?
        BATCH_SIZE = 128

        question_optim = torch.optim.Adam(self.question_model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        question_scheduler = get_guesser_scheduler(question_optim, 100, NUM_EPOCHS * (len(train_dataset) // BATCH_SIZE))

        question_input_ids = self.question_tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        question_embeddings = self.question_model(question_input_ids).pooler_output

        context_input_ids = self.context_tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        context_embeddings = self.context_model(context_input_ids).pooler_output

        ### THEN, BUILD THE FAISS INDEX OF CONTEXT VECTORS ###


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

    def load(self, model_identifier: str, max_model_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_identifier, num_labels=2).to(device)

    def train(self):
        """Fill this method with code that finetunes Sequence Classification task on QuizBowl questions and passages.
        Feel free to change and modify the signature of the method to suit your needs."""
        pass

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
            logits = model_outputs.logits[:, 1] # Label 1 means they are similar

            return torch.argmax(logits, dim=-1)


class Retriever:
    """The component that indexes the documents and retrieves the top document from an index for an input open-domain question.
    
    It uses two systems:
     - Guesser that fetches top K documents for an input question, and
     - ReRanker that then reranks these top K documents by comparing each of them with the question to produce a similarity score."""

    def __init__(self, guesser: BaseGuesser, reranker: BaseReRanker, wiki_lookup: Union[str, WikiLookup], max_n_guesses=10) -> None:
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

    def load(self, model_identifier: str, max_model_length: int = 512):

        # You don't need to re-train the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, max_model_length=max_model_length)

        # Finetune this model for QuizBowl questions
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_identifier).to(device)

    def train(self):
        """Fill this method with code that finetunes Answer Extraction task on QuizBowl examples.
        Feel free to change and modify the signature of the method to suit your needs."""
        pass

    def extract_answer(self, question: Union[str,List[str]], ref_text: Union[str, List[str]]) -> List[str]:
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
        
