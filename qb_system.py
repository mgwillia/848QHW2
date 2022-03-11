import torch
from tqdm import tqdm

from qbdata import QantaDatabase
from tfidf_guesser import TfidfGuesser
from models import AnswerExtractor, Retriever, ReRanker, WikiLookup, Guesser


class QuizBowlSystem:

    def __init__(self) -> None:
        """Fill this method to create attributes, load saved models, etc
        Don't have any arguments to this constructor.
        If you really want to have arguments, they should have some default values set.
        """
        print('Loading the Guesser model...')
        guesser = TfidfGuesser()
        guesser.load('models/tfidf_full.pickle')
        #guesser = Guesser()
        #guesser.load()
        #guesser.train(QantaDatabase('data/qanta.train.2018.json'))
        #guesser.build_faiss_index()

        print('Loding the Wiki Lookups...')
        self.wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')

        print('Loading the Reranker model...')
        first_sent_reranker = ReRanker()
        last_sent_reranker = ReRanker()
        first_sent_path = 'models/reranker-first_sent-finetuned-full_2'
        last_sent_path = 'models/reranker-last_sent-finetuned-full_2'
        identifier = 'amberoad/bert-multilingual-passage-reranking-msmarco'
        first_sent_reranker.load(identifier, first_sent_path)
        last_sent_reranker.load(identifier, last_sent_path)

        self.retriever = Retriever(guesser, first_sent_reranker, last_sent_reranker, wiki_lookup=self.wiki_lookup)

        answer_extractor_base_model = "csarron/bert-base-uncased-squad-v1"
        self.answer_extractor = AnswerExtractor()
        print('Loading the Answer Extractor model...')
        self.answer_extractor.load(answer_extractor_base_model, "models/extractor")

    def retrieve_page(self, question: str, disable_reranking=False, is_first_sent=False) -> str:
        """Retrieves the wikipedia page name for an input question."""
        with torch.no_grad():
            page = self.retriever.retrieve_answer_document(question, disable_reranking=disable_reranking, is_first_sent=is_first_sent)
            return page

    def execute_query(self, question: str, *, get_page=True, is_first_sent=False) -> str:
        """Populate this method to do the following:
        1. Use the Retriever to get the top wikipedia page.
        2. Tokenize the question and the passage text to prepare inputs to the Bert-based Answer Extractor
        3. Predict an answer span for each question and return the list of corresponding answer texts."""
        with torch.no_grad():
            page = self.retrieve_page(question, disable_reranking=False, is_first_sent=is_first_sent)
            reference_text = self.wiki_lookup[page]['text']
            answer = self.answer_extractor.extract_answer(question, reference_text)[0] # singleton list
            return answer, page if get_page else answer


if __name__ == "__main__":
    qa = QuizBowlSystem()
    qanta_db_dev = QantaDatabase('data/qanta.dev.2018.json')
    small_set_questions = qanta_db_dev.all_questions[:10]

    for question in tqdm(small_set_questions):
        answer = qa.execute_query(question.first_sentence)
