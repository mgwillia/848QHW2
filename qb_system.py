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
        guesser = TfidfGuesser()
        #guesser = Guesser()
        print('Loading the Guesser model...')
        guesser.load('models/tfidf_full.pickle')
        #guesser.load()
        #guesser.build_faiss_index()

        print('Loding the Wiki Lookups...')
        self.wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')

        print('Loading the Reranker model...')
        reranker = ReRanker()
        path = 'models/reranker-finetuned-full_1'
        identifier = 'amberoad/bert-multilingual-passage-reranking-msmarco'
        reranker.load(identifier, path)

        self.retriever = Retriever(guesser, reranker, wiki_lookup=self.wiki_lookup)

        answer_extractor_base_model = "csarron/bert-base-uncased-squad-v1"
        self.answer_extractor = AnswerExtractor()
        print('Loading the Answer Extractor model...')
        self.answer_extractor.load(answer_extractor_base_model, "models/extractor")

    def retrieve_page(self, question: str, disable_reranking=False) -> str:
        """Retrieves the wikipedia page name for an input question."""
        with torch.no_grad():
            page = self.retriever.retrieve_answer_document(question, disable_reranking=disable_reranking)
            return page

    def execute_query(self, question: str, *, get_page=True) -> str:
        """Populate this method to do the following:
        1. Use the Retriever to get the top wikipedia page.
        2. Tokenize the question and the passage text to prepare inputs to the Bert-based Answer Extractor
        3. Predict an answer span for each question and return the list of corresponding answer texts."""
        with torch.no_grad():
            page = self.retrieve_page(question, disable_reranking=False)
            reference_text = self.wiki_lookup[page]['text']
            answer = self.answer_extractor.extract_answer(question, reference_text)[0] # singleton list
            return answer, page if get_page else answer


if __name__ == "__main__":
    qa = QuizBowlSystem()
    qanta_db_dev = QantaDatabase('data/qanta.dev.2018.json')
    small_set_questions = qanta_db_dev.all_questions[:10]

    for question in tqdm(small_set_questions):
        answer = qa.execute_query(question.first_sentence)
