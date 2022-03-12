import argparse
import random

from models import ReRanker
from qbdata import QantaDatabase


def compute_retrieval_accuracy(reranker: ReRanker, evaluation_data: QantaDatabase, num_guesses: int, first_sent: bool):
    if first_sent:
        questions = [x.sentences[0] for x in evaluation_data.guess_dev_questions]
    else:
        questions = [x.sentences[-1] for x in evaluation_data.guess_dev_questions]
    answers = [x.page for x in evaluation_data.guess_dev_questions]

    print(f'Eval on {len(questions)} questions')

    num_correct = 0
    data_index = 0
    for question, correct_answer in zip(questions, answers):
        correct_answer_index = random.randrange(0, num_guesses)
        ref_texts = random.choices(answers, k=num_guesses)
        ref_texts[correct_answer_index] = correct_answer
        guess_index = reranker.get_best_document(question, ref_texts)
        if correct_answer_index == guess_index.item():
            num_correct += 1
        data_index += 1
        if data_index % 100 == 0:
            print(f'{data_index}/{len(questions)} for computing accuracy')
    return num_correct / data_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dev_data", default="data/qanta.dev.2018.json", type=str)
    parser.add_argument("--epoch_num", default=0, type=int)
    parser.add_argument("--type", default='full', type=str)

    flags = parser.parse_args()

    print("Loading %s" % flags.dev_data)
    guessdev = QantaDatabase(flags.dev_data)

    full_path = f'models/reranker-finetuned-full_{flags.epoch_num}'
    first_sent_path = f'models/reranker-first_sent-finetuned-full_{flags.epoch_num}'
    last_sent_path = f'models/reranker-last_sent-finetuned-full_{flags.epoch_num}'
    identifier = 'amberoad/bert-multilingual-passage-reranking-msmarco'
    reranker = ReRanker()
    if flags.type == 'full':
        reranker.load(identifier, full_path)
    elif flags.type == 'first_sent':
        reranker.load(identifier, first_sent_path)
    elif flags.type == 'last_sent':
        reranker.load(identifier, last_sent_path)
    else:
        reranker.load(identifier)

    for num_guesses in [5, 10, 15, 20]:
        print(f'epoch: {flags.epoch_num}, type: {flags.type}, guesses: {num_guesses}, first_sent_accuracy: {compute_retrieval_accuracy(reranker, guessdev, num_guesses, True)}')
        print(f'epoch: {flags.epoch_num}, type: {flags.type}, guesses: {num_guesses}, last_sent_accuracy: {compute_retrieval_accuracy(reranker, guessdev, num_guesses, False)}')
