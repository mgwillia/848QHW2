import argparse

from models import Guesser
from qbdata import QantaDatabase


def compute_retrieval_accuracy(guesser, evaluation_data: QantaDatabase, num_guesses: int):
    questions = [x.sentences[-1] for x in evaluation_data.guess_dev_questions]
    answers = [x.page for x in evaluation_data.guess_dev_questions]

    print(f'Eval on {len(questions)} questions')

    num_correct = 0
    data_index = 0
    raw_guesses = guesser.guess(questions, max_n_guesses=num_guesses)
    for i, answer in enumerate(answers):
        guesses = raw_guesses[i]
        for guess in guesses:
            if guess[0] == answer:
                num_correct += 1
                break
        data_index += 1
        if data_index % 100 == 0:
            print(f'{data_index}/{len(raw_guesses)} for computing accuracy')
    return num_correct / data_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", default="data/qanta.train.2018.json", type=str)
    parser.add_argument("--dev_data", default="data/qanta.dev.2018.json", type=str)
    parser.add_argument("--epoch_num", default=0, type=int)
    parser.add_argument("--sentence_splitter", action='store_true')

    flags = parser.parse_args()

    print("Loading %s" % flags.train_data)
    guesstrain = QantaDatabase(flags.train_data)
    guessdev = QantaDatabase(flags.dev_data)

    guesser = Guesser()
    if flags.sentence_splitter:
        guesser.load(True, f'models/guesser_question_encoder_{flags.epoch_num}.pth.tar', f'models/guesser_context_encoder_{flags.epoch_num}.pth.tar')
    else:
        guesser.load(True, f'models/guesser_question_encoder_new_{flags.epoch_num}.pth.tar', f'models/guesser_context_encoder_new_{flags.epoch_num}.pth.tar')
    #guesser.train(guesstrain, f'models/context_embeddings_{flags.epoch_num}_{int(flags.sentence_splitter == True)}.pth.tar')
    guesser.build_faiss_index(f'models/context_embeddings_{flags.epoch_num}_{int(flags.sentence_splitter == True)}.pth.tar')

    for num_guesses in [1, 5, 10, 20]:
        print(f'epoch: {flags.epoch_num}, splitting: {flags.sentence_splitter}, guesses: {num_guesses}, accuracy: {compute_retrieval_accuracy(guesser, guessdev, num_guesses)}')
