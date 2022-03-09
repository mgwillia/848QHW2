from models import AnswerExtractor

extractor = AnswerExtractor()
extractor.load('csarron/bert-base-uncased-squad-v1')
extractor.train()