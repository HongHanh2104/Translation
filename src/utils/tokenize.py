import spacy

class Tokenizer:
    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]
    
    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]