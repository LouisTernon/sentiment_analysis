import re
import numpy as np 
import spacy 

class SentencePreprocessor:

    def __init__(self, spacy_model=None, lemmatize=False, min_letter=2):
        """
        Preprocessing class. Applied to each sentence individually. 
        """
        if spacy_model : self.spacy_nlp = spacy.load(spacy_model)
        self.lemmatize = lemmatize
        self.min_letter = min_letter


    def process(self, s:str, pretrained_embedding:bool):
        """
        Main function of the class: takes as a sentence, and iteratively perform all the preprocessing
        Return a string, or an array of shape (n_words, n_emb) if pretrained_embedding=True
        """
        if self.lemmatize:
            s_ = self.clean(s)
        else:
            s_ = self.replace_ponctuation(s)
            s_ = self.lower(s_)
        if self.min_letter: s_ = self.filter_min_letter(s_, self.min_letter)
        if pretrained_embedding: s_ = self.embed(s_)
        return s_


    @staticmethod
    def filter_min_letter(s:str, min_letter:int) -> str:
        """
        Remove words from sentence with less than min_letter letters
        """
        filtered = " ".join(list(filter(lambda w: len(w)>=min_letter,  s.split(" "))))
        return filtered

    @staticmethod
    def replace_ponctuation(s:str) -> str:
        """
        Replace ponc
        """
        s_ = re.sub(r"[^\w\s]", " ", s)
        return s_

    @staticmethod
    def lower(s:str) -> str:
        s_ = s.lower()
        return s_

    def clean(self, s: str) -> str:
        """
        filter, lower and lemmatize words
        :param s (str): single sentence 
        :param spacy_nlp: spacy model
        :return: a list of lists of words
        """
        string = re.sub('[^a-zA-Z ]+', " ", s.lower())
        spacy_tokens = self.spacy_nlp(string)
        res = [token.lemma_ for token in spacy_tokens if token.lemma_ != '-PRON-' and token.lemma_.strip()]
        cleaned_s = " ".join(res)
        return cleaned_s

    def embed(self, s:str) -> np.array:
        """
        Embedd a single sentence using spacy_nlp embedder
        :param s: str, single sentence
        :return: array (len(s), 300)
        """
        spacy_tokens = self.spacy_nlp(s)
        embedded = np.array([token.vector for token in spacy_tokens if token.has_vector])
        return embedded