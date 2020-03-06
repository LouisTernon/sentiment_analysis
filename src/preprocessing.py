import spacy
import numpy as np
import re


def load_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        return np.array([line.strip().split("\t") for line in f if line.strip()])

class Preprocessing(object):
    def __init__(self, filename, spacy_nlp):

        data = load_data(filename)

        self.labels = parse_labels(data[:, 0])
        self.category = data[:, 1]
        self.term = data[:, 2]
        self.pos = data[:, 3]
        self.sentences = clean(data[:, 4], spacy_nlp)
        self.embeddings = embeddings(data[:, 4], spacy_nlp)


def embeddings(sentences, spacy_nlp):
    """
    construct the embeddings of a list of sentences
    :param sentences: list of sentences (as strings)
    :param spacy_nlp: spacy model, must be "xx_core_web_md" or "xx_core_web_lg"
    :return: a list of np array of size (len(sentence)*300) embeddings for each sentence
    """
    embed = []
    for raw_string in sentences:
        string = re.sub('[^a-zA-Z ]+', " ", raw_string.lower())
        spacy_tokens = spacy_nlp(string)
        embed.append(np.array([token.vector for token in spacy_tokens if token.has_vector]))
    return embed


def clean(sentences, spacy_nlp):
    """
    filter, lower and lemmatize words
    :param sentences: list of sentences (as strings)
    :param spacy_nlp: spacy model
    :return: a list of lists of words
    """
    res = []
    for raw_string in sentences:
        string = re.sub('[^a-zA-Z ]+', " ", raw_string.lower())
        spacy_tokens = spacy_nlp(string)
        res.append([token.lemma_ for token in spacy_tokens if token.lemma_ != '-PRON-'])
    return res


def parse_labels(labels_as_string):
    labels = []
    for lab_str in labels_as_string:
        if lab_str == "positive":
            labels.append(1)
        elif lab_str == "negative":
            labels.append(-1)
        else:
            labels.append(0)
    return np.array(labels)
