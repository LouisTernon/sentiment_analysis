import torch
import torch.nn as nn
from collections import Counter
import re
import torch.optim as optim
torch.manual_seed(1)


class SentencePreprocessor:

    def __init__(self):
        pass

    def process(self, s):
        s_ = self.replace_ponctuation(s)
        return s_

    @staticmethod
    def replace_ponctuation(s):
        s_ = re.sub(r"[^\w\s]", " ", s)
        return s_


class Dataset:

    def __init__(self):
        self.len = None
    
    def open_file(self, path):
        with open(path, 'r') as f:
            labels, class_, target, position, sentences = [], [], [], [], []
            lines = [line for line in f if line.strip()]
            for line in lines:
                splitted = line.strip().split("\t")
                labels.append(splitted[0])
                class_.append(splitted[1])
                target.append(splitted[2])
                position.append(splitted[3])
                sentences.append(splitted[4:][0])
        
        return labels, class_, target, position, sentences



    def load(self, path):
        """
        l : labels ("positive", "neutral", "negative")
        c : class ("---#---")
        t : target word
        p : position
        s : sentence

        """
        self.l, self.c, self.t, self.p, self.s = self.open_file(path)


    @staticmethod
    def label_encoder(labels):
        l = list(set(labels))
        encoded = list(map(lambda x: l.index(x), labels))
        return l, encoded

    def load_encode(self, path):
        l, c, t, p, s = self.open_file(path)
        self.gt_mapping, self.l = self.label_encoder(l)
        self.c_mapping, self.c = self.label_encoder(c)
        self.s = s
        self.p = p 
        self.t = t


    def compute_vocab_stat(self, processor):
        assert self.s, 'Load first'
        words = [w.strip() for s in self.s for w in processor.process(s).split(" ")]
        self.words_count = Counter(words)
        self.vocab_size = len(self.words_count)

    def create_word_to_id(self, sentenceprocessor):
        word_to_ix = {}
        for s in self.s:
            for word in sentenceprocessor.process(s).split(" "):
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        self.word_to_ix = word_to_ix
        return word_to_ix



class SentenceEncoder(nn.Module):
    
    def __init__(self, n_emb, n_hidden, vocab_size, tagset_size):
        super(SentenceEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, n_emb)
        self.lstm = nn.LSTM(n_emb, n_hidden)
        self.dense = nn.Linear(n_hidden, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.dense(lstm_out.view(len(sentence), -1))
        tag_scores = nn.LogSoftmax()(tag_space, dim=1)
        return tag_scores

    @staticmethod
    def seq_to_id_list(s, word_to_id):
        idxs = [word_to_id[w] for w in s.split(" ")]
        return idxs
         
        
class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        dataset = Dataset()
        dataset.load_encode(trainfile)

        sentence_processor = SentencePreprocessor()
        dataset.compute_vocab_stat(sentence_processor)
        dataset.create_word_to_id(sentence_processor)

        model = SentenceEncoder(32, 32, dataset.vocab_size, 3)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(20):  
            for s, l in zip(dataset.s, dataset.l):
               
                model.zero_grad()

                sentence_in = model.seq_to_id_list(s, dataset.word_to_ix)

                tag_scores = model(sentence_in)

                loss = loss_function(tag_scores, l)
                loss.backward()
                optimizer.step()


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """




if __name__ == "__main__":


    import os
    data_path = os.path.join("..", "data", "devdata.csv")
  
    classifier = Classifier()
    classifier.train(data_path)

