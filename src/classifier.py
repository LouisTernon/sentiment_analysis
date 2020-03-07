import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import spacy

from collections import Counter
import re
import os 
from time import time
torch.manual_seed(1)


class SentencePreprocessor:

    def __init__(self, spacy_nlp=None, lemmatize=False, min_letter=2):
        """
        """
        self.spacy_nlp = spacy_nlp
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
    def filter_min_letter(s, min_letter):
        filtered = " ".join(list(filter(lambda w: len(w)>=min_letter,  s.split(" "))))
        return filtered

    @staticmethod
    def replace_ponctuation(s):
        s_ = re.sub(r"[^\w\s]", " ", s)
        return s_

    @staticmethod
    def lower(s):
        s_ = s.lower()
        return s_

    def clean(self, s):
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

class Dataset:

    def __init__(self):
        self.len = None
        self.n_labels = None
    
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
        
        self.len = len(self.l)
        self.n_labels = len(set(self.l.tolist()))

    @staticmethod
    def label_encoder(labels):
        l = list(set(labels))
        encoded = list(map(lambda x: l.index(x), labels))
        return l, np.array(encoded)

    def load_encode(self, path):
        l, c, t, p, s = self.open_file(path)
        self.l_mapping, self.l = self.label_encoder(l)
        self.c_mapping, self.c = self.label_encoder(c)
        self.s = s
        self.p = p 
        self.t = t

        self.len = len(self.l)
        self.n_labels = len(set(self.l.tolist()))

    def compute_vocab_stat(self, processor):
        assert self.s, 'Load first'
        words = [w.strip() for s in self.s for w in processor.process(s, pretrained_embedding=False).split(" ")]
        self.words_count = Counter(words)
        self.vocab_size = len(self.words_count)

    def create_word_to_id(self, sentenceprocessor):
        word_to_id = {}
        for s in self.s:
            for word in sentenceprocessor.process(s, pretrained_embedding=False).split(" "):
                if word not in word_to_id:
                    word_to_id[word] = len(word_to_id)
        self.word_to_id = word_to_id
        return word_to_id

    def seq_to_id_list(self, s):
        assert self.word_to_id, "run dataset.create_word_to_id first"
        idxs = [self.word_to_id.get(w, len(self.word_to_id)+1) for w in s.split(" ")]
        return torch.tensor(idxs, dtype=torch.int64)

    @staticmethod
    def padd_seq(s, max_length): 
        if isinstance(s, np.ndarray) and s.ndim == 2: # embedding
            padded = np.zeros((max_length, s.shape[1]))
            padded[-len(s):, :] = s[:max_length, :]
        else: #label encoder
            padded = np.zeros((max_length,))
            padded[-len(s):] = np.array(s)[:max_length]
        return padded

    def preprocess_sentences(self, sentenceprocessor, pretrained_embedding, max_length=20):
        assert self.s, "Load first"
        # Preprocess
        self.s_processed = [sentenceprocessor.process(s, pretrained_embedding) for s in self.s]
        # Replace words by indexes
        if not pretrained_embedding: self.s_processed = [self.seq_to_id_list(s) for s in self.s_processed]
        # Padd
        self.s_processed = np.array([self.padd_seq(s, max_length) for s in self.s_processed])
        
        return self.s_processed

    @staticmethod
    def label_to_oh(labels) -> np.array:
        n_labels = len(set(labels))
        def oh(el):
            oh = np.array([0]*n_labels)
            oh[el] = 1
            return oh

        labels_oh = np.array(list(map(oh, labels)))
        return labels_oh


class SentenceEncoder(nn.Module):
    
    def __init__(self, n_emb, n_hidden, n_layers, vocab_size, tagset_size, category_size, pretrained_embedding, dropout=0, bias=True):
        super(SentenceEncoder, self).__init__()
        if not pretrained_embedding: self.word_embeddings = nn.Embedding(vocab_size, n_emb)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(n_emb, n_hidden, n_layers, bias=bias, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(n_hidden + category_size, tagset_size)
        self.activation = nn.Sigmoid()

        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.pretrained_embedding = pretrained_embedding

    def forward(self, input_, categories, hidden):
        if not self.pretrained_embedding: 
            input_ = input_.long()
            embeds = self.word_embeddings(input_)
        else :
            embeds = input_
        out, hidden = self.lstm(embeds, hidden)
        out = out[:, -1, :] # Get final hidden state
        out = self.dropout(out)
        if categories is not None: out = torch.cat((out, categories), 1)
        out = self.dense(out)
        out = self.activation(out)
        return out, hidden 

    def init_hidden(self, batch_size):
        device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden
         
        
class Classifier:
    """The Classifier"""

    #############################################
    def train(self, trainfile,  n_epochs=20, batch_size=32, encoder_args=None, pretrained_embedding=False, lemmatize=False, clip=0, verbose=0):
        """Trains the classifier model on the training set stored in file trainfile"""

        if verbose: print("   Loading data..")
        dataset = Dataset()
        dataset.load_encode(trainfile)

        if verbose: print("   Preprocessing..")
        spacy_nlp = spacy.load("en_core_web_md") if (lemmatize or pretrained_embedding) else None
        sentence_processor = SentencePreprocessor(spacy_nlp=spacy_nlp, lemmatize=lemmatize)
        dataset.compute_vocab_stat(sentence_processor) # TODO : compute stat after preprocessing to reduce redundancy
        dataset.create_word_to_id(sentence_processor)
        train_sentences = dataset.preprocess_sentences(sentence_processor, pretrained_embedding=pretrained_embedding)
        
        category_oh = dataset.label_to_oh(dataset.c)
        train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(category_oh), torch.from_numpy(dataset.l))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

        if verbose: print("   Training..")
        device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        encoder_args = {"n_hidden":128, "n_layers":1, "tagset_size":3}
        encoder_args["n_emb"] = 150 if not pretrained_embedding else 300
        self.model = SentenceEncoder(vocab_size=dataset.vocab_size+1, pretrained_embedding=pretrained_embedding, category_size=len(dataset.c_mapping), **encoder_args)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=5e-3)

        start_time = time()
        self.model.train()
        self.model.float()
        for epoch in range(n_epochs):  
            e_loss = 0
            h = self.model.init_hidden(batch_size)

            for s, c, l in train_loader:

                self.model.zero_grad()
                s, c, l = s.to(device), c.to(device), l.to(device)
                h = tuple([e.data for e in h])
                
                output, h = self.model(s.float(), c.float(), h)


                loss = loss_function(output.squeeze(), l.long())
                loss.backward()
                if clip: nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                optimizer.step()

                e_loss += loss.item()

            if verbose: print("         Epoch : {}/{} | Loss : {:.3f} | Eta : {:.2f}s".format(epoch, n_epochs, e_loss/dataset.len, (n_epochs-epoch+1)*(time()-start_time)/(epoch+1)))

    def predict(self, datafile, pretrained_embedding=False, lemmatize=False):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        dataset = Dataset()
        dataset.load_encode(datafile)

        spacy_nlp = spacy.load("en_core_web_md") if (lemmatize or pretrained_embedding) else None
        sentence_processor = SentencePreprocessor(spacy_nlp=spacy_nlp, lemmatize=lemmatize)
        dataset.compute_vocab_stat(sentence_processor) # TODO : compute stat after preprocessing to reduce redundancy
        dataset.create_word_to_id(sentence_processor)
        test_sentences = dataset.preprocess_sentences(sentence_processor, pretrained_embedding)
        
        category_oh = dataset.label_to_oh(dataset.c)
        test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(category_oh), torch.from_numpy(dataset.l))
        batch_size = len(test_data)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

        device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        
        h = self.model.init_hidden(batch_size)
        self.model.eval()
        for s, c, l in test_loader:
            h = tuple([e.data for e in h])
            s, c, l = s.to(device), c.to(device), l.to(device)
            pred_logits, _ = self.model(s.float(), c.float(), h)
            pred_labels = torch.argmax(pred_logits, dim=1)
            
        pred_labels = pred_labels.numpy()
        pred_labels = [dataset.l_mapping[p] for p in pred_labels] # Turn 0,1,2 to 'positive', ..
        return pred_labels


if __name__ == "__main__":

    import os
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "traindata.csv")
  
    # dataset = Dataset()
    # dataset.load_encode(data_path)

    # sentence_processor = SentencePreprocessor()
    # dataset.compute_vocab_stat(sentence_processor)
    # dataset.create_word_to_id(sentence_processor)
    # train_sentences = dataset.preprocess_sentences(sentence_processor)

    classifier = Classifier()
    classifier.train(data_path, n_epochs=200, verbose=1)

