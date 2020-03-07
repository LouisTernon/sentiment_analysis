import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from collections import Counter
import re
import os 
from time import time
torch.manual_seed(1)


class SentencePreprocessor:

    def __init__(self):
        pass

    def process(self, s):
        s_ = self.replace_ponctuation(s)
        s_ = self.lower(s)
        return s_

    @staticmethod
    def replace_ponctuation(s):
        s_ = re.sub(r"[^\w\s]", " ", s)
        return s_

    @staticmethod
    def lower(s):
        s_ = s.lower()
        return s_



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
        words = [w.strip() for s in self.s for w in processor.process(s).split(" ")]
        self.words_count = Counter(words)
        self.vocab_size = len(self.words_count)

    def create_word_to_id(self, sentenceprocessor):
        word_to_id = {}
        for s in self.s:
            for word in sentenceprocessor.process(s).split(" "):
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
        padded = np.zeros((max_length,))
        padded[-len(s):] = np.array(s)[:max_length]
        return padded

    def preprocess_sentences(self, sentenceprocessor, max_length=20):
        assert self.s, "Load first"
        # Preprocess
        self.s_processed = [sentenceprocessor.process(s) for s in self.s]
        # Replace words by indexes
        self.s_processed = [self.seq_to_id_list(s) for s in self.s_processed]
        # Padd
        self.s_processed = np.array([self.padd_seq(s, max_length) for s in self.s_processed])
        
        return self.s_processed

    def label_to_oh(self) -> np.array:

        def oh(el):
            oh = np.array([0]*self.n_labels)
            oh[el] = 1
            return oh

        labels_oh = np.array(list(map(oh, self.l)))
        return labels_oh


class SentenceEncoder(nn.Module):
    
    def __init__(self, n_emb, n_hidden, n_layers, vocab_size, tagset_size, dropout=0, bias=True):
        super(SentenceEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, n_emb)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(n_emb, n_hidden, n_layers, bias=bias, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(n_hidden, tagset_size)
        self.activation = nn.Sigmoid()

        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_layers = n_layers


    def forward(self, input_, hidden):
        input_ = input_.long()
        embeds = self.word_embeddings(input_)
        out, hidden = self.lstm(embeds, hidden)
        out = out[:, -1, :] # Get final hidden state
        
        out = self.dropout(out)
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
    def train(self, trainfile, n_epochs=20, batch_size=32, clip=0, verbose=0):
        """Trains the classifier model on the training set stored in file trainfile"""
        dataset = Dataset()
        dataset.load_encode(trainfile)

        sentence_processor = SentencePreprocessor()
        dataset.compute_vocab_stat(sentence_processor) # TODO : compute stat after preprocessing to reduce redundancy
        dataset.create_word_to_id(sentence_processor)
        train_sentences = dataset.preprocess_sentences(sentence_processor)
        
        train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(dataset.l))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

        device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

        self.model = SentenceEncoder(150, 64, 1, dataset.vocab_size+1, 3)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=5e-2)

        start_time = time()
        self.model.train()
        for epoch in range(n_epochs):  
            e_loss = 0
            h = self.model.init_hidden(batch_size)

            for s, l in train_loader:

                self.model.zero_grad()
                s, l = s.to(device), l.to(device)
                h = tuple([e.data for e in h])
                
                output, h = self.model(s, h)

                loss = loss_function(output.squeeze(), l.long())
                loss.backward()
                if clip: nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                optimizer.step()

                e_loss += loss.item()

            if verbose: print("Epoch : {}/{} | Loss : {:.3f} | Eta : {:.2f}s".format(epoch, n_epochs, e_loss/dataset.len, (n_epochs-epoch+1)*(time()-start_time)/(epoch+1)))

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        dataset = Dataset()
        dataset.load_encode(datafile)

        sentence_processor = SentencePreprocessor()
        dataset.compute_vocab_stat(sentence_processor) # TODO : compute stat after preprocessing to reduce redundancy
        dataset.create_word_to_id(sentence_processor)
        test_sentences = dataset.preprocess_sentences(sentence_processor)
        
        test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(dataset.l))
        batch_size = len(test_data)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

        device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        
        h = self.model.init_hidden(batch_size)
        self.model.eval()
        for s, l in test_loader:
            h = tuple([e.data for e in h])
            s, l = s.to(device), l.to(device)
            pred_logits, _ = self.model(s, h)
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

