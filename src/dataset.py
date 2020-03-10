import numpy as np 

from collections import Counter

import torch

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
        """
        Use label encoder to encode class & label
        """
        l, c, t, p, s = self.open_file(path)
        self.l_mapping, self.l = self.label_encoder(l)
        self.c_mapping, self.c = self.label_encoder(c)
        self.s = s
        self.p = p 
        self.t = t

        self.len = len(self.l)
        self.n_labels = len(set(self.l.tolist()))

    def compute_vocab_stat(self, sentenceprocessor):
        """
        Compute vocav size and word counts
        """
        assert self.s, 'Load first'
        words = [w.strip() for s in self.s for w in sentenceprocessor.process(s, pretrained_embedding=False).split(" ")]
        self.words_count = Counter(words)
        self.vocab_size = len(self.words_count)

    def create_word_to_id(self, sentenceprocessor):
        """
        Create word to index mapping. Used to encode words when pretrained embedding are not used
        """
        word_to_id = {}
        for s in self.s:
            for word in sentenceprocessor.process(s, pretrained_embedding=False).split(" "):
                if word not in word_to_id:
                    word_to_id[word] = len(word_to_id)
        self.word_to_id = word_to_id
        return word_to_id

    def seq_to_id_list(self, s: str) -> torch.tensor:
        """
        Encode a sentence using word to index mapping
        """
        assert self.word_to_id, "run dataset.create_word_to_id first"
        idxs = [self.word_to_id.get(w, len(self.word_to_id)+1) for w in s.split(" ")]
        return torch.tensor(idxs, dtype=torch.int64)

    @staticmethod
    def padd_seq(s, max_length): 
        """
        Padd a sequence to the desired length (pre-padding applied)
        """
        if isinstance(s, np.ndarray) and s.ndim == 2: # embedding
            padded = np.zeros((max_length, s.shape[1]))
            padded[-len(s):, :] = s[:max_length, :]
        else: #label encoder
            padded = np.zeros((max_length,))
            padded[-len(s):] = np.array(s)[:max_length]
        return padded

    def preprocess_sentences(self, sentenceprocessor, pretrained_embedding, max_length=30):
        """
        Dataset main preprocessing function. Apply lemmatization/cleaning, transform using word_to_id/embedding
        and padding
        """
        assert self.s, "Load first"
        # Preprocess
        self.s_processed = [sentenceprocessor.process(s, pretrained_embedding) for s in self.s]
        # Replace words by indexes
        if not pretrained_embedding: self.s_processed = [self.seq_to_id_list(s) for s in self.s_processed]
        # Padd
        self.s_processed = np.array([self.padd_seq(s, max_length) for s in self.s_processed])
        
        return self.s_processed

    @staticmethod
    def label_to_oh(labels:list) -> np.array:
        """
        Util function to transform feature to one hot.
        """
        n_labels = len(set(labels))
        def oh(el):
            oh = np.array([0]*n_labels)
            oh[el] = 1
            return oh

        labels_oh = np.array(list(map(oh, labels)))
        return labels_oh
