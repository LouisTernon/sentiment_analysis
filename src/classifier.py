import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(1)

import numpy as np

import os 
from time import time

from dataset import Dataset
from preprocessor import SentencePreprocessor
from model import LTSMSentenceEncoder
         
        
class Classifier:
    """The Classifier"""

    #############################################
    def train(self, trainfile,  n_epochs=20, batch_size=32, encoder_args=None, pretrained_embedding=False, lemmatize=False, clip=0, verbose=0):
        """Trains the classifier model on the training set stored in file trainfile"""

        if verbose: print("   Loading data..")
        dataset = Dataset()
        dataset.load_encode(trainfile)

        if verbose: print("   Preprocessing..")
        spacy_model = "en_core_web_md" if (lemmatize or pretrained_embedding) else None
        sentence_processor = SentencePreprocessor(spacy_model=spacy_model, lemmatize=lemmatize)
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
        self.model = LTSMSentenceEncoder(vocab_size=dataset.vocab_size+1, pretrained_embedding=pretrained_embedding, category_size=len(dataset.c_mapping), **encoder_args)
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

        spacy_model = "en_core_web_md" if (lemmatize or pretrained_embedding) else None
        sentence_processor = SentencePreprocessor(spacy_model=spacy_model, lemmatize=lemmatize)
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

    def return_prediction_errors(self, datafile, pretrained_embedding=False, lemmatize=False):
        """
        Util function to analyse prediction errors. Returns list of tuples (sentence, gt label, pred label, class)
        """
        dataset = Dataset()
        dataset.load_encode(datafile)

        spacy_model = "en_core_web_md" if (lemmatize or pretrained_embedding) else None
        sentence_processor = SentencePreprocessor(spacy_model=spacy_model, lemmatize=lemmatize)
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

        gt_labels = [dataset.l_mapping[gt] for gt in dataset.l]

        err_ind, *_ = np.where(np.array(pred_labels) != np.array(gt_labels))

        errors = [(dataset.s[i], gt_labels[i], pred_labels[i], dataset.c_mapping[dataset.c[i]]) for i in err_ind]
        return errors

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

