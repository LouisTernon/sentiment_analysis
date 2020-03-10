import torch.nn as nn
import torch

class LTSMSentenceEncoder(nn.Module):
    
    def __init__(self, n_emb, n_hidden, n_layers, vocab_size, tagset_size, category_size, pretrained_embedding, dropout=0, bias=True):
        super(LTSMSentenceEncoder, self).__init__()
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