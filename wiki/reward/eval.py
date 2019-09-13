import os
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import numpy as np
import pickle as pkl

from tqdm import tqdm
from sklearn.metrics import f1_score

IS_NEG_ZERO = True
BATCH_SIZE = 128
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
INCLUDE_LABELS = False

# This is redefined here for ease of use. Not Ideal.
class Model(nn.Module):
    def __init__(self, hidden_dim, ent_vocab_size, rel_vocab_size, embedding_dim):
        super().__init__()
        self.ent_embedding = nn.Embedding(ent_vocab_size, embedding_dim)
        self.rel_embedding = nn.Embedding(rel_vocab_size, embedding_dim)
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.mlp = nn.Linear(embedding_dim*3, 100)
        self.output = nn.Linear(100, 1)
        if IS_NEG_ZERO: self.act = nn.Sigmoid()
        else: self.act = nn.Tanh()

    def forward(self, subject, relation, object):
        s = self.ent_embedding(subject)
        o = self.ent_embedding(object)
        r = self.rel_embedding(relation)
        _, (h, c) = self.lstm(r)

        # print(s.size(), h[-1].size(), o.size())
        concat = torch.cat([torch.squeeze(s, 1), h[-1], torch.squeeze(o, 1)], dim=1)
        out = F.relu(self.mlp(concat))
        return self.act(self.output(out))


def get_iterator(ent_vocab, rel_vocab, data_path=None, triples=None):
    rel = data.Field(sequential=True, lower=True, batch_first=True)
    ent = data.Field(sequential=True, lower=False, batch_first=True)
    if INCLUDE_LABELS:
        label = data.Field(sequential=False, lower=False, use_vocab=False, batch_first=True)
        datafields = [('subject', ent), ('relation', rel),
                      ('object', ent), ('label', label)]
    else:
        datafields = [('subject', ent), ('relation', rel),
                      ('object', ent), ('label', None)]    
    if data_path:
        ds = data.TabularDataset(path=data_path, format='tsv', 
            skip_header=False, fields=datafields)
    elif triples:
        examples = []
        for triple in triples:
            examples.append(data.Example.fromlist(triple, datafields))
            ds = data.Dataset(examples, datafields)
    else:
        raise NotImplementedError()
    
    ent.vocab = ent_vocab
    rel.vocab = rel_vocab

    return data.Iterator(ds, batch_size=BATCH_SIZE, device=DEVICE, shuffle=False)
    

def load_checkpoint(saved_path):
    state_dict = torch.load(saved_path, map_location=DEVICE)
    ent_vocab_size = state_dict['ent_embedding.weight'].size()[0]
    rel_vocab_size = state_dict['rel_embedding.weight'].size()[0]
    model = Model(200, ent_vocab_size, rel_vocab_size, 200)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model
    

def load_vocabs(path):
    with open(path, 'rb') as f:
        vocabs = pkl.load(f)
    return vocabs['ent_vocab'], vocabs['rel_vocab']


def get_f1(y, preds):
    y, preds = y.cpu(), preds.cpu()
    if not IS_NEG_ZERO:
        y = (y == 1).float()
    return f1_score(y, preds) 


def get_arguments():
    parser = argparse.ArgumentParser(description='Parsing arguments like a boss!')
    parser.add_argument('model', type=str, help='model to load')
    parser.add_argument('--dataset', type=str, help='dataset to test on (defaults to triples)')
    return parser.parse_args()


def eval_file(dataset, model, ent_vocab, rel_vocab):
    valid_path = '../../data/{}_valid.tsv'.format(dataset)
    loader = get_iterator(ent_vocab, rel_vocab, data_path=valid_path)

    correct, f1 = 0.0, 0.0
    for batch in tqdm(loader):
        bs = batch.subject.size(0)
        preds = model(batch.subject, batch.relation, batch.object)
        preds = preds.squeeze()
        if INCLUDE_LABELS:
            y = batch.label.float()
        else:    
            y = torch.ones(bs)
        preds = (preds >= THRESHOLD).float()
        f1 += get_f1(y, preds)
        correct += ((preds == y).sum().item() / len(y))

    acc = correct / len(loader)
    f1 = f1 / len(loader)
    return acc, f1


def eval_triples(triples, model, ent_vocab, rel_vocab):
    if type(model) == list:
        reward = []
        for i in range(len(model)):
            reward.append(eval_triples(triples, model[i], ent_vocab[i], rel_vocab[i]))
        return reward
    else:
        # print(triples)
        if not triples:
            return 0.0
        loader = get_iterator(ent_vocab, rel_vocab, triples=triples)
        reward = 0.0
        for batch in loader:
            preds = model(batch.subject, batch.relation, batch.object)
            # print(preds)
            reward += preds.sum().item()
        return reward/len(triples)

if __name__ == '__main__':
    args = get_arguments()

    samples = [['Q82955', 'served as', 'Q11696'],               # Politician served as POTUS
              ['Q76', 'is', 'Q40348'],                          # Obama is Lawyer
              ['Q76', 'is', 'Q30'],                             # Obama is USA
              ['Q76', 'served as', 'Q13217683'],                # Obama served as US Senator
              ['Q76', 'previously served as', 'Q13217683']]     # Obama previously served as US Senator
    
    model_path = './models/{}'.format(args.model)
    model = load_checkpoint(os.path.join(model_path, 'best_model.pt'))

    ent_vocab, rel_vocab = load_vocabs(os.path.join(model_path, 'vocab.pkl'))

    if args.dataset:
        if re.match(r'gold_openie_[\d]+k_docs_balanced', args.dataset):
            INCLUDE_LABELS = True
        acc, f1 = eval_file(args.dataset, model, ent_vocab, rel_vocab)
        print('Validation accuracy: {:.4f}, F1: {:.4f}'.format(acc, f1))
    else:
        print(eval_triples(samples, model, ent_vocab, rel_vocab))
    
    