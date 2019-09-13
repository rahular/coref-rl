import os
import json
import torch
import argparse

import pickle as pkl
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
import torch.nn.functional as F

from tqdm import tqdm
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score

# for reproducibility
torch.manual_seed(42)

# tunable parameters; may become flags later
BATCH_SIZE = 128
EPOCHS = 25
THRESHOLD = 0.5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_PRETRAINED = True
FREEZE_EMB = False

# loss function; defined here so that it may become a flag later
LOSS = nn.L1Loss()

# for some losses, the negatives must be -1 instead of 0
IS_NEG_ZERO = True


def get_pbg(qids, emb_path, emb_file):
    if os.path.isfile(os.path.join(emb_path, emb_file + '.pt')):
        return Vectors(name=emb_file, cache=emb_path)
    
    # create temporary file with only the required embeddings and return
    with open(os.path.join(emb_path, emb_file), 'w', encoding='utf-8') as new_file:
        pbg_file = open(os.path.join(
            emb_path, 'wikidata_embeddings_tranlation_v1.tsv'), 'r', encoding='utf-8')
        index_file = open(os.path.join(
            emb_path, 'wikidata_qcodes_pointers_v1.json'), 'r', encoding='utf-8')
        index = json.load(index_file)
        found = 0
        for qid in qids:
            pos = index.get(qid, None)
            if pos:
                found += 1
                pbg_file.seek(pos)
                line = ' '.join(pbg_file.readline().strip().split('\t'))
                new_file.write(line + '\n')
        index_file.close()
        pbg_file.close()
    print('Created smaller PBG embedding file with {}/{} QIDs...'.format(found, len(qids)))
    return Vectors(name=emb_file, cache=emb_path)


def get_iterator(data_path, emb_path, train_file, valid_file, emb_file):
    rel = data.Field(sequential=True, lower=True, batch_first=True)
    ent = data.Field(sequential=True, lower=False, batch_first=True)
    datafields = [('subject', ent), ('relation', rel),
                  ('object', ent), ('confidence', None)]
    train_ds, valid_ds = data.TabularDataset.splits(
        path=data_path, train=train_file, validation=valid_file,
        format='tsv', skip_header=False, fields=datafields)

    rel.build_vocab(train_ds, valid_ds)
    ent.build_vocab(train_ds, valid_ds)
    if USE_PRETRAINED:
        glove = Vectors(name='glove.6B.200d.txt', cache=emb_path)
        rel.vocab.set_vectors(glove.stoi, glove.vectors, glove.dim)
        pbg = get_pbg(ent.vocab.freqs.keys(), emb_path, emb_file)
        ent.vocab.set_vectors(pbg.stoi, pbg.vectors, pbg.dim)

    train_iter, valid_iter = data.BucketIterator.splits(
        (train_ds, valid_ds), batch_sizes=(BATCH_SIZE, BATCH_SIZE),
        sort_key=lambda x: len(x.relation),
        sort_within_batch=True, device=DEVICE)
    if USE_PRETRAINED:
        print('Entity vocab size: {} and Relation string vocab size: {}'.format(
            ent.vocab.vectors.size(), rel.vocab.vectors.size()))
    else:
        print('Entity vocab size: {} and Relation string vocab size: {}'.format(
            len(ent.vocab), len(rel.vocab)))
    return ent, rel, train_iter, valid_iter, len(ent.vocab), len(rel.vocab)


class Model(nn.Module):
    def __init__(self, hidden_dim, ent_vocab_size, ent_emb, rel_vocab_size, rel_emb, embedding_dim=100):
        super().__init__()
        self.ent_embedding = nn.Embedding(ent_vocab_size, embedding_dim)
        self.rel_embedding = nn.Embedding(rel_vocab_size, embedding_dim)
        
        if USE_PRETRAINED:
            self.ent_embedding.from_pretrained(ent_emb)
            self.rel_embedding.from_pretrained(rel_emb)
        if FREEZE_EMB:
            self.ent_embedding.weight.requires_grad = False
            
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
        concat = torch.cat([s.squeeze(), h[-1], o.squeeze()], dim=1)
        out = F.relu(self.mlp(concat))
        return self.act(self.output(out))


def create_negative_batch(batch, bs):
    subject, relation, object = batch.subject, batch.relation, batch.object
    neg_subject, neg_relation, neg_object = batch.subject, batch.relation, batch.object[torch.randperm(bs)]

    subject = torch.cat([subject, neg_subject], dim=0)
    relation = torch.cat([relation, neg_relation], dim=0)
    object = torch.cat([object, neg_object], dim=0)
    if IS_NEG_ZERO:
        y = torch.cat([torch.ones(bs, 1), torch.zeros(bs, 1)], dim=0)
    else:
        y = torch.cat([torch.ones(bs, 1), -torch.ones(bs, 1)], dim=0)

    return subject.to(DEVICE), relation.to(DEVICE), object.to(DEVICE), y.to(DEVICE)


def get_arguments():
    parser = argparse.ArgumentParser(description='Parsing arguments like a boss!')
    parser.add_argument('dataset', type=str, help='dataset to train on')
    return parser.parse_args()

def get_f1(y, preds):
    y, preds = y.cpu(), preds.cpu()
    if not IS_NEG_ZERO:
        y = (y == 1).float()
    return f1_score(y, preds)


def save_vocab(path, ent_vocab, rel_vocab):
    with open(path, 'wb') as f:
        pkl.dump({
            'ent_vocab': ent_vocab,
            'rel_vocab': rel_vocab
        }, f, pkl.HIGHEST_PROTOCOL)
    print('Saved vocabularies for future use...')

if __name__ == '__main__':
    args = get_arguments()

    data_path = '../data/'
    emb_path = '../embeddings/'
    model_path = './models/{}'.format(args.dataset)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_file = '{}_train.tsv'.format(args.dataset)
    valid_file = '{}_valid.tsv'.format(args.dataset)
    emb_file = '{}_pbg.txt'.format(args.dataset)
    vocab_file = os.path.join(model_path, 'vocab.pkl'.format(args.dataset))

    ent_field, rel_field, train_loader, valid_loader, ent_vocab_size, rel_vocab_size = get_iterator(
        data_path, emb_path, train_file, valid_file, emb_file)
    save_vocab(vocab_file, ent_field.vocab, rel_field.vocab)

    model = Model(200, ent_vocab_size, ent_field.vocab.vectors,
                  rel_vocab_size, rel_field.vocab.vectors, 200)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_f1, best_epoch, best_acc = -1.0, -1.0, -1.0

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            bs = batch.subject.size(0)

            subject, relation, object, y = create_negative_batch(batch, bs)
            preds = model(subject, relation, object)
            loss = LOSS(preds, y)

            loss.backward()
            optimizer.step()
            running_loss += (loss.item() / bs)
        epoch_loss = running_loss / len(train_loader)

        valid_loss, correct, f1 = 0.0, 0.0, 0.0
        model.eval()
        for batch in valid_loader:
            bs = batch.subject.size(0)
            y = torch.ones(bs, 1).to(DEVICE)
            preds = model(batch.subject, batch.relation, batch.object)

            loss = LOSS(preds, y)
            valid_loss += (loss.item() / bs)

            preds = (preds >= THRESHOLD).float()
            f1 += get_f1(y, preds)
            correct += ((preds == y).sum().item() / len(y))

        valid_loss = valid_loss / len(valid_loader)
        acc = correct / len(valid_loader)
        f1 = f1 / len(valid_loader)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, Avg. F1: {:.4f}'.format(
            epoch, epoch_loss, valid_loss, acc, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pt'))
    
    print('\nBest epoch was {} with {:.4f} F1 and {:.4f}% accuracy.'.format(best_epoch, best_f1, best_acc * 100))
