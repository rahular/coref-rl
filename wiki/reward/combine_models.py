import os
import torch
import pickle as pkl

from collections import Counter
from torchtext.vocab import Vocab
from eval import Model, load_checkpoint, load_vocabs
from model import get_pbg, save_vocab

BASE_PATH = './models/unified'

def unify_ents(e1, e2):
    e = set(e1).union(set(e2))
    e.remove('<unk>')
    e.remove('<pad>')
    return e

def create_vocabs(ents, rel_vocab):
    ent_vocab = Vocab(Counter(ents))
    vectors = get_pbg(ents, '../../embeddings', 'unified_embs.txt')
    ent_vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
    save_vocab(os.path.join(BASE_PATH, 'vocab.pkl'), ent_vocab, rel_vocab)

def combine_models(model1, model2, beta=0.5):
    params1 = dict(model1)
    params2 = dict(model2)

    for name in params2.keys():
        if name in params1:
            if name == 'ent_embedding.weight' or name == 'rel_embedding.weight':
                continue
            print('Combining layer {}... {} {}'.format(name, params1[name].data.size(), params2[name].data.size()))
            params1[name].data.copy_((1-beta)*params2[name].data + beta*params1[name].data)
    
    return params1

if __name__ == '__main__':
    model_names = ['wiki_gold_1M', 'gold_openie_50k_docs_balanced', 'openie_50k_docs']
    model_paths = ['./models/{}'.format(name) for name in model_names]
    
    wikidata, combined, wikipedia = [load_checkpoint(os.path.join(path, 'best_model.pt')) for path in model_paths]
    (ent_wikidata, rel_wikidata), (_, _), (ent_wikipedia, rel_wikipedia) = [load_vocabs(os.path.join(path, 'vocab.pkl')) for path in model_paths]
    
    state_dict = combine_models(wikidata.state_dict(), combined.state_dict(), beta=0.8)
    state_dict = combine_models(state_dict, wikipedia.state_dict(), beta=0.5)
    state_dict['ent_embedding.weight'] = wikipedia.state_dict()['ent_embedding.weight']
    state_dict['rel_embedding.weight'] = wikipedia.state_dict()['rel_embedding.weight']

    umodel = Model(200, len(ent_wikipedia), len(rel_wikipedia), 200)
    umodel.load_state_dict(state_dict)
    torch.save(umodel.state_dict(), os.path.join(BASE_PATH, 'best_model.pt'))
