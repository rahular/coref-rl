import os
import io
import sys
import json
import signal
import corenlp
import requests

from . import eval as E

SLING_IP = '10.66.202.12'
timeout = 10
inp = [
    'Barack Obama is an American attorney and politician who served as the 44th president of the United States from 2009 to 2017.',
    'A member of the Democratic Party, Barack Obama was the first African American to be elected to the presidency.',
    'Barack Obama previously served as a United States senator from Illinois from 2005 to 2008.'
]

class AintNobodyGotTimeFoDat(Exception):
    pass

class supress_output():
    def __enter__(self):
        text_trap = io.StringIO()
        sys.stdout = text_trap
    def __exit__(self, type, value, traceback):
        sys.stdout = sys.__stdout__

def timeout_handler(signum, frame):
    raise AintNobodyGotTimeFoDat

def get_openie_triples(client, document):
    output = []
    signal.alarm(timeout)
    try:
        ann = client.annotate(document)
        for sent in ann.sentence:
            if len(sent.openieTriple) > 0:
                triples = sent.openieTriple
                for t in triples:
                    output.append([t.subject, t.relation, t.object])
    except Exception as e:
        # ain't nobody got time fo annotating big docs
        print(e)
    else:
        signal.alarm(0)
    return output

def get_linked_triples(triples):
    url = 'http://{}:5000/get_kb_triples'.format(SLING_IP)
    payload = {
        'triples': triples
    }
    headers = {
        'Content-Type': "application/json"
    }
    response = requests.request('POST', url, data=json.dumps(payload), headers=headers)
    return json.loads(response.text)[0]['output']

def get_reward(inputs, reward_model, ent_vocab, rel_vocab, corenlp_client):
    triples = []
    for doc in inputs:
        with supress_output():
            triples.extend(get_openie_triples(corenlp_client, doc.lower()))
    # print(len(inputs), len(triples))
    return E.eval_triples(get_linked_triples(triples), reward_model, ent_vocab, rel_vocab)

def corenlp_start():
    os.environ['CORENLP_HOME'] = './corenlp'
    corenlp_client = corenlp.CoreNLPClient(endpoint="http://localhost:9000", memory='16G', 
        annotators="tokenize ssplit pos lemma depparse natlog openie".split())
    corenlp_client.annotate('This is a dummy input sentence to initialize all CoreNLP components.')
    print('(Re)initialized coreNLP server...')
    return corenlp_client

def corenlp_stop(corenlp_client):
    corenlp_client.stop()

def corenlp_restart(corenlp_client):
    # corenlp slows down after a while, probably due to bad garbage collection
    # so restart it periodically
    corenlp_stop(corenlp_client)
    return corenlp_start()

def reward_model_start(reward_model_name):
    reward_model_path = './wiki/reward/models/{}'.format(reward_model_name)
    reward_model = E.load_checkpoint(os.path.join(reward_model_path, 'best_model.pt'))
    ent_vocab, rel_vocab = E.load_vocabs(os.path.join(reward_model_path, 'vocab.pkl'))
    print('Initialized reward model...')
    return reward_model, ent_vocab, rel_vocab

def init(reward_model_name):
    # reward model
    if reward_model_name == 'all':
        reward_model, ent_vocab, rel_vocab = [], [], []
        for name in ['wiki_gold_1M', 'openie_50k_docs', 'gold_openie_50k_docs_balanced']:
            rm, ev, rv = reward_model_start(name)
            reward_model.append(rm)
            ent_vocab.append(ev)
            rel_vocab.append(rv)
    else:
        reward_model, ent_vocab, rel_vocab = reward_model_start(reward_model_name)

    # coreNLP
    corenlp_client = corenlp_start()

    signal.signal(signal.SIGALRM, timeout_handler)

    return (reward_model, ent_vocab, rel_vocab, corenlp_client)

if __name__ == '__main__':
    reward_model_name = 'openie_50k_docs'
    reward_model, ent_vocab, rel_vocab, corenlp_client = init(reward_model_name)
    print(get_reward(inp, reward_model, ent_vocab, rel_vocab, corenlp_client))
    corenlp_stop(corenlp_client)