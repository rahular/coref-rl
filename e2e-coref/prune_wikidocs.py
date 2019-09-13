import json
import statistics as s
import numpy as np

def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def get_stats(data):
    num_sents, max_len, all_lens = [], [], []
    for dp in data:
        num_sents.append(len(dp['sentences']))
        max_len.append(max([len(sent) for sent in dp['sentences']]))
        all_lens.extend([len(sent) for sent in dp['sentences']])
    return num_sents, max_len, all_lens

def prune(data, sent_th, word_th):
    for i, dp in enumerate(data):
        data[i]['sentences'] = [sent for sent in dp['sentences'] if len(sent) < word_th]
        data[i]['speakers'] = [sent for sent in dp['speakers'] if len(sent) < word_th]
        data[i]['sentences'] = dp['sentences'][:sent_th]
        data[i]['speakers'] = dp['speakers'][:sent_th]
    data = [dp for dp in data if dp['sentences']]
    with open('./temp/wiki_10k_docs_pruned.jsonlines', 'w', encoding='utf-8') as f:
        for dp in data:
            f.write(json.dumps(dp) + '\n')

if __name__ == '__main__':
    data = read('./temp/wiki_10k_docs.jsonlines')
    num_sents, max_len, all_lens = get_stats(data)
    print('mean, median, max of number of sentences: {}, {}, {}'.format(s.mean(num_sents), s.median(num_sents), max(num_sents)))
    print('mean, median, max of maximum lengths: {}, {}, {}'.format(s.mean(max_len), s.median(max_len), max(max_len)))

    num_sents = np.array(num_sents)
    all_lens = np.array(all_lens)
    sent_th = 50
    word_th = 50
    print('{}/{} documents have less that {} sentences'.format((num_sents <= 50).sum(), len(num_sents), sent_th))
    print('{}/{} sentences have less that {} words'.format((all_lens <= 50).sum(), len(all_lens), word_th))

    prune(data, sent_th, word_th)
