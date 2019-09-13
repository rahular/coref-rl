import json
import numpy as np

from tqdm import tqdm
from time import time
from itertools import islice


def make_pointers(path):
	with open(path, 'r', encoding='utf-8') as f:
		tokens = {}
		row = f.readline()		# ignore the first line
		counter = 0
		while True:
			pos = f.tell()
			row = f.readline()
			if not row:
				break
			
			counter += 1
			row = row.strip()
			token = row.split('\t')[0]
			if token.startswith('<http://www.wikidata.org/entity/Q'):
				key = token.split('/')[-1][:-1]
				tokens[key] = pos
			if counter % 10**6 == 0:
				print('Processed {} vectors...'.format(counter))
	return tokens


def save_pointers(tokens, path):
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(tokens, f)


def load_pointers(path):
	print('Loading QID to line mapping...')
	start_load = time()
	with open(path, 'r', encoding='utf-8') as f:
		pointers = json.load(f)
	print('Done. Took {0:.2f} seconds.'.format(time()-start_load))
	return pointers


def file_open(path):
	return open(path, 'r', encoding='utf-8')


def file_close(fp):
	fp.close()


def get_embedding(fp, n):
	start_load = time()
	fp.seek(n)
	emb = fp.readline()
	emb = emb.strip().split('\t')
	qid = emb[:1][0]
	emb = np.array(emb[1:], dtype=np.float)
	print('Accessed {} at byte {} in {:.2f} seconds.'.format(qid, n, time()-start_load))
	return emb


if __name__ == '__main__':
	embeddings_path = '../data/wikidata_embeddings_tranlation_v1.tsv'
	pointers_path = '../data/wikidata_qcodes_pointers_v1.json'

	tokens = make_pointers(embeddings_path)
	save_pointers(tokens, pointers_path)

	emb_file = file_open(embeddings_path)
	lines = load_pointers(pointers_path)
	qids = ['Q42', 			# Douglas Adams
			'Q691283',		# (was educated at) St John's College
			'Q159288',		# (died in) Santa Barbara
			'Q9865'			# (unrelated) Sint Anthonis
	]
	embs = []
	for qid in qids:
		emb = get_embedding(emb_file, lines[qid])
		embs.append(emb)
	
	print('Douglas Adams and St John\'s college (high)', np.dot(embs[0], embs[1]))
	print('Douglas Adams and Santa Barbara (high)', np.dot(embs[0], embs[2]))
	print('Douglas Adams and Sint Anthonis (low)', np.dot(embs[0], embs[3]))

	file_close(emb_file)