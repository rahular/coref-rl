from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys

import h5py
import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder


def cache_dataset(data_path, elmo, out_file):
    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file.readlines()):
            example = json.loads(line)
            sentences = example["sentences"]
            text_len = np.array([len(s) for s in sentences])
            lm_emb = elmo.embed_sentences(sentences, batch_size=80)
            file_key = example["doc_key"].replace("/", ":")
            group = out_file.create_group(file_key)
            for i, (e, l) in enumerate(zip(lm_emb, text_len)):
                e = np.transpose(e, (1, 2, 0))  # to make it compatible with e2e-coref
                e = e[:l, :, :]
                group[str(i)] = e
            if doc_num % 10 == 0:
                print("Cached {} documents in {}".format(doc_num + 1, data_path))


if __name__ == "__main__":
    elmo = ElmoEmbedder(cuda_device=0 if torch.cuda.is_available() else -1)
    with h5py.File(sys.argv[1], "w") as out_file:
        for json_filename in sys.argv[2:]:
            cache_dataset(json_filename, elmo, out_file)
