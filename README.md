## Rewarding Coreference Resolvers for Being Consistent with World Knowledge

To appear in EMNLP 2019.

### Datasets

For convinience, create a symlink: `cd e2e-coref && ln -s ../wiki ./wiki`

For pre-training the coreference resolution system, OntoNotes 5.0 is required. [[Download](https://catalog.ldc.upenn.edu/LDC2013T19)] [[Create splits](https://github.com/rahular/coref-rl/blob/master/e2e-coref/setup_training.sh)]

Data for training the reward models and fine-tuning the coreference resolver (place in `<PROJECT_HOME>/data`):

- 2M triples for RE-Text [[Train](https://drive.google.com/open?id=1OkmeevtBBke2iNCBEwtbY52LhAb5eY3S)] [[Dev](https://drive.google.com/open?id=17-0fyHHiwiVE8m_4Rrqhbig7Vj8IEuxh)]
- 12M triples for RE-KG [[Train](https://drive.google.com/open?id=1fyIAecXKhfo6yy5LylJw4I7OLrRuA9-f)] [[Dev](https://drive.google.com/open?id=17-0fyHHiwiVE8m_4Rrqhbig7Vj8IEuxh)]
- 60k triples for RE-Joint [[Train](https://drive.google.com/open?id=1UKLKJN_6WuTBqMTOG5VQIOe0Ef5EGQpF)] [[Dev](https://drive.google.com/open?id=17-0fyHHiwiVE8m_4Rrqhbig7Vj8IEuxh)]
- 10k wikipedia summaries for fine-tuning [[Download](https://drive.google.com/open?id=1twtOxrCGRUnEHzk8VD8obeZS7N3Ms8mK)]

*Note*: If you want to make these files from scratch, follow the instructions in the `triples` folder.

### Pre-trained models

- Best performing reward model (RE-Distill) [[Download](https://drive.google.com/open?id=1ewyia0ai28j9rOixJNyXUPdYfS4Z46v1)]
- Best performing coreference resolver (Coref-Distill) [[Download](https://drive.google.com/open?id=1KkNHOqUfSNwgD0bITI-5HCzq04nSvYIR)]

### Evaluation

Unzip `Coref-Distill` into `e2e-coref/logs` folder and run `GPU=x python evaluate.py final`

### Training

#### Reward models
- Download pytorch big-graph embeddings (~40G, place in `<PROJECT_HOME>/embeddings`) [[Download](https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_vectors.npy.gz)]
- Run `wiki/embs.py` to create an index of the embeddings (you need to do this only once)
- Run reward module training with `cd wiki/reward && python train.py <dataset-name>`

#### Coreference resolver
##### Pre-training
- Follow `e2e-coref/README.md` to setup environment, create ELMO embeddings, etc.
- Run coreference pre-training with `cd e2e-coref && GPU=x python train.py <experiment>` 

##### Fine-tuning
- Start the sling server with `python wiki/reward/sling_server.py`
- Change `SLING_IP` in `wiki/reward/reward.py` to the IP of the sling server
- Run coreference fine-tuning with `cd e2e-coref && GPU=x python finetune.py <experiment>` (see `e2e-coref/experiments.conf` for the different configurations)

### Misc
- `wiki/reward/combine_models.py` can be used to distill the various reward models
- `e2e-coref/save_weights.py` can be used to save the weights of the fine-tuned coreference models so that they can be combined by setting the `distill` flag in the configuration file

### Citation
```
@article{DBLP:journals/corr/abs-1909-02392,
  author    = {Rahul Aralikatte and
               Heather Lent and
               Ana Valeria Gonz{\'{a}}lez{-}Gardu{\~{n}}o and
               Daniel Hershcovich and
               Chen Qiu and
               Anders Sandholm and
               Michael Ringaard and
               Anders S{\o}gaard},
  title     = {Rewarding Coreference Resolvers for Being Consistent with World Knowledge},
  journal   = {CoRR},
  volume    = {abs/1909.02392},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.02392},
  archivePrefix = {arXiv},
  eprint    = {1909.02392},
  timestamp = {Mon, 16 Sep 2019 17:27:14 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1909-02392},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
