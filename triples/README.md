## Pipeline for extracting triples from wikipedia/wikidata
This assumes that you have already installed [sling](https://github.com/google/sling) and [corenlp](https://stanfordnlp.github.io/CoreNLP/download.html).
Sling only works on Python 2 and so the files which import sling also work only on Python 2!

For the below pipeline to work, our Python 2 and Python 3 environments had the following dependencies:

Python 2 - [sling](https://github.com/google/sling). 

Python 3 - [stanford-corenlp](https://pypi.org/project/stanford-corenlp/), `spacy`, and `beautifulsoup4`.

To run start to finish, edit the `run.sh` script with the names of your Python2 and Python3 environments, and run. Otherwise, follow the step by step directions below.

#### From Wikipedia

* Step 0 - edit the `config.yaml` as appropriate
* Step 1 - **(py2)** run `wikidump.py` which uses `sling` (Python2) to access the wikipedia documents already parsed by sling into a `.txt` file
* Step 2 - run `cleanWikidump.py` which takes the previously generated `.txt` file and cleans it up; produces a clean `.txt` file
* Step 2.5 - start CoreNLP server (see `getTriples.py` for more details)
* Step 3 - run `getTriples.py` which takes this clean `.txt` and uses OpenIE (corenlp) to extract triples and saves it as a `.tsv` file

This `.tsv` file can be used to train the `RE-Text` reward model

* Step 4 - **(py2)** run `pruneCorenlpTriples.py` which removes triples whose entities are not in wikidata and produces a `.tsv` file

This `.tsv` file can be used to train the `RE-Joint` reward model

#### From Wikidata

Download pid2str.json and triples-1M.pkl from [here](https://drive.google.com/drive/folders/1KwS4mzGtJdsEFt931ZFdpU5IyCqT3P0-).

* `convertGold.py` uses `pid2str.json` to convert 1 million triples scraped from wikidata present in `triples-1M.pkl` into the format: (q1, "relation as string", q2) and stores it in a `.tsv`

This `.tsv` file can be used to train the `RE-KG` reward model
