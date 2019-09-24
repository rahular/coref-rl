#!/usr/bin/env bash
py2_env="py27" #pythons with the name of your python 2 Anaconda environment, where sling is installed
py3_env="py36" #edit this with the name of your python 3 Anaconda environment, with corenlp and beautifulsoup4

source activate $py2_env
echo "* run: wikidump.py"
python wikidump.py
echo "* completed: wikidump.py"
source deactivate

source activate $py3_env
echo "* run: cleanWikidump.py"
python cleanWikidump.py
echo "* completed: cleanWikidump.py"
echo "* starting NLP server ..."
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 120000
wget --post-data 'The quick brown fox jumped over the lazy dog.' 'localhost:9000/?properties={"annotators":"tokenize,ssplit,pos","outputFormat":"json"}' -O -
python getTriples.py
echo "* completed: getTriples.py"
source deactivate

source activate $py2_env
echo "* run: pruneCorenlpTriples"
python pruneCorenlpTriples.py
echo "* completed: pruneCorenlpTriples"

echo "* run: convertGold.py"
python convertGold.py #doesn't matter which environment you use for this file
echo "* completed: convertGold.py"
source deactivate
