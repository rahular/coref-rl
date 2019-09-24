import time
import csv
import yaml
import os
import corenlp

"""
Get triples with CoreNLP's OpenIE (SRO)

NB: Before running this code, start an NLP server:

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 120000

To get the server running properly for the script, give the NLP Server an example first:

wget --post-data 'The quick brown fox jumped over the lazy dog.' 'localhost:9000/?properties={"annotators":"tokenize,ssplit,pos","outputFormat":"json"}' -O -

This will ensure that the NLP server doesn't time out when making calls with Python. 
"""


def getTriples(corpus):
    """
    Input: corpus: output of cleanWikidump.py,

    Output:
        CORENLP: triples: set( (tuple[string], confidence score[float]) )"""

    all_triples = set()  # init results
    t1 = time.time()  # start the clock

    print("* Using CoreNLP to extract triples ... ")
    for document in corpus:
        triples = get_openIE_triples(document)
        all_triples.update(triples)
    print("* Triples extracted with OpenIE in %0.3fs." % (time.time() - t1))

    return all_triples


def get_openIE_triples(document):
    # CoreNLP OpenIE
    """
    Input: document: str
    Output: tripleSet: set(tripple-tuple, confidence)
    """
    tripleSet = set()

    try:
        with corenlp.CoreNLPClient(annotators="tokenize,ssplit,pos,lemma,depparse,natlog,openie".split()) as client:
            ann = client.annotate(document)  # 'doc.CoreNLP_pb2.Document'>

        for sent in ann.sentence:
            if len(sent.openieTriple) > 0:  # if there are any triples...
                # <class 'google.protobuf.pyext._message.RepeatedCompositeContainer'>
                triples = sent.openieTriple
                for t in triples:
                    triple_tuple = (t.subject, t.relation, t.object)
                    c = t.confidence
                    out = (triple_tuple, c)
                    tripleSet.add(out)
    except:
        # corenlp server probably timed out because the document was too big
        # TODO: grab a smaller section and annotate that.
        pass

    return tripleSet


def main(cfg):
    input_file = os.path.join(
        cfg["all"]["path_to_data"], "clean_wiki_" + str(cfg['all']['n_articles']) + ".txt")

    with open(input_file, "r") as inFile:
        corpus = inFile.readlines()

    all_triples = getTriples(corpus)

    output_file = os.path.join(cfg["all"]["path_to_data"], "re-text.tsv")
    with open(output_file, "w") as outFile:
        writer = csv.writer(outFile, delimiter='\t')
        for num, t in enumerate(all_triples):
            s = t[0][0]  # subject
            r = t[0][1]  # relation
            o = t[0][2]  # object
            c = t[1]  # confidence score
            writer.writerow([s, r, o, c])
            if num % 10000 == 0:
                print("* Reached num: " + str(num) + " triple")


if __name__ == '__main__':
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)  # dict

    main(cfg)
