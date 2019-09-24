import sling  # on leia, use the usr/local/bin python
import time
import csv
import yaml
import os

# Load config
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)  # dict


def confirmEntities(names, entity1, entity2):
    """
    ******** This method needs to be called in a python env that has sling!!! *******
    :param names: sling.PhraseTable
    :param entity1: str
    :param entity2: str
    :return: Boolean, True if both entities exist in KB; False if either is not in KB
             Qcode1: Str, returns the q code if the True, otherwise returns a dummy empty string
             Qcode2: Str, "

    """
    throwaway = ['he', 'she', 'him', 'her', 'it', 'they', 'their', 'them']

    if entity1.lower() in throwaway:
        return False, '', ''
    if entity2.lower() in throwaway:
        return False, '', ''

    candidates1 = names.lookup(entity1)
    try:
        qcode1 = candidates1[0].id  # str
    except Exception as e1:  # candidates list was empty, so return false
        return False, '', ''

    candidates2 = names.lookup(entity2)
    try:
        qcode2 = candidates2[0].id  # str
    except Exception as e2:
        return False, '', ''

    if qcode1 and qcode2:
        return True, qcode1, qcode2


def pruneTriples(triple_file, output_file):
    """
    ******** This method needs to be called in a python env that has sling!!! *******
    Input: 1. name of file with tab sepperated SRO triples
           2. name of the output file
    Output: file containing triples where both entities have QCodes.
    """

    t1 = time.time()  # start the clock
    base_path = cfg['all']['base_path']
    kb = sling.Store()
    kb.load(base_path + "local/data/e/wiki/kb.sling")
    names = sling.PhraseTable(
        kb, base_path + "local/data/e/wiki/en/phrase-table.repo")
    kb.freeze()
    print("* Sling KB loaded in %0.3fs." % (time.time() - t1))

    verified_triples = []

    with open(triple_file, "r") as inFile:
        tsvreader = csv.reader(inFile, delimiter="\t")
        for triple in tsvreader:
            entity1 = triple[0]
            relation = triple[1]
            entity2 = triple[2]
            score = triple[3]
            hasBoth, qcode1, qcode2 = confirmEntities(names, entity1, entity2)
            if (hasBoth):
                q_triple = (qcode1, relation, qcode2, score)
                verified_triples.append(q_triple)

    with open(output_file, "w") as outFile:
        writer = csv.writer(outFile, delimiter='\t')
        for t in verified_triples:
            s = t[0]  # subject
            r = t[1]  # relation
            o = t[2]  # object
            c = t[3]  # confidence score
            writer.writerow([s, r, o, c])


def main():
    # give it the output of coref-rl/wiki/getTriples
    input_file = os.path.join(
        cfg['all']['path_to_data'], "re-text.tsv")
    output_file = os.path.join(
        cfg['all']['path_to_data'], "re-joint.tsv")
    pruneTriples(input_file, output_file)


if __name__ == '__main__':
    main()
