 # Python3
import os
import sys
import nltk
import yaml

pointer_to_wiki = os.path.join(os.path.abspath(
    os.path.join(__file__, "../..")), 'wiki')
sys.path.append(pointer_to_wiki)  # make the path up and over
from wikiapi import scrub_html, remove_spurious_parts 

"""
Clean up the wikidumps from wikidump.py, so that it can be input 
to OpenIE for extracting triples. 
"""


def cleanWiki(documents, cfg):
    output_file_name = os.path.join(cfg["all"]["path_to_data"], "clean_wiki_" + str(cfg['all']['n_articles']) + ".txt")
    with open(output_file_name, "w") as outFile:
        for doc in documents:
            sentences = []
            passages, _ = scrub_html(doc)  # clean out html markup

            try:
                sent_text = nltk.sent_tokenize(
                    passages[0])  # grab only first passage
                if len(sent_text) > 0 and len(sent_text) < 100:
                    [sentences.append(s) for s in sent_text]
                elif len(sent_text) > 0 and len(sent_text) > 100:
                    # only keep 100 sentences
                    [sentences.append(s) for s in sent_text[:100]]
                text = ' '.join(sentences)
                text = text.strip()
                text = ' '.join(text.split())  # make sure its all one line
                if text != '':
                    outFile.write(text + "\n")
            except:
                pass  # If it gets here, its because passages was completely empty


def main(cfg):
    path_to_wikidump = cfg['all']['path_to_data']
    name_of_wikidump = 'wikidump_' + str(cfg['all']['n_articles']) + '.txt'
    # output of wikidump.py
    wikidump = os.path.join(path_to_wikidump, name_of_wikidump)
    with open(wikidump, "r") as inFile:
        wikitexts = inFile.read()
        documents = wikitexts.split(
            "#######################")  # List of strings
        cleanWiki((documents), cfg)


if __name__ == '__main__':
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)  # dict
    main(cfg)
