import os
import sling
import yaml

"""
Load some articles from sling's wiki dumps.
Dump these articles into a txt file. 
(This txt file is the input to cleanWikidump.py)
"""

def getWiki(n_articles, sling_path, data_path):
    """ Input - n_articles: int. The number of desired articles
        Output: dumps the HTML of the articles into a dummy.txt WITH A DOCUMENT SEPERATOR for further processing
        (since Sling is py2.7, but our Wikipedia scripts are for py3*)"""

    output_file_name = os.path.join(data_path, "wikidump_" + str(n_articles) + ".txt")
    with open(output_file_name, "w") as outFile:
        for num, document in enumerate(sling.Corpus(sling_path + "local/data/e/wiki/en/documents@10.rec")):
            if num < n_articles:
                raw_text = document.text  # str
                outFile.write(raw_text)
                outFile.write("#######################")  # doc sepperator!!!
            else:
                break


def main():
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)  # dict

    sling_path = cfg['all']['sling_path']
    n_articles = cfg['all']['n_articles']
    data_path = cfg['all']['path_to_data']

    getWiki(n_articles, sling_path, data_path)


if __name__ == '__main__':
    main()
