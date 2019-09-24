import json
import pickle
import csv
import yaml
import os


def main(cfg):
    # PID to string mappings. E.g. {"P31": ["member of", "type", "is a", "is a specific"]}
    with open(os.path.join(cfg["all"]["path_to_data"], "pid2str.json", "r")) as inJson:
        dictdump = json.loads(inJson.read())

    # gold triples from wikidata
    with open(os.path.join(cfg["all"]["path_to_data"], "triples-1M.pkl", "rb"), 'rb') as inGold:
        gold = pickle.load(inGold)

        with open(os.path.join(cfg["all"]["path_to_data"], "re-kg.tsv", "w")) as outFile:
            writer = csv.writer(outFile, delimiter='\t')
            for g in gold:
                q1 = g[0]
                q2 = g[2].strip("\n")
                pid = g[1]
                pid2str = dictdump[pid]
                for s in pid2str:  # output q1 "relation" q2 for all possible relations for a pid
                    writer.writerow([q1, s, q2])


if __name__ == '__main__':
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)  # dict

    main(cfg)
    print("* Done! ")
