import os
import sling

from flask import Flask
from flask import request, jsonify

DEBUG_FLAG = False
app = Flask(__name__)

@app.route('/get_kb_triples', methods = ['POST'])
def api_call():
    body = request.get_json(force=True, silent=True)
    triples = body.get('triples', None)
    if not triples or not isinstance(triples, list):
        return jsonify({
            'output': []
        }, 200)

    return jsonify({
        'output': get_kb_triples(triples)
    }, 200)

def get_kb_triples(triples):
    pronouns = ['he', 'she', 'him', 'her', 'it', 'they', 'their', 'them']
    qtriples = []
    for triple in triples:
        s = triple[0]
        r = triple[1]
        o = triple[2]
        if s in pronouns or o in pronouns:
            continue
        
        try:
            qs = names.lookup(s.encode('ascii', 'ignore'))
            qo = names.lookup(o.encode('ascii', 'ignore'))
            if qs and qo:
                qtriples.append([qs[0].id, r, qo[0].id])
        except Exception as e:
            print(e)
            continue

    return qtriples

def init_phrase_table():
    base_path = '../../sling/'
    kb = sling.Store()
    kb.load(os.path.join(base_path, 'local/data/e/wiki/kb.sling'))
    names = sling.PhraseTable(kb, os.path.join(base_path, 'local/data/e/wiki/en/phrase-table.repo'))
    kb.freeze()
    return names

if __name__ == '__main__':
    names = init_phrase_table()
    print('Loaded Wikidata. Starting server...')
    app.run(host="0.0.0.0", debug=DEBUG_FLAG)