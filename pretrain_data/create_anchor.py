import requests
import sys
import json 
import os
from multiprocessing import Pool, Lock
from nltk.tokenize import sent_tokenize
import math
import pdb
from tqdm import tqdm


anchors = json.load(open("pretrain_data/all_anchors_name.json"))
part = int(math.ceil(len(anchors) / 256.)) # IMPORTANT! This number must be consistant with workers' number
anchors = [anchors[i:i+part] for i in range(0, len(anchors), part)]
print(len(anchors))

def run_proc(idx, n, input_names):
    folder = "pretrain_data/anchor"
    target = "{}/{}".format(folder, idx)
    fout = open(target+"_anchor2id", "a+")
    for input_name in tqdm(input_names):
        try:
            entity_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles={input_name}&format=json"
            entity_info = requests.get(entity_url)
            id = list(entity_info.json()['query']['pages'].items())[0][1]['pageprops']['wikibase_item']
        except:
            id = '#UNK#'
        fout.write(f"{input_name}\t{id}\n")
    fout.close()

folder = "pretrain_data/anchor"
if not os.path.exists(folder):
    os.makedirs(folder)

n = int(sys.argv[1])
p = Pool(n)
for i in range(n):
    p.apply_async(run_proc, args=(i, n, anchors[i]))
p.close()
p.join()



