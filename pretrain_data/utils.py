import os 
import sys
import json 
from collections import Counter
from tqdm import tqdm, trange

def get_all_anchors_name():
    input_folder = "pretrain_data/ann"
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))

    anchors = {}
    for i in trange(len(file_list)):
        input_name = file_list[i]
        fin = open(input_name, "r")
        for doc in fin.readlines():
            doc = doc.strip()
            segs = doc.split("[_end_]")
            map_segs = segs[1:]
            maps = {}
            for x in map_segs:
                v = x.split("[_map_]")
                if len(v) != 2:
                    continue
                if anchors.get(v[1], -1) != -1:
                    continue
                anchors[v[1]] = 1
        fin.close()
    print(len(anchors))
    json.dump(list(anchors.keys()), open("pretrain_data/all_anchors_name.json", 'w'))

def aggregate_anchor2id():
    fout = open("anchor2id.txt", 'w')
    files = os.listdir("pretrain_data/anchor")
    for file in files:
        f = open(os.path.join("pretrain_data/anchor", file))
        fout.write(f.read())
        f.close()
    fout.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == 'get_anchors':
        get_all_anchors_name()
    elif mode == 'agg_anchors':
        aggregate_anchor2id()


