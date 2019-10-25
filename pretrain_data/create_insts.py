import sys
import os
from multiprocessing import Pool
import math
import subprocess

input_folder = "pretrain_data/raw"
output_folder = "pretrain_data/data"
file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))

file_list = list(set(["_".join(x.split("_")[:-1]) for x in file_list]))

def run_proc(idx, n, file_list):
    for i in range(len(file_list)):
        if i % n == idx:
            target = file_list[i].replace("raw", "data")
            print(file_list[i])
            command = "python3 code/create_instances.py --input_file_prefix {} --output_file {} --vocab_file ernie_base/vocab.txt --dupe_factor 1 --max_seq_length 256 --max_predictions_per_seq 40"
            subprocess.run(command.format(file_list[i], target).split())

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n = int(sys.argv[1])
p = Pool(n)
for i in range(n):
    p.apply_async(run_proc, args=(i,n, file_list))
p.close()
p.join()