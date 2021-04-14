from bs4 import BeautifulSoup
import sys
from urllib import parse
import os
import logging
from multiprocessing import Pool

logging.basicConfig(level=logging.DEBUG)
input_folder = "pretrain_data/output"

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))


def run_proc(idx, n, file_list):
    for input_name in file_list[idx::n]:
        target = input_name.replace(input_folder, "pretrain_data/ann")
        folder = '/'.join(target.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        elif os.path.exists(target):
            # Already done parsing
            continue

        soup = BeautifulSoup(open(input_name), features="html.parser")
        docs = soup.find_all('doc')

        with open(target, 'w') as fout:
            for doc in docs:
                content = doc.get_text(" sepsepsep ")
                while content[0] == "\n":
                    content = content[1:]
                content = [x.strip() for x in content.split("\n")]
                content = "".join(content[1:])

                try:
                    lookup = [(x.get_text().strip(), x.get('href'))
                              for x in doc.find_all("a")]
                    lookup = "[_end_]".join(
                        ["[_map_]".join(x) for x in lookup])
                    fout.write(content+"[_end_]"+lookup+"\n")
                except Exception as e:
                    logging.warning(
                        'Error {} when parsing file {}'.format(str(e), input_name))
        logging.info('Finished {}'.format(target))


n = int(sys.argv[1])
p = Pool(n)
for i in range(n):
    p.apply_async(run_proc, args=(i, n, file_list))
p.close()
p.join()
