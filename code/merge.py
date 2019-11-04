import indexed_dataset
import os

builder = indexed_dataset.IndexedDatasetBuilder('pretrain_data/merge.bin')
for filename in os.listdir("pretrain_data/data"):
    if filename[-4:] == '.bin':
        builder.merge_file_("pretrain_data/data/"+filename[:-4])
builder.finalize("pretrain_data/merge.idx")