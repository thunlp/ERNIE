# ERNIE (sub-project of OpenSKL)

ERNIE is a sub-project of OpenSKL, providing an open-sourced toolkit (**E**nhanced language **R**epresentatio**N** with **I**nformative **E**ntities) for augmenting pre-trained language models with knowledge graph representations.

## Overview

ERNIE contains the source code and dataset for "[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)", and is an effective and efficient toolkit for augmenting pre-trained language models with knowledge graph representations.

## Models
We include the vanilla BERT model as well as our knowledge-enhanced version ERNIE in this toolkit.

## Evaluation

We validate the effectiveness of ERNIE on entity typing and relation classification tasks through fine-tuning.

### Settings
We use the following datasets: FIGER and OpenEntity for entity typing, FewRel and TACRED for relation classification. We will fine-tune the models (BERT and ERNIE) first, and then evaluate their accuracies and F1 scores.

### Results

Here we report the main results on the above datasets. From this table, we observe that ERNIE effectively improves the performance of BERT on these knowledge-driven tasks.

|       | FIGER | OpenEntity | FewRel | TACRED |
|-------|-------|------------|--------|--------|
|       | Acc.  | F1         | F1     | F1     |
| BERT  | 52.04 | 73.56      | 84.89  | 66.00  |
| ERNIE | 57.19 | 75.56      | 88.32  | 67.97  |


## Usage

### Requirements:

* Pytorch>=0.4.1
* Python3
* tqdm
* boto3
* requests
* apex (If you want to use fp16, you should make sure the commit is `79ad5a88e91434312b43b4a89d66226be5f2cc98`.)

### Prepare Pre-train Data

Run the following command to create training instances.

```shell
  # Download Wikidump
  wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
  # Download anchor2id
  wget -c https://cloud.tsinghua.edu.cn/f/6318808dded94818b3a1/?dl=1 -O anchor2id.txt
  # WikiExtractor
  python3 pretrain_data/WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -o pretrain_data/output -l --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 4
  # Modify anchors with 4 processes
  python3 pretrain_data/extract.py 4
  # Preprocess with 4 processes
  python3 pretrain_data/create_ids.py 4
  # create instances
  python3 pretrain_data/create_insts.py 4
  # merge
  python3 code/merge.py
```

If you want to get anchor2id by yourself, run the following code(this will take about half a day) after `python3 pretrain_data/extract.py 4`
```shell
  # extract anchors
  python3 pretrain_data/utils.py get_anchors
  # query Mediawiki api using anchor link to get wikibase item id. For more details, see https://en.wikipedia.org/w/api.php?action=help.
  python3 pretrain_data/create_anchors.py 256 
  # aggregate anchors 
  python3 pretrain_data/utils.py agg_anchors
```

Run the following command to pretrain:

```
  python3 code/run_pretrain.py --do_train --data_dir pretrain_data/merge --bert_model bert_base --output_dir pretrain_out/ --task_name pretrain --fp16 --max_seq_length 256
```

We use 8 NVIDIA-2080Ti to pre-train our model and there are 32 instances in each GPU. It takes nearly one day to finish the training (1 epoch is enough).

### Pre-trained Model

Download pre-trained knowledge embedding from [Google Drive](https://drive.google.com/open?id=14VNvGMtYWxuqT-PWDa8sD0e7hO486i8Y)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/229e8cccedc2419f987e/) and extract it.

```shell
tar -xvzf kg_embed.tar.gz
```

Download pre-trained ERNIE from [Google Drive](https://drive.google.com/open?id=1DVGADbyEgjjpsUlmQaqN6i043SZvHPu5)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a763616323f946fd8ff6/) and extract it.

```shell
tar -xvzf ernie_base.tar.gz
```

Note that the extraction may be not completed in Windows.

### Fine-tuning

As most datasets except FewRel don't have entity annotations, we use [TAGME](<https://tagme.d4science.org/tagme/>) to extract the entity mentions in the sentences and link them to their corresponding entitoes in KGs. We provide the annotated datasets [Google Drive](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6ec98dbd931b4da9a7f0/).

```shell
tar -xvzf data.tar.gz
```

In the root directory of the project, run the following codes to fine-tune ERNIE on different datasets.

**FewRel:**

```bash
python3 code/run_fewrel.py   --do_train   --do_lower_case   --data_dir data/fewrel/   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel   --fp16   --loss_scale 128
# evaluate
python3 code/eval_fewrel.py   --do_eval   --do_lower_case   --data_dir data/fewrel/   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel   --fp16   --loss_scale 128
```

**TACRED:**

```bash
python3 code/run_tacred.py   --do_train   --do_lower_case   --data_dir data/tacred   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --fp16   --loss_scale 128 --threshold 0.4
# evaluate
python3 code/eval_tacred.py   --do_eval   --do_lower_case   --data_dir data/tacred   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --fp16   --loss_scale 128 --threshold 0.4
```

**FIGER:**

```bash
python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/FIGER   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir output_figer  --gradient_accumulation_steps 32 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2
# evaluate
python3 code/eval_figer.py    --do_eval   --do_lower_case   --data_dir data/FIGER   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir output_figer  --gradient_accumulation_steps 32 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2
```

**OpenEntity:**

```bash
python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 128   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --fp16 --loss_scale 128
# evaluate
python3 code/eval_typing.py   --do_eval   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 128   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --fp16 --loss_scale 128
```

Some code is modified from the **pytorch-pretrained-BERT**. You can find the explanation of most parameters in [pytorch-pretrained-BERT](<https://github.com/huggingface/pytorch-pretrained-BERT>). 

As the annotations given by TAGME have confidence score, we use `--threshlod` to set the lowest confidence score and choose the annotations whose scores are higher than `--threshold`. In this experiment, the value is usually `0.3` or `0.4`.

The script for the evaluation of relation classification just gives the accuracy score. For the macro/micro metrics, you should use `code/score.py` which is from [tacred repo](<https://github.com/yuhaozhang/tacred-relation>).

```shell
python3 code/score.py gold_file pred_file
```

You can find `gold_file` and `pred_file` on each checkpoint in the output folder (`--output_dir`).

### New Tasks

If you want to use ERNIE in new tasks, you should follow these steps:

* Use an entity-linking tool like TAGME to extract the entities in the text
* Look for the Wikidata ID of the extracted entities
* Take the text and entities sequence as input data

Here is a quick-start example (`code/example.py`) using ERNIE for Masked Language Model. We show how to annotate the given sentence with TAGME and build the input data for ERNIE. Note that it will take some time (around 5 mins) to load the model.

```shell
# If you haven't installed tagme
pip install tagme
# Run example
python3 code/example.py
```

## Citation

If you use the code, please cite this paper:

```
@inproceedings{zhang2019ernie,
  title={{ERNIE}: Enhanced Language Representation with Informative Entities},
  author={Zhang, Zhengyan and Han, Xu and Liu, Zhiyuan and Jiang, Xin and Sun, Maosong and Liu, Qun},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```
******************
## About OpenSKL
OpenSKL project aims to harness the power of both structured knowledge and natural languages via representation learning. All sub-projects of OpenSKL, under the categories of **Algorithm**, **Resource** and **Application**, are as follows.

- **Algorithm**: 
  - [OpenKE](https://www.github.com/thunlp/OpenKE)
    - An effective and efficient toolkit for representing structured knowledge in large-scale knowledge graphs as embeddings, with <a href="https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350"> TransR</a> and  <a href="https://aclanthology.org/D15-1082.pdf">PTransE</a> as key features to handle complex relations and relational paths.
    - This toolkit also includes three repositories:
       - [KB2E](https://www.github.com/thunlp/KB2E)
       - [TensorFlow-Transx](https://www.github.com/thunlp/TensorFlow-Transx)
       - [Fast-TransX](https://www.github.com/thunlp/Fast-TransX)
  - [ERNIE](https://github.com/thunlp/ERNIE)
    - An effective and efficient toolkit for augmenting pre-trained language models with knowledge graph representations.
  - [OpenNE](https://www.github.com/thunlp/OpenNE)
    - An effective and efficient toolkit for representing nodes in large-scale graphs as embeddings, with [TADW](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) as key features to incorporate text attributes of nodes.
  - [OpenNRE](https://www.github.com/thunlp/OpenNRE)
    - An effective and efficient toolkit for implementing neural networks for extracting structured knowledge from text, with [ATT](https://aclanthology.org/P16-1200.pdf) as key features to consider relation-associated text information.
    - This toolkit also includes two repositories:
      - [JointNRE](https://www.github.com/thunlp/JointNRE)
      - [NRE](https://github.com/thunlp/NRE)
- **Resource**:
  - The embeddings of large-scale knowledge graphs pre-trained by OpenKE, covering three typical large-scale knowledge graphs: Wikidata, Freebase, and XLORE. The embeddings are free to use under the [MIT license](https://opensource.org/license/mit/), and please click the following link to submit [download requests](http://139.129.163.161/download/wikidata).
  - OpenKE-Wikidata
    - Wikidata is a free and collaborative database, collecting structured data to provide support for Wikipedia. The original Wikidata contains 20,982,733 entities, 594 relations and 68,904,773 triplets. In particular, Wikidata-5M is the core subgraph of Wikidata, containing  5,040,986 high-frequency entities from Wikidata with their corresponding 927 relations and 24,267,796 triplets.
    - TransE version: Knowledge embeddings of Wikidata pre-trained by OpenKE. 
    - [TransR version](https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/transr.npy) of Wikidata-5M: Knowledge embeddings of Wikidata-5M pre-trained by OpenKE.
  - OpenKE-Freebase
    - Freebase was a large collaborative knowledge base consisting of data composed mainly by its community members. It was an online collection of structured data harvested from many sources. Freebase contains 86,054,151 entities, 14,824 relations and 338,586,276 triplets.
    - TransE version: Knowledge embeddings of Freebase pre-trained by OpenKE. 
  - OpenKE-XLORE
    - XLORE is one of the most popular Chinese knowledge graphs developed by THUKEG. XLORE contains 10,572,209 entities, 138,581 relations and 35,954,249 triplets.
    - TransE version: Knowledge embeddings of XLORE pre-trained by OpenKE.
- **Application**:   
    - [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin)
      - An effective and efficient toolkit of plug-and-play knowledge injection for pre-trained language models. Knowledge-Plugin is general for all kinds of knowledge graph embeddings mentioned above. In the toolkit, we plug the TransR version of Wikidata-5M into BERT as an example of applications. With the TransR embedding, we enhance the knowledge ability of BERT without fine-tuning the original model, e.g., up to 8% improvement on question answering.
