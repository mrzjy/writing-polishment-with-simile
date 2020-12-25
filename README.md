# WPS - Writing Polishment with Simile
The official implementation of the AAAI2021 paper ["Writing Polishment with Simile: Task, Dataset and A Neural Approach"](https://arxiv.org/abs/2012.08117 "Preprint version")

## Task Description
Writing Polishment with Simile (WPS)—This task aims to polish plain text with similes (明喻), which we believe is a critical step towards figurative writing polishment.

For instance:
 
Before polishment:
> Looking at his bloodthirsty eyes, everyone felt horrible and couldn’t help but step back.

After polishment:
> Looking at his bloodthirsty eyes, everyone felt horrible ***as if they were being stared at by a serpent,*** and couldn’t help but step back.

In the paper, we cast this task as a two-staged process:
1. Decide where to put a simile within plain input text
2. Figure out what content to generate as a coherent simile.

## Data and Code
### Chinese Simile (CS) Dataset
This dataset is constructed and based on the online free-access fictions that are tagged with sci-fi, urban novel, love story, youth, etc.

All similes are extracted by rich regular expression, and the extraction precision is estimated as 92% by labelling 500 random extracted samples. Further data filtering as well as processing is truly encouraged!

The data split in paper is as follows (You could find more details in the paper):

| Train      | Dev     | Test     |
| ---------- | :-----------:  | :-----------: |
| 5,485,721     | 2,500     | 2,500     |

You could download the data here:

["GoogleDrive"](https://drive.google.com/file/d/1IfVLuOMsoVkZp34An6Ii44zsIModzWb3/view?usp=sharing  "link") 
or
["BaiduDisk"](https://pan.baidu.com/s/1_hJgBXspqTsyfgwx9hn-Pg  "link") (code: fslg)

Put them into data folder to proceed:

~~~
data
 |- train.txt
 |- dev.txt
 |- test.txt 

# format for each line: (similes are delimited with " || ") 
#   similes \t context_containing_similes
~~~

### Code
#### Requirements
- Python 3.6
- Tensorflow >= 1.14
- Jieba (for evaluation only)

#### Commands
##### Train

Remember to download a pretrained BERT checkpoint to initialize your training. In our paper, we used ["BERT-wwm-ext"](https://github.com/ymcui/Chinese-BERT-wwm  "link").  
~~~
# Remember to set your $your_model_dir and $PATH_TO_BERT variables
python -u run_WPS.py --output_dir=$your_model_dir \
    --train_file=data/train.txt --do_train \
    --epochs=15 --batch_size=128 \
    --max_input_length=128 --max_target_length=16 \
    --init_checkpoint=$PATH_TO_BERT/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt 
~~~

##### Predict

~~~
# (There are already some generation examples provided in output folder)

# Interpolate simile given plain contexts in test/dev set
python -u run_WPS.py --output_dir=$your_model_dir \
    --predict_file=data/test.txt --do_predict \
    --batch_size=128 --max_input_length=128 --max_target_length=16 \
    --beam_size=20

# This ends in multiple output files in output folder, namely:
output 
 |- bleu_out         # char-tokenized generated simile
 |- bleu_ref         # char-tokenized ground-truth simile
 |- contexts         # char-tokenized ground-truth plain context
 |- generations.txt  # context_left \t generated_simile \t context_right 
 |- groundtruth.txt  # context_left \t ground-truth_simile \t context_right 
~~~

##### Evaluation

Note: we used pretrained Tencent word embedding when evaluating embedding similarity, you could download their original version, or use a smaller version here ["GoogleDrive"](https://drive.google.com/file/d/1NHIB_GasLqJL4CQgawn3uMEOExtQLl45/view?usp=sharing  "link"), where we only retain embeddings for words in config/vocab_seg_as_jieba.txt  

~~~
# Suppose the generation files described previously already exist in output folder 
# (and let's take output/beam_20 for instance)

# 1. Evaluate generation word overlap
perl output/multi-bleu.perl output/beam_20/bleu_ref < output/beam_20/bleu_out

# 2. Evaluate generation length
python eval_length.py --f output/beam_20/bleu_out

# 3. Evaluate generation diversity
python eval_diversity.py --f output/beam_20/bleu_out

# 4. Evaluate embedding similarity
python eval_embedding.py --embeddings=embeddings.pkl \
    --ref=output/beam_20/contexts \
    --out=output/beam_20/bleu_out \
    --vocab_file=config/vocab_seg_as_jieba.txt

# 5. Evaluate PPL (do the e^(-neg_log_perplexity) calculation yourself)
# You should download a pretrained word embedding file in order to run this
python -u run_WPS.py --output_dir=$your_model_dir \
    --test_file=data/test.txt \
    --do_eval --batch_size=128 \
    --max_input_length=128 \
    --max_target_length=16
~~~

##### Playing around with pretrained Locate&Gen
You could download our pretrained model here:

["GoogleDrive"](https://drive.google.com/file/d/1P1JG5xFMJE0Mq_bCnjuqgK8o0wgJCQUj/view?usp=sharing  "link")
or 
["BaiduDisk"](https://pan.baidu.com/s/1KHKylbMR-uD_34wVbiRpyg  "link") (code: 9nws)

**First**, extract the downloaded checkpoint into pretrained_model directory:

~~~
pretrained_model
 |- checkpoint
 |- graph.pbtxt
 |- model.ckpt-642858.data-00000-of-00001
 |- model.ckpt-642858.index
 |- model.ckpt-642858.meta
~~~

**Second**, prepare your own input texts to be polished (line by line), e.g.,

~~~
# your_input.txt (3 random sentences, note that the last one is from table 7 in the paper)

我打开窗帘发现外面阳光明媚，于是开始准备出去玩。
我打开窗帘发现外面阴云密布，出游计划估计泡汤了。
梁师这一脚好生了得，王宇感觉难受，身体就要不顾自己的控制飞出悬崖，还好他及时用双手抓在水泥地上，只抓划着地面发出了令人发指的声音。
~~~

**Third**, start playing around:

- Apply beam-search decoding (confident, but lack of diversity)

~~~
python -u run_WPS.py --output_dir=pretrained_model \
    --predict_file=your_input.txt --do_predict \
    --batch_size=128 --max_input_length=128 --max_target_length=16 \
    --beam_size=20

# output (1st one is a bad case, 2nd is OK, 3rd is a bit weird)
~~~

The reproducible output: 

> ***就像现在*** 我打开窗帘发现外面阳光明媚，于是开始准备出去玩。
>
> 我打开窗帘发现外面阴云密布， ***好像要下雨一样*** 出游计划估计泡汤了。
>
> 梁师这一脚好生了得，王宇感觉 ***像是被火车撞了一样*** 难受，身体就要不顾自己的控制飞出悬崖，还好他及时用双手抓在水泥地上，只抓划着地面发出了令人发指的声音。

- Apply top-p sampling (more diverse, but risky)

~~~
python -u run_WPS.py --output_dir=pretrained_model \
    --predict_file=your_input.txt --do_predict \
    --batch_size=128 --max_input_length=128 --max_target_length=16 \
    --sampling --top_p=0.25 \
    --beam_size=1
~~~

A sampled output: 

> ***仿佛是心有灵犀一般*** 我打开窗帘发现外面阳光明媚，于是开始准备出去玩。
>
> 我打开窗帘发现外面阴云密布， ***就像天要塌下来一样*** 出游计划估计泡汤了。
> 
> 梁师这一脚好生了得，王宇感觉 ***就像是踢在了钢板上一样*** 难受，身体就要不顾自己的控制飞出悬崖，还好他及时用双手抓在水泥地上，只抓划着地面发出了令人发指的声音。

- Apply top-p beam decoding: (As a matter of fact, feel free to apply top-p sampling together with beam search, higher top_p might be better in this case)

~~~
python -u run_WPS.py --output_dir=pretrained_model \
    --predict_file=your_input.txt --do_predict \
    --batch_size=128 --max_input_length=128 --max_target_length=16 \
    --sampling --top_p=0.65 \
    --beam_size=2
~~~

A sampled output: 

> ***就好像是一个小孩子一样*** 我打开窗帘发现外面阳光明媚，于是开始准备出去玩。
>
> 我打开窗帘发现外面阴云密布， ***好像随时都要下雨一样*** 出游计划估计泡汤了。
>
> 梁师这一脚好生了得，王宇感觉 ***像是被巨石砸中一样*** 难受，身体就要不顾自己的控制飞出悬崖，还好他及时用双手抓在水泥地上，只抓划着地面发出了令人发指的声音。

- More sampled similes for the last sentence:

~~~
就像是要散架了一般的
就像是断了一样
好像被人撕裂了一样
就跟吃了一只苍蝇一样  # bad case: incoherent 
就像做迁徙的动物一样  # bad case: incoherent 
像是踢到了钢板上一样的
就好像被一辆大卡车撞到一样
就像被铁锤砸中一样
就像是被火车撞了一样
就像是什么东西被踢##惯了一样  # bad case: disfluent
就像是一拳打在了棉花上一样
就像是被人抽了一巴掌一样
像是被人迎面扇了一巴掌一样
就像是被火烧了一样
像是被火车撞了一样
就像被大象踩了一脚一样
就像是被铁棍敲中了一般
就像是被车子辗过后一样的
如同被刀子剜##艾一样  # bad case: disfluent
~~~

## Copyright Disclaimer

- The original copyrights of all the articles and sentences belong to the source owners.
- The copyright of simile tagging belongs to our group, and they are free to the public.
- All the contents in this dataset does not represent the authors' opinion. We are not responsible for any contents generated using our model.
- The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes.


## Citation
Please kindly cite this repository using the following reference:

~~~
@inproceedings{Zhang2020WritingPW,
  title={Writing Polishment with Simile: Task, Dataset and A Neural Approach},
  author={Jiayi Zhang and Z. Cui and Xiaoqiang Xia and Ya-Long Guo and Yanran Li and Chen Wei and Jianwei Cui},
  booktitle={AAAI},
  year={2021}
}
~~~