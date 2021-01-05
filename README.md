# On Explaining Your Explanations of BERT: An Empirical Study with Sequence Classification
Pytorch implmentation of Gradient Sensitivity, Graident X Input, Layerwise Revelance Propagation and Layerwise Attention Tracing for BERT-based models in just a single `model.backward()` call.


### Citation

[Zhengxuan Wu](http://zen-wu.social) and [Desmond C. Ong](https://web.stanford.edu/~dco/). 2020. [On Explaining Your Explanations of BERT: An Empirical Study with Sequence Classification](https://arxiv.org/abs/2101.00196). Ms., Stanford University and National University of Singapore.

```stex
  @article{wu-ong-2020-explain,
    title={On Explaining Your Explanations of BERT: An Empirical Study with Sequence Classification},
    author={Wu, Zhengxuan and Ong, Desmond C.},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/2101.00196},
    year={2020}}
```

### Abstract
BERT, as one of the pretrianed language models, attracts the most attention in recent years for creating new benchmarks across NLP tasks via fine-tuning. One pressing issue is to open up the blackbox and explain the decision makings of BERT. A number of attribution techniques have been proposed to explain BERT models, but are often limited to sequence to sequence tasks. In this paper, we adapt existing attribution methods on explaining decision makings of BERT in sequence classification tasks. We conduct extensive analyses of four existing attribution methods by applying them to four different datasets in sentiment analysis. We compare the reliability and robustness of each method via various ablation studies. Furthermore, we test whether attribution methods explain generalized semantics across semantically similar tasks. Our work provides solid guidance for using attribution methods to explain decision makings of BERT for downstream classification tasks.

### Introduction
Regular BERT training is for accuracy, but if you are interested in understanding why BERT made that decision, and which tokens contribute more towards that model decision, this repo will help you understand more about the BERT model. If you want to know the decision of your trained BERT, simple import our BERT model, and train with a gradient hook enabled. Your attribution scores will be calculated with a simple ``backward()`` call:
```python
from BERT import *
# initialize model with our own BERT module
model = BERT()
# training loop
model.train()
# after training, simply call respective attribuiton method's backward function
model.backward()
```

### Install Requirements
You will have to clone this repo, and install all the dependencies. You can skip this step if you have torch and cuda installed. That is all you need. You can also mannually install these without going through this installation headache that ``requirements.txt`` may give you.
```bash
cd BERT_LRP/code/
pip install -r requirements.txt
```

### Download Pretrained BERT Model
You will have to download pretrained BERT model in order to execute the fine-tune pipeline. We recommand to use models provided by the official release on BERT from [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert). Note that their model is in tensorflow format. To convert tensorflow model to pytorch model, you can use the helper script to do that. For example,
```bash
cd BERT_LRP/code/
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

### BERT Model and Pretrain
Our BERT model is adapted from [huggingface](https://github.com/huggingface/transformers) BERT model for text classification. If you want to take a look at the original model please search for [BertForSequenceClassification](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py). If you want to fine-tune or train a BERT classifier, you can either use their pipeline line or ours. Ours is tested against theirs before publishing. It is bug-free. To train a model, you can do something like this,
```bash
cd code/
CUDA_VISIBLE_DEVICES=0.1,2 python run_classifier.py \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--seed 123 \
--task_name SST5 \
--data_dir ../datasets/SST5/ \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--output_dir ../results/SST5-NewSeed/ \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin
```
Take a look at ``code/util/processor.py`` to see how we process different datasets. We currently supports almost 10 different dataset loadings. You can create your own within 1 minute for loading data. You can specify your directories info above in the command.

### Analyze, Attribution, Relevance and More
Once you have your model ready, save it to a location that you know (e.g., ``../results/SST5/checkpoint.bin``). Our example code how to get relevance scores is in a jupyter notebook format, which is much easier to read. This is how you will open it,
```bash
cd code/notebook/
jupyter notebook
```
Inside ``lrp_visualize``, we provide an example on how to get relevance scores! In short, it is really easy, when you create your BERT model, just provide an extra argument which enables a variety of gradient hooks, by setting ``init_lrp=True``. And then, in your evaluation loop, you can call ``model.backward_lrp(relevance_score)``.
