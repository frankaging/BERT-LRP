# This is using hugging face transformer to baseline the pipeline
CUDA_VISIBLE_DEVICES=3,4,5 python run_glue.py \
--model_name_or_path bert-base-uncased \
--task_name SST2 \
--do_train \
--do_eval \
--max_seq_length 128 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--output_dir ../results/SST2Baseline/ \
--evaluate_during_training \
--load_best_model_at_end

# This is for transformer-like training without any pretrained model
CUDA_VISIBLE_DEVICES=7,8,9 python run_classifier.py \
--task_name SST5 \
--data_dir ../data/dataset/SST/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--model_type BERT \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 512 \
--eval_batch_size 512 \
--learning_rate 1e-5 \
--num_train_epochs 100 \
--output_dir ../results/SST-Transformer/ \
--seed 42

# This is a template for training the first time
CUDA_VISIBLE_DEVICES=3,4,5 python run_classifier.py \
--task_name AdvSA \
--data_dir ../data/dataset/AdvSA/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 36 \
--eval_batch_size 36 \
--learning_rate 2e-5 \
--num_train_epochs 20 \
--output_dir ../results/AdvSA/ \
--seed 42 \
--init_checkpoint ../data/uncased_L-12_H-768_A-12/pytorch_model.bin

# This is a template for running LRP analysis
CUDA_VISIBLE_DEVICES=7 python run_lrp_bert.py \
--task_name SemEval \
--data_dir ../data/dataset/SemEval/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type BERTPretrain \
--do_lower_case \
--max_seq_length 512 \
--eval_batch_size 1 \
--seed 42 \
--output_dir ../results/SemEval/ \
--init_checkpoint ../results/SemEval/checkpoint.bin \
--eval_size 2000 \
--no_cuda