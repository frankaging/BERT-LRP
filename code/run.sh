# This is a template for training the first time
CUDA_VISIBLE_DEVICES=4,5,7,8 python run_classifier_sa.py \
--task_name IMDb \
--data_dir ../data/dataset/IMDb/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--eval_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6 \
--output_dir ../results/IMDb/ \
--seed 42 \
--init_checkpoint ../data/uncased_L-12_H-768_A-12/pytorch_model.bin
--load_checkpoint_model

# This is a template for running LRP analysis
CUDA_VISIBLE_DEVICES=0 python run_lrp_bert.py \
--task_name SST5 \
--data_dir ../data/dataset/SST/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type BERTPretrain \
--do_lower_case \
--max_seq_length 512 \
--eval_batch_size 5 \
--seed 42 \
--output_dir ../results/SST5/ \
--init_checkpoint ../results/SST5/checkpoint.bin