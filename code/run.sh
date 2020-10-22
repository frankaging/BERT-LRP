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


CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 python run_classifier_sa.py \
--task_name Yelp5 \
--data_dir ../data/dataset/Yelp5/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 6 \
--output_dir ../results/Yelp5/ \
--seed 42 \
--init_checkpoint ../data/uncased_L-12_H-768_A-12/pytorch_model.bin