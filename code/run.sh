# This is a template for training the first time
CUDA_VISIBLE_DEVICES=7,8,9 python run_classifier.py \
--task_name SST5 \
--data_dir ../datasets/SST5/ \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--eval_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6 \
--output_dir ../results/SST5/ \
--seed 42 \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin