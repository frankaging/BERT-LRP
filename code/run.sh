# This is a template for training the first time
CUDA_VISIBLE_DEVICES=0,5,6 python run_classifier.py \
--task_name Yelp5 \
--data_dir ../datasets/Yelp5/ \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 18 \
--eval_batch_size 18 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir ../results/Yelp5/ \
--seed 42 \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin