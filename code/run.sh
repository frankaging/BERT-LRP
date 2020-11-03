# This is a template for training the first time
CUDA_VISIBLE_DEVICES=1,2,4 python run_classifier.py \
--task_name AdvSA \
--data_dir ../data/dataset/AdvSA/ \
--vocab_file ../data/BERT-Google/vocab.txt \
--bert_config_file ../data/BERT-Google/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--eval_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6 \
--output_dir ../results/AdvSA/ \
--seed 42 \
--init_checkpoint ../data/BERT-Google/pytorch_model.bin