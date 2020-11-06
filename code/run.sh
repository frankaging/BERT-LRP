# This is a template for training the first time
CUDA_VISIBLE_DEVICES=0,4,5,6,7,8 python run_classifier.py \
--task_name SemEval \
--data_dir ../datasets/SemEval/ \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--model_type BERTPretrain \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 48 \
--eval_batch_size 48 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir ../results/SemEval/ \
--seed 42 \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin