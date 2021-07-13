for ((i=0;i<1;i++));
do

python train.py \
--model_type bert \
--do_train \
--do_eval \
--do_eval_during_train \
--model_name_or_path E:\\PreTrainM\\bert-base-cased \
--data_dir ../data/data/K_clean_data/data_clean_$i \
--output_dir ../checkpoints/test/$i \
--max_seq_length 70 \
--learning_rate 5e-5 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 2 \
--num_train_epochs 1 \
--gradient_accumulation_steps 1

done
cmd /k




#python predict.py \
#--model_type albert \
#--max_seq_length 70 \
#--vote_model_paths ../checkpoints/albert_clean_70/ \
#--predict_file ../data/data/Test_clean.csv \
#--predict_result_file ../prediction_result/albert_clean_70_result.csv
#cmd /k

