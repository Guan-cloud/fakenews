CURRENT_DIR=/media/lab1510/hb/T7/task7
#for ((i=0;i<1;i++));
#do
#
#python ../train_re.py \
#--model_type bert \
#--do_train \
#--do_eval \
#--do_eval_during_train \
#--model_name_or_path /home/lab1510/桌面/guan/Bert/task1/model/bert \
#--data_dir $CURRENT_DIR/data/data_$i \
#--output_dir $CURRENT_DIR/checkpoints/bert_re/$i \
#--max_seq_length 50 \
#--learning_rate 3e-5 \
#--per_gpu_train_batch_size 32 \
#--per_gpu_eval_batch_size 32 \
#--num_train_epochs 1 \
#--gradient_accumulation_steps 1 \
#--max 5 \
#--min 0
#
#done





#python ../predict_re.py \
#--model_type bert \
#--max_seq_length 50 \
#--vote_model_paths $CURRENT_DIR/checkpoints/test/ \
#--predict_file $CURRENT_DIR/data/data_0/dev.csv \
#--predict_result_file $CURRENT_DIR/prediction_result/bert_re_result.csv \
#--max 5 \
#--min 0

