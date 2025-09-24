# export CUDA_VISIBLE_DEVICES=0

model_name=EST

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 1 \
  --memory_units 4 \
  --memory_dim 128 \
  --d_model 64 \
  --dropout 0 \
  --memory_connectivity 0.05 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001



python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 1 \
  --memory_units 4 \
  --memory_dim 100 \
  --d_model 64 \
  --dropout 0 \
  --memory_connectivity 0.05 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 1 \
  --memory_units 16 \
  --memory_dim 64 \
  --d_model 64 \
  --dropout 0 \
  --memory_connectivity 0.125 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 1 \
  --memory_units 2 \
  --memory_dim 512 \
  --d_model 128 \
  --dropout 0 \
  --memory_connectivity 0.025 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001
