# export CUDA_VISIBLE_DEVICES=0

model_name=EST

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 3 \
  --memory_units 4 \
  --memory_dim 32 \
  --d_model 64 \
  --dropout 0 \
  --memory_connectivity 0.25 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 3 \
  --memory_units 4 \
  --memory_dim 32 \
  --d_model 64 \
  --dropout 0 \
  --memory_connectivity 0.25 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001
