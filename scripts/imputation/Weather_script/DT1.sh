# export CUDA_VISIBLE_DEVICES=0

model_name=DT

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
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
  --model_id weather_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001
