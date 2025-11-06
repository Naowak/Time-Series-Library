# export CUDA_VISIBLE_DEVICES=2

model_name=DT

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --num_layers 2 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 32 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --num_layers 2 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 32 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --num_layers 2 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 32 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --num_layers 2 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 32 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1