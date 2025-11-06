# export CUDA_VISIBLE_DEVICES=1

model_name=DT

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --num_layers 4 \
  --memory_units 4 \
  --memory_dim 64 \
  --d_model 64 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'