model_name=Mamba

for pred_len in 96 192 336 720
# for pred_len in 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$pred_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $pred_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 321 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 321 \
  --d_model 16 \
  --des 'Exp' \
  --itr 1 \

done
