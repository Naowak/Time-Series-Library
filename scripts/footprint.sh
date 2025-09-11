python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model EST \
    --data custom \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --num_layers 1 \
    --memory_units 4 \
    --memory_dim 64 \
    --d_model 192 \
    --dropout 0 \
    --memory_connectivity 0.125 \
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
  --model_id ECL_96_96 \
  --model Transformer \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
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
  --model_id ECL_96_96 \
  --model Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 321 \
  --expand 2 \
  --d_ff 256 \
  --d_conv 4 \
  --c_out 321 \
  --d_model 192 \
  --des 'Exp' \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model Reformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_model 280 \
  --n_heads 8 \
  --d_ff 512 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 256 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 32 \
  --num_kernels 6 \
  --d_ff 28 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1





































