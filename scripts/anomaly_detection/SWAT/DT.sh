# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWAT \
  --model_id SWAT \
  --model DT \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --num_layers 2 \
  --memory_units 4 \
  --memory_dim 32 \
  --d_model 32 \
  --n_heads 4 \
  --dropout 0.1 \
  --memory_connectivity 0.2 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --use_gpu 1 \
  --gpu_type cuda 