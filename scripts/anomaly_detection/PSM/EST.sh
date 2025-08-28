# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model EST \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --num_layers 2 \
  --memory_units 8 \
  --memory_dim 64 \
  --d_model 64 \
  --dropout 0 \
  --memory_connectivity 0.1 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --use_gpu 1 \
  --gpu_type cuda 