# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model EST \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --num_layers 4 \
  --memory_units 2 \
  --memory_dim 64 \
  --d_model 128 \
  --dropout 0 \
  --memory_connectivity 0.125 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 10 \
  --use_gpu 1 \
  --gpu_type cuda 