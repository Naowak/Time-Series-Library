# export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model DT \
  --data MSL \
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
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 10 \
  --use_gpu 1 \
  --gpu_type cuda 