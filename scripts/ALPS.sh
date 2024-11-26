# 代码目的：使用 accelerate 工具在多 GPU 和混合精度模式下运行长时预测任务的训练
model_name=ALPS
train_epochs=20
learning_rate=1e-4
llama_layers=12

batch_size=4     # 进一步减小批量大小
d_model=128        # 模型的维度
d_ff=256           # 前馈神经网络的维度

comment='ALPS-Alibaba'

# 设置环境变量以减少内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 使用 fp16 进行混合精度训练
accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision bf16 run_main.py \
  --is_training 1 \
  --model_id ALibaba_128_24 \
  --model $model_name \
  --data Alibaba \
  --features M \
  --seq_len 32\
  --label_len 0 \
  --pred_len 7 \
  --patch_len 30 \
  --factor 3 \
  --enc_in 100 \
  --dec_in 100 \
  --c_out 100 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
