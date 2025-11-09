for num in 2000
do
  deepspeed --master_port=4006 \
  train_ptune.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path ./data/summary/xsum_train_${num}.json \
  --output_dir saved_models/summary/xsum_${num} \
  --report_to none \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --save_steps 50 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0 \
  --lr_scheduler_type "linear" \
  --logging_steps 1 \
  --bf16 True \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,k_proj
done

