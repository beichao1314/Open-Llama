max_length = 1024
train_batch_size = 2
num_training_steps = 1000000
num_warmup_steps = 2000
initializer_range = 1e-2
lr = 2e-4
weight_decay = 1e-1
tokenizer_model_path = 'configs/10w_vocab_wudao5_pile10.model'
patterns = [
    'data/pretrain_data/part-*.jsonl.zst'
]
# global step
log_interval = 5
eval_interval = 200
save_interval = 800
work_dir = 'data/saved_ckpt/'