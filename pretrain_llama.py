'''
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-17 14:27:28
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-27 01:07:25
FilePath: /Open-Llama/pretrain_llama.py
Description: 
pretrain GPT
Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
'''
import os
import time
import wandb
import torch
import random
import sentencepiece as spm
from torchinfo import summary
from accelerate import Accelerator
from datasets import IterableDataset
from torch.utils.data import DataLoader
from deepspeed.ops.adam import FusedAdam
from transformers import LlamaForCausalLM, LlamaConfig, get_cosine_schedule_with_warmup

from dataset.validation import val_set
from dataset.tokenizer import Tokenizer
from dataset.data_iter import create_shard_kwargs, create_data_iter
from dataset.pretrain_dataset import preprocess_the_pile_gen, preprocess_wudao_gen, pretrain_collate_fn_gen
from configs.train_config import *

accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(
        project='LLAMA Pretrain'
    )

log_interval *= accelerator.gradient_accumulation_steps
eval_interval *= accelerator.gradient_accumulation_steps
save_interval *= accelerator.gradient_accumulation_steps

sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
tokenizer = Tokenizer(sp_model)

paths = create_shard_kwargs(patterns)
random.shuffle(paths)
transform_dict = {
    'wudao': preprocess_wudao_gen(tokenizer, max_length), 
    'pile': preprocess_the_pile_gen(tokenizer, max_length)
}
data_set = IterableDataset.from_generator(create_data_iter, gen_kwargs={
    'paths': paths, 
    'transform_dict': transform_dict,
    'process_index': accelerator.process_index, 
    'num_processes': accelerator.num_processes
})
train_loader = DataLoader(data_set, batch_size=train_batch_size, num_workers=1, 
collate_fn=pretrain_collate_fn_gen(tokenizer, max_length), drop_last=True)
# smaller initializer_range make training more stable
# add stabel embedding to token embedding
raw_model = LlamaForCausalLM(LlamaConfig(vocab_size=tokenizer.vocab_size, 
                                         initializer_range=initializer_range, 
                                         pad_token_id=tokenizer.pad_id, 
                                         rms_norm_eps=1e-5, 
                                         hidden_dropout_prob=0.1, 
                                         attention_dropout_prob=0.1, 
                                         use_stable_embedding=True, 
                                         shared_input_output_embedding=True))
raw_model.eval()
with torch.no_grad():
    summary(raw_model.cuda(), input_data=torch.ones(1, 64, dtype=torch.int64).cuda())
no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in raw_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in raw_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optim = FusedAdam(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95))
optim.zero_grad()
factor = accelerator.num_processes / accelerator.gradient_accumulation_steps
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps * factor, 
                                            num_training_steps=num_training_steps * factor)

_, model, optim, scheduler = accelerator.prepare(
    train_loader, raw_model, optim, scheduler
)
print('start training...')
train_loader_iter = iter(train_loader)
global_step = 0
start_time = time.time()
for data_step in range(num_training_steps):
    model.train()
    with accelerator.accumulate(model):
        batch = next(train_loader_iter)
        for k, v in batch.items():
            batch[k] = v.to(accelerator.device)
        labels = batch['input_ids'].clone()
        labels[labels==tokenizer.pad_id] = -100
        out = model(**batch, labels=labels)
        total_loss = out.loss
        losses = {
            'total_loss': total_loss
        }
        accelerator.backward(total_loss)
        optim.step()
        scheduler.step()
        optim.zero_grad()
        if accelerator.sync_gradients:
            global_step += 1
    if data_step % log_interval == 0 and data_step > 0 and accelerator.is_main_process:
        cost_time = time.time() - start_time
        start_time = time.time()
        tokens = train_batch_size * log_interval * max_length
        wandb.log({'Training/Token per second per gpu': tokens/cost_time})
        for k, v in losses.items():
            wandb.log({'Losses/{}'.format(k): v})
        current_lr = optim.param_groups[0]['lr']
        wandb.log({'Training/LR': current_lr})
        if optim.scaler is not None:
            wandb.log({'Training/Loss Scale': optim.scaler.get_scale()})
        wandb.log({'Training/Data Step': data_step})
        wandb.log({'Training/Global Step': global_step})
        accelerator.print('Global Step: {}, Data Step: {}, Loss: {}, Token per second per gpu: {}'.format(
            global_step, data_step, losses['total_loss'], tokens/cost_time))
    if data_step % eval_interval == 0 and accelerator.is_main_process:
        text_table = wandb.Table(columns=['question', 'pred'])
        model.eval()
        with torch.no_grad():
            for data in val_set:
                raw_inputs = data
                inputs_len = len(raw_inputs)
                inputs = tokenizer(raw_inputs, return_tensors=True, add_special_tokens=False)
                for k, v in inputs.items():
                    inputs[k] = v.to(accelerator.device)
                pred = model.generate(**inputs, max_new_tokens=256, do_sample=True, repetition_penalty=2.0)
                pred = tokenizer.decode(pred.cpu())[0]
                pred = pred[inputs_len:]
                text_table.add_data(raw_inputs, pred)
        wandb.log({'Predictions on {}'.format(global_step) : text_table})
    if data_step % save_interval == 0 and data_step > 0 and accelerator.is_main_process:
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
        torch.save(raw_model.state_dict(), '{}/{}.pt'.format(work_dir, global_step))
wandb.finish()
