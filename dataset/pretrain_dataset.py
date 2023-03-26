'''
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-17 20:41:25
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-26 23:07:56
FilePath: /Open-Llama/dataset/pretrain_dataset.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
'''
import math
import torch

def preprocess_wudao_gen(tokenizer, segment_max_length=1024):
    def preprocess_wudao(line):
        '''
        The format of the data is roughly as follows.
        {'id': 1, 'dataType': '百科', 'title': 'some title', 'content': 'some content'}
        Split the data based on the tokenized length according to the maximum length.
        '''
        total = line['title'] + '\n' + line['content']
        out = tokenizer(total)
        input_ids = out['input_ids']
        return [input_ids[i*segment_max_length: (i+1)*segment_max_length] 
        for i in range(math.ceil(len(input_ids)/segment_max_length))]
    return preprocess_wudao

def preprocess_the_pile_gen(tokenizer, segment_max_length=1024):
    def preprocess_the_pile(line):
        '''
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        '''
        total = line['text']
        out = tokenizer(total)
        input_ids = out['input_ids']
        return [input_ids[i*segment_max_length: (i+1)*segment_max_length] 
        for i in range(math.ceil(len(input_ids)/segment_max_length))]
    return preprocess_the_pile

def pretrain_collate_fn_gen(tokenizer, segment_max_length=1024):
    '''
    Organize data into tensors by padding based on the preset maximum length.
    '''
    pad_id = tokenizer.pad_id
    def pretrain_collate_fn(batch):
        input_ids = []
        for i in batch:
            input_len = len(i)
            input_ids.append(i+[pad_id]*(segment_max_length-input_len))
        inputs = {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
        }
        return inputs
    return pretrain_collate_fn

if __name__ == '__main__':
    import sentencepiece as spm
    from datasets import IterableDataset
    from torch.utils.data import DataLoader

    from dataset.tokenizer import Tokenizer
    from dataset.data_iter import create_shard_kwargs, create_data_iter
    
    sp_model = spm.SentencePieceProcessor(model_file='configs/10w_vocab_wudao5_pile10.model')
    tokenizer = Tokenizer(sp_model)
    patterns = [
        'data/pretrain_data/part-*.jsonl.zst'
    ]
    paths = create_shard_kwargs(patterns)
    transform_dict = {
        'wudao': preprocess_wudao_gen(tokenizer), 
        'pile': preprocess_the_pile_gen(tokenizer)
    }
    data_set = IterableDataset.from_generator(create_data_iter, gen_kwargs={'paths': paths, 'transform_dict': transform_dict})
    train_loader = DataLoader(data_set, batch_size=8, num_workers=4, 
    collate_fn=pretrain_collate_fn_gen(tokenizer), drop_last=True)
    for batch in train_loader:
        for k, v in batch.items():
            print(k, v.shape)
        break