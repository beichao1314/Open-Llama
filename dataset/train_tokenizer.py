'''
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-24 20:49:03
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-26 23:43:59
FilePath: /Open-Llama/dataset/train_tokenizer.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
'''
import random
from dataset.data_iter import create_data_iter, create_shard_kwargs

wudao_patterns = [
    'data/pretrain_data/part-wudao-*.jsonl.zst',
]
wudao_paths = create_shard_kwargs(wudao_patterns)
random.shuffle(wudao_paths)

pile_patterns = [
    'data/pretrain_data/part-pile-*.jsonl.zst',
]
pile_paths = create_shard_kwargs(pile_patterns)
random.shuffle(pile_paths)
paths = wudao_paths[: 5] + pile_paths[: 10]
transform_dict = {
    'wudao': lambda line: [(line['title'] + '\n' + line['content'])],
    'pile': lambda line: [line['text']]
}
data_iter = create_data_iter(paths, transform_dict)

import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
spm.SentencePieceTrainer.train(
  sentence_iterator=data_iter, model_writer=model, shuffle_input_sentence=False, train_extremely_large_corpus=True, 
  # hyperparameters of tokenizer
  max_sentence_length=16384, pad_id=3, model_type='BPE', vocab_size=100000, 
  # split digits and fallback to byte same as Llama. 
  # set split_by_unicode_script to True to avoid grouping punctuation and characters together.
  split_digits=True, split_by_unicode_script=True, byte_fallback=True,
  # reserve whitespace and \n and \t etc. for code generation
  allow_whitespace_only_pieces=True, remove_extra_whitespaces=False, normalization_rule_name='nfkc')

# Serialize the model as file.
with open('configs/10w_vocab_wudao5_pile10.model', 'wb') as f:
  f.write(model.getvalue())

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.decode(sp.encode('只因你太美🤗▃     \n  1')))