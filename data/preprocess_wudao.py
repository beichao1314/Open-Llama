'''
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-16 22:10:44
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-26 22:59:55
FilePath: /Open-Llama/data/preprocess_wudao.py
Description: 
Parse the dataset from the raw files and split them into different jsonl files based on the preset maximum number of lines, 
making it easy for parallel training to perform streaming reads.
Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
'''
import json
from glob import glob
from tqdm import tqdm
import zstandard as zstd

paths = glob('data/WuDaoCorpus2.0_base_200G/part*')
write_path = 'data/pretrain_data/part-wudao-{}.jsonl.zst'
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), 'wb', encoding='utf-8')
for path in tqdm(paths, total=len(paths)):
    with open(path, 'r') as fp:
        data = json.load(fp)
    for line in data:
        if total_num % 16384 == 0 and total_num > 0:
            file_num += 1
            wfp.close()
            wfp = zstd.open(write_path.format(file_num), 'wb', encoding='utf-8')
        wfp.write(json.dumps(line).encode('utf-8'))
        wfp.write('\n'.encode('utf-8'))
        total_num += 1
wfp.close()
print('total line: {}\ntotal files: {}'.format(total_num, file_num))