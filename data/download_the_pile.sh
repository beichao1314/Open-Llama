#!/bin/bash
###
 # @Author: LiangSong(sl12160010@gmail.com)
 # @Date: 2023-03-16 21:21:38
 # @LastEditors: LiangSong(sl12160010@gmail.com)
 # @LastEditTime: 2023-03-26 22:58:02
 # @FilePath: /Open-Llama/data/download_the_pile.sh
 # @Description: 
 # download the pile dataset and preprocess
 # Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
### 
start=0
end=29
mkdir data/the_pile
for (( i=$start; i<=$end; i++ ))
do
    url="https://the-eye.eu/public/AI/pile/train/$(printf "%02d" $i).jsonl.zst"
    echo "Downloading file: $url"
    curl -C - $url -o data/the_pile/"$(printf "%02d" $i).jsonl.zst"
done

wait

echo "All files downloaded successfully."
mkdir data/pretrain_data
python3 data/preprocess_the_pile.py