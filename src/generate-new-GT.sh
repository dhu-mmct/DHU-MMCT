#!/usr/bin/env bash


for ((i = 1; i <= 12; i++));
do
  if [ $i == 1 -o $i == 6 -o $i == 7 -o i == 2 ]
  then
    echo "运行"${i}"场景"
    python3 genmtmc.py \
      --ts ~/Downloads/github/dataset/label_transfer_server/${i}/gt/ \
      --cpd ~/Downloads/github/dataset/label_transfer_server/${i}/cpd.txt \
      --out ~/Downloads/github/dataset/label_transfer_server/GT/ \
      --filename S${i}.txt
  else
    echo "运行"${i}"场景"
    python3 genmtmc.py \
      --ts ~/Downloads/github/dataset/label_transfer_server/${i}/gt/  \
      --cpd ~/Downloads/github/dataset/label_transfer_server/${i}/cpd.txt \
      --out ~/Downloads/github/dataset/label_transfer_server/GT/ \
      --filename S${i}.txt
  fi
done