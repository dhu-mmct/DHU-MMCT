#!/usr/bin/env bash


for ((i = 1; i <= 12; i++));
do
  if [ $i == 1 -o $i == 6 -o $i == 7 -o i == 2 ]
  then
    echo "运行"${i}"场景"
    python3 genmtmc.py \
      --ts /mnt/disk/shuanghong/dataset/new_dataset_fair_metrics/ts-dir-img/${i}/ts/ \
      --cpd /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_reid/${i}/cpd.txt \
      --out /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/TS/ \
      --filename S${i}.txt
  else
    echo "运行"${i}"场景"
    python3 genmtmc.py \
      --ts /mnt/disk/shuanghong/dataset/new_dataset_fair_metrics/ts-dir-img/${i}/ts/ \
      --cpd /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_reid/${i}/cpd.txt \
      --out /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/TS/ \
      --filename S${i}.txt
  fi
done