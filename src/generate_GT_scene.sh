#!/usr/bin/env bash


for ((i = 1; i <= 6; i++));
do
  if [ $i == 5 -o $i == 6 ]
  then
    echo "运行"${i}"场景"
    python3 genmtmc.py \
      --ts /home/shuanghong/Downloads/github/dataset/fair_metrics/ts-dir/${i}/ts/ \
      --cpd /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_scene/${i}/cpd.txt \
      --out /home/shuanghong/Downloads/github/dataset/scene_copy/TS_icme_529_1.5/ \
      --filename S${i}.txt
  else
    echo "运行"${i}"场景"
    python3 genmtmc.py \
      --ts /home/shuanghong/Downloads/github/dataset/fair_metrics/ts-dir/${i}/ts/ \
      --cpd /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_scene/${i}/cpd.txt \
      --out /home/shuanghong/Downloads/github/dataset/scene_copy/TS_icme_529_1.5/ \
      --filename S${i}.txt
  fi
done