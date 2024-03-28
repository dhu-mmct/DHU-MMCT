#!/usr/bin/env bash


for ((i = 1; i <= 12; i++));
do
  if [ $i == 1 -o $i == 6 -o $i == 7 ]
  then
    echo "运行"${i}"场景"
    python3 mctrack_offline.py \
      --a ../embedding_reid/${i}/A/A.pickle \
      --b ../embedding_reid/${i}/C/C.pickle \
      --change 1  \
      > /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_reid/${i}/cpd.txt
  elif [ $i == 2 ]
  then
    echo "运行"${i}"场景"
    python3 mctrack_offline.py \
      --a ../embedding_reid/${i}/A/A.pickle \
      --b ../embedding_reid/${i}/B/B.pickle \
      --change 1 \
      > /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_reid/${i}/cpd.txt
  else
    echo "运行"${i}"场景"
    python3 mctrack_offline.py \
      --a ../embedding_reid/${i}/A/A.pickle \
      --b ../embedding_reid/${i}/B/B.pickle \
      --change 1 \
      > /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_reid/${i}/cpd.txt
    #添加重定向
    python3 mctrack_offline.py \
      --a ../embedding_reid/${i}/A/A.pickle \
      --b ../embedding_reid/${i}/C/C.pickle \
      --change 2 \
      >> /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_reid/${i}/cpd.txt
  fi
done


