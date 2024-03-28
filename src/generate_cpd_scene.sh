#!/usr/bin/env bash


#先变量每个场景下的A
for ((i = 1; i <= 6; i++));
do
  if [ $i == 5 -o $i == 6 ]
  then
     echo "运行"${i}"场景"
  python3 mctrack_offline.py \
    --a ../embedding_scene/${i}/A/A.pickle \
    --b ../embedding_scene/${i}/B/B.pickle \
    --change 1  \
    > /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_scene/${i}/cpd.txt
  python3 mctrack_offline.py \
    --a ../embedding_scene/${i}/A/A.pickle \
    --b ../embedding_scene/${i}/C/C.pickle \
    --change 2 \
    >> /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_scene/${i}/cpd.txt

  else
     echo "运行"${i}"场景"
  python3 mctrack_offline.py \
    --a ../embedding_scene/${i}/A/A.pickle \
    --b ../embedding_scene/${i}/B/B.pickle \
    --change 1  \
    > /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_scene/${i}/cpd.txt
   fi
done





