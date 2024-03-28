#!/usr/bin/env bash


#先变量每个场景下的A
for ((i = 1; i <= 10; i++));
do
  echo "运行"${i}"A场景"
  python3 demo_img.py mot \
    --result_file /mnt/disk/shuanghong/dataset/new_dataset_fair_metrics/ts-dir-img/${i}A.txt \
    --input_img /mnt/disk/shuanghong/dataset/scene_version_2/${i}/A/jpg \
    --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/A/A.pickle \
    --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result/${i}/A
done
#
for ((i = 1; i <= 12; i++))
do
  if [ $i == 1 -o $i == 6 -o $i == 7 ]
  then
    echo ${i}"没有B场景"
  else
      echo "运行"${i}"B场景"
      python3 demo_img.py mot \
    --result_file /mnt/disk/shuanghong/dataset/new_dataset_fair_metrics/ts-dir-img/${i}B.txt \
    --input_img /mnt/disk/shuanghong/dataset/scene_version_2/${i}/B/jpg \
    --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/B/B.pickle \
    --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result/${i}/B
  fi
done

for ((i = 1; i <= 12; i++))
do
  if [ $i == 2 ]
  then
    echo ${i}"没有C场景"
  else
      echo "运行"${i}"C场景"
      python3 demo_img.py mot \
    --result_file /mnt/disk/shuanghong/dataset/new_dataset_fair_metrics/ts-dir-img/${i}C.txt \
    --input_img /mnt/disk/shuanghong/dataset/scene_version_2/${i}/C/jpg \
    --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/C/C.pickle \
    --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result/${i}/C
  fi
done