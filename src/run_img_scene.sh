#!/usr/bin/env bash


#先变量每个场景下的A
for ((i = 1; i <= 6; i++));
do
  echo "运行"${i}"A场景"
  python3 demo_img.py mot \
    --result_file /mnt/disk/shuanghong/dataset/fair_metrics/ts-dir/${i}A.txt \
    --input_img /mnt/disk/shuanghong/dataset/scene_copy/${i}/A \
    --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_scene/${i}/A/A.pickle \
    --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result_scene/${i}/A
done
#
for ((i = 1; i <= 6; i++));
do
  echo "运行"${i}"B场景"
  python3 demo_img.py mot \
    --result_file /mnt/disk/shuanghong/dataset/fair_metrics/ts-dir/${i}B.txt \
    --input_img /mnt/disk/shuanghong/dataset/scene_copy/${i}/B \
    --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_scene/${i}/B/B.pickle \
    --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result_scene/${i}/B
done

for ((i = 5; i <= 6; i++))
do
    echo "运行"${i}"C场景"
    python3 demo_img.py mot \
  --result_file /mnt/disk/shuanghong/dataset/fair_metrics/ts-dir/${i}A.txt \
  --input_img /mnt/disk/shuanghong/dataset/scene_copy/${i}/C\
  --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_scene/${i}/C/C.pickle \
  --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result_scene/${i}/C
done