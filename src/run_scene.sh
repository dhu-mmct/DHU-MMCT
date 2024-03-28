#!/usr/bin/env bash


#先变量每个场景下的A
for ((i = 1; i <= 6; i++));
do
    echo "运行"${i}"A场景"
  python3 demo.py mot \
    --result_file /home/shuanghong/Downloads/github/dataset/fair_metrics/ts-dir/${i}A.txt \
    --input-video /home/shuanghong/Downloads/github/dataset/scene_copy/$i/A/jpg.mp4 \
    --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_scene/${i}/A/A.pickle \
#    --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result/${i}/A
done

for ((i = 1; i <= 6; i++))
do
      echo "运行"${i}"B场景"
    python3 demo.py mot \
  --result_file /home/shuanghong/Downloads/github/dataset/fair_metrics/ts-dir/${i}B.txt \
  --input-video /home/shuanghong/Downloads/github/dataset/scene_copy/$i/B/jpg.mp4 \
  --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_scene/${i}/B/B.pickle \
#  --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result/${i}/B
done

for ((i = 5; i <= 6; i++))
do
      echo "运行"${i}"C场景"
    python3 demo.py mot \
  --result_file /home/shuanghong/Downloads/github/dataset/fair_metrics/ts-dir/${i}C.txt \
  --input-video /home/shuanghong/Downloads/github/dataset/scene_copy/$i/C/jpg.mp4 \
  --embedding_result_filename /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_scene/${i}/C/C.pickle \
#  --video_result_file /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/video_result/${i}/C
done