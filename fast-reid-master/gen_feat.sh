#!/usr/bin/env bash


#先变量每个场景下的A
for ((i = 1; i <= 12; i++));
do
  if [ $i == 3 -o $i == 4 -o $i == 5 -o $i == 8 -o $i == 9 -o $i == 10 -o $i == 11 -o $i == 12 ]
  then
    echo "运行"${i}"场景(A,B,C)"
    python3 gen_feat.py \
      --a /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/A/A.pickle \
      --b /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/B/B.pickle \
      --c /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/C/C.pickle \
      --embedding_result_filename_A /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/A/A.pickle \
      --embedding_result_filename_B /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/B/B.pickle \
      --embedding_result_filename_C /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/C/C.pickle \
      --crop_dir /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/crop_reid
  fi
done

for ((i = 1; i <= 12; i++))
do
  if [ $i == 1 -o $i == 6 -o $i == 7 ]
  then
    echo "运行"${i}"场景（A,C）"
    python3 gen_feat.py \
    --a /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/A/A.pickle \
    --c /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/C/C.pickle \
    --embedding_result_filename_A /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/A/A.pickle \
    --embedding_result_filename_C /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/C/C.pickle \
    --crop_dir /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/crop_reid

  fi
done

for ((i = 1; i <= 12; i++))
do
  if [ $i == 2 ]
  then
    echo "运行"${i}"场景（A,B）"
    python3 gen_feat.py \
    --a /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/A/A.pickle \
    --b /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_img/${i}/B/B.pickle \
    --embedding_result_filename_A /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/A/A.pickle \
    --embedding_result_filename_B /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_reid/${i}/B/B.pickle \
    --crop_dir /home/shuanghong/Downloads/github/project/FairMOT_new_dataset/crop_reid

  fi
done