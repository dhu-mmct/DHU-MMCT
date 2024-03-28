#! /bin/bash
set -u
set -e

# 这个程序是为了py-motmetric而写的辅助程序。
# 这个程序可以生成motmetric所需要的gt-dir的文件目录结构
# 将所有的结果放在gt-dir下面,进入gt-dir,运行此程序
#mkdir gt-dir
#for i in *;do mkdir -p ${i:0:-4}/gt; done
#for i in *.txt;do mv $i ./${i:0:-4}/gt/gt.txt; done
#!/usr/bin/env bash


#先变量每个场景下的A
for ((i = 1; i <= 12; i++));
do
  mkdir ${i}
  cd ${i}
  mkdir A
  cd .. 
done

for ((i = 1; i <= 12; i++))
do
  if [ $i == 1 -o $i == 6 -o $i == 7 ]
  then
    echo ${i}"没有B场景"
  else
    cd ${i}
    mkdir B
    cd ..
  fi
done

for ((i = 1; i <= 12; i++))
do
  if [ $i == 2 ]
  then
    echo ${i}"没有C场景"
  else
    cd ${i}
    mkdir C
    cd .. 
  fi
done
 



