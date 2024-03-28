'''
此程序生mcmt的验证输入数据。
输入：
	1,多个相机的gt，或者是多个相机生成的test。
	2,一个相机之间的obj的对应。
	    camera_a camera_b a_id b_id

输出：
	单个mcmt验证文件，包含以下列。
	camera_id, obj_id, frame_id, xmin, ymin, width, heigth, xword(-1), yword(-1)
	其中，frame_id是指的当前视频中的frame.

用法：cat
	python genmcmt.py --ts dirpath-to-gts or dirpath-to-tests --cpd file --out path-to-outdir
	其中，ts中命名为camera_id.txt

Author: Syo
'''



from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd
from os.path import join, exists
import os

def solve(opt):
    # read gts to dictionary gts
    gts={}
    s=listdir(opt.ts)
    if not exists(opt.out): os.mkdir(opt.out)


    for i in s:
        if i == '.DS_Store' or i == '.1.txt.swp':
            continue
        fpath=opt.ts+str(i)
        df = pd.read_csv(
            fpath,
            index_col=None,
            skipinitialspace=True,
            header=None,
            names=[ 'FrameId','Id', 'X', 'Y', 'Width', 'Height', '_1', '_2','_3', '_4'],
            engine='python'
        )
        print(df)
        #gt_B.txt
        gts[int(str(i).split('.')[0])]=df

    #read cpd file
    cpd=pd.read_csv(opt.cpd,index_col=None,header=None,names=['camera_a','camera_b','a_id','b_id'],engine='python')
    print(cpd)

    for i in cpd.index:
        camera_a=cpd.loc[i]['camera_a']
        camera_b=cpd.loc[i]['camera_b']
        a_id=cpd.loc[i]['a_id']
        b_id=cpd.loc[i]['b_id']

        if camera_b < camera_a:
            cpd.loc[i]['camera_a']=camera_b
            cpd.loc[i]['camera_b']=camera_a
            cpd.loc[i]['a_id']=b_id
            cpd.loc[i]['b_id']=a_id

    #update id
    # 先解决相机之中的id冲突问题。
    # 对于每个相机中的id，全部重新发行。
    # max_id=0
    # for camera in cameras:
    #     对所有的id，加上max_id
    #     用最大的id,更新max_id
    # 读取cpd文件
    # 对于每一个匹配，更新更大的camera_id的列
    cameras=[]
    for i in gts.keys():
        cameras.append(i)
    cameras.sort()

    max_id=0
    for i in cameras:
        # first, update cpd
        cpd.loc[cpd['camera_a']==i,'a_id']+=max_id
        cpd.loc[cpd['camera_b']==i,'b_id']+=max_id

        # second, update gts
        gts[i]['Id']+=max_id
        max_id=gts[i]['Id'].max()

    for i in cpd.index:
        camera_a=cpd.loc[i]['camera_a']
        camera_b=cpd.loc[i]['camera_b']
        a_id=cpd.loc[i]['a_id']
        b_id=cpd.loc[i]['b_id']

        gts[camera_b].loc[gts[camera_b]['Id']==b_id,'Id']=a_id



    for i in cameras:
        gts[i]['camera_id']=i
        gts[i]['wx']=-1
        gts[i]['wy']=-1
    out= pd.concat([gts[i][['camera_id','Id','FrameId', 'X', 'Y', 'Width', 'Height','wx','wy']] for i in cameras],
                   axis=0)
    print(out)
    out.to_csv(opt.out + opt.filename,index=False,header=False)
    # out.to_csv(opt.out + 'GT.txt', index=False, header=False)



#python genmcmt.py --ts dirpath-to-gts or dirpath-to-tests --cpd file --out path-to-outdir
#--ts /home/shuanghong/Downloads/github/dataset/scene/1/
#--cpd /home/shuanghong/Downloads/github/dataset/scene/1/fair_pcd.txt
#--out /home/shuanghong/Downloads/github/dataset/scene/1/

#python genmtmc.py --ts /home/shuanghong/Downloads/github/dataset/new_dataset_fair_metrics/gt-dir/ts/ --cpd /home/shuanghong/Downloads/github/dataset/scene_version_2/data/label/3/cpd.txt --out /home/shuanghong/Downloads/github/dataset/scene_version_2/data/label/3/

#--ts /home/shuanghong/Downloads/github/dataset/scene/6/ts/ --cpd /home/shuanghong/Downloads/github/dataset/scene/6/pcd_pose.txt --out /home/shuanghong/Downloads/github/dataset/scene/6/
# --ts /home/shuanghong/Downloads/github/dataset/old-dataset-result-wqx/old-jde-v2/ts --cpd /home/shuanghong/Downloads/github/project/strongsortPure/AFLink/cpd/3.txt --out /home/shuanghong/Downloads/github/dataset/old-dataset-result-wqx/old-jde-v2/TS/

#/home/shuanghong/Downloads/github/FairMOT/demo_result/2B_demo.txt
#/home/shuanghong/Downloads/github/FairMOT/demo_result/1A_demo.txt
#注意命名问题

#如/6/gts/下存的是demo.txt改名后的1.txt
#(FairMOT) shuanghong@dhu-Precision-7920-Tower:~/Downloads/github/dataset/scene/6/gts$ ls
#1.txt  2.txt  3.txt
#cp /home/shuanghong/Downloads/github/dataset/metrics/ts-dir/4A.txt ts/1.txt
#cp /home/shuanghong/Downloads/github/dataset/metrics/ts-dir/4B.txt ts_crop/2.txt

#scp -r shuanghong@10.199.154.162:/home/shuanghong/Downloads/github/dataset/scene/2/GT_ts_spatial.txt /Users/wsh/PycharmProjects/pythonProject/eval_dir/2



if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='genmtmc.py')
    parser.add_argument('--ts',type=str,default="./gts/",help="gts' or tests' directory path")
    parser.add_argument('--cpd',type=str,default="./cpd.txt")
    parser.add_argument('--out',type=str,default="./")
    parser.add_argument('--filename',type=str,default="./")


    opt=parser.parse_args()
    print(opt)
    solve(opt)

# python genmtmc.py --ts ~/Downloads/github/dataset/label_transfer_server/2/gt/ --cpd ~/Downloads/github/dataset/label_transfer_server/${i}/cpd.txt --out ~/Downloads/github/dataset/label_transfer_server/
