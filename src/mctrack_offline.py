'''
to do :
 看一下结果
 改进算法
 网络deepCC
 评估
'''

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from termcolor import colored
import time

import argparse
from tracker.matching import linear_assignment
import os
import pickle
import numpy as np
from numba import jit
from collections import deque
import torch
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from models import *
from tracker import matching
from tracking_utils import visualization as vis
import logging
import pickle
import scipy
import xlwt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment as sklearn_linear_assignment




def load_track_list(save_file_path):
    if os.path.getsize(save_file_path) > 0:
        with open(save_file_path, 'rb') as f:
            return pickle.load(f)


def normalize(x):
    norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
    return x / norm

#embedding average
def embedding_distance(tracks_a, tracks_b, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks_a), len(tracks_b)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    tracks_a_features = np.asarray([track.smooth_feat for track in tracks_a], dtype=np.float)
    tracks_b_features = np.asarray([track.smooth_feat for track in tracks_b], dtype=np.float)
    cost_matrix = np.maximum(0.0, scipy.spatial.distance.cdist(tracks_a_features, tracks_b_features))

    return cost_matrix


def _sum(deque):
    if len(deque) == 0:
        return None
    sum=0
    for i in deque:
        sum+=i
    return sum/len(deque)


def average_embedding_distance(tracks_a,tracks_b, metric='euclidean'):
    #求两个track之间的cost_matrix
    cost_matrix = np.zeros((len(tracks_a), len(tracks_b)), dtype=float)
    if cost_matrix.size==0:
        return cost_matrix
    #_sum(track.features)
    tracks_a_features=np.asarray([_sum(track.features) for track in tracks_a if _sum(track.features) is not None], dtype=float)
    tracks_b_features = np.asarray([_sum(track.features) for track in tracks_b if _sum(track.features) is not None], dtype=float)

    metric = 'euclidean'
    cost_matrix =  np.maximum(0.0, scipy.spatial.distance.cdist(tracks_a_features, tracks_b_features, metric))

    return cost_matrix

def average_reid_distance(tracks_a,tracks_b, metric='cosine'):
    #求两个track之间的cost_matrix
    cost_matrix = np.zeros((len(tracks_a), len(tracks_b)), dtype=float)
    if cost_matrix.size==0:
        return cost_matrix

    tracks_a_reid = np.asarray([_sum(track.reid) for track in tracks_a if _sum(track.reid) is not None], dtype=float)
    tracks_b_reid = np.asarray([_sum(track.reid) for track in tracks_b if  _sum(track.reid) is not None], dtype=float)

    metric='cosine'
    cost_matrix =  np.maximum(0.0, scipy.spatial.distance.cdist(tracks_a_reid,  tracks_b_reid, metric))
 
    return cost_matrix



# #简单Hungarian Algorithm
def multimatch(opt):


    path_A=opt.a
    path_C=opt.b

    tracks_A=load_track_list(path_A)
    tracks_C=load_track_list(path_C)

    #去除短的轨迹
    # tracks_A=[i for i in tracks_A if len(i.features)>5]
    # tracks_C=[i for i in tracks_C if len(i.features)>5]

    #dist即cost_matrix
    start = time.perf_counter()
    dists=average_reid_distance(tracks_A,tracks_C)

    # todo : average embedding
    matches, u_track, u_detection = linear_assignment(dists, thresh=0.4)# 1 before  #0.8  # 0.7 orignal
    end = time.perf_counter()
    dur = end - start
    # print(dur)
    k=0
    for i,j in matches:
        k+=1

        if opt.change == 1:
            print(1,',',2,',',tracks_A[i].track_id, ',',tracks_C[j].track_id)   #gai！！！
        elif opt.change == 2:
            print(1,',',3,',',tracks_A[i].track_id, ',',tracks_C[j].track_id)   #gai！！！


    #print(k)
#===========================输出图片序列===================================
    # k = 0

    # for i in tracks_C:
    #     for j in i.img0s:
    #         cv2.imwrite('./tmp/' + str(i) + str(k) + '.jpg', j)
    #         # print(str(i))
    #         k += 1
#===================================================================







def main():
    parser = argparse.ArgumentParser(prog='mctrack_offline.py')
    parser.add_argument('--a', type=str, default = '/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_new/4/A/A.pickle',help='')
    parser.add_argument('--b', type=str, default = '/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_new/4/B/B.pickle',help='')
    parser.add_argument('--change', type=int, help='')

    opt = parser.parse_args()
    multimatch(opt)


#根据不同相机的特征pickle文件，来得到pcd匹配文件
if __name__=="__main__":
    main()



