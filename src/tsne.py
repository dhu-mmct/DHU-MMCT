'''This file discribe the distance of person's embeddings.
'''

import _init_paths
import os
import pickle
from  tracker.matching import linear_assignment
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
import tracker.multitracker
# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold._t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state.
RS = 20300101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns

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


def average_embedding_distance(tracks_a,tracks_b,metric='cosine'):
    cost_matrix = np.zeros((len(tracks_a), len(tracks_b)), dtype=np.float)
    if cost_matrix.size==0:
        return cost_matrix
    tracks_a_features=np.asarray([_sum(track.features) for track in tracks_a if _sum(track.features) is not None], dtype=np.float)
    tracks_b_features = np.asarray([_sum(track.features) for track in tracks_b if _sum(track.features) is not None], dtype=np.float)
    cost_matrix =  np.maximum(0.0, scipy.spatial.distance.cdist(tracks_a_features, tracks_b_features))
    return cost_matrix


# 把一个相机的某一个人的reid特征和另一个相机所有的id的特征比一下
def average_embedding_tracks_distance(embedding, tracks, metric='euclidean'):
    # 求两个track之间的cost_matrix
    cost_matrix = np.zeros((1, len(tracks_b)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    # _sum(track.features)
    embedding_features = np.asarray([_sum(track.features) for track in embedding if _sum(track.features) is not None], dtype=np.float)
    tracks_b_features = np.asarray([_sum(track.features) for track in tracks if _sum(track.features) is not None],
                                   dtype=float)
    metric = 'euclidean'
    cost_matrix = np.maximum(0.0, scipy.spatial.distance.cdist(embedding_features, tracks_b_features, metric))
    return cost_matrix



def _sum(deque):
    if len(deque) == 0:
        return None
    sum=0
    for i in deque:
        sum+=i
    return sum/len(deque)

def load_track_list(save_file_path):
    if os.path.getsize(save_file_path) > 0:
        with open(save_file_path, 'rb') as f:
            return pickle.load(f)

if __name__=='__main__':
    path_A='../embedding/10/A/A.pickle'
    path_B='../embedding/10/B/B.pickle'
    path_C='../embedding/10/C/C.pickle'

    tracks_A=load_track_list(path_A)
    tracks_B=load_track_list(path_B)
    tracks_C=load_track_list(path_C)

    tracks_a=[i for i in tracks_A if len(i.features)>29]
    tracks_b=[i for i in tracks_B if len(i.features)>29]
    tracks_c=[i for i in tracks_C if len(i.features)>29]


    '''
    for i in tracks_a:
        if i.track_id == 2:
            list_a = [i]
            dist = average_embedding_tracks_distance(list_a, tracks_b)
    '''

#
#cam1   cam2    cam3
# 5      112     149
# 423    629      131
# 473    455      4
# 899    402      21

    A_a=[]  #store features
    A_b=[]  #store id
    for i in tracks_a:
        id=i.track_id
        if id in [5,2,246,422,423,473,457,402,491,497,508,522,846,915,899,514,524,525]:
            for j in i.features:
                if len(j)!=0:
                    A_a.append(j)
                    A_b.append(id)
    #a is used to store features of each person,
    # b to store corresponding id.so the len(a) = sum_i(len(featurns))(i = 1...len(tracks_a))

#2B 66, 77, 75, 11, 19, 2, 5, 9, 1, 4, 42, 36, 47, 37
    B_a = []
    B_b = []
    for i in tracks_b:
        id=i.track_id
        if id in [112,307,205,311,629,455,494,217,320,373,664,303,317,325,402]:
            for j in i.features:
                if len(j)!=0:
                    B_a.append(j)
                    B_b.append(id)

    C_a = []
    C_b = []
    for i in tracks_c:
        id=i.track_id
        if id in [149,131,4,58,41,34,207,21]:
            for j in i.features:
                if len(j)!=0:
                    C_a.append(j)
                    C_b.append(id)




    #transform 128 dim to 2 dim
    tsne_A = TSNE(random_state=RS).fit_transform(A_a)
    tsne_B = TSNE(random_state=RS).fit_transform(B_a)
    tsne_C = TSNE(random_state=RS).fit_transform(C_a)


    #设成40吧
    palette = np.array(sns.color_palette("hls", 25))
    #fig(640 * 480)
    fig, ax = plt.subplots()


    n=-1
    p=[]
    n_id=0
    for i in range(len(A_b)): #len(b) is the total num of features of all people
        if n!=A_b[i]:
            if len(p)!=0:
                p=np.array(p)
                ax.scatter(p[:,0],p[:,1],c=palette[n_id],alpha=0.7,label=n_id,marker='^')  #,marker='x'
                n_id+=1
            n=A_b[i]
            p=[]
        p.append(tsne_A[i])
    #ax.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    #plt.show()

    n = -1
    p = []
    n_id = 0
    for i in range(len(B_b)):  # len(b) is the total num of features of all peoplw
        if n != B_b[i]:
            if len(p) != 0:
                p = np.array(p)
                ax.scatter(p[:, 0], p[:, 1], c=palette[n_id], alpha=0.7, label=n_id,marker='s')  # ,marker='x'
                n_id += 1
            n = B_b[i]
            p = []
        p.append(tsne_B[i])


    n = -1
    p = []
    n_id = 0
    for i in range(len(C_b)):  # len(b) is the total num of features of all peoplw
        if n != C_b[i]:
            if len(p) != 0:
                p = np.array(p)
                ax.scatter(p[:, 0], p[:, 1], c=palette[n_id], alpha=0.7, label=n_id,marker='o')  # ,marker='x'
                n_id += 1
            n = C_b[i]
            p = []
        p.append(tsne_C[i])






    plt.xticks([])
    plt.yticks([])
    plt.grid(True,axis='x')
    fig.savefig('../tsne/tsne_10_same_dif.png')
    print(set(A_b))
    print("--------")
    print(set(B_b))
    print("--------")
    print(set(C_b))


'''
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)
'''