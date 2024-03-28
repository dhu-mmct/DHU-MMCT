from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets

from track import eval_seq

from tracker.multitracker import STrack
from tracker.multitracker import joint_stracks

import pickle
from termcolor import colored
import cv2
import numpy as np





logger.setLevel(logging.INFO)

def load_track_list(save_file_path):
    if os.path.getsize(save_file_path) > 0:
        with open(save_file_path, 'rb') as f:
            return pickle.load(f)


#convert -resize 200x200 * +append ../append_img/6A_output_22.jpg

if __name__=='__main__':
    path_A='/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding/6/A/A.pickle'
    path_B='/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding/6/C/C.pickle'

    # path_A='/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_transfer/10/A/A.pickle'
    # path_B='/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_transfer/10/B/B.pickle'
    # path_C ='../embedding/5/C/C.pickle'

    tracks_A=load_track_list(path_A)
    tracks_B=load_track_list(path_B)
    # tracks_C = load_track_list(path_C)


    tracks_a=[i for i in tracks_A if len(i.features)>=9]
    tracks_b=[i for i in tracks_B if len(i.features)>=9]
    # tracks_c = [i for i in tracks_C if len(i.features) >= 10]



    result_root = '/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/cpd_img'

    for i in tracks_a:
            id = i.track_id
            st_frame = i.start_frame
            result_dir = os.path.join(result_root, str(id))  # Gai
            if result_dir:
                mkdir_if_missing(result_dir)
            for j in i.img0s:
                if(j.size == 0):
                    continue
                cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'Acrop.jpg'), j)
                st_frame = st_frame + 1
            print(colored("successed!!", "green", attrs=["bold"]))

    for p in tracks_b:
            id = p.track_id
            st_frame = p.start_frame
            result_dir = os.path.join(result_root, str(id))  # Gai
            if result_dir:
                mkdir_if_missing(result_dir)
            for k in p.img0s:
                if(k.size == 0):
                    continue
                cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'Bcrop.jpg'), k)
                st_frame = st_frame + 1
            print(colored("successed!!", "green", attrs=["bold"]))







    # for i in tracks_a:
    #     if(i.track_id == 14):
    #         id = i.track_id
    #         st_frame = i.start_frame
    #         result_dir = os.path.join(result_root, str(id))  # Gai
    #         if result_dir:
    #             mkdir_if_missing(result_dir)
    #         for j in i.img0s:
    #             cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'crop.jpg'), j)
    #             st_frame = st_frame + 1
    #         print(colored("successed!!", "green", attrs=["bold"]))
    #
    # for p in tracks_b:
    #     if (p.track_id == 4):
    #         id = p.track_id
    #         st_frame = p.start_frame
    #         result_dir = os.path.join(result_root, str(id))  # Gai
    #         if result_dir:
    #             mkdir_if_missing(result_dir)
    #         for k in p.img0s:
    #             cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'crop.jpg'), k)
    #             st_frame = st_frame + 1
    #         print(colored("successed!!", "green", attrs=["bold"]))
    #

    # for i in tracks_a:
    #     if(i.track_id == 16):
    #         tracka_smooth_feat = i.smooth_feat
    #         break
    #
    # for p in tracks_b:
    #     if (p.track_id == 53):
    #         trackb_smooth_feat = p.smooth_feat
    #         break

# features = np.vstack((tracka_smooth_feat,trackb_smooth_feat))
# features = np.concatenate((tracka_smooth_feat, trackb_smooth_feat),axis=1)
# features = np.ndarray(tracka_smooth_feat, trackb_smooth_feat)
# cosine_similarity = np.matmul(features, features.transpose([1,0]))
# print('cosine similarity as below: ')
# print(cosine_similarity)


'''
def demo(opt):
    #result_root = opt.output_root if opt.output_root != '' else '.'
    result_root = '/home/shuanghong/Downloads/github/FairMOT/crop_img'

    
    #把tracker里面的track连结起来
    track_A = joint_stracks(joint_stracks(tracker.tracked_stracks, tracker.lost_stracks), tracker.removed_stracks)

    #挑选特征长度大于20的track
    tracks_a = [i for i in track_A if len(i.features) > 20]

    flag = 32
    for i in tracks_a:
        id = i.track_id
        result_dir = os.path.join(result_root, str(i.frame_id))  # gai
        if result_dir:
            mkdir_if_missing(result_dir)
        for j in i.img0s:
            cv2.imwrite(os.path.join(result_dir, str(flag) + 'crop.jpg'), j)
            flag = flag + 1
        print(colored("successed!!", "green", attrs=["bold"]))

            #cv2.imwrite('/home/shuanghong/Downloads/github/FairMOT/crop_img/6C.jpg', online_im)

#convert -resize 200x200 * +append ./output_img/output.jpg
#convert -resize 200x200 * +append output.jpg


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
    
'''