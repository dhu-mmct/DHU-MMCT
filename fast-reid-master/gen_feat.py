"""
@Author: Du Yunhao
@Filename: generate_detections.py
@Contact: dyh_bupt@163.com
@Time: 2021/11/8 17:02
@Discription: 生成检测特征
"""
import os
import os.path as osp
import cv2
import sys
import argparse
import pickle
import glob
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms
from os.path import join, exists, split
from termcolor import colored
sys.path.append('.')
sys.path.append('../src/lib')
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch


def load_track_list(save_file_path):
    if os.path.getsize(save_file_path) > 0:
        with open(save_file_path, 'rb') as f:
            return pickle.load(f)

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_model(cfg):
    model = DefaultTrainer.build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    return model

def get_transform(size=(256, 128)):
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # norm,
    ])
    return transform

def gen_feat(args):
    print(datetime.now())
    '''配置信息'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



    args.config_file = './configs/DukeMTMC/bagtricks_R101-ibn.yml'

    args.eval_only = True
    print("Command Line Args:", args)
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.WEIGHTS = 'fastreid/models/duke_bot_R101-ibn.pth'

    model = get_model(cfg)

    transform = get_transform((256, 128))

    thres_score = 0.6
    min_box_area = 100


    if(args.a != ''):
        tracks_A = load_track_list(args.a)
        # tracks_A = [i for i in tracks_A if len(i.features) > 29] # 去除短的轨迹
        for j in tracks_A:
            tracks_A_features = []
            st_frame = j.start_frame
            result_dir = os.path.join(args.crop_dir, str(j.track_id))  # Gai
            if result_dir:
                mkdir_if_missing(result_dir)
            for i in j.img0s:
                # if(score < thres_score):
                #     continue
                w = i.shape[1]
                h = i.shape[0]
                # vertical = w / h > 1.6
                if h!=0 and w * h > min_box_area and w / h < 1.6 and w / h > 0.3:
                    cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'crop.jpg'), i)
                    img = Image.open(os.path.join(result_dir, str(st_frame) + 'crop.jpg'))
                    input = transform(img) * 255.
                    input = input.unsqueeze(0).cuda()
                    outputs = model(input).detach().cpu().numpy()
                    tracks_A_features.append(outputs.reshape(-1))
                st_frame += 1
            tracks_A_features = np.asarray(tracks_A_features)
            j.reid = tracks_A_features

        with open(args.embedding_result_filename_A, 'wb') as f:
            ttt = tracks_A
            pickle.dump(ttt, f, protocol=5)

        print(colored("A EMBDEDDING dump successed!!", "green", attrs=["bold"]))


    if (args.b != ''):
        tracks_B = load_track_list(args.b)
        # tracks_B = [i for i in tracks_B if len(i.features) > 29]

        for j in tracks_B:
            tracks_B_features = []
            st_frame = j.start_frame
            result_dir = os.path.join(args.crop_dir, str(j.track_id))  # Gai
            if result_dir:
                mkdir_if_missing(result_dir)
            for i in j.img0s:
                # if (score < thres_score):
                #     continue
                w = i.shape[1]
                h = i.shape[0]
                # vertical = w / h > 1.6
                if h!=0 and w * h > min_box_area and w / h < 1.6 and w / h > 0.3:
                    cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'crop.jpg'), i)
                    img = Image.open(os.path.join(result_dir, str(st_frame) + 'crop.jpg'))
                    input = transform(img) * 255.
                    input = input.unsqueeze(0).cuda()
                    outputs = model(input).detach().cpu().numpy()
                    tracks_B_features.append(outputs.reshape(-1))
                st_frame += 1

            tracks_B_features = np.asarray(tracks_B_features)
            j.reid = tracks_B_features

        with open(args.embedding_result_filename_B, 'wb') as f:
            ttt = tracks_B
            pickle.dump(ttt, f, protocol=5)

        print(colored("B EMBDEDDING dump successed!!", "green", attrs=["bold"]))


    if(args.c != ''):
        tracks_C = load_track_list(args.c)
        # tracks_C = [i for i in tracks_C if len(i.features) > 29]

        for j in tracks_C:
            tracks_C_features = []
            st_frame = j.start_frame
            result_dir = os.path.join(args.crop_dir, str(j.track_id))  # Gai
            if result_dir:
                mkdir_if_missing(result_dir)
            for i in j.img0s:
                w = i.shape[1]
                h = i.shape[0]
                # vertical = w / h > 1.6
                if h!=0 and w * h > min_box_area and w / h < 1.6 and w / h > 0.3:
                    cv2.imwrite(os.path.join(result_dir, str(st_frame) + 'crop.jpg'), i)
                    img = Image.open(os.path.join(result_dir, str(st_frame) + 'crop.jpg'))
                    input = transform(img) * 255.
                    input = input.unsqueeze(0).cuda()
                    outputs = model(input).detach().cpu().numpy()
                    tracks_C_features.append(outputs.reshape(-1))
                st_frame += 1
            tracks_C_features = np.asarray(tracks_C_features)
            j.reid = tracks_C_features

        with open(args.embedding_result_filename_C, 'wb') as f:
            ttt = tracks_C
            pickle.dump(ttt, f, protocol=5)

        print(colored("C EMBDEDDING dump successed!!", "green", attrs=["bold"]))





def main():
    args = default_argument_parser().parse_args()

    # args.a = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_transfer/7/A/A.pickle"
    # args.b = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_transfer/7/C/C.pickle"
    # args.c = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding/12/C/C.pickle"
    #
    # args.embedding_result_filename_A = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_new/7/A/A.pickle"
    # args.embedding_result_filename_B = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_new/7/C/C.pickle"
    # args.embedding_result_filename_C = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_new/12/C/C.pickle"
    #
    #
    # args.crop_dir = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/crop_reid"
    gen_feat(args)


if __name__ == '__main__':
    main()








    # args.a = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding/7/A/A.pickle"
    # args.b = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding/7/C/C.pickle"
    # args.c = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding/12/C/C.pickle"

    # args.embedding_result_filename_A = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_new/7/A/A.pickle"
    # args.embedding_result_filename_B = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_new/7/C/C.pickle"
    # args.embedding_result_filename_C = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/embedding_newdataset_new/12/C/C.pickle"


    # args.crop_dir = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/crop_reid"


    # duke_agw_R101 - ibn.pth  S4 60.4
    # duke_mgn_R50 - ibn.pth      61.1
    # duke_sbs_R101-ibn.pth       61.1   63.2
    # market_bot_S50.pth          60.4
    #market_sbs_R101 - ibn.pth    59.5



    # cfg.OUTPUT_DIR = '/data/dyh/checkpoints/FastReID/tmp_log'

    # cfg.MODEL.BACKBONE.WITH_IBN = False












    # root_img = '/data/dyh/data/MOTChallenge/MOT17/train'
    # root_img = '/data/dyh/data/MOTChallenge/MOT17/test'
    # root_img = '/data/dyh/data/MOTChallenge/MOT20/test'
    # dir_in_det = '/data/dyh/results/StrongSORT/Detection/YOLOX_ablation_nms.8_score.1'
    # dir_in_det = '/data/dyh/results/StrongSORT/TEST/MOT17_YOLOX_nms.8_score.1'
    # dir_in_det = '/data/dyh/results/StrongSORT/TEST/MOT20_YOLOX_nms.8_score.1'
    # dir_out_det = '/data/dyh/results/StrongSORT/Features/YOLOX_nms.8_score.6_BoT-S50_DukeMTMC_again'
    # dir_out_det = '/data/dyh/results/StrongSORT/TEST/MOT17_YOLOX_nms.8_score.1_BoT-S50'
    # dir_out_det = '/data/dyh/results/StrongSORT/TEST/MOT20_YOLOX_nms.8_score.1_BoT-S50'
    # if not exists(dir_out_det): os.mkdir(dir_out_det)

    # transform = get_transform((384, 128))



#----------------------------------------------






# # ---------------------------------------------




#----------------------------------------------
#














    # transform = get_transform((384, 128))

    # files = sorted(glob.glob(join(dir_in_det, '*.txt')))
    # for i, file in enumerate(files, start=1):
    #     # if i <= 5: continue
    #     video = split(file)[1][:-4]
    #     print('processing the video {}...'.format(video))
    #     dir_img = join(root_img, '{}/img1'.format(video))
    #     detections = np.loadtxt(file, delimiter=',')
    #     detections = detections[detections[:, 6] >= thres_score]
    #     mim_frame, max_frame = int(min(detections[:, 0])), int(max(detections[:, 0]))
    #     list_res = list()
    #     for frame in range(mim_frame, max_frame + 1):
    #         # print('  processing the frame {}...'.format(frame))
    #         img = Image.open(join(dir_img, '%06d.jpg' % frame))
    #         detections_frame = detections[detections[:, 0] == frame]
    #         batch = [img.crop((b[2], b[3], b[2] + b[4], b[3] + b[5])) for b in detections_frame]
    #         batch = [transform(patch) * 255. for patch in batch]
    #         if batch:
    #             batch = torch.stack(batch, dim=0).cuda()
    #             outputs = model(batch).detach().cpu().numpy()
    #             list_res.append(np.c_[(detections_frame, outputs)])
    #     res = np.concatenate(list_res, axis=0)
    #     np.save(join(dir_out_det, video + '.npy'), res, allow_pickle=False)
    # print(datetime.now())