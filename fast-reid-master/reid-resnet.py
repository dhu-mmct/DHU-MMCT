"""
@Author: Du Yunhao
@Filename: generate_detections.py
@Contact: dyh_bupt@163.com
@Time: 2021/11/8 17:02
@Discription: 生成检测特征
"""
import os
import cv2
import sys
import glob
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms
from os.path import join, exists, split

sys.path.append('../')
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image



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
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # norm,
    ])
    return transform

if __name__ == '__main__':
    print(datetime.now())
    '''配置信息'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args = default_argument_parser().parse_args()
    args.eval_only = True
    # args.config_file = './configs/DukeMTMC/bagtricks_R101-ibn.yml'
    args.config_file = './configs/DukeMTMC/AGW_R101-ibn.yml'
    print("Command Line Args:", args)
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    # cfg.MODEL.WEIGHTS = 'fastreid/models/duke_bot_R101-ibn.pth'
    cfg.MODEL.WEIGHTS = 'fastreid/models/duke_agw_R101-ibn.pth'
    cfg.OUTPUT_DIR = 'fastreid/output'

    # cfg.MODEL.BACKBONE.WITH_IBN = False

    # thres_score = 0.6
    # root_img = '/home/shuanghong/Downloads/github/dataset/MOT15/images/all'
    # root_img = '/data/dyh/data/MOTChallenge/MOT17/train'
    # root_img = '/data/dyh/data/MOTChallenge/MOT17/test'
    # root_img = '/home/shuanghong/Downloads/github/dataset/MOT20/train'
    root_img = '/mnt/disk/shuanghong/MOT17/test/'
    # root_img = '/home/shuanghong/Downloads/github/dataset/scene_copy/'
    # root_img = '/home/shuanghong/Downloads/github/dataset/scene_version_2/'

    dir_in_det = '/mnt/disk/shuanghong/tracktor_v2_motchallenge/MOT17/Tracktor++/test'
    dir_out_det = '/mnt/disk/shuanghong/tracktor_v2_motchallenge/MOT17/Tracktor++/reid-resnet'
    # dir_in_det = '/home/shuanghong/Downloads/github/dataset/ISSAP/TMOH/train'
    # dir_out_det = '/home/shuanghong/Downloads/github/dataset/ISSAP/TMOH/train-refined'
    # dir_in_det = '/home/shuanghong/Downloads/github/dataset/transcenter_motchallenge/TransCenter_mot20/train'
    # dir_out_det = '/home/shuanghong/Downloads/github/dataset/transcenter_motchallenge/TransCenter_mot20/train-aff-reid-AGW'



    if not exists(dir_out_det): os.mkdir(dir_out_det)
    # model = get_model(cfg)

    resnet50_feature_extractor = models.resnet50(pretrained=True).to('cuda:0')  # 导入ResNet50的预训练模型
    resnet50_feature_extractor.fc = nn.Linear(2048, 2048).to('cuda:0')  # 重新定义最后一层
    torch.nn.init.eye(resnet50_feature_extractor.fc.weight)  # 将二维tensor初始化为单位矩阵


#     x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#
#     y = resnet50_feature_extractor(x)
#     y = y.data.numpy()  # 将提出出来的特征向量转化成numpy形式便于后面存储
#     data_list.append(y)  # 依次将每一张图片提出出来的向量放在列表中
#     data_npy = np.array(data_list)
#     print(data_npy.shape)
# np.save('list.npy', data_list)  # 存储为.npy文件


    transform = get_transform((256, 128))
    # transform = get_transform((384, 128))

    files = sorted(glob.glob(join(dir_in_det, '*.txt')))
    for i, file in enumerate(files, start=1):
        # if i <= 5: continue
        video = split(file)[1][:-4]
        print('processing the video {}...'.format(video))
        dir_img = join(root_img, '{}/img1'.format(video))
        detections = np.loadtxt(file, delimiter=',')
        # detections = detections[detections[:, 6] >= thres_score]
        mim_frame, max_frame = int(min(detections[:, 0])), int(max(detections[:, 0]))
        list_res = list()
        for frame in range(mim_frame, max_frame + 1):
            # print('  processing the frame {}...'.format(frame))
            for param in resnet50_feature_extractor.parameters():
                param.requires_grad = False

            img = Image.open(join(dir_img, '%06d.jpg' % frame))
            detections_frame = detections[detections[:, 0] == frame]
            batch = [img.crop((b[2], b[3], b[2] + b[4], b[3] + b[5])) for b in detections_frame]
            batch = [transform(patch) * 255. for patch in batch]
            if batch:
                batch = torch.stack(batch, dim=0).float().cuda()
                outputs = resnet50_feature_extractor(batch).data.detach().cpu().numpy()
                list_res.append(np.c_[(detections_frame, outputs)])
        res = np.concatenate(list_res, axis=0)
        np.save(join(dir_out_det, video + '.npy'), res, allow_pickle=False)
    print(datetime.now())

    # for newdataset
    # for i, file in enumerate(files, start=1):
    #     # if i <= 5: continue
    #     video = split(file)[1][:-4]
    #     print('processing the video {}...'.format(video))
    #     # dir_img = join(root_img, '{}/img1'.format(video))
    #     dir_img = join(root_img, '{}/{}/jpg'.format(video[:-1], video[-1]))
    #     detections = np.loadtxt(file, delimiter=',')
    #     # detections = detections[detections[:, 6] >= thres_score]
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

    # for old_dataset
    # for i, file in enumerate(files, start=1):
    #     # if i <= 5: continue
    #     video = split(file)[1][:-4]
    #     print('processing the video {}...'.format(video))
    #     # dir_img = join(root_img, '{}/img1'.format(video))
    #     dir_img = join(root_img, '{}/{}'.format(video[0], video[1]))
    #     detections = np.loadtxt(file, delimiter=',')
    #     # detections = detections[detections[:, 6] >= thres_score]
    #     mim_frame, max_frame = int(min(detections[:, 0])), int(max(detections[:, 0]))
    #     list_res = list()
    #     for frame in range(mim_frame, max_frame + 1):
    #         # print('  processing the frame {}...'.format(frame))
    #         if video[0] == '3' or video[0] == '6' or (video[0] == '4' and video[1] == 'B'):
    #             img = Image.open(join(dir_img, '%06d.png' % frame))
    #         else:
    #             img = Image.open(join(dir_img, '%06d.jpg' % frame))
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