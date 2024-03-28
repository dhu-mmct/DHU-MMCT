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


logger.setLevel(logging.INFO)


def demo(opt):

    logger.info('Starting tracking...')
    opt.conf_thres = 0.3

    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    print("input_video: {}".format(opt.input_video))


    frame_rate = dataloader.frame_rate

    opt.output_format = 'text'  #video or text
    frame_dir = None if opt.output_format == 'text' else osp.join(opt.video_result_file, 'frame2A')
    tracker, nf, ta, tc  = eval_seq(opt, dataloader, 'mot', opt.result_file,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])


    if opt.output_format == 'video':
        output_video_path = osp.join(opt.video_result_file, '1Avv.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(opt.video_result_file, 'frame2A'), output_video_path)
        os.system(cmd_str)

    with open(opt.embedding_result_filename, 'wb') as f:
        ttt = joint_stracks(joint_stracks(tracker.tracked_stracks, tracker.lost_stracks), tracker.removed_stracks)
        pickle.dump(ttt, f, protocol=5)

    print(colored("EMBDEDDING dump successed!!", "green", attrs=["bold"]))





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
