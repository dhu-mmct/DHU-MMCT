#!/usr/bin/env python


import cv2
import numpy as np
import os

# def load_mot(detections):
#     """
#     Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
#     'w', 'h', 'score']).
#     Args:
#         detections
#     Returns:
#         list: list containing the detections for each frame.
#     """
#
#     data = []
#     if type(detections) is str:
#         raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
#     else:
#         # assume it is an array
#         assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
#         raw = detections.astype(np.float32)
#
#     raw = raw[np.argsort(raw[:, 0])]
#     end_frame = int(np.max(raw[:, 0]))
#     len = raw.shape[0]
#     # for i in range(1, len+1):
#         # det.txt 1, -1, 1797, 205, 65.8, 191.2, 1
#         # gt.txt 4,1,1362,568,103,241,1,1,0.86173
#         # print(i,idx)
#     bbox = raw[:, 2:6]
#     bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
#     # scores = raw[idx, 6] #when det.txt
#     scores = raw[:, 8]
#     id = raw[:,1]
#
#     frame = raw[:, 0]
#     dets = []
#
#     is_pedestrian = raw[:, 6]
#
#     for bb, i, s , is_p in zip(bbox, id, scores, is_pedestrian):
#         #源代码没有除以2  由于下载的视频相较与标签分辨率缩小了一半，所以除以2
#         dets.append({'bbox': (int(bb[0]/2), int(bb[1]/2), int(bb[2]/2), int(bb[3]/2)), 'id': i,  'score': s, 'is_pedestrian': is_p ,'frame': frame})
#     # data.append(dets)
#
#     # return data
#     return dets

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

# def plot_tracking(image, tlwhs, obj_ids):
#
#
# def main():
#     video_path="MOT17-04-SDP.mp4"
#     vid=cv2.VideoCapture(video_path)
#     # dets=load_mot("mot17/train/MOT17-04-SDP/det/det.txt")
#     # dets = load_mot("MOT17-04-SDP/det/det.txt")
#     dets = load_mot("MOT17-04-SDP/gt/gt.txt")
#     save_dir = './results'
#     # return_value, frame = vid.read()
#
#     for i in range(1050):
#         return_value, frame = vid.read()
#         idx = raw[:, 0] == i
#         img = plot_tracking(frame, tlwhs, dets[:,dets[:,0]=1])
#
#
#     # for i in range(len(dets)):
#
#     for frame_num, detections_frame in enumerate(dets, start=0):
#         #读取视频
#         return_value,frame=vid.read()
#
#         text_scale = max(1, frame.shape[1] / 800.)
#         text_thickness = 2
#         line_thickness = max(1, int(frame.shape[1] / 250.))
#
#         cv2.imwrite(os.path.join(save_dir, 'frame{:05d}.jpg'.format(frame_num)), frame)
#         online_img = frame
#         # （1080，1920，3）
#         # text_scale = max(1, frame.shape[1] / 1600.)
#         # line_thickness = max(1, int(frame.shape[1] / 500.))
#         # for a in range(len(detections_frame)):
#         if int(detections_frame[frame_num]["is_pedestrian"]) == 0:
#             continue
#
#         bbox = detections_frame[frame_num]["bbox"]
#         obj_id = int(detections_frame[frame_num]["id"])
#         id_text = '{}'.format(int(obj_id))
#         color = get_color(abs(obj_id))
#
#         # print(frame_num,"bbox:x1, y1, x2, y2", bbox)
#     # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
#         #将解析处理的矩形框，绘制在视频上，实时显示
#         online_img = cv2.rectangle(online_img,bbox[:2],bbox[2:],color=color, thickness=line_thickness)
#         online_img = cv2.putText(online_img, id_text, (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
#                     thickness=text_thickness)
#
#         cv2.imwrite(os.path.join(save_dir, 'frame{:05d}.jpg'.format(frame_num)), frame)
#         cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_num)), online_img)
#         print('done:', frame_num)

            # cv2.imshow("frame", frame)
        # 键盘控制视频播放  waitKey(x)控制视频显示速度
        # key=cv2.waitKey(100)& 0xFF
        # if key == ord(' '):
        #     cv2.waitKey(0)
        # if key == ord('q'):
        #     break


def load_mot(detections):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).
    Args:
        detections
    Returns:
        list: list containing the detections for each frame.
    """

    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)

    end_frame = int(np.max(raw[:, 0]))
    min_visibility = np.min(raw[:,8])
    for i in range(1, end_frame + 1):
        idx = raw[:, 0] == i
        # print(i,idx)
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = raw[idx, 6]
        id = raw[idx, 1]

        dets = []
        for bb, s, i in zip(bbox, scores, id):
            # 源代码没有除以2  由于下载的视频相较与标签分辨率缩小了一半，所以除以2
            dets.append({'bbox': (int(bb[0] / 2), int(bb[1] / 2), int(bb[2] / 2), int(bb[3] / 2)), 'score': s, 'id': i})
            #  dets.append({'bbox': (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), 'score': s, 'id': i})
        data.append(dets)

    return data


def main():
    # video_path="/home/shuanghong/Downloads/github/dataset/scene_version_2/6/C/jpg.mp4"
    video_path = "/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/visualize_mot17/MOT17-04-SDP.mp4"
    vid=cv2.VideoCapture(video_path)
    # dets=load_mot("/home/shuanghong/Downloads/github/dataset/new_dataset_fair_metrics/ts-dir-spatio-change-aflink-60point0/11B.txt")

    # dets=load_mot("/home/shuanghong/Downloads/github/dataset/new_dataset_fair_metrics/ts-dir/6/ts/2.txt")
    # dets = load_mot("/home/shuanghong/Downloads/github/dataset/transcenter_motchallenge/TransCenter_mot17/train/MOT17-04-SDP.txt")
    dets = load_mot("/home/shuanghong/Downloads/github/dataset/transcenter_motchallenge/TransCenter_mot17/reid-refined/MOT17-04-SDP.txt")
    # dets = load_mot("/home/shuanghong/Downloads/github/dataset/transcenter_motchallenge/TransCenter_mot17/train/")
    # dets=load_mot("/home/shuanghong/Downloads/github/dataset/former_metrics/ts-dir-aflink-spatio-linker/11B.txt")
    # dets=load_mot("/home/shuanghong/Downloads/github/dataset/jde_metrics/jde/4A.txt")
    # dets=load_mot("/home/shuanghong/Downloads/github/dataset/tracktor_metrics/tracktor/1C.txt")
    # dets=load_mot("/home/shuanghong/Downloads/github/dataset/former_metrics/ts-dir-aflink-spatio-linker/11B.txt")

    save_dir = './results/MOT17-04-MDP-transcenter-iaff'

    for frame_num, detections_frame in enumerate(dets, start=1):
        #读取视频
        return_value,frame=vid.read()
        text_scale = max(1, frame.shape[1] / 800.)
        text_thickness = 2
        for a in range(len(detections_frame)):
            bbox=detections_frame[a]["bbox"]
            obj_id = int(detections_frame[a]["id"])
            id_text = '{}'.format(int(obj_id))

            color = get_color(abs(obj_id))

            print(frame_num,"bbox:x1, y1, x2, y2", bbox)
        # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
            #将解析处理的矩形框，绘制在视频上，实时显示
            # if obj_id == 2:
            cv2.rectangle(frame,bbox[:2],bbox[2:],color,2)
            # frame = cv2.putText(frame, id_text, (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale,
            #                          (0, 0, 255), thickness=text_thickness)
        # cv2.imshow("frame", frame)
        cv2.imwrite(os.path.join(save_dir, '{:06d}.jpg'.format(frame_num)), frame)
        # 键盘控制视频播放  waitKey(x)控制视频显示速度





if __name__ == '__main__':
    main()

    # src_path = './results/4C'
    # output_video_path = './results/11B_gt/11_gt.mp4'
    # cmd_str = 'ffmpeg -f image2 -i {}/%06d.jpg -b 5000k -c:v mpeg4 {}'.format(src_path, output_video_path)
    # os.system(cmd_str)

    # src_path = '/home/shuanghong/Downloads/github/dataset/MOT17/train/MOT17-11-DPM/img1'
    # output_video_path = '/home/shuanghong/Downloads/github/dataset/MOT17/train/MOT17-11-DPM/MOT17-11-DPM.mp4'
    # cmd_str = 'ffmpeg -f image2 -i {}/%06d.jpg -b 5000k -c:v mpeg4 {}'.format(src_path, output_video_path)
    # os.system(cmd_str)
