"""
    This script is to draw trajectory prediction as in Fig.6 of the paper
"""

import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np
import os
# traj_file = sys.argv[1]
# name = sys.argv[2]

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



def plot_traj(plt, traj_file_gt, traj_file_original, traj_file_original_linker, name, color, shape, marker):
    # for i in os.listdir(traj_file):
    #     print(i)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200) #figsize=(12, 6)

    trajs = np.loadtxt(traj_file_gt, delimiter=",")
    track_ids = np.unique(trajs[:,1])
    for tid in track_ids:
        # 16
        if tid == 35:
            t_color = get_cmap(tid)
            traj = trajs[np.where(trajs[:,1]==tid)]

            frames = traj[10:100, 0]
            boxes = traj[10:100, 2:6]
            # frames = traj[:100, 0]
            # boxes = traj[:100, 2:6]
            boxes_x = boxes[:,0]
            boxes_y = boxes[:,1]
            # plt.plot(boxes_x, boxes_y, color = color, marker = marker)
            # plt.plot(boxes_x, boxes_y, "ko")
            ax.plot(boxes_x, boxes_y, color = "dimgrey", marker = "o", label="GT", linewidth = 2.5)


            box_num = boxes_x.shape[0]
            for bind in range(0, box_num-1):
                frame_l = frames[bind]
                frame_r = frames[bind+1]
                box_l = boxes[bind]
                box_r = boxes[bind+1]
                if frame_r == frame_l + 1:
                    l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="dimgrey", linewidth = 2.5)
                    ax.add_line(l)
                    # import pdb; pdb.set_trace()
                    # ax.plot(box_l[0], box_l[1], box_r[0], box_r[1], color='red')
                else:
                    l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="dimgrey", linewidth = 2.5)
                    ax.add_line(l)

#---------------------------------------------------------
    # trajs = np.loadtxt(traj_file_original, delimiter=",")
    # track_ids = np.unique(trajs[:,1])
    # for tid in track_ids:
    #     #  77
    #     if tid == 698:
    #         t_color = get_cmap(tid)
    #         traj = trajs[np.where(trajs[:,1]==tid)]
    #         # fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    #         frames = traj[:100, 0] #fairmot
    #         boxes = traj[:100, 2:6]
    #         # frames = traj[5:105, 0]
    #         # boxes = traj[5:105, 2:6]
    #         boxes_x = boxes[:,0]
    #         boxes_y = boxes[:,1]
    #         # plt.plot(boxes_x, boxes_y, color = color, marker = marker)
    #         # plt.plot(boxes_x, boxes_y, "b*")
    #         ax.plot(boxes_x, boxes_y, color = "red", marker = "^", label="FairMOT", linewidth = 2.5)
    #
    #
    #         box_num = boxes_x.shape[0]
    #         for bind in range(0, box_num-1):
    #             frame_l = frames[bind]
    #             frame_r = frames[bind+1]
    #             box_l = boxes[bind]
    #             box_r = boxes[bind+1]
    #             if frame_r == frame_l + 1:
    #                 l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="red", linewidth = 2.5)
    #                 ax.add_line(l)
    #                 # import pdb; pdb.set_trace()
    #                 # ax.plot(box_l[0], box_l[1], box_r[0], box_r[1], color='red')
    #             else:
    #                 l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="red", linewidth = 2.5)
    #                 ax.add_line(l)


    #----------------------
    trajs = np.loadtxt(traj_file_original_linker, delimiter=",")
    track_ids = np.unique(trajs[:,1])
    for tid in track_ids:
        #75 trackformer-linker
        if tid == 698:
            t_color = get_cmap(tid)
            traj = trajs[np.where(trajs[:,1]==tid)]
            # fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
            # frames = traj[7:105, 0]
            # boxes = traj[7:105, 2:6]
            frames = traj[0:100, 0]
            boxes = traj[0:100, 2:6]
            boxes_x = boxes[:,0]
            boxes_y = boxes[:,1]
            # plt.plot(boxes_x, boxes_y, color = color, marker = marker)
            # plt.plot(boxes_x, boxes_y, "g^"),  markersize = 50
            ax.plot(boxes_x, boxes_y, color = "red", marker = "^", label="FairMOT+Linker", linewidth = 2.5)


            box_num = boxes_x.shape[0]
            for bind in range(0, box_num-1):
                frame_l = frames[bind]
                frame_r = frames[bind+1]
                box_l = boxes[bind]
                box_r = boxes[bind+1]
                if frame_r == frame_l + 1:
                    l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color = "red", linewidth = 2.5)
                    ax.add_line(l)
                    # import pdb; pdb.set_trace()
                    # ax.plot(box_l[0], box_l[1], box_r[0], box_r[1], color='red')
                elif frame_r == 757:
                    l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="green", linewidth = 3, label="Association by Linker")
                    ax.add_line(l)
                else:
                    l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="red", linewidth = 3)
                    ax.add_line(l)


    # #------------
    # trajs = np.loadtxt(traj_file_trackformer, delimiter=",")
    # track_ids = np.unique(trajs[:,1])
    # for tid in track_ids:
    #     if tid == 7:
    #         t_color = get_cmap(tid)
    #         traj = trajs[np.where(trajs[:,1]==tid)]
    #         # fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    #         frames = traj[:100, 0]
    #         boxes = traj[:100, 2:6]
    #         boxes_x = boxes[:,0]
    #         boxes_y = boxes[:,1]
    #         # plt.plot(boxes_x, boxes_y, color = color, marker = marker)
    #         # plt.plot(boxes_x, boxes_y, "rD")
    #         ax.plot(boxes_x, boxes_y, color = "pink", marker = "D", label="TrackFormer")
    #
    #         box_num = boxes_x.shape[0]
    #         for bind in range(0, box_num-1):
    #             frame_l = frames[bind]
    #             frame_r = frames[bind+1]
    #             box_l = boxes[bind]
    #             box_r = boxes[bind+1]
    #             if frame_r == frame_l + 1:
    #                 l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="pink")
    #                 ax.add_line(l)
    #                 # import pdb; pdb.set_trace()
    #                 # ax.plot(box_l[0], box_l[1], box_r[0], box_r[1], color='red')
    #             else:
    #                 l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="red")
    #                 ax.add_line(l)
    #         plt.plot(-0.3, -0.5, 0.8, 0.8, marker='o', color='red')

    plt.xticks([], linewidth = 2)
    plt.yticks([], linewidth = 2)
    plt.legend(fontsize = 20, loc = "upper left")
    plt.grid(True,axis='x', linewidth = 4)
    ax = plt.gca()

    bwidth = 2
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth)
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)

    fig.savefig("./traj_compare/11B_traj/new_2/11B—gt-fairmot-LINKER-red-label-finalv3.png")






if __name__ == "__main__":
    gt_src = "/home/shuanghong/Downloads/github/dataset/label_transfer_server/gt/11B/gt/gt.txt"
    # original_src = "/home/shuanghong/Downloads/github/dataset/former_metrics/ts-dir-47point2/11B.txt"
    # original_linker_src = "/home/shuanghong/Downloads/github/dataset/former_metrics/ts-dir-aflink-spatio-linker/11B.txt"

    original_src = "/home/shuanghong/Downloads/github/dataset/new_dataset_fair_metrics/ts-dir/11/ts/2.txt"
    original_linker_src = "/home/shuanghong/Downloads/github/dataset/new_dataset_fair_metrics/ts-dir-spatio-change-aflink-60point0/11B.txt"



    # fairmot = "/home/shuanghong/Downloads/github/dataset/new_dataset_fair_metrics/ts-dir/1/ts/2.txt"
    # jde = "/home/shuanghong/Downloads/github/dataset/jde_metrics/jde/4A.txt"
    # tracktor = "/home/shuanghong/Downloads/github/dataset/tracktor_metrics/tracktor/1C.txt"
    # trackformer = "/home/shuanghong/Downloads/github/dataset/former_metrics/ts-dir-47point2/1C.txt"
    seq = "1C"
    name = "gt_{}".format(seq)
    os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
    plot_traj(plt, gt_src, original_src, original_linker_src, name = "gt", color = "red", shape="-", marker='o')
    # plt_2 = plot_traj(plt_1, fairmot, name = "fairmot", color = "green", shape=":", marker='*')
    # plt_3 = plot_traj(plt_2, aflink, name = "aflink", color = "black", shape="-.", marker='^')
    # plt_4 = plot_traj(plt_3, linker, name = "linker", color = "orange", shape="--", marker='D')








