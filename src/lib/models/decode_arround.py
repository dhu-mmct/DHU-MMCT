from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    #keep[0][0]: 152*272 1*1*152*272
    '''
    flag_top = torch.zeros([1, 1, 152, 272]).to("cuda")
    flag_bottom = torch.zeros([1, 1, 152, 272]).to("cuda")
    flag_left = torch.zeros([1, 1, 152, 272]).to("cuda")
    flag_right = torch.zeros([1, 1, 152, 272]).to("cuda")

    for i in range(152):
        for j in range(272):
            if(keep[0][0][i][j] == 1.):
                    if(i > 0):
                        flag_top[0][0][i - 1][j] = 1.
                    if(i < 151):
                        flag_bottom[0][0][i + 1][j] = 1.
                    if(j < 271):
                        flag_right[0][0][i][j + 1] = 1.
                    if (j > 0):
                       flag_left[0][0][i][j - 1] = 1.
    surrounding = []
    surrounding.append(flag_top)
    surrounding.append(flag_bottom)
    surrounding.append(flag_right)
    surrounding.append(flag_left)'''


    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs


#scores: heat
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    #topk_inds 1*1*500
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    tmp_inds = topk_inds

    topk_inds = topk_inds % (height * width)
    judge = (topk_inds == tmp_inds)

    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    inds_top = torch.zeros([1, 500]).to("cuda")
    inds_topLeft = torch.zeros([1, 500]).to("cuda")
    inds_topRight = torch.zeros([1, 500]).to("cuda")
    inds_bottom = torch.zeros([1, 500]).to("cuda")
    inds_bottomLeft = torch.zeros([1, 500]).to("cuda")
    inds_bottomRight = torch.zeros([1, 500]).to("cuda")
    inds_left = torch.zeros([1, 500]).to("cuda")
    inds_right = torch.zeros([1, 500]).to("cuda")

    inds_top = inds_top.to(torch.int64)
    inds_topLeft = inds_topLeft.to(torch.int64)
    inds_topRight = inds_topRight.to(torch.int64)
    inds_bottomLeft = inds_bottomLeft.to(torch.int64)
    inds_bottomRight = inds_bottomRight.to(torch.int64)
    inds_bottom = inds_bottom.to(torch.int64)
    inds_left = inds_left.to(torch.int64)
    inds_right = inds_right.to(torch.int64)

    for i in range(500):
        if(topk_inds[0][i] > 271):
            inds_top[0][i] = topk_inds[0][i] - 272
        else: inds_top[0][i] = topk_inds[0][i]

        if(topk_inds[0][i] > 271 and topk_inds[0][i] % 272 != 0):
            inds_topLeft[0][i] = topk_inds[0][i] - 272 - 1
        else: inds_topLeft[0][i] = topk_inds[0][i]

        if(topk_inds[0][i] > 271 and (topk_inds[0][i] - 271) % 272 != 0):
            inds_topRight[0][i] = topk_inds[0][i] - 272 + 1
        else: inds_topRight[0][i] = topk_inds[0][i]

        if(topk_inds[0][i] < 41072 and topk_inds[0][i] % 272 != 0):
            inds_bottomLeft[0][i] = topk_inds[0][i] + 272 - 1
        else:
            inds_bottomLeft[0][i] = topk_inds[0][i]

        if(topk_inds[0][i] < 41072 and (topk_inds[0][i] - 271) % 272 != 0):
            inds_bottomRight[0][i] = topk_inds[0][i] + 272 + 1
        else:
            inds_bottomRight[0][i] = topk_inds[0][i]

        if(topk_inds[0][i] < 41072):
            inds_bottom[0][i] = topk_inds[0][i] + 272
        else: inds_bottom[0][i] = topk_inds[0][i]

        if ((topk_inds[0][i] - 271) % 272 != 0):
            inds_right[0][i] = topk_inds[0][i] + 1
        else: inds_right[0][i] = topk_inds[0][i]

        if(topk_inds[0][i] % 272 != 0):
            inds_left[0][i] = topk_inds[0][i] - 1
        else: inds_left[0][i] = topk_inds[0][i]

    topk_list = []
    topk_list.append(topk_inds)
    topk_list.append(inds_top)
    topk_list.append(inds_bottom)
    topk_list.append(inds_left)
    topk_list.append(inds_right)

    topk_list.append(inds_topLeft)
    topk_list.append(inds_topRight)

    topk_list.append(inds_bottomLeft)
    topk_list.append(inds_bottomRight)




    return topk_score, topk_list, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps

    heat = _nms(heat)

    scores, topk_list, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, topk_list[0])
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, topk_list[0])
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, topk_list
