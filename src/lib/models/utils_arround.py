from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _sum(fear):
    if len(feat) == 0:
        return None
    sum=0
    for i in feat:
        sum+=i
    return sum/len(feat)


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _gather_feat_post(feat, ind, mask=None):
    dim  = feat.size(2)

    ind_center  = ind[0].unsqueeze(2).expand(ind[0].size(0), ind[0].size(1), dim)
    feat_center = feat.gather(1, ind_center)

    ind_top = ind[1].unsqueeze(2).expand(ind[1].size(0), ind[1].size(1), dim)
    feat_top = feat.gather(1, ind_top)

    ind_bottom = ind[2].unsqueeze(2).expand(ind[2].size(0), ind[2].size(1), dim)
    feat_bottom = feat.gather(1, ind_bottom)

    ind_left = ind[3].unsqueeze(2).expand(ind[3].size(0), ind[3].size(1), dim)
    feat_left = feat.gather(1, ind_left)

    ind_right = ind[4].unsqueeze(2).expand(ind[4].size(0), ind[4].size(1), dim)
    feat_right = feat.gather(1, ind_right)

    inds_topLeft = ind[5].unsqueeze(2).expand(ind[5].size(0), ind[5].size(1), dim)
    feat_topLeft = feat.gather(1, inds_topLeft)
    inds_topRight = ind[6].unsqueeze(2).expand(ind[6].size(0), ind[6].size(1), dim)
    feat_topRight = feat.gather(1, inds_topRight)
    inds_bottomLeft = ind[7].unsqueeze(2).expand(ind[7].size(0), ind[7].size(1), dim)
    feat_bottomLeft = feat.gather(1, inds_bottomLeft)
    inds_bottomRight = ind[8].unsqueeze(2).expand(ind[8].size(0), ind[8].size(1), dim)
    feat_bottomRight = feat.gather(1, inds_bottomRight)





    feat_ave = torch.zeros([1, 500, 64]).to("cuda")

    for i in range(500):
        feat_sum = 0.8 * feat_center[0][i] + 0.025 * feat_top[0][i] + 0.025 * feat_bottom[0][i] + 0.025 * feat_left[0][i] + 0.025 * feat_right[0][i]+ 0.025 * feat_topLeft[0][i]+ 0.025 * feat_topRight[0][i]+ 0.025 * feat_bottomLeft[0][i]+ 0.025 * feat_bottomRight[0][i]

        feat_ave[0][i] = feat_sum

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)

    return feat_ave

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _tranpose_and_gather_feat_post(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat_post(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)