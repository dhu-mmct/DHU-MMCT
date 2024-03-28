from PIL import Image
from torchvision import transforms

def get_transform(size=(256, 128)):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # norm,
    ])
    return transform

transform = get_transform((256, 128))

img = Image.open("/home/shuanghong/Downloads/github/project/FairMOT_new_dataset/crop_scene/6scene/1/10crop.jpg")
batch = transform(img) * 255.
print(batch)
# import motmetrics as mm
# import numpy as np
# import os
#
# #评价指标
# metrics = list(mm.metrics.motchallenge_metrics)
#
# #导入gt和ts文件
# gt_file = '/home/shuanghong/Downloads/github/eval/1/A/1A_gt.txt'
# ts_file = '/home/shuanghong/Downloads/github/eval/1/A/1A_demo.txt'
#
# gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
# ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
# name=os.path.splitext(os.path.basename(ts_file))[0]
#
# #计算单个acc
# acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
# mh = mm.metrics.create()
# summary = mh.compute(acc, metrics=metrics, name=name)
# print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))