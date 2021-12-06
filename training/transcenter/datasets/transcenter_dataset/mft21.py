## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2017 NVIDIA CORPORATION. (Apache License, Version 2.0: https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
## (3) 2019 Simon Niklaus. (GNU General Public License v3.0: https://github.com/sniklaus/pytorch-liteflownet)
## (4) 2018 Tak-Wai Hui. (Copyright (c), see details in the LICENSE file: https://github.com/twhui/LiteFlowNet)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import platform

try:
    from .generic_dataset import GenericDataset
except:
    from generic_dataset import GenericDataset


class MFT21(GenericDataset):
    num_classes = 1
    num_joints = 17
    default_resolution = [640, 1088]
    max_objs = 20
    class_name = ['fish']
    cat_ids = {1: 1}

    def __init__(self, opt, split):
        super(MFT21, self).__init__()
        data_dir = opt.data_dir
        if split == 'test':
            img_dir = os.path.join(
                data_dir, 'test')
        else:
            img_dir = os.path.join(
                data_dir, 'train')

        if split == 'train' or split == 'val':
            ann_path = os.path.join(data_dir, 'annotations',
                                    '{}_half.json').format(split)
        else:
            ann_path = os.path.join(data_dir, 'annotations',
                                    '{}.json').format(split)

        print('==> initializing MFT21 {} data.'.format(split))

        self.images = None
        # load image list and coco
        super(MFT21, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def _save_results(self, records, fpath):
        with open(fpath, 'w') as fid:
            for record in records:
                line = json.dumps(record) + '\n'
                fid.write(line)
        return fpath

    def convert_eval_format(self, all_bboxes):
        detections = []
        person_id = 1
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                # newest format
                dtboxes = []
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    if item['class'] != person_id:
                        continue
                    bbox = item['bbox']
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "tag": 1,
                        "box": bbox_out,
                        "score": float("{:.2f}".format(item['score']))
                    }
                    dtboxes.append(detection)
            img_info = self.coco.loadImgs(ids=[image_id])[0]
            file_name = img_info['file_name']
            detections.append({'ID': file_name[:-4], 'dtboxes': dtboxes})
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        self._save_results(self.convert_eval_format(results),
                           '{}/results_crowdhuman.odgt'.format(save_dir))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        try:
            os.system('python tools/crowdhuman_eval/demo.py ' + \
                      '../data/crowdhuman/annotation_val.odgt ' + \
                      '{}/results_crowdhuman.odgt'.format(save_dir))
        except:
            print('Crowdhuman evaluation not setup!')


if __name__ == '__main__':
    import argparse
    import torch

    def get_args_parser():
        parser = argparse.ArgumentParser('Deformable DETR Dataloader', add_help=False)

        # dataset parameters
        parser.add_argument('--dataset_file', default='mft21')
        if(platform.system()=='Windows'):
            parser.add_argument('--data_dir', default='D:\\data\\dataset\\track', type=str)
        elif(platform.system()=='Linux'):
            parser.add_argument('--data_dir', default='/home/data/HJZ/MFT', type=str)

        parser.add_argument('--coco_panargsic_path', type=str)
        parser.add_argument('--remove_difficult', action='store_true')

        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=1, type=int)

        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--num_workers', default=1, type=int)
        parser.add_argument('--cache_mode', default=False, action='store_true',
                            help='whether to cache images on memory')


        # centers
        parser.add_argument('--num_classes', default=1, type=int)
        parser.add_argument('--input_h', default=480, type=int)
        parser.add_argument('--input_w', default=480, type=int)
        parser.add_argument('--down_ratio', default=1, type=int)
        parser.add_argument('--dense_reg', type=int, default=1, help='')
        parser.add_argument('--trainval', action='store_true',
                            help='include validation in training and '
                                 'test on test set')

        parser.add_argument('--K', type=int, default=20,
                            help='max number of output objects.')

        parser.add_argument('--debug', default=True, action='store_true')

        # noise
        parser.add_argument('--not_rand_crop', action='store_true',
                            help='not use the random crop data augmentation'
                                 'from CornerNet.')
        parser.add_argument('--not_max_crop', action='store_true',
                            help='used when the training dataset has'
                                 'inbalanced aspect ratios.')
        parser.add_argument('--shift', type=float, default=0.05,
                            help='when not using random crop'
                                 'apply shift augmentation.')
        parser.add_argument('--scale', type=float, default=0.05,
                            help='when not using random crop'
                                 'apply scale augmentation.')
        parser.add_argument('--rotate', type=float, default=0,
                            help='when not using random crop'
                                 'apply rotation augmentation.')
        parser.add_argument('--flip', type=float, default=0.5,
                            help='probability of applying flip augmentation.')

        parser.add_argument('--no_color_aug', action='store_true',
                            help='not use the color augmenation '
                                 'from CornerNet')

        parser.add_argument('--image_blur_aug', default=True, action='store_true',
                            help='blur image for aug.')
        parser.add_argument('--aug_rot', type=float, default=0,
                            help='probability of applying '
                                 'rotation augmentation.')
        # parser.add_argument('--heads', default=['hm', 'reg', 'wh', 'center_offset', 'tracking'], type=str, nargs='+')
        parser.add_argument('--heads', default=['hm', 'reg', 'wh', 'center_offset'], type=str, nargs='+')

        # tracking
        parser.add_argument('--max_frame_dist', type=int, default=3)
        parser.add_argument('--merge_mode', type=int, default=1)
        parser.add_argument('--tracking', default=False, action='store_true')
        parser.add_argument('--pre_hm', action='store_true')
        parser.add_argument('--same_aug_pre', action='store_true')
        parser.add_argument('--zero_pre_hm', action='store_true')
        parser.add_argument('--hm_disturb', type=float, default=0.05)
        parser.add_argument('--lost_disturb', type=float, default=0.4)
        parser.add_argument('--fp_disturb', type=float, default=0.1)
        parser.add_argument('--pre_thresh', type=float, default=-1)
        parser.add_argument('--track_thresh', type=float, default=0.3)
        parser.add_argument('--new_thresh', type=float, default=0.3)
        parser.add_argument('--ltrb_amodal', action='store_true')
        parser.add_argument('--ltrb_amodal_weight', type=float, default=0.1)
        parser.add_argument('--public_det', action='store_true')
        parser.add_argument('--no_pre_img', action='store_true')
        parser.add_argument('--zero_tracking', action='store_true')
        parser.add_argument('--hungarian', action='store_true')
        parser.add_argument('--max_age', type=int, default=-1)
        parser.add_argument('--out_thresh', type=float, default=-1,
                            help='')
        return parser


    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    dataset_train = MFT21(args, 'train')
    # input output shapes
    args.input_h, args.input_w = dataset_train.default_resolution[0], dataset_train.default_resolution[1]
    print(args.input_h, args.input_w)
    args.output_h = args.input_h // args.down_ratio
    args.output_w = args.input_w // args.down_ratio
    args.input_res = max(args.input_h, args.input_w)
    args.output_res = max(args.output_h, args.output_w)
    # threshold
    args.out_thresh = max(args.track_thresh, args.out_thresh)
    args.pre_thresh = max(args.track_thresh, args.pre_thresh)
    args.new_thresh = max(args.track_thresh, args.new_thresh)
    args.adaptive_clip = True
    print(args)
    print("trainset #samples: ", len(dataset_train))

    ret = dataset_train.__getitem__(50)
