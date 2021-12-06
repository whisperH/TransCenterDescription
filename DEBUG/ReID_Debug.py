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
import os
import torch

import numpy as np
from datasets.transcenter_dataset.mot17_val_save_mem import MOT17_val
from reid.resnet import resnet50
import csv
import os.path as osp
import yaml
import argparse

from shutil import copyfile
import platform

torch.set_grad_enabled(False)

curr_pth = '/'.join(osp.dirname(__file__).split('/'))

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--learnable_queries', action='store_true',
                        help="If true, we use learnable parameters.")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--heads', default=['hm', 'reg', 'wh', 'center_offset', 'tracking'], type=str, nargs='+')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--hm_weight', default=1, type=float)
    parser.add_argument('--off_weight', default=1, type=float)
    parser.add_argument('--wh_weight', default=0.1, type=float)
    parser.add_argument('--tracking_weight', default=1, type=float)
    parser.add_argument('--ct_offset_weight', default=0.1, type=float)
    parser.add_argument('--boxes_weight', default=0.5, type=float)
    parser.add_argument('--giou_weight', default=0.4, type=float)
    parser.add_argument('--norm_factor', default=1.0, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='mot17')
    parser.add_argument('--data_dir', default='/home/data/HJZ/MOT/MOT17', type=str)

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
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # centers
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_h', default=640, type=int)
    parser.add_argument('--input_w', default=1088, type=int)
    parser.add_argument('--down_ratio', default=4, type=int)
    parser.add_argument('--dense_reg', type=int, default=1, help='')
    parser.add_argument('--trainval', action='store_true',
                        help='include validation in training and '
                             'test on test set')

    parser.add_argument('--K', type=int, default=300,
                        help='max number of output objects.')

    parser.add_argument('--debug', action='store_true')

    # noise
    parser.add_argument('--not_rand_crop', action='store_true',
                        help='not use the random crop data augmentation'
                             'from CornerNet.')
    parser.add_argument('--not_max_crop', action='store_true',
                        help='used when the training dataset has'
                             'inbalanced aspect ratios.')
    parser.add_argument('--shift', type=float, default=0.0,
                        help='when not using random crop'
                             'apply shift augmentation.')
    parser.add_argument('--scale', type=float, default=0.0,
                        help='when not using random crop'
                             'apply scale augmentation.')
    parser.add_argument('--rotate', type=float, default=0,
                        help='when not using random crop'
                             'apply rotation augmentation.')
    parser.add_argument('--flip', type = float, default=0.0,
                        help='probability of applying flip augmentation.')
    parser.add_argument('--no_color_aug', action='store_true',
                        help='not use the color augmenation '
                             'from CornerNet')
    parser.add_argument('--aug_rot', type=float, default=0,
                        help='probability of applying '
                             'rotation augmentation.')

    # tracking
    parser.add_argument('--max_frame_dist', type=int, default=3)
    parser.add_argument('--merge_mode', type=int, default=1)
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--pre_hm', action='store_true')
    parser.add_argument('--same_aug_pre', action='store_true')
    parser.add_argument('--zero_pre_hm', action='store_true')
    parser.add_argument('--hm_disturb', type=float, default=0.00)
    parser.add_argument('--lost_disturb', type=float, default=0.0)
    parser.add_argument('--fp_disturb', type=float, default=0.0)
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

def load_batch():
    if (platform.system() == 'Windows'):
        # ROOT_PATH = 'D:\\data\\dataset\\track'
        id_path = 'D:\\data\\perdestrain\\id_info'
    elif (platform.system() == 'Linux'):
        id_path = '/home/data/HJZ/MOT/id_info'
    id_list = os.listdir(id_path)



def write_results(all_tracks, out_dir, seq_name=None, frame_offset=0):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 submission

    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    assert seq_name is not None, "[!] No seq_name, probably using combined database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, seq_name+'.txt')

    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow([frame+frame_offset, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])

    # copy to FRCNN, DPM.txt, private setting
    copyfile(file, file[:-7]+"FRCNN.txt")
    copyfile(file, file[:-7]+"DPM.txt")

def main(tracktor, reid):
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    # load model
    main_args = get_args_parser().parse_args()
    main_args.node0 = True
    main_args.private = True
    ds = MOT17_val(main_args, 'test')

    main_args.input_h, main_args.input_w = ds.default_resolution[0], ds.default_resolution[1]
    print(main_args.input_h, main_args.input_w)
    main_args.output_h = main_args.input_h // main_args.down_ratio
    main_args.output_w = main_args.input_w // main_args.down_ratio
    main_args.input_res = max(main_args.input_h, main_args.input_w)
    main_args.output_res = max(main_args.output_h, main_args.output_w)
    # threshold
    main_args.track_thresh = tracktor['tracker']["track_thresh"]
    main_args.pre_thresh = tracktor['tracker']["pre_thresh"]
    main_args.new_thresh = max(tracktor['tracker']["track_thresh"], tracktor['tracker']["new_thresh"])

    # load reid network
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    print(f"Loading Reid Model {tracktor['reid_weights']}")
    reid_network.load_state_dict(torch.load(curr_pth + "/model_zoo/" + tracktor['reid_weights'],
                                            map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # new_det_pos 之前帧的目标检测值

    blob, new_det_pos = load_batch()

    det_feature = reid_network.test_rois(blob['img'], new_det_pos).detach()


with open(curr_pth + '/cfgs/detracker_reidV3.yaml', 'r') as f:
    tracktor = yaml.load(f)['tracktor']

with open(curr_pth+ '/cfgs/reid.yaml', 'r') as f:
    reid = yaml.load(f)['reid']
    # print(reid)

main(tracktor, reid)
