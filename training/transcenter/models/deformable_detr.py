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
"""
Deformable DETR model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from losses.utils import _sigmoid
from losses.losses import FastFocalLoss, RegWeightedL1Loss, loss_boxes, ContrastiveLoss
from util import box_ops
from util.plot_utils import plot_grad_flow_v2
from util.misc import NestedTensor, inverse_sigmoid
from post_processing.decode import generic_decode
from post_processing.post_process import generic_post_process
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
import copy
from .dla import IDAUpV3
from torch import Tensor
from .matcher import build_matcher
import itertools

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class GenericLoss(torch.nn.Module):
    def __init__(self, opt, weight_dict):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.opt = opt
        self.weight_dict = weight_dict
        self.feat_box_h = 68
        self.feat_box_w = 40

        if self.opt.cl_appearance:
            # need q, k, appear_queue
            self.crit_cl = ContrastiveLoss(
                prject_in_feat=(self.feat_box_h, self.feat_box_w)
            )

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'pre_hm' in output:
            output['pre_hm'] = _sigmoid(output['pre_hm'])
        return output

    def gatherByIndex(self, feature_map, gt_ind, _groups):
        '''
        feature_map's shape: <bs, channel, width, height>
        gt_ind's shape: <bs, max_query_obj, 6>
        '''
        bs, channel, height, width = feature_map.shape
        height_ratio = self.opt.input_h / height
        width_ratio = self.opt.input_w / width
        assert height_ratio == width_ratio, 'ratio of input‘s width != height'


        scale_box_w = self.opt.output_w / width
        scale_box_h = self.opt.output_h / height

        for ibs in range(bs):
            box_list = gt_ind[ibs].cpu().numpy()
            feat_map = feature_map[ibs]

            for box_info in box_list:
                frame_id, track_id, bb_cx, bb_cy, bb_w, bb_h = box_info
                if bb_w == bb_h == 0:
                    continue
                # track_list.append([frame_id, track_id])
                bb_tlx = math.ceil((bb_cx-0.5*bb_w)/scale_box_w)
                bb_tly = math.ceil((bb_cy-0.5*bb_h)/scale_box_h)
                bb_brx = math.ceil((bb_cx+0.5*bb_w)/scale_box_w)
                bb_bry = math.ceil((bb_cy+0.5*bb_h)/scale_box_h)

                amod_box = feat_map[:, bb_tly:bb_bry, bb_tlx:bb_brx]

                # _groups[(frame_id, track_id)] = F.interpolate(
                #     amod_box,
                #     size=[self.feat_box_h, self.feat_box_w]
                # )

                left_padding = self.feat_box_w - amod_box.shape[2]
                top_padding = self.feat_box_h - amod_box.shape[1]
                padding_box = nn.ZeroPad2d((0, int(left_padding), 0, int(top_padding)))
                _groups[(frame_id, track_id)] = padding_box(amod_box)

        return _groups

    def buildContrastGroup(self, _groups: dict):
        contra_g = {}
        track_list = list(_groups.keys())

        for q_id, k_id in itertools.combinations(track_list, 2):
            # print(q_id, k_id)
            q_frameid, q_trk_id = q_id
            k_frameid, k_trk_id = k_id
            if q_trk_id not in contra_g:
                contra_g[q_trk_id] = {
                    'label': [],
                    'feature': []
                }
            if abs(q_frameid - k_frameid) <= self.opt.max_frame_dist:
                if q_trk_id == k_trk_id:
                    contra_g[q_trk_id]['label'].append(1)
                else:
                    contra_g[q_trk_id]['label'].append(0)

                contra_g[q_trk_id]['feature'].append(
                    torch.stack([
                        _groups[(q_frameid, q_trk_id)],
                        _groups[(k_frameid, k_trk_id)]
                    ])
                )
        return contra_g

    def forward(self, outputs, batch, appear_queue=None):
        opt = self.opt
        regression_heads = ['reg', 'wh', 'tracking', 'center_offset']
        losses = {}
        # change: modify hm'sigmoid and pre_hm's sigmoid
        outputs = self._sigmoid_output(outputs)

        # _groups = [{}, {}, {},...]
        # track_list = [{id1}, {id2}, {id3},...]
        _groups = {}
        for s in range(opt.dec_layers):
            if s < opt.dec_layers - 1:
                end_str = f'_{s}'
            else:
                end_str = ''

            if self.opt.cl_appearance:
                # gatherNeedDict = {
                #     'wh': outputs['wh'][s],
                #     'hm': outputs['hm'][s],
                # }
                # pred_pos = self.extractHMPos(gatherNeedDict)
                _groups = self.gatherByIndex(
                    # current frame
                    outputs['feature_map'][f'layer_{s}'][0],
                    batch['sm_queue'][:, :opt.num_queries, :],
                    _groups
                )
                _groups = self.gatherByIndex(
                    # prev frame
                    outputs['feature_map'][f'layer_{s}'][1],
                    batch['sm_queue'][:, opt.num_queries:, :],
                    _groups
                )
                contra_g = self.buildContrastGroup(_groups)
                if len(contra_g) > 0:
                    print(f"contrastive pair length {len(contra_g)}")
                    losses['ctr' + end_str] = self.crit_cl(
                        contra_g
                    )
                else:
                    print("no contrastive pair")

            # only 'hm' is use focal loss for heatmap regression. #
            if 'hm' in outputs:
                losses['hm' + end_str] = self.crit(
                    outputs['hm'][s], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / opt.norm_factor

            for head in regression_heads:
                if head in outputs:
                    # print(head)
                    losses[head + end_str] = self.crit_reg(
                        outputs[head][s], batch[head + '_mask'],
                        batch['ind'], batch[head]
                    ) / opt.norm_factor

            losses['boxes' + end_str], losses['giou' + end_str] = loss_boxes(outputs['boxes'][s], batch)
            losses['boxes' + end_str] /= opt.norm_factor
            losses['giou' + end_str] /= opt.norm_factor

        return losses

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, args, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, learnable_queries=False, input_shape=(640, 1088)):
        """ Initializes the model.
        """
        super().__init__()
        self.transformer = transformer
        self.args = args

        hidden_dim = transformer.d_model

        self.ida_up = IDAUpV3(
            # 64, [256, 256, 256, 256],
            64, [hidden_dim, hidden_dim, hidden_dim, hidden_dim],
            [2, 4, 8, 16])

        # different ida up for tracking and detection
        self.ida_up = _get_clones(self.ida_up, 2)

        '''
        (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        '''

        self.hm = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.ct_offset = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.reg = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.wh = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        # future tracking offset
        self.tracking = nn.Sequential(
            nn.Conv2d(129, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.cl_feat = nn.Sequential(
            GAM_Attention(in_channels=64, out_channels=64),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=True),
            GAM_Attention(in_channels=16, out_channels=16),
            nn.Conv2d(16, 1, kernel_size=3, stride=4, padding=1, bias=True),
        )

        # init weights #
        # prior bias
        self.hm[-1].bias.data.fill_(-4.6)
        fill_fc_weights(self.reg)
        fill_fc_weights(self.wh)
        fill_fc_weights(self.ct_offset)
        fill_fc_weights(self.tracking)

        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss

        # init weights #
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers

        self.transformer.decoder.reg = None
        self.transformer.decoder.ida_up = None
        self.transformer.decoder.wh = None

        self.query_embed = None

    def forward(self, samples: NestedTensor, pre_samples: NestedTensor, pre_hm: Tensor):
        assert isinstance(samples, NestedTensor)
        # resnet输出多尺度的特征图，在这一步需要把特征图搞成统一大小
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        with torch.no_grad():
            pre_features, pre_pos = self.backbone(pre_samples)
            pre_srcs = []
            pre_masks = []

        for l, (feat, pre_feat) in enumerate(zip(features, pre_features)):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

            # xyh pre
            pre_src, pre_mask = pre_feat.decompose()
            pre_srcs.append(self.input_proj[l](pre_src))
            pre_masks.append(pre_mask)

            assert mask is not None
            assert pre_mask is not None
            assert pre_src.shape == src.shape

        # make mask, src, pos embed
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                    pre_src = self.input_proj[l](pre_features[-1].tensors)

                else:
                    src = self.input_proj[l](srcs[-1])
                    pre_src = self.input_proj[l](pre_srcs[-1])
                assert pre_src.shape == src.shape
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

                # pre
                pre_m = pre_samples.mask
                pre_mask = F.interpolate(pre_m[None].float(), size=pre_src.shape[-2:]).to(torch.bool)[0]
                pre_pos_l = self.backbone[1](NestedTensor(pre_src, pre_mask)).to(src.dtype)
                pre_srcs.append(pre_src)
                pre_masks.append(pre_mask)
                pre_pos.append(pre_pos_l)

        if self.query_embed is not None:
            query_embed = self.query_embed.weight
        else:
            query_embed = None
        merged_hs = self.transformer(srcs, masks, pos, query_embed, pre_srcs=pre_srcs, pre_masks=pre_masks,
                                     pre_hms=None, pre_pos_embeds=pre_pos)

        hs = []
        pre_hs = []

        pre_hm_out = F.interpolate(pre_hm.float(), size=(
            pre_hm.shape[2] // self.args.down_ratio, pre_hm.shape[3] // self.args.down_ratio))

        for hs_m, pre_hs_m in merged_hs:
            hs.append(hs_m)
            pre_hs.append(pre_hs_m)

        outputs_coords = []
        outputs_hms = []
        outputs_pre_hms = []
        outputs_regs = []
        outputs_whs = []
        outputs_ct_offsets = []
        outputs_tracking = []
        feature_map = {}

        for layer_lvl in range(len(hs)):
            hs[layer_lvl] = self.ida_up[0](hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]
            pre_hs[layer_lvl] = self.ida_up[1](pre_hs[layer_lvl], 0, len(pre_hs[layer_lvl]))[-1]
            feature_map[f'layer_{layer_lvl}'] = [
                self.cl_feat(hs[layer_lvl]),
                self.cl_feat(pre_hs[layer_lvl])
            ]
            # Object Size Branch
            ct_offset = self.ct_offset(hs[layer_lvl])
            wh_head = self.wh(hs[layer_lvl])
            reg_head = self.reg(hs[layer_lvl])
            # Center Heatmap Branch
            hm_head = self.hm(hs[layer_lvl])
            pre_hm_head = self.hm(pre_hs[layer_lvl])
            # tracking branch
            # 拼接上一帧hm:pre_hm_out 、TF_t、DF_t
            tracking_head = self.tracking(torch.cat(
                [
                    # hs[layer_lvl]被detach掉了，但是只要不改变hs[layer_lvl]的值就可以进行反向传播
                    # https://blog.csdn.net/qq_27825451/article/details/95498211?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
                    hs[layer_lvl].detach(),
                    pre_hs[layer_lvl],
                    pre_hm_out
                ], dim=1)
            )

            outputs_whs.append(wh_head)
            outputs_ct_offsets.append(ct_offset)
            outputs_regs.append(reg_head)

            outputs_hms.append(hm_head)
            outputs_pre_hms.append(pre_hm_head)

            outputs_tracking.append(tracking_head)

            # b,2,h,w => b,4,h,w
            outputs_coords.append(torch.cat([reg_head + ct_offset, wh_head], dim=1))
            torch.cuda.empty_cache()

        if self.args.cl_appearance:
            out = {
                'hm': torch.stack(outputs_hms),
                'pre_hm': torch.stack(outputs_pre_hms),
                'feature_map': feature_map,
                'boxes': torch.stack(outputs_coords),
                'wh': torch.stack(outputs_whs),
                'reg': torch.stack(outputs_regs),
                'center_offset': torch.stack(outputs_ct_offsets),
                'tracking': torch.stack(outputs_tracking),
            }
        else:
            out = {
                'hm': torch.stack(outputs_hms),
                'boxes': torch.stack(outputs_coords),
                'wh': torch.stack(outputs_whs),
                'reg': torch.stack(outputs_regs),
                'center_offset': torch.stack(outputs_ct_offsets),
                'tracking': torch.stack(outputs_tracking)
            }
        return out


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args, valid_ids):
        self.args = args
        self._valid_ids = valid_ids
        print("valid_ids: ", self._valid_ids)
        print("self.args in post processor before:", self.args)
        super().__init__()

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    @torch.no_grad()
    def forward(self, outputs, target_sizes, target_c=None, target_s=None, not_max_crop=False, filter_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        # for map you dont need to filter
        if filter_score:
            out_thresh = self.args.out_thresh
        else:
            out_thresh = 0.0
        # get the output of last layer of transformer
        output = {k: v[-1].cpu() for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output = self._sigmoid_output(output)

        dets = generic_decode(output, K=self.args.K, opt=self.args)

        if target_c is None and target_s is None:
            target_c = []
            target_s = []
            for target_size in target_sizes:
                # get image centers
                c = np.array([target_size[1].cpu() / 2., target_size[0].cpu() / 2.], dtype=np.float32)
                # get image size or max h or max w
                s = max(target_size[0], target_size[1]) * 1.0 if not self.args.not_max_crop \
                    else np.array([target_size[1], target_size[0]], np.float32)
                target_c.append(c)
                target_s.append(s)
        else:
            target_c = target_c.cpu().numpy()
            target_s = target_s.cpu().numpy()

        results = generic_post_process(self.args, dets,
                                       target_c, target_s,
                                       output['hm'].shape[2], output['hm'].shape[3], filter_by_scores=out_thresh)
        # print(len(results))
        coco_results = []
        for btch_idx in range(len(results)):
            boxes = []
            scores = []
            labels = []
            tracking = []
            for det in results[btch_idx]:
                boxes.append(det['bbox'])
                scores.append(det['score'])
                labels.append(self._valid_ids[det['class'] - 1])
                tracking.append(det['tracking'])
            if len(boxes) > 0:
                coco_results.append({'scores': torch.as_tensor(scores).float(),
                                     'labels': torch.as_tensor(labels).int(),
                                     'boxes': torch.as_tensor(boxes).float(),
                                     'tracking': torch.as_tensor(tracking).float()})
            else:
                coco_results.append({'scores': torch.zeros(0).float(),
                                     'labels': torch.zeros(0).int(),
                                     'boxes': torch.zeros(0, 4).float(),
                                     'tracking': torch.zeros(0, 2).float()})
        return coco_results


def build(args):
    num_classes = args.num_classes if args.dataset_file != 'coco' else 80

    if args.dataset_file == 'coco':
        valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
    else:

        valid_ids = [1]

    device = torch.device(args.device)
    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    print("num_classes", num_classes)
    model = DeformableDETR(
        args,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        learnable_queries=args.learnable_queries,
        input_shape=(args.input_h, args.input_w),
    )

    # weights
    weight_dict = {
        'ctr': args.ctr_weight,
        'hm': args.hm_weight,
        'reg': args.off_weight,
        'wh': args.wh_weight,
        'boxes': args.boxes_weight,
        'giou': args.giou_weight,
        'center_offset': args.ct_offset_weight,
        'tracking': args.tracking_weight
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = GenericLoss(args, weight_dict).to(device)
    postprocessors = {'bbox': PostProcess(args, valid_ids)}
    matcher = build_matcher(args)

    return model, criterion, postprocessors, matcher
