import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import platform
import pandas as pd

SPLITS = ['train_half', 'val_half', 'train', 'test']  # --> split training data to train_half and val_half.
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

def gen_COCO_Anno(data_path, out_path, split):
    '''
    将MOT格式的数据转换成COCO格式的数据，并在image_info字典中新增帧间的信息。
    data_path: 数据的根目录
    out_path： 输出的标注信息的目录
    split：划分策略，一段数据的前半部分为训练集，后面部分为验证集
    '''
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'fish'}],
           'videos': []}
    # seqs: [track1, track2, ..., files under the folder "train"]
    if split == "test":
        data_path = os.path.join(data_path, 'test')
    else:
        data_path = os.path.join(data_path, 'train')
    seqs = os.listdir(data_path)
    # image's counter
    image_cnt = 0
    # annotation's counter
    ann_cnt = 0
    # sub-dataset's counter
    video_cnt = 0
    # traverse each sub_dataset
    for seq in sorted(seqs):
        video_cnt += 1

        # locate track"i"/
        seq_path = '{}/{}/'.format(data_path, seq)
        img_path = seq_path + 'img1/'
        ann_path = seq_path + 'gt/gt.txt'
        images = os.listdir(img_path)

        num_images = len([image for image in images if 'PNG' in image])
        if HALF_VIDEO and ('half' in split):
            image_range = [0, num_images // 2] if 'train' in split else \
                [num_images // 2 + 1, num_images - 1]
        else:
            image_range = [0, num_images - 1]

        # 用于存储单帧验证数据的文件列表
        image_filelist = []
        # 用于存储抽5帧验证数据的文件列表
        image_x_filelist = []

        for i in range(num_images):
            if (i < image_range[0] or i > image_range[1]):
                continue
            img_filename = img_path + '{:06d}.PNG'.format(i + 1)
            img = cv2.imread(img_filename)
            image_info = {
                'file_name': img_filename,
                'id': image_cnt + i + 1,
                'frame_id': i + 1 - image_range[0],
                'prev_image_id': image_cnt + i if i > 0 else -1,
                f'prev_{Posterize_NUM}_image_id': image_cnt + i - Posterize_NUM + 1 if i - Posterize_NUM > 0 else -1,
                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                f'next_{Posterize_NUM}_image_id': image_cnt + i + Posterize_NUM + 1 if i < num_images - Posterize_NUM else -1,
                'video_id': video_cnt,
                'width': img.shape[1],
                'height': img.shape[0],
            }
            image_filelist.append(img_filename)
            if i % Posterize_NUM == 0:
                image_x_filelist.append(img_filename)
            out['images'].append(image_info)

        out['videos'].append({
            'id': video_cnt,
            'file_name': seq,
            'image_list': image_filelist,
            f'image_{Posterize_NUM}_list': image_x_filelist,
        })
        print('{}: {} images'.format(seq, num_images))

        if split != 'test':
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            anns = anns[np.argsort(anns[:, 0])]

            # 生成 COCO格式
            print(' {} ann images'.format(int(anns[:, 0].max())))
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                    continue
                track_id = int(anns[i][1])
                ann_cnt += 1

                category_id = 1
                area = (anns[i][4] - anns[i][2]) * (anns[i][5] - anns[i][3])

                ann = {
                    'id': ann_cnt,
                    'category_id': category_id,
                    'image_id': image_cnt + frame_id,
                    'track_id': track_id,
                    'bbox': anns[i][2:6].tolist(),
                    'conf': float(anns[i][6]),
                    'area': area.item() / 2.0,
                    'iscrowd': int(anns[i][8])
                }
                out['annotations'].append(ann)

        image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
        split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))


class DemoData(object):
    '''
    展示图片数据类
    '''

    def __init__(self,
                 COCO_PATH,
                 show_image_nums=10,
                 image_freq=1,
                 start_img_id=1,
                 ):
        '''
        show_image_nums:展示图片数据的数量
        image_freq：取值为1或5，1代表不抽帧的情况，5代表抽5帧的情况
        start_img_id：图片开始的id
        '''
        with open(COCO_PATH, 'r', encoding='utf8') as fp:
            self.data = json.load(fp)

        image_nums = len(self.data['images'])
        assert show_image_nums > 0, 'show_image_nums must in (0, {})'.format(image_nums)
        assert start_img_id > 0 and start_img_id + image_freq < image_nums, 'image_freq must be 1 or 5'

        self.show_image_nums = show_image_nums
        self.image_freq = image_freq
        self.start_img_id = start_img_id
        self.end_img_id = start_img_id + image_freq * show_image_nums

    def getImgInfo(self, Img_ids: list):
        assert len(Img_ids) >= 1, 'list: Img_ids must contain 1 image id'
        Img_infos = {}

        for iImg in self.data['images']:
            if iImg['id'] in Img_ids:
                Img_infos[iImg['id']] = iImg
            else:
                continue
        return sorted(Img_infos.items(), key=lambda d: d[0])

    def getCOCOAnnoInfo(self, Img_ids: list):
        assert len(Img_ids) >= 1, 'list: Img_ids must contain 1 image id'
        Anno_infos = {}

        for iAnno in self.data['annotations']:
            if iAnno['image_id'] in Img_ids:
                if iAnno['image_id'] not in Anno_infos:
                    Anno_infos[iAnno['image_id']] = []
                Anno_infos[iAnno['image_id']].append(iAnno)
            else:
                continue
        return sorted(Anno_infos.items(), key=lambda d: d[0])

    def drawCOCOAnnoInfoImg(self, Img_info, Anno_info, Img_id):
        Img_path = Img_info['file_name']
        image = cv2.imread(os.path.join(DATA_PATH, Img_path))
        for i in range(len(Anno_info)):
            annotation = Anno_info[i]
            bbox = annotation['bbox']  # (x1, y1, w, h)
            x, y, w, h = bbox

            # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
            # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
            cv2.rectangle(
                image, (int(x), int(y)), (int(x + w), int(y + h)),
                (0, 255, 255), 2
            )
            cv2.putText(
                image, str(annotation['track_id']),
                (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 255), 2
            )
            # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def show(self):
        Img_ids = [_ for _ in range(self.start_img_id, self.end_img_id, self.image_freq)]
        Img_infos = self.getImgInfo(Img_ids)
        Anno_infos = self.getCOCOAnnoInfo(Img_ids)

        for idx, image_id in enumerate(Img_ids):
            self.drawCOCOAnnoInfoImg(Img_infos[idx][1], Anno_infos[idx][1], image_id)
            #             print(Anno_infos[idx][1])
            print("=====================")


class AdjustKP(object):
    def __init__(self, image, anno):

        self.img = image
        self.anno = anno
        self.track_idinfo = {_['track_id']: _ for _ in anno}
        self.track_idlist = list(self.track_idinfo.keys())

    def getObject(self, obj_id, show=True, save=False):
        if obj_id not in self.track_idinfo:
            print("tracker id is not annotations or not exist!")
            exit(-1)
        tl_x, tl_y, w, h = [int(_) for _ in self.track_idinfo[obj_id]['bbox']]
        obj_img = self.img[tl_y:tl_y + h, tl_x:tl_x + w, :]
        if show:
            cv2.imshow("single object image", obj_img)
            cv2.waitKey(0)

        return obj_img

    def iou_batch(self, bb_test, bb_gt):
        """
        From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return (o)

    def get_OverlappingMat(self):
        overlapping_dict = {}
        boxinfo = pd.DataFrame(self.anno)['bbox'].apply(pd.Series)
        boxinfo[3] += boxinfo[1]
        boxinfo[2] += boxinfo[0]
        overlap_mat = pd.concat([pd.DataFrame(self.anno)['track_id'], boxinfo], axis=1)
        for iobj in self.track_idlist:
            overlapping_dict[iobj] = []
            # 选出第iobj个bbox
            iobj_bbox = overlap_mat[overlap_mat['track_id'] == iobj]
            # 计算所有重叠的obj
            overlapping_ids = self.iou_batch(iobj_bbox.values[:, 1:], boxinfo.values)
            for flag, track_id in zip(list((overlapping_ids>0)[0]), self.track_idlist):
                if flag and track_id != iobj:
                    overlapping_dict[iobj].append(track_id)
        return overlapping_dict

    def run(self, obj_id):
        # obj_img = self.getObject(obj_id)
        overlapping_dict = self.get_OverlappingMat()
        print(f'obj {obj_id} is shelter from {overlapping_dict[obj_id]}')


def TEST_AdjustKP():
    # test for function <reJustifyKP>
    img_idx = 1
    with open(coco_path, 'r', encoding='utf8') as fp:
        data = json.load(fp)

    ImageInfos = data['images']
    Anno_info = []
    for iAnno in data['annotations']:
        if iAnno['image_id'] == img_idx:
            Anno_info.append(iAnno)
        else:
            continue

    image = cv2.imread(
        os.path.join(DATA_PATH, ImageInfos[img_idx]['file_name'])
    )

    AdjustKP(image, Anno_info).run(1)


if __name__ == '__main__':

    if (platform.system() == 'Windows'):
        ROOT_PATH = 'D:\\data\\dataset\\track'
        # ROOT_PATH = 'D:\\data\\perdestrain'
    elif (platform.system() == 'Linux'):
        ROOT_PATH = '/home/data/HJZ/MFT'

    race_path = 'train'  # 'train' or 'test'
    Posterize_NUM = 5
    DATA_PATH = os.path.join(ROOT_PATH, race_path)
    coco_path = os.path.join(ROOT_PATH, 'annotations/train.json')
    # coco_path = os.path.join(ROOT_PATH, 'annotations_onlySDP/train.json')


    #
    # for split in SPLITS:
    #     out_path = os.path.join(ROOT_PATH, 'annotations/{}.json'.format(split))
    #     gen_COCO_Anno(ROOT_PATH, out_path, split)
    # print("done!")

    # DemoData(
    #     COCO_PATH=coco_path,
    #     show_image_nums=3,
    #     image_freq=Posterize_NUM,
    #     start_img_id=1
    # ).show()

    TEST_AdjustKP()
