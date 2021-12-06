import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import platform
import pandas as pd
from data.tools import builder

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

# 选取重叠的目标
class GenerateObjsImage(object):
    def __init__(self, image_filename, anno, overlap_threld_list=(0, 0.2, 0.4, 0.6, 1)):
        self.image_filename = image_filename
        self.img = cv2.imread(
            os.path.join(DATA_PATH, image_filename)
        )
        self.anno = anno
        self.track_idinfo = {_['track_id']: _ for _ in anno}
        self.track_idlist = list(self.track_idinfo.keys())

        self.thred_range = []
        for idx_thred in range(len(overlap_threld_list)-1):
            min_thred, max_thred = overlap_threld_list[idx_thred], overlap_threld_list[idx_thred+1]
            self.thred_range.append(f"{str(min_thred)}_{str(max_thred)}")

        self.overlapping_dict = self.get_UnOverlappingMat(self.thred_range)


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

    def get_UnOverlappingMat(self, thred_range):
        overlapping_dict = {}
        boxinfo = pd.DataFrame(self.anno)['bbox'].apply(pd.Series)
        boxinfo[3] += boxinfo[1]
        boxinfo[2] += boxinfo[0]
        overlap_mat = pd.concat([pd.DataFrame(self.anno)['track_id'], boxinfo], axis=1)
        for iobj in self.track_idlist:
            for ithred_range in thred_range:
                if ithred_range not in overlapping_dict:
                    overlapping_dict[ithred_range] = {
                        _: [] for _ in self.track_idlist
                    }
                # 选出第iobj个bbox
                iobj_bbox = overlap_mat[overlap_mat['track_id'] == iobj]
                # 计算所有重叠的obj
                overlapping_ids = self.iou_batch(iobj_bbox.values[:, 1:], boxinfo.values)
                # print(f"overlapping relationship with {iobj} and others: {overlapping_ids}")
                max_overlap_tracker = np.argsort(-overlapping_ids)[0][1]
                max_overlap_value = overlapping_ids[0][max_overlap_tracker]
                min_thred, max_thred = [float(_) for _ in ithred_range.split("_")]
                if max_overlap_value >= min_thred and max_overlap_value < max_thred:
                    overlapping_dict[ithred_range][iobj].append(max_overlap_tracker)
                else:
                    continue
        return overlapping_dict

    def selectObj(self, obj_id, thred_range, show=True, save=False):
        thred_range = f"{str(thred_range[0])}_{str(thred_range[1])}"
        assert thred_range in self.thred_range, f"thred_range must in {self.thred_range}"

        print(f"object id {obj_id}")

        thred_info = self.overlapping_dict[thred_range]
        if len(thred_info[obj_id]) > 0:
            if obj_id not in self.track_idinfo:
                print("tracker id is not annotations or not exist!")
                exit(-1)
            tl_x, tl_y, w, h = [int(_) for _ in self.track_idinfo[obj_id]['bbox']]
            obj_img = self.img[tl_y:tl_y + h, tl_x:tl_x + w, :]
            if show:
                cv2.imshow("single object image", obj_img)
                cv2.waitKey(0)
            if save:
                save_file_name = "tracker"+str(obj_id)+"_"+self.image_filename.replace("/img1/", "_")
                self.output_path = os.path.join(ROOT_PATH, thred_range)
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)
                print(f"filter result is saved in {self.output_path}")
                cv2.imwrite(f"{os.path.join(self.output_path, save_file_name)}", obj_img)
            return obj_img

# 将所有id相同的目标选出来
class GenerateIDImage(object):
    def __init__(self, image_filename, anno):
        self.image_filename = image_filename
        self.img = cv2.imread(
            os.path.join(DATA_PATH, image_filename)
        )
        self.anno = anno
        self.track_idinfo = {_['track_id']: _ for _ in anno}
        self.track_idlist = list(self.track_idinfo.keys())


    def selectObj(self, obj_id, show=True, save=False):
        if obj_id not in self.track_idinfo:
            print("tracker id is not annotations or not exist!")
            exit(-1)
        tl_x, tl_y, w, h = [int(_) for _ in self.track_idinfo[obj_id]['bbox']]
        obj_img = self.img[tl_y:tl_y + h, tl_x:tl_x + w, :]
        try:
            if show:
                cv2.imshow("single object image", obj_img)
                cv2.waitKey(0)
            if save:
                save_file_name = "tracker"+str(obj_id)+"_"+self.image_filename.replace("/img1/", "_")
                self.output_path = os.path.join(ROOT_PATH, 'id_info', str(obj_id))
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)
                print(f"filter result is saved in {self.output_path}")
                cv2.imwrite(f"{os.path.join(self.output_path, save_file_name)}", obj_img)
            return obj_img
        except:
            return

def LocalKP(img, processer_name):
    if type(img) is str:
        img = cv2.imread(img)
    process_res = builder.build_processer(img, processer_name)

def getOverlapObjsImage():
    # test for function <reJustifyKP>
    overlap_threld_list=(0, 0.1, 0.2, 0.4, 0.6, 1)
    with open(coco_path, 'r', encoding='utf8') as fp:
        data = json.load(fp)

    ImageInfos = data['images']
    for img_idx in ImageInfos[0:150]:
        Anno_info = []
        for iAnno in data['annotations']:
            if iAnno['image_id'] == img_idx['id']:
                Anno_info.append(iAnno)
            else:
                continue
        for anno in Anno_info:
            Obj_img = GenerateObjsImage(
                img_idx['file_name'], Anno_info,
                overlap_threld_list=overlap_threld_list
            ).selectObj(anno['track_id'], (0.2, 0.4), save=True, show=False)
            print("======================================")

def getIDObjsImage():
    # test for function <reJustifyKP>
    with open(coco_path, 'r', encoding='utf8') as fp:
        data = json.load(fp)

    ImageInfos = data['images']
    for img_idx in ImageInfos:
        Anno_info = []
        for iAnno in data['annotations']:
            if iAnno['image_id'] == img_idx['id']:
                Anno_info.append(iAnno)
            else:
                continue
        for anno in Anno_info:
            GenerateIDImage(
                img_idx['file_name'], Anno_info,
            ).selectObj(anno['track_id'], save=True, show=False)
            print("======================================")
if __name__ == '__main__':

    if (platform.system() == 'Windows'):
        # ROOT_PATH = 'D:\\data\\dataset\\track'
        ROOT_PATH = 'D:\\data\\perdestrain'
    elif (platform.system() == 'Linux'):
        ROOT_PATH = '/home/data/HJZ/MOT'

    race_path = 'train'  # 'train' or 'test'
    DATA_PATH = os.path.join(ROOT_PATH, race_path)
    # coco_path = os.path.join(ROOT_PATH, 'annotations/train.json')
    coco_path = os.path.join(ROOT_PATH, 'annotations_onlySDP/train.json')
    # coco_path = os.path.join(ROOT_PATH, 'annotations_onlySDP/train.json')

    # DemoData(
    #     COCO_PATH=coco_path,
    #     show_image_nums=100,
    #     image_freq=1,
    #     start_img_id=1
    # ).show()
    # # 截取重叠度为 overlap_threld_list 的目标
    # getOverlapObjsImage()
    # LocalKP(
    #     "D:\\data\\dataset\\track\\0.2_0.4\\tracker4_cruise1_000001.PNG",
    #     'SLIC'
    # )


    # 按照id截取目标
    getIDObjsImage()
