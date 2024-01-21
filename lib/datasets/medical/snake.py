import os
from lib.utils.snake import snake_voc_utils, snake_config, visualize_utils
from lib.utils.getedge import binary_mask_to_polygon
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg
import sys
import torch
from  torchvision import utils as vutils
import time
import random


class Dataset(data.Dataset):
    def __init__(self,  data_root, ann_file, split, ff_num1=cfg.ff_num):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.data_root = cfg.data_path
        self.split = split

        self.coco = []
        self.anns = []
        self.json_category_id_to_contiguous_id = []

        #编写medical数据读取
        img_names = os.listdir(self.data_root)
        self.img_paths = []
        self.mask_paths = []
        
        img_num=len(img_names)/2

        

        for i in range(int(img_num)):
            self.img_paths.append(cfg.data_path+'/'+str(i)+'_image.jpg')
            self.mask_paths.append(cfg.data_path+'/'+str(i)+'_mask.jpg')

        self.ff_num = ff_num1     #五折训练，测试集折数
        self.train_img_path = []  #用于存放训练图像
        self.train_mask_path = [] #用于存放训练掩码
        self.test_img_path = []   #用于测试训练图像
        self.test_mask_path = []    #用于存放测试掩码

        split_num = int(len(img_names)/10)   #计算每一折的数据量
        
        
        #用切片的方法取出所有的训练数据，注意第4600组数据全部归为训练集
        self.train_img_path.extend(self.img_paths[0:split_num*self.ff_num])
        self.train_img_path.extend(self.img_paths[split_num*(self.ff_num+1):])
        self.train_mask_path.extend(self.mask_paths[0:split_num*self.ff_num])
        self.train_mask_path.extend(self.mask_paths[split_num*(self.ff_num+1):])
        
        #用切片的方法取出所有的测试数据
        if self.ff_num!=4:
            self.test_img_path.extend(self.img_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])
            self.test_mask_path.extend(self.mask_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])
        else:
            self.test_img_path.extend(self.img_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])
            self.test_mask_path.extend(self.mask_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])



        print("训练数据量：{}".format(len(self.train_img_path)))

        

    def mask2poly(self, mask_path):
        poly_to_return=[]
        poly = []
        poly.append(binary_mask_to_polygon(mask_path))
        poly_to_return.append(poly)
        return [poly]

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            
            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_voc_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_voc_utils.filter_tiny_polys(instance)
            polys = snake_voc_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_voc_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
        
        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_detection_(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]  #cls_id 对应类别
        
        ct_cls.append(cls_id)  

        x_min, y_min, x_max, y_max = box
        box_ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

        x_min_int, y_min_int = int(x_min), int(y_min)
        h_int, w_int = math.ceil(y_max - y_min_int) + 1, math.ceil(x_max - x_min_int) + 1
        max_h, max_w = ct_hm.shape[0], ct_hm.shape[1]
        h_int, w_int = min(y_min_int + h_int, max_h) - y_min_int, min(x_min_int + w_int, max_w) - x_min_int

        mask_poly = poly - np.array([x_min_int, y_min_int])
        mask_ct = box_ct - np.array([x_min_int, y_min_int])
        ct, off, xy = snake_voc_utils.prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int)

        xy += np.array([x_min_int, y_min_int])
        ct += np.array([x_min_int, y_min_int])

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_voc_utils.get_init(box)
        img_init_poly = snake_voc_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_voc_utils.get_octagon(extreme_point)
        img_init_poly = snake_voc_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)

    def __getitem__(self, index):
        
        cls_ids=[0]
        if self.split=="train":
            mask_path =self.train_mask_path[index]
            img_path = self.train_img_path[index]
        else:
            mask_path =self.test_mask_path[index]
            img_path = self.test_img_path[index]


        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        mask = cv2.imread(img_path.replace('_image', "_mask"))
        mask = cv2.resize(mask, (128, 128))

        instance_polys = self.mask2poly(mask_path)
        
        height, width = img.shape[0], img.shape[1]#图像的长宽

        #数据增强操作
        orig_img, mask, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                img, mask, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        mask = cv2.resize(mask, (128, 128))
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        extreme_points = self.get_extreme_points(instance_polys) #instance的上下左右四个极值点



        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        for i in range(len(cls_ids)):#遍历图片中标记的物体
            cls_id = cls_ids[i]                  # cls_id:图像中被分割物体的类别
            instance_poly = instance_polys[i]    # instance_poly图像中物体的边界点
            instance_points = extreme_points[i]  # instance_points图像中物体上下左右四个方位的极点                
            
            
            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]  #由poly的上下左右四个极值点确定
                h, w = y_max - y_min + 1, x_max - x_min + 1 #bbox的长宽
                if h <= 1 or w <= 1:                 
                    continue

                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.prepare_init(bbox, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
        ret = {'inp': inp}
        orimg = {"orig_img" : orig_img}
        maskdict = {"mask" : np.float32(mask)}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        ret.update(orimg)
        ret.update(maskdict)
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)
        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': [], 'ann': [], 'ct_num': ct_num}
        ret.update({'meta': meta})
        
        return ret

    def __len__(self):
        if self.split == 'train':
            a=len(self.train_img_path)
        else:
            a=len(self.test_img_path)
        return a



