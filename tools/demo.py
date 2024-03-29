import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_config
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from process.metrics import FBound_metric, WCov_metric
import random
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def calculate_iou_from_mask_and_box(mask, detected_box):
    # 确保 mask 是 NumPy 数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # 提取边界框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    gold_standard_box = np.array([xmin, ymin, xmax, ymax])

    # 确保 detected_box 是在 CPU 上的 NumPy 数组
    if detected_box.is_cuda:
        detected_box = detected_box.cpu().numpy()
    else:
        detected_box = detected_box.numpy()

    # 四舍五入检测框的坐标
    detected_box_rounded = np.round(detected_box)

    # 计算交集
    x1_inter = max(gold_standard_box[0], detected_box_rounded[0])
    y1_inter = max(gold_standard_box[1], detected_box_rounded[1])
    x2_inter = min(gold_standard_box[2], detected_box_rounded[2])
    y2_inter = min(gold_standard_box[3], detected_box_rounded[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 计算并集
    gold_standard_area = (gold_standard_box[2] - gold_standard_box[0]) * (gold_standard_box[3] - gold_standard_box[1])
    detected_box_area = (detected_box_rounded[2] - detected_box_rounded[0]) * (detected_box_rounded[3] - detected_box_rounded[1])
    union_area = gold_standard_area + detected_box_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()


        self.img_paths=[]
        img_names = os.listdir(cfg.data_path)
        self.img_paths = []
        self.mask_paths = []
        
        img_num=len(img_names)/2

        

        for i in range(int(img_num)):
            self.img_paths.append(cfg.data_path+"/"+str(i)+'_image.jpg')
            self.mask_paths.append(cfg.data_path+"/"+str(i)+'_mask.jpg')

        self.ff_num = int(cfg.ff_num)
        self.train_img_path = []
        self.train_mask_path = []
        self.test_img_path = []
        self.test_mask_path = []

        split_num = int(len(img_names)/10)


        if self.ff_num!=4:
            self.test_img_path.extend(self.img_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])
            self.test_mask_path.extend(self.mask_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])
        else:
            self.test_img_path.extend(self.img_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])
            self.test_mask_path.extend(self.mask_paths[split_num*self.ff_num:split_num*(self.ff_num+1)])

        self.imgs=self.test_img_path
        

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        
        img = cv2.resize(img,(512,512))
        imgreturn = img
        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = 512
        input_h = 512
        
        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        img_org = inp
        img_org = (img_org.astype(np.float32))
        inp = self.normalize_image(inp)
        
        
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': ''}
        ret.update({"orig_img":img_org})
        ret.update({'meta': meta})

        return ret, img_path, imgreturn

    def __len__(self):
        return len(self.imgs)

def draw_and_save(points, image, save_path):
    points = np.round(points.cpu().numpy()).astype(np.int32)
    reshaped_points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [reshaped_points], True, (255, 0, 255), 2)
    cv2.imwrite(save_path, image)


def poly2mask(ex):

    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() * snake_config.down_ratio

    img = np.zeros((512,512))
    ex = np.array(ex)
    ex = ex.astype(np.int32)
    img = cv2.polylines(img,[ex[0]],True,1,1)
    img = cv2.fillPoly(img, [ex[0]], 1)
    return img

def calculate_stats_np(data_list):
    if not data_list:  # 检查列表是否为空
        return "List is empty. Cannot compute statistics."

    # 将列表转换为 NumPy 数组
    data_array = np.array(data_list)

    # 计算方差、最大值和最小值
    variance = np.var(data_array, ddof=1)  # 设置 ddof=1 以获得样本方差
    max_value = np.max(data_array)
    min_value = np.min(data_array)

    return variance, max_value, min_value

def cal_iou(mask, gtmask):
    jiaoji = mask*gtmask
    bingji = ((mask+gtmask)!=0).astype(np.int16)
    return jiaoji.sum()/bingji.sum()

def cal_dice(iou):
    return 2*iou/(iou+1)

def cal_fbound(mask, gt_mask):
    return FBound_metric(mask, gt_mask)

def cal_wcov(mask, gt_mask):
    return WCov_metric(mask, gt_mask)

def demo():
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    
    network.eval()
    nodect=0   
    dataset = Dataset()
    print(len(dataset))
    iousum=0
    dice_sum=0
    fbound_sum=0
    wc_sum=0
    img_num=0 
    counter=0
    bbox_iou = 0
    bbox_iou_list = []
    seg_iou_list = []
    ioudict = {}
    for batch, path, img in tqdm.tqdm(dataset):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        batch['orig_img'] = torch.FloatTensor(batch['orig_img'])[None].cuda()
        with torch.no_grad():
            output= network(batch['inp'],  batch)
        tosave_name = os.path.basename(path)
        
        output = output

        # try:

        mask = poly2mask(output['py'])
        mask = cv2.resize(mask,(128,128))

        gt_mask_path = path.replace("_image","_mask")
        gt_mask = cv2.imread(gt_mask_path)
        gt_mask = cv2.resize(gt_mask,(128,128))
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        gt_mask = cv2.threshold(gt_mask, 100, 1, cv2.THRESH_BINARY)[1]
        iou = cal_iou(mask, gt_mask)
        seg_iou_list.append(iou)
        dice_sum += cal_dice(iou)
        fbound_sum += cal_fbound(mask, gt_mask)
        wc_sum += cal_wcov(mask, gt_mask)
        biou = calculate_iou_from_mask_and_box(gt_mask, output["detection"][0])
        bbox_iou_list.append(biou)

  

        ioudict[os.path.basename(gt_mask_path)]=str(biou)+"_"+str(cal_iou(mask, gt_mask))
        bbox_iou+= biou

        
        
        counter+=1

        iousum+=iou
        img_num+=1
        # except:
        #     counter+=1
        #     nodect+=1
        #     continue
        
    fangcha, max1,min1 = calculate_stats_np(bbox_iou_list)
    print("mean iou is :",iousum/img_num)
    print("mean dice is :",dice_sum/img_num)
    print("mean boundf is :",fbound_sum/img_num)
    print("mean wc is :",wc_sum/img_num)
    print("bboxiou wc is :",bbox_iou/img_num)
    print('方差 最大值 最小值：', fangcha, max1, min1)
    # print("小于0.8的biou时的iou:",ioudict)
    
    print("nodect",nodect) 
