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


class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.imgs=[os.path.join(cfg.demo_path, img_name) for img_name in os.listdir(cfg.demo_path)]

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

def poly2mask(ex):

    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() * snake_config.down_ratio

    img = np.zeros((512,512))
    ex = np.array(ex)
    ex = ex.astype(np.int32)
    img = cv2.polylines(img,[ex[0]],True,1,1)
    img = cv2.fillPoly(img, [ex[0]], 1)
    return img


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

def visual():
    
        network = make_network(cfg).cuda()
        print(cfg.model_dir)
        load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
        
        network.eval()
        dataset = Dataset()

        for batch, path, img in tqdm.tqdm(dataset):
            try:
                print(path)
                batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
                batch['orig_img'] = torch.FloatTensor(batch['orig_img'])[None].cuda()
                with torch.no_grad():
                    output= network(batch['inp'],  batch)
                img = cv2.imread(path)
                print(len(output['py']))
                for i in range(128):
                    cv2.line(img, (int(output['py'][2][0,i,0].item()),int(output['py'][2][0,i,1].item())), (int(output['py'][2][0,(i+1)%128,0].item()),int(output['py'][2][0,(i+1)%128,1].item())), color=(255, 0, 255), thickness=2)
                cv2.imwrite(path.replace("input","output3"),img)
            except:
                continue

        


        


