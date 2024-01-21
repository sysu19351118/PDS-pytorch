import torch.nn as nn
from lib.utils import net_utils
import torch
import sys
import numpy as np
import cv2
from lib.config import cfg
import numpy as np
import numpy.linalg as LA





def intermediate_signal(gtpoly):
    decerate = cfg.layer1rate
    circle_rate=cfg.circle_rate
    sup_polys=[]
    for rate in decerate:
        for i in range(gtpoly.shape[0]):
            gtpyiter = gtpoly[i,:,:]      #128*2
            # 首先计算poly的中心 扩展成128
            center = torch.mean(gtpyiter,dim=0)
            center_vector = center.repeat(128,1)
            #计算每个点到中心的欧式距离 以及均值
            pdist = nn.PairwiseDistance(p=2)
            pdist_result = pdist(gtpyiter,center_vector)
            mean_pdist = torch.mean(pdist_result)
            #计算poly中心点向每个点发从的单位矢量 并乘以均值得到一个圆
            vector_poly = gtpyiter-center
            pdist_result1 = pdist_result.unsqueeze(dim=1).repeat(1,2)         
            vector_poly_circle = mean_pdist*vector_poly/pdist_result1
            #计算每个点到center距离与均值的差值
            gap_dist = pdist_result-mean_pdist
            #计算缩放权重
            percentage1 = 1+ rate*gap_dist/mean_pdist
            percentage1 = percentage1.unsqueeze(dim=1)
            percentage1 = percentage1.repeat(1,2)
            percentage1 = circle_rate*percentage1
            #根据缩放权重计算简化后的值
            if i==0:
                layer1_sup_poly = vector_poly_circle.mul(percentage1)+center_vector
                layer1_sup_poly = layer1_sup_poly.unsqueeze(dim=0)
            else:
                layer1_sup_poly_toappend = vector_poly_circle.mul(percentage1)+center_vector
                layer1_sup_poly_toappend = layer1_sup_poly_toappend.unsqueeze(dim=0)
                layer1_sup_poly = torch.cat((layer1_sup_poly,layer1_sup_poly_toappend),dim=0)
        sup_polys.append(layer1_sup_poly)   
    return sup_polys




class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss


    def forward(self, batch):
        output  = self.net(batch['inp'], batch) #需要用原图输入的话output  = self.net(batch['inp'], batch['orig_img'], batch)
        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += cfg.ct_weight*ct_loss
        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        scalar_stats.update({'wh_loss': wh_loss})
        loss += cfg.wh_weight*wh_loss

        ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        scalar_stats.update({'ex_loss': ex_loss})
        loss += ex_loss
        py_loss = 0        
        #我们的创新点，使用loss来约束每个演化阶段的输出
        sup_polys=intermediate_signal(output['i_gt_py'])
        
        ifmultistage=cfg.ifmultistage
        if ifmultistage:
            sup_singal = []
            for poly in sup_polys:
                sup_singal.append(poly)
            sup_singal.append(output['i_gt_py'])
            for i in range(len(output['py_pred'])):
                py_loss += self.py_crit(output['py_pred'][i], sup_singal[i]) / len(output['py_pred'])
        else:   
            output['py_pred'] = [output['py_pred'][-1]]
            for i in range(len(output['py_pred'])):
                py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss
        

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

