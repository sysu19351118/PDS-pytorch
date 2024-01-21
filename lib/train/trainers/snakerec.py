import torch.nn as nn
from lib.utils import net_utils
import torch
import sys
import numpy as np
import cv2
from lib.config import cfg
from matplotlib import pyplot as plt


def get_rec(mean_pdist, vector_poly):
    opreator = vector_poly/torch.abs(vector_poly)
    vector_poly=torch.abs(vector_poly)
    pi=3.1415926535897
    #gt上的点到中点的均值
    dist_sum = mean_pdist*vector_poly.shape[0]
    #计算夹角教tan值和cot值, sin 
    vector_tan=vector_poly[:,1]/vector_poly[:,0]
    vector_cot=vector_poly[:,0]/vector_poly[:,1]
    vector_sin=vector_poly[:,1]/torch.sqrt(vector_poly[:,0]**2+vector_poly[:,1]**2)
    vector_cos = vector_poly[:,0]/torch.sqrt(vector_poly[:,0]**2+vector_poly[:,1]**2)
    #计算正方形边长 dist_sum*(2*sin(135-theta的值)).sum
    a = dist_sum/((1/(np.sqrt(2)*torch.abs(vector_cos)+np.sqrt(2)*torch.abs(vector_sin) )).sum())

    
    rec = vector_poly
    #print(a/(np.sqrt(2)*(vector_cot+1 )))
    rec[:,0] = a*vector_poly[:,0]/(np.sqrt(2)*(vector_poly[:,0]+vector_poly[:,1]))
    rec[:,1] = a*vector_poly[:,1]/(np.sqrt(2)*(vector_poly[:,0]+vector_poly[:,1]))
    rec=rec*opreator
    
    
    return rec

def intermediate_signal(gtpoly):
    decerate = cfg.layer1rate
    circle_rate=cfg.circle_rate

    for i in range(gtpoly.shape[0]):
        gtpyiter = gtpoly[i,:,:]      #128*2
        
        # 首先计算poly的中心 扩展成128
        center = torch.mean(gtpyiter,dim=0)
        center_vector = center.repeat(128,1)

        #计算每个点到中心的欧式距离 以及均值
        pdist = nn.PairwiseDistance(p=2)
        pdist_result = pdist(gtpyiter,center_vector)
        mean_pdist = torch.mean(pdist_result)

        
        #得到基准正方形
        vector_poly = gtpyiter-center
        pdist_result1 = pdist_result.unsqueeze(dim=1).repeat(1,2)
        
        vector_poly_rec = get_rec(mean_pdist,vector_poly)
        
        #计算每个点到正方形上的差值
        #print(pdist_result.shape,pdist(vector_poly_rec.shape))
        gap_dist = pdist_result-pdist(vector_poly_rec,torch.zeros_like(center_vector))
        #print(pdist(vector_poly_rec,center_vector))
        #计算缩放权重
        percentage1 = 1+ decerate[0]*gap_dist/pdist(vector_poly_rec,torch.zeros_like(center_vector))
        percentage1 = percentage1.unsqueeze(dim=1)
        percentage1 = percentage1.repeat(1,2)

        percentage2 = 1+ decerate[1]*gap_dist/pdist(vector_poly_rec,torch.zeros_like(center_vector))
        percentage2 = percentage2.unsqueeze(dim=1)
        percentage2 = percentage2.repeat(1,2)
        
        percentage1 = circle_rate*percentage1
       
        if i==0:
            ac =gtpyiter
        #根据缩放权重计算简化后的值
        if i==0:
            layer1_sup_poly = vector_poly_rec.mul(percentage1)+center_vector
            layer2_sup_poly = vector_poly_rec.mul(percentage2)+center_vector
            layer1_sup_poly = layer1_sup_poly.unsqueeze(dim=0)
            layer2_sup_poly = layer2_sup_poly.unsqueeze(dim=0)
        else:
            layer1_sup_poly_toappend = vector_poly_rec.mul(percentage1)+center_vector
            layer2_sup_poly_toappend = vector_poly_rec.mul(percentage2)+center_vector
            layer1_sup_poly_toappend = layer1_sup_poly_toappend.unsqueeze(dim=0)
            layer2_sup_poly_toappend = layer2_sup_poly_toappend.unsqueeze(dim=0)
            layer1_sup_poly = torch.cat((layer1_sup_poly,layer1_sup_poly_toappend),dim=0)
            layer2_sup_poly = torch.cat((layer2_sup_poly,layer2_sup_poly_toappend),dim=0)
    if circle_rate>1:
        up_thresh = torch.ones((layer1_sup_poly.shape))*544
        up_thresh = up_thresh.to("cuda",dtype = torch.float32)
        down_thresh = torch.zeros((layer1_sup_poly.shape)).to('cuda')
        layer1_sup_poly = torch.where(layer1_sup_poly > 544, up_thresh, layer1_sup_poly)
        layer1_sup_poly = torch.where(layer1_sup_poly < 0 , down_thresh, layer1_sup_poly)

    '''
    inp=cv2.imread("/home/amax/Titan_Five/TZX/snake_envo_num/visual_result/dark1.png")

    a=np.array(layer1_sup_poly[0].cpu()).astype(int)
    b=np.array(ac.cpu()).astype(int)
    c=np.array(layer2_sup_poly[0].cpu()).astype(int)
    #print(a)
    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(inp)
    ax.plot(a[:, 0], a[:, 1],"o", color='white', linewidth=5)
    ax.plot(b[:, 0], b[:, 1],"o", color='white', linewidth=5)
    #ax.plot(c[:, 0], c[:, 1],"o", color='white', linewidth=5)
    plt.savefig("./visual_result/rectL1.png", bbox_inches='tight', pad_inches=0)
    '''
    return layer1_sup_poly, layer2_sup_poly




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
        
        output = self.net(batch['inp'], batch)
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

        #output['py_pred'] = [output['py_pred'][-1]]

        #我们的创新点，使用loss来约束每个演化阶段的输出
        layer1_sup,layer2_sup=intermediate_signal(output['i_gt_py'])
        sup_singal = [layer1_sup,layer2_sup,output['i_gt_py']]
        for i in range(len(output['py_pred'])):
            py_loss += self.py_crit(output['py_pred'][i], sup_singal[i]) / len(output['py_pred'])

        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

