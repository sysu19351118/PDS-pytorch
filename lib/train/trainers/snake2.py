import torch.nn as nn
from lib.utils import net_utils
import torch
import sys
import numpy as np
import cv2
from lib.config import cfg
from matplotlib import pyplot as plt

def intermediate_signal(gtpoly):
    decerate = [0.3, 0.3]
    circle_rate=1.0

    

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
        percentage1 = 1+ decerate[0]*gap_dist/mean_pdist
        percentage1 = percentage1.unsqueeze(dim=1)
        percentage1 = percentage1.repeat(1,2)

        percentage2 = 1+ decerate[1]*gap_dist/mean_pdist
        percentage2 = percentage2.unsqueeze(dim=1)
        percentage2 = percentage2.repeat(1,2)
        
        percentage1 = circle_rate*percentage1


        

        #根据缩放权重计算简化后的值
        if i==0:
            layer1_sup_poly = vector_poly_circle.mul(percentage1)+center_vector
            layer2_sup_poly = vector_poly_circle.mul(percentage2)+center_vector
            layer1_sup_poly = layer1_sup_poly.unsqueeze(dim=0)
            layer2_sup_poly = layer2_sup_poly.unsqueeze(dim=0)
        else:
            if i==2:
                A=vector_poly_circle.mul(percentage1)+center_vector
                B=gtpyiter
                P=percentage1
                print(percentage1)
            layer1_sup_poly_toappend = vector_poly_circle.mul(percentage1)+center_vector
            layer2_sup_poly_toappend = vector_poly_circle.mul(percentage2)+center_vector
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
    a=np.array(layer2_sup_poly[0].cpu()).astype(int)
    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(inp)
    ax.plot(a[:, 0], a[:, 1],"o", color='white', linewidth=5)
    plt.savefig("./visual_result/layer2.png", bbox_inches='tight', pad_inches=0)
    '''

    #画向量图
    inp=255-cv2.imread("/home/amax/Titan_Five/TZX/snake_envo_num/visual_result/dark1.png")
    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(inp)
    A=np.array(A.cpu()).astype(int)
    B=np.array(B.cpu()).astype(int)
    print(A.shape,B.shape)
    for i in range(128):
        if P[i,0]>1:
            ax.plot([A[i,0],B[i,0]], [A[i, 1],B[i, 1]], color='orange', linewidth=5)
        else:
            ax.plot([A[i,0],B[i,0]], [A[i, 1],B[i, 1]], color='cornflowerblue', linewidth=5)
    plt.savefig("./visual_result/offset.png", bbox_inches='tight', pad_inches=0)
    sys.exit(0)

    
    
    
        
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
        #print(output['py_pred'][3].shape)
        scalar_stats = {}
        loss = 0

        print(">>>>>>>>>>>>",batch.keys())
        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss
        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        scalar_stats.update({'wh_loss': wh_loss})
        loss += 0.1*wh_loss

        ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        scalar_stats.update({'ex_loss': ex_loss})
        loss += ex_loss

        py_loss = 0

        #output['py_pred'] = [output['py_pred'][-1]]

        #我们的创新点，使用loss来约束每个演化阶段的输出

        layer1_sup,layer2_sup=intermediate_signal(output['i_gt_py'])
        sup_singal = [layer1_sup, layer2_sup, output['i_gt_py']]
        for i in range(len(output['py_pred'])):
            py_loss += self.py_crit(output['py_pred'][i], sup_singal[i]) / len(output['py_pred'])

        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

