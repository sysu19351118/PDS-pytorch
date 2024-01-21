import torch.nn as nn
from .dla import DLASeg
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode
import torch
import torch.nn.functional as F
from lib.config import cfg 
from .unethead import UNet
import time
from torchvision.utils import save_image
import sys


from math import sqrt
import torch
import torch.nn as nn

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention

class CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query和key，第二个参数用于计算value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num
        assert all_head_size % head_num == 0
        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)
        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,x,y,attention_mask):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q和k的输入，y作为v的输入
        """
        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        k_s = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        attention_mask = attention_mask.eq(0)
        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        output = self.linear_output(attention)

        return output


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)
        self.gcn = Evolution()
        self.unet = UNet()
        self.attention = CrossAttention()

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def use_gt_detection(self, output, batch):
        _, _, height, width = output['ct_hm'].size()
        ct_01 = batch['ct_01'].byte()

        ct_ind = batch['ct_ind'][ct_01]

        xs, ys = ct_ind % width, ct_ind // width
        xs, ys = xs[:, None].float(), ys[:, None].float()
        ct = torch.cat([xs, ys], dim=1)

        wh = batch['wh'][ct_01]
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=1)
        score = torch.ones([len(bboxes)]).to(bboxes)[:, None]
        ct_cls = batch['ct_cls'][ct_01].float()[:, None]
        detection = torch.cat([bboxes, score, ct_cls], dim=1)
        
        output['ct'] = ct[None]
        output['detection'] = detection[None]
        return output

    def forward(self, x, unet_input, batch=None):
        #x:[batch_size/num_gpus, 3, 512, 512]
        unet_input = unet_input.to(torch.float32)
        unet_input = unet_input.transpose(2,3)
        unet_input = unet_input.transpose(1,2)
        x_input_unet = F.interpolate(unet_input, scale_factor=0.25,mode='nearest')
        save_image(x_input_unet/255, "/data/tzx/AADebug_img/x_input_unet.jpg")

        output, cnn_feature = self.dla(x)

        mapE, mapA, mapB = self.unet(x_input_unet)
        
        unet_mapfeature = torch.cat([mapE, mapA, mapB ], dim=1)
        zeros = torch.zeros_like(unet_mapfeature)
        unet_mapfeature = torch.where(unet_mapfeature < 0, zeros, unet_mapfeature)
        unet_mapfeature = 255*(unet_mapfeature-unet_mapfeature.min())/(unet_mapfeature.max()-unet_mapfeature.min())
  

        

        with torch.no_grad():
            ct, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

        if cfg.use_gt_det:
            self.use_gt_detection(output, batch)
        cnn_feature = self.attention(cnn_feature, unet_mapfeature) #使用Cross attention进行融合
        #cnn_feature = torch.cat([cnn_feature, unet_mapfeature], dim=1) #原本的，直接concat
        output = self.gcn(output, cnn_feature, batch)
        return output, mapE, mapA, mapB 


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
