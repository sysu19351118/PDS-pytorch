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
        pretrain_model_dict = torch.load(cfg.unet_pretrain_model)
        self.unet.load_state_dict(pretrain_model_dict)

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
        #save_image(unet_input[0]/255,'/data/tzx/snake_envo_num/visual_result/debug/unetinput.png')
        x_input_unet = F.interpolate(unet_input, scale_factor=0.25,mode='nearest')
        output, cnn_feature = self.dla(x)

        unet_mapfeature = self.unet(x_input_unet)
        #print(unet_mapfeature,unet_mapfeature.max(),unet_mapfeature.min(), unet_mapfeature.mean())
        #sys.exit(0)
        zeros = torch.zeros_like(unet_mapfeature)
        unet_mapfeature = torch.where(unet_mapfeature < 0, zeros, unet_mapfeature)
        unet_mapfeature = 255*(unet_mapfeature-unet_mapfeature.min())/(unet_mapfeature.max()-unet_mapfeature.min())
 
        #print(unet_mapfeature.shape)
        
        #save_image(x[0],'/data/tzx/snake_envo_num/visual_result/debug/x.png')
        #save_image(unet_mapfeature[0],'/data/tzx/snake_envo_num/visual_result/debug/mape_visual.png')
        #sys.exit(0)
        #print(cnn_feature.shape)

        #cnn feature [batch_size/num_gpus, 64, 128, 128]
        #out put ['ct_hm', 'wh']  [-, 2, 128, 128] [-, 2, 128, 128]

        

        with torch.no_grad():
            ct, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

        if cfg.use_gt_det:
            self.use_gt_detection(output, batch)
            #print(output['ct_hm'].shape,output['wh'].shape,output['ct'].shape,output['detection'].shape)
        cnn_feature = torch.cat([cnn_feature, unet_mapfeature], dim=1)
        output = self.gcn(output, cnn_feature, batch)
        #print(output['wh'].shape) #['ct_hm', 'wh', 'ct', 'detection', 'it_ex', 'ex', 'it_py', 'py']

        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
