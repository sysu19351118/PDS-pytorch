import torch
import torch.nn as nn
import torch.nn.functional as F
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class CrossAttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.attention_block = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=8)
        factor = 2 if bilinear else 1
        #self.down4 = (Down(512, 1024 // factor))
        #self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, xpre, xmid, xlate):
        #计算第一个Unet块的特征
        xmid1 = self.inc(xmid)
        xpre1 = self.inc(xpre)
        xlate1 = self.inc(xlate)
        #计算第一个Unet下采样的结果
        xmid2 = self.down1(xmid1)
        xpre2 = self.down1(xpre1)
        xlate2 = self.down1(xlate1)
        #第2个
        xmid3 = self.down2(xmid2)
        xpre3 = self.down2(xpre2)
        xlate3 = self.down2(xlate2)
        #第3个
        xmid4 = self.down3(xmid3)
        xpre4 = self.down3(xpre3)
        xlate4 = self.down3(xlate3)
        #view成attention需要的形状
        bs,ntoken,_,_ = xmid4.shape
        xmid4=xmid4.view(bs,ntoken,-1)
        xpre4=xpre4.view(bs,ntoken,-1)
        xlate4=xlate4.view(bs,ntoken,-1)
        xmid4 = self.attention_block(xmid4,xpre4,xlate4)
        xmid4 = xmid4.view(bs,ntoken,16,16)
        #1上采样
        xmid = self.up2(xmid4, xmid3)

        #2上采样
        xmid = self.up3(xmid, xmid2)

        #3上采样
        xmid = self.up4(xmid, xmid1)
        return xmid


    
if __name__=="__main__":
    a=torch.zeros((32,64,128,128))
    net = UNet(64,2)
    print(net(a,a,a).shape)