import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import cv2

def gaussian_filter(shape, sigma):
    x, y = [int(np.floor(edge / 2)) for edge in shape]
    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    filt = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    filt /= np.sum(filt)
    var = np.zeros((1, 1, shape[0], shape[1]))
    var[0, 0, :, :] = filt
    return torch.tensor(np.float32(var))


class UNet(nn.Module):
    def __init__(self, in_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.fcEp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcE = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)
        self.gaussE = torch.nn.Conv2d(1, 1, (9, 9), bias=False, padding_mode='reflect', padding=4)
        self.gaussE.weight = torch.nn.Parameter(gaussian_filter((9, 9), 2), requires_grad=False)

        self.fcAp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcA = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)
        self.fcBp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcB = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 通道预测
        h_fcEp = self.fcEp(dec1)
        h_fcE = self.fcE(h_fcEp)
        #filteredE = torch.nn.functional.conv2d(h_fcE, self.gaussE.weight, padding=4)

        h_fcAp = self.fcAp(dec1)
        h_fcA = self.fcA(h_fcAp)
        h_fcBp = self.fcBp(dec1)
        h_fcB = self.fcB(h_fcBp)

        return h_fcE, h_fcA, h_fcB

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ =="__main__":
    import numpy as np
    net = UNet().cpu()
    #pretrain_model_dict = torch.load('/home/amax/Titan_Five/TZX/PDS/lib/networks/snake/ADMIRE_model_19.pth',map_location='cpu')
    #net.load_state_dict(pretrain_model_dict)
    #img = cv2.imread("/home/amax/Titan_Five/TZX/deep_sanke/images/0_image.jpg")/255
    #img = torch.tensor(img).cpu().unsqueeze(dim=0).transpose(1,3).to(torch.float32)
    img=torch.zeros((8,3,128,128))
    a = net(img)
    print(a.shape)