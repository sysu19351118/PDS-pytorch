import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageMath
import sys

def draw_poly(poly,values,im_shape,brush_size):
    """ Returns a MxN (im_shape) array with values in the pixels crossed
    by the edges of the polygon (poly). total_points is the maximum number
    of pixels used for the linear interpolation.
    """
    u = poly[:,0]
    v = poly[:,1]
    b = np.round(brush_size/2)
    image = Image.fromarray(np.zeros(im_shape))
    image2 = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    if type(values) is int:
        values = np.ones(np.shape(u)) * values  # 全1矩阵再乘上values
    for n in range(len(poly)):
        d.ellipse([(v[n]-b,u[n]-b),(v[n]+b,u[n]+b)], fill=values[n])  # 好像在画一个椭圆
        image2 = ImageMath.eval("convert(max(a, b), 'F')", a=image, b=image2)
    return torch.from_numpy(np.array(image2))  # 蛇上点及其4邻域的点上的值是5，其他地方的值是0

def draw_poly_fill(poly,im_shape,values=1):
    """Returns a MxN (im_shape) array with 1s in the interior of the polygon
    defined by (poly) and 0s outside."""
    u = poly[:, 0]
    v = poly[:, 1]
    image = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    if not values == 1:
        if (values) is int:
            values = np.ones(np.shape(u)) * values
    d.polygon(np.column_stack((v, u)).reshape(-1).tolist(), fill=values, outline=1)
    return np.array(image)

def batch_mask_convert(contours, im_shape):
    '''
    Returns masks in (imH, imW, batchno), 0-1 binary, PyTorch Tensor
    '''
    batch_mask = np.zeros([contours.shape[0], im_shape[0], im_shape[1]])

    for i in range(contours.shape[0]):
        batch_mask[i,:,:] = draw_poly_fill(contours[i,:,:].detach().cpu().numpy(), im_shape, values=1)

    return torch.from_numpy(batch_mask)


def GTpoly(poly, im_shape, brush_size, GTmask):
    GTmask = GTmask.cpu().numpy()
    # GTmask是一个0-1之间的float，np.round之后变为准确掩膜
    #ret, GTmask = cv2.threshold(GTmask, 80, 255, cv2.THRESH_BINARY)
    side = cv2.Canny(GTmask.astype(np.uint8), 200, 255)
    mask_contour, thresh = cv2.findContours(side, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = poly.cpu().numpy()

    imsize = im_shape[0]

    u = poly[:,0]
    v = poly[:,1]
    b = np.round(brush_size/2)
    image = Image.fromarray(np.zeros(im_shape))
    image2 = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)

    values = np.ones(np.shape(u))
    for i in range(v.shape[0]):
        dist = cv2.pointPolygonTest(mask_contour[0],[v[i],u[i]],True)
        if dist==0: # 在边界上
            values[i] = 0
        elif abs(dist) > 5: # 向内/外偏移得较远
            values[i] = abs(dist/imsize * 10)
        else:
            values[i] = abs(dist/imsize * 3)

    for n in range(len(poly)):
        d.ellipse([(v[n]-b,u[n]-b),(v[n]+b,u[n]+b)], fill=values[n])  # 实际上list里画的是椭圆形的边界框，这样子就是在上边点点
        image2 = ImageMath.eval("convert(max(a, b), 'F')", a=image, b=image2)
    return torch.from_numpy(np.array(image2))  # 蛇上点及其4邻域的点上的值是5，其他地方的值是0

def derivatives_poly(poly):
    """
    :param poly: the Lx2 polygon array [u,v]
    :return: der1, der1, Lx2 derivatives arrays
    """
    
    poly = poly.cpu().numpy()
    u = poly[:, 0]
    v = poly[:, 1]
    L = len(u)
    der1_mat = -np.roll(np.eye(L), -1, axis=1) + \
               np.roll(np.eye(L), -1, axis=0)  # first order derivative, central difference
    # 上句构造的矩阵，主对角线是0，然后上面一层都是1，下面一层都是-1，其他的都是0。那和原有向量乘完了，就是一阶隔项差分用的。
    
    der2_mat = np.roll(np.eye(L), -1, axis=0) + \
               np.roll(np.eye(L), -1, axis=1) - \
               2 * np.eye(L)  # second order derivative, central difference
    # 主对角线是-2，然后上面一层和下面一层都是1，其他的都是0。那和原有向量乘完了，就是二阶差分用的。
    der1 = np.sqrt(np.power(np.matmul(der1_mat, u), 2) + \
                   np.power(np.matmul(der1_mat, v), 2))  # 蛇上每一点的一阶差分的模(对u坐标和v坐标上的差分，平方-相加-开方)，长度为L。
    der2 = np.sqrt(np.power(np.matmul(der2_mat, u), 2) + \
                   np.power(np.matmul(der2_mat, v), 2))  # 蛇上每一点的二阶差分的模(对u坐标和v坐标上的差分，平方-相加-开方)，长度为L。
    return torch.from_numpy(der1), torch.from_numpy(der2)