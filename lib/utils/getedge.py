#  -*- coding: utf-8 -*- 

import cv2
import os
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt 

def uniformsample(pgtnp_px2, newpnum):
    print(pgtnp_px2)
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)
    print(edgeidxsort_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2

    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(root, tolerance=0):

    mask = cv2.imread(root,0)
    mask = np.array(mask)
    mask = mask/128
    binary_mask = mask.astype(np.int)
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    #唐梓轩添加，因为默认处理单连通区域，所以这里取最大的单连通区域
    max=0
    if len(contours)!=1:
        for i,counter1 in enumerate(contours):
            if counter1.shape[0]>max:
                max=counter1.shape[0]
                contours=[counter1]
        
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    polys=[]
    for i in range(int(len(polygons[0])/2)):
        polys.append([polygons[0][2*i],polygons[0][2*i+1]])
    
    return np.array(polys)









if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    root = '/home/amax/Titan_Five/TZX/deep_sanke/images/4426_mask.jpg'	# 修改为你对应的文件路径
    poly = binary_mask_to_polygon(root)
    #poly=poly[0:256:2,:]
    #poly=uniformsample(poly,128)
    print(poly.shape)

    inp=cv2.imread("/home/amax/Titan_Five/TZX/snake_envo_num/visual_result/dark.png")
    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(inp)
    ax.plot(poly[:, 0], poly[:, 1], color='white', linewidth=5)
    plt.savefig('./visual_result/demo_result.png', bbox_inches='tight', pad_inches=0)


    #a=np.array(instance_polys[0]).astype(int)
    #print(a.shape)
    #a=a*4
    #for i in range(a.shape[1]):
    #    point=(a[0,i,0],a[0,i,1])
    #    print(point)
    #    cv2.circle(img1, point, 1, (255, 0, 0), 1)
    #cv2.imwrite('./visual_result/4_inp_with_instance_poly2.jpg', img1)
    #print("instance_polys",)
    #sys.exit(0)

    #Edge_Extract(root)  #shape(278,2)