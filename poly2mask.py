import cv2
import numpy as np

x = [70,222,280,330,467,358,392,280,138,195]

y = [190,190,61,190,190,260,380,308,380,260]

cor_xy = np.vstack((x, y)).T
print(cor_xy.shape)
img=np.zeros((512,512))
print((img==1).sum())

img = cv2.polylines(img,[cor_xy],True,1,1)
print((img==1).sum())

img = cv2.fillPoly(img, [cor_xy], 1)
print((img==1).sum())

cv2.imwrite('/home/amax/Titan_Five/TZX/snake-master/visual_result/poly2mask.jpg',img*255)
