mport cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline
image_path='/home/aistudio/train/domain1/000049601024979.jpg'
xml_path='/home/aistudio/train/domain1/XML/000049601024979.xml'
xmin=[83,180,189]
ymin=[364,361,410]
xmax=[181,260,286]
ymax=[407,402,452]
im = cv2.imread(image_path) 
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(xmin)):
    cv2.rectangle(im, (xmin[i], ymin[i]), (xmax[i], ymax[i]), (0, 255, 0), 4)
    #cv2.putText(im, '{:s} {:.3f}'.format(cls, score), (xmin, ymin), font, 0.5, (255, 0, 0), thickness=2)

cv2.imwrite('result_real.jpg', im)
plt.figure(figsize=(15,12))
plt.imshow(im[:, :, [2,1,0]])
plt.show()
