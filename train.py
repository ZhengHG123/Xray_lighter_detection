lso add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
#import sys 
#sys.path.append('/home/aistudio/external-libraries')
print("####")
####################################
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

os.chdir('/home/aistudio/work/')
####################################
from random import shuffle

base = 'train'

domain_num = 1

raw_imgs = []

for i in range(1, domain_num+1):

  xml_base = os.path.join(base, 'domain{}/XML'.format(i))
  print(xml_base)
  pts = os.listdir(xml_base)
  for pt in pts:
    if pt.endswith('.xml'):
      pt = os.path.join('domain{}/XML'.format(i), pt)
      img_pt = pt.replace('XML/', '').replace('.xml', '.jpg')

      raw_imgs.append((img_pt, pt))

print('total_num:', len(raw_imgs))
print(raw_imgs[0])
####################################
shuffle(raw_imgs)

with open(os.path.join(base, 'train_list.txt'), 'w') as f:
    for im in raw_imgs[:-200]:
        info = '{} {}\n'.format(im[0], im[1]).replace('train/', '')
        f.write(info)

print(info)

with open(os.path.join(base, 'val_list.txt'), 'w') as f:
    for im in raw_imgs[-200:]:
        info = '{} {}\n'.format(im[0], im[1]).replace('train/', '')
        f.write(info)

print(info)
#############################################
from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608,interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])
#############################################
base = './train/'

train_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=base+'train_list.txt',
    label_list='label_list.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=base+'val_list.txt',
    label_list='label_list.txt',
    transforms=eval_transforms)
###############################################3
## 使用ET模块解析xml标注文件

import xml.etree.ElementTree as ET
import numpy as np

tar_size = 608

def load_one_info(name):

    filename = os.path.join(base, name)

    tree = ET.parse(filename)
    size = tree.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    ratio = min(tar_size / width, tar_size / height)

    Objects = tree.findall('object')
    objs_num = len(Objects)
    Boxes = np.zeros((objs_num, 4), dtype=np.float32)
    True_classes = np.zeros((objs_num), dtype=np.float32)
    
    result = []
    for i, obj in enumerate(Objects):

        bbox = obj.find('bndbox')

        x_min = float(bbox.find('xmin').text) - 1
        y_min = float(bbox.find('ymin').text) - 1
        x_max = float(bbox.find('xmax').text) - 1
        y_max = float(bbox.find('ymax').text) - 1

        w = ratio * (x_max - x_min)
        h = ratio * (y_max - y_min)
        result.append([w, h])

    return result
###############################################
def iou(box, clusters):

    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        return 0
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area +
                          cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):

    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):

    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):

    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(
                boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou
#############################################3
result = []
for _, name in raw_imgs:  
    result.extend(load_one_info(name))

result = np.array(result)
anchors, ave_iou = get_kmeans(result, 9)

anchor_string = ''
anchor_sizes = []
for anchor in anchors:
    anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_sizes.append([anchor[0], anchor[1]])
anchor_string = anchor_string[:-2]

print('anchors are:')
print(anchor_string)
print('the average iou is:')
print(ave_iou)

num_classes = len(train_dataset.labels)
print('class num:', num_classes)
model = pdx.det.YOLOv3(
    num_classes=num_classes, 
    backbone='DarkNet53', 
    nms_iou_threshold=0.1,
    anchors=anchor_sizes)
model.train(
    num_epochs=700,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[20, 40],
    save_interval_epochs=10,
    log_interval_steps=100,
    # save_dir='./yolov3_darknet53',
    pretrain_weights='./output/breakpoint',
    # pretrain_weights='IMAGENET',
    use_vdl=True)
#os.system('rm -rf /home/aistudio/work/output/epoch_1*')
#os.system('rm -rf /home/aistudio/work/output/epoch_2*')

#模型评估
model.evaluate(eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)

import cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline
def test(image_name):
    #image_name = './test1/000117001025653.jpg'
    #image_name = '/home/aistudio/test1/000028901014074.jpg'
    start = time.time()
    result = model.predict(image_name, eval_transforms)
    #print(result)
    print('infer time:{:.6f}s'.format(time.time()-start))
    print('detected num:', len(result))
    print('##############################################################')

    im = cv2.imread(image_name)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.00
    
    for value in result:
        xmin, ymin, w, h = np.array(value['bbox']).astype(np.int)
        cls = value['category']
        score = value['score']
        if score < threshold:
            continue
        print(np.array(value['bbox']).astype(np.int),end=' class=')
        print(cls,end=' score=')
        print(score)
        cv2.rectangle(im, (xmin, ymin), (xmin+w, ymin+h), (0, 255, 0), 4)
        cv2.putText(im, '{:s} {:.3f}'.format(cls, score),
                        (xmin, ymin), font, 0.5, (255, 0, 0), thickness=2)

    cv2.imwrite('result.jpg', im)
    plt.figure(figsize=(15,12))
    plt.imshow(im[:, :, [2,1,0]])
    plt.show()

def test2(image_name):
    start = time.time()
    result = model.predict(image_name, eval_transforms)
    print('detected num:', len(result))
    im = cv2.imread(image_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.09
    first,second=image_name.split('.')
    f1,f2,f3,f4,name=first.split('/')
    file=open('result.txt','a')
    for value in result:
        xmin, ymin, w, h = np.array(value['bbox']).astype(np.int)
        cls = value['category']
        score = value['score']
        if score < threshold:
            continue
        print(np.array(value['bbox']).astype(np.int),' class=',cls,' score=',score)
        file.write(name+' '+str(score)+' '+str(xmin)+' '+str(ymin)+' '+str(xmin+w)+' '+str(ymin+h)+' \n')
        #cv2.rectangle(im, (xmin, ymin), (xmin+w, ymin+h), (0, 255, 0), 4)
        #cv2.putText(im, '{:s} {:.3f}'.format(cls, score),
        #                (xmin, ymin), font, 0.5, (255, 0, 0), thickness=2)
    file.close()

image_name='/home/aistudio/test1/'
#image_name='/home/aistudio/test2/'
import glob
image_list=glob.glob(image_name+'*.jpg')
for path in image_list:
    test2(path)
