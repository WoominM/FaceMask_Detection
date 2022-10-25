#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os
import dataload_yolo
import joblib
import pylab
import random


# In[2]:


# import psutil

# def memory_usage():
#     mem_available = psutil.virtual_memory().available
#     mem_process = psutil.Process(os.getpid()).memory_info().rss
#     return round(mem_process / 1024 / 1024, 2), round(mem_available / 1024 / 1024, 2)


# In[3]:


# import pynvml
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(1)
# meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)


# In[4]:


def conv1_3(indim, middim, outdim):
    layer = nn.Sequential(
        nn.Conv2d(indim, middim, kernel_size=1),
        nn.BatchNorm2d(middim),
        nn.LeakyReLU(),
        nn.Conv2d(middim, outdim, kernel_size=3, padding=1),
        nn.BatchNorm2d(outdim),
        nn.LeakyReLU(),
    )
    return layer

class my_YOLO(nn.Module):
    def __init__(self):
        super(my_YOLO, self).__init__()
        self.premodel = nn.Sequential(*list(model.children()))[:-3]
        self.Convlayer1 = nn.Sequential(
            conv1_3(1024, 512, 1024),
            conv1_3(1024, 512, 1024),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.Convlayer2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.fclayer = nn.Sequential(
            nn.Linear(S*S*1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, S*S*(B*5+C)),
        )   
#         for layer in self.premodel:
#             layer.requires_grad = False
        
    def forward(self, x):
        out = self.premodel(x)
        out = self.Convlayer1(out)
        out = self.Convlayer2(out)
        out = self.fclayer(out.view(out.size(0),-1))
        out = torch.sigmoid(out)
        return out.view(-1, S, S, B*5+C)


# In[5]:


def Yololoss(pred, gt):   
    losscor, lossobj, lossnoobj, losscls = [0,0,0,0]
    batchsize = gt.shape[0]
    for bs in range(batchsize): 
        for i in range(7):
            for j in range(7):  
                if gt[bs,i,j,4]==1:
                    pred_ = pred.clone()
                    gt_ = gt.clone()
                    pred_[bs,i,j,0] = (pred_[bs,i,j,0]+j)/S
                    pred_[bs,i,j,1] = (pred_[bs,i,j,1]+i)/S
                    pred_[bs,i,j,5] = (pred_[bs,i,j,5]+j)/S
                    pred_[bs,i,j,6] = (pred_[bs,i,j,6]+i)/S
                    gt_[bs,i,j,0] = (gt_[bs,i,j,0]+j)/S
                    gt_[bs,i,j,1] = (gt_[bs,i,j,1]+i)/S
                    box1 = wh2xy(pred_[bs,i,j,:4])
                    box2 = wh2xy(pred_[bs,i,j,5:9])
                    gtxy = wh2xy(gt_[bs,i,j,:4])
                    iou1 = my_IoU(box1, gtxy)
                    iou2 = my_IoU(box2, gtxy)
                    if iou1 >= iou2:
                        losscorxy = torch.pow(pred[bs,i,j,0:2]-gt[bs,i,j,0:2], 2).sum()
                        losscorwh = torch.pow(pred[bs,i,j,2:4].sqrt()-gt[bs,i,j,2:4].sqrt(), 2).sum()
                        losscor += lambda_cor*(losscorxy+losscorwh)
                        lossobj += lambda_obj*torch.pow(pred[bs,i,j,4]-iou1, 2)
                        lossnoobj += lambda_noobj*torch.pow(pred[bs,i,j,9]-iou2, 2)
                    else:
                        losscorxy = torch.pow(pred[bs,i,j,5:7]-gt[bs,i,j,5:7], 2).sum()
                        losscorwh = torch.pow(pred[bs,i,j,7:9].sqrt()-gt[bs,i,j,7:9].sqrt(), 2).sum()
                        losscor += lambda_cor*(losscorxy+losscorwh)
                        lossobj += lambda_obj*torch.pow(pred[bs,i,j,9]-iou2, 2)
                        lossnoobj += lambda_noobj*torch.pow(pred[bs,i,j,4]-iou1, 2)
                    losscls += lambda_cls*torch.pow(pred[bs,i,j,10:]-gt[bs,i,j,10:], 2).sum()
                else:  
                    lossnoobj += lambda_noobj*torch.pow(pred[bs,i,j,[4,9]], 2).sum()
    totalloss = (losscor+lossobj+lossnoobj+losscls)/batchsize
    return totalloss


# In[6]:


def my_IoU(anchor, gt):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    if gt.ndim == 1:
        gt = gt.unsqueeze(0)
    anchor = np.array(anchor.tolist())
    gt = np.array(gt.tolist())
    IoU = np.zeros((len(anchor),len(gt)))
    for i in range(len(gt)):                 
        IoU_W = np.maximum(np.min((anchor[:,0], anchor[:,2], gt[i,0]*np.ones(len(anchor)), gt[i,2]*np.ones(len(anchor))),0) + anchor[:,2]-anchor[:,0] + gt[i,2]-gt[i,0] - np.max((anchor[:,0], anchor[:,2], gt[i,0]*np.ones(len(anchor)), gt[i,2]*np.ones(len(anchor))), 0), 1e-100)
        IoU_H = np.maximum(np.min((anchor[:,1], anchor[:,3], gt[i,1]*np.ones(len(anchor)), gt[i,3]*np.ones(len(anchor))),0) + anchor[:,3]-anchor[:,1] + gt[i,3]-gt[i,1] - np.max((anchor[:,1], anchor[:,3], gt[i,1]*np.ones(len(anchor)), gt[i,3]*np.ones(len(anchor))), 0), 1e-100)
        IoU[:,i] = (IoU_W*IoU_H)/((anchor[:,3]-anchor[:,1])*(anchor[:,2]-anchor[:,0]) + (gt[i,3]-gt[i,1])*(gt[i,2]-gt[i,0]) - IoU_W*IoU_H)
        for j in range(len(anchor)):
                if (gt[i] == anchor[j]).all():
                    IoU[j,i] = 1
    IoU = torch.Tensor(IoU)
    IoU[(IoU>=0)&(IoU<=1)==False] = 0
    return IoU.squeeze()


# In[7]:


def xy2wh(anchor):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    w = anchor[:,2]-anchor[:,0]
    h = anchor[:,3]-anchor[:,1]    
    w[w<=0] = 1e-100
    h[h<=0] = 1e-100
    
    ctrx = anchor[:,0]+w/2
    ctry = anchor[:,1]+h/2
    xywh = torch.stack((ctrx,ctry,w,h),1)
    
    return xywh


# In[8]:


def wh2xy(anchor):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    anchor[:,2][anchor[:,2]<=0] = 1e-100
    anchor[:,3][anchor[:,3]<=0] = 1e-100
    xy = torch.zeros(anchor.shape)
    xy[:,0] = anchor[:,0]-anchor[:,2]/2 #x1
    xy[:,2] = anchor[:,0]+anchor[:,2]/2 #x2
    xy[:,1] = anchor[:,1]-anchor[:,3]/2 #y1
    xy[:,3] = anchor[:,1]+anchor[:,3]/2 #y2
    return xy


# In[9]:


def make_target(bndbox, boxlabel):
    target = torch.zeros(S, S, 5*B+C)
    box = xy2wh(bndbox/inputsize)
    ctrx = box[:,0].unsqueeze(1)
    ctry = box[:,1].unsqueeze(1)
    indx = (ctrx*S).long()
    indy = (ctry*S).long()
    w = box[:,2].unsqueeze(1)
    h = box[:,3].unsqueeze(1)
    delctrx = ctrx*S-indx
    delctry = ctry*S-indy
    clslabel = torch.zeros(len(box), C)
    clslabel[torch.arange(len(box)), boxlabel] = 1
    newbox = torch.cat([delctrx, delctry, w, h, torch.ones((len(box), 1))], 1)
    newbox = torch.cat((newbox, newbox), 1).view((len(box), -1))
    newbox = torch.cat((newbox, clslabel), 1)
    for i in range(len(box)):
        target[indy.long()[i],indx.long()[i],:] = newbox[i]
    return target
# make_target(torch.Tensor([[1,2,3,4],[4,5,6,7]]), [1,2]).shape


# In[10]:


def getbatch(dataloader, batchsize, flag):
    img_batch = torch.zeros((batchsize, 3, inputsize, inputsize))
    bndbox_batch = []
    boxlabel_batch = []
    i = 0
    j = 0
    while True:
        if i-j == batchsize:
            break  
        img, bndbox, boxlabel, stopflag = dataload_yolo.Dataloader(dataloader, flag*batchsize+i)
        i += 1
        if stopflag == 1:
            j += 1
            print('Abnormal data!')
            continue
        else:
            boxlabel_ = []
            for targ in boxlabel:
                if targ == 'face':
                    boxlabel_.append(1)
                else:
                    boxlabel_.append(2)
            boxlabel = torch.LongTensor(boxlabel_)
            bndbox = torch.LongTensor(bndbox)
            img = img.float()

        _, H, W = img.shape
        img  = cv2.resize(img.permute(1,2,0).numpy(), (inputsize,inputsize))
        bndbox = dataload_yolo.resize_box(bndbox.tolist(), (H, W), (inputsize, inputsize))
        bndbox = torch.from_numpy(bndbox).float()
        img_batch[i-j-1] = torch.from_numpy(img).permute(2,0,1)
        bndbox_batch.append(bndbox)
        boxlabel_batch.append(boxlabel)
        
    return img_batch, bndbox_batch, boxlabel_batch


# In[11]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flag = 0
batchsize = 5
epochs = 200
lr = 1e-5
train_flag = 6000
valid_flag = 6240
test_flag = 7955
inputsize = 448
dataloader = dataload_yolo.load_data('./FaceMaskDataset/FaceMaskDataset')
model = torchvision.models.googlenet(pretrained=True)

S = 7
B = 2
C = 3
crit = Yololoss
dirs = './'
def loadmodel(pretrained=True):
    if pretrained == True:
        yolo = joblib.load(dirs+'/modelYOLO.pkl')
        yolo = yolo.to(device)
    else:
        yolo = my_YOLO().to(device)
    for layer in yolo.premodel:
        layer.requires_grad = True
    optimizer = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    return yolo, optimizer, scheduler


# In[12]:


lambda_cor = 1
lambda_obj = 5
lambda_cls = 1
lambda_noobj = 1


# In[13]:


def train():
    yolo, optimizer, scheduler = loadmodel(pretrained=False)
    loss_ = []
    for epoch in range(epochs):
    #     if epoch == 50:
    #         lambda_cor = 5
    #         lambda_obj = 1
    #         lambda_cls = 3
    #         lambda_noobj = 1
        st = time.time()
        print('='*100)
        print('epoch:', epoch)
        for flag in range(train_flag//batchsize):
            optimizer.zero_grad()
            img, box, label = getbatch(dataloader, batchsize, flag)
            img = img.to(device)

            target = torch.zeros(batchsize, S, S, 5*B+C).to(device)
            for i in range(batchsize):
                target[i] = make_target(box[i], label[i])
            output = yolo(img)
            loss = crit(output, target)
            loss_.append(loss.item())
            print('batch: %d / %d '%(flag, train_flag//batchsize))
            print('loss:', loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        ed = time.time()
        print('time: %.2f' %(ed-st))



# In[ ]:


# dirs = './'
# if not os.path.exists(dirs):
#     os.makedirs(dirs)

# joblib.dump(yolo, dirs+'/mod3___.pkl')


# In[ ]:


# loss_ = torch.Tensor(loss_)
# plt.plot(loss_,label = 'training')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('task C')
# xt = loss_.shape[0]-1
# yt = loss_[xt]
# plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
# plt.show()


# In[ ]:


dataloader = dataload_yolo.load_data('./FaceMaskDataset/FaceMaskDataset', train=False)
def prediction(randpic=False):
    yolo, optimizer, scheduler = loadmodel(pretrained=True)
    picnum = 0
    while True:
        if picnum == 3: break
        if randpic == True:
            pic = random.randint(valid_flag, test_flag)  
        else:
            if picnum == 0:
                pic = 7280
            else:
                pic += 1
        img, box, label = getbatch(dataloader, 1, pic)
        img = img.to(device)
        st = time.time()
        output = yolo(img)

        pred = output[0].clone().to(device)
        y, x = torch.meshgrid(torch.arange(7), torch.arange(7))
        pred[:,:,[0,5]] = (pred[:,:,[0,5]]+x.repeat(2,1,1).permute(1,2,0).to(device))/S
        pred[:,:,[1,6]] = (pred[:,:,[1,6]]+y.repeat(2,1,1).permute(1,2,0).to(device))/S
        box1 = wh2xy(pred[:,:,:4].view(-1,4))
        box2 = wh2xy(pred[:,:,5:9].view(-1,4))
        predbox = torch.cat((box1, box2), 0)

        cls = output[0][:,:,10:].view(-1, C)
        conf = output[0][:,:,[4,9]].view(-1, 2)
        score1 = cls*conf[:,0].unsqueeze(1)
        score2 = cls*conf[:,1].unsqueeze(1)
        predscore = torch.cat((score1, score2), 0)
        predconf = torch.cat((conf[:,0], conf[:,1]), 0)

        lbl = predscore.argmax(1)
        indsort = predscore[torch.arange(len(predscore)),lbl].argsort(descending=True)
        thrscore = torch.where(predconf.sort(descending=True)[0]>0.2)[0]
        if torch.Size(thrscore) == torch.Size([]):
            predlabel = lbl[indsort][:3]
            box_nms = predbox[indsort][:3]
            predconf = predconf[indsort][:3]
        else:
            picnum += 1
            predlabel = lbl[indsort][thrscore]
            box_nms = predbox[indsort][thrscore]
            predconf = predconf[indsort][thrscore]

        maxind = torch.arange(len(box_nms))
        ind_nms = []
        thr_nms = 0.1
        box_nms_ = box_nms.clone()
        while(len(box_nms_)>=1):
            gt_nms = box_nms_[0]
            ind_nms.append(maxind[0])
            iou_nms = my_IoU(box_nms_, gt_nms)
            box_nms_ = box_nms_[(iou_nms<thr_nms).squeeze()]
            maxind = maxind[(iou_nms<thr_nms).squeeze()]
        predbox = box_nms[torch.LongTensor(ind_nms)][:5]*inputsize
        predlabel = predlabel[torch.LongTensor(ind_nms)][:5]
        predconf = predconf[torch.LongTensor(ind_nms)][:5]

        ed = time.time()
#         print('time: %.2f s' %(ed-st))

        ##
        img, bndbox, boxlabel, stopflag = dataload_yolo.Dataloader(dataloader, pic)
        if stopflag == 1:
            print('Abnormal data!')
            continue
        else:
            boxlabel_ = []
            for targ in boxlabel:
                if targ == 'face':
                    boxlabel_.append(1)
                else:
                    boxlabel_.append(2)
            boxlabel = torch.LongTensor(boxlabel_)
            bndbox = torch.LongTensor(bndbox)
            img = img.float()

        _, H, W = img.shape
        img  = cv2.resize(img.permute(1,2,0).numpy(), (inputsize,inputsize))
        bndbox = dataload_yolo.resize_box(bndbox.tolist(), (H, W), (inputsize, inputsize))
        bndbox = torch.from_numpy(bndbox).float()

#         for j in range(len(bndbox)):
#             cv2.rectangle(img, (bndbox[j][0], bndbox[j][1]), (bndbox[j][2], bndbox[j][3]), (0,255,255), 2)
#             text = 'face'
        #     cv2.putText(img, text, (bndbox[j][0],bndbox[j][1]), 1, 1.5, (0,255,255), 2)
        for j in range(len(predbox)):
            cv2.rectangle(img, (predbox[j][0], predbox[j][1]), (predbox[j][2], predbox[j][3]), (0,255,0), 2)
            if predlabel[j] == 1:
                text = 'face'
            elif predlabel[j] == 2:
                text = 'mask'
            score = '%.2f%%' %(predconf[j]*100) 
            cv2.putText(img, text, (predbox[j][0],predbox[j][1]), 1, 1.8, (0,255,0), 2)
            cv2.putText(img, score, (predbox[j][2],predbox[j][3]), 0, 0.5, (0,255,0), 1)
        img = dataload_yolo.Unnormalize_Orgsizeimg(img, (H, W))
        plt.figure(figsize = (10,10))
        plt.imshow(img)
        pylab.show()



# In[ ]:


# PR = []
# thr = 0.1
# det_boxes = []
# det_labels = []
# det_scores = []
# true_boxes = []
# true_labels = []
# valid_flag = 6240
# test_flag = 7955
# for pic in range(valid_flag, test_flag):
#     img, box, label = getbatch(dataloader,1, pic)
#     img = img.to(device)
#     output = yolo(img)

#     true_boxes.append(box[0])
#     true_labels.append(label[0])

#     pred = output[0].clone().to(device)
#     y, x = torch.meshgrid(torch.arange(7), torch.arange(7))
#     pred[:,:,[0,5]] = (pred[:,:,[0,5]]+x.repeat(2,1,1).permute(1,2,0).to(device))/S
#     pred[:,:,[1,6]] = (pred[:,:,[1,6]]+y.repeat(2,1,1).permute(1,2,0).to(device))/S
#     box1 = wh2xy(pred[:,:,:4].view(-1,4))
#     box2 = wh2xy(pred[:,:,5:9].view(-1,4))
#     predbox = torch.cat((box1, box2), 0)

#     cls = output[0][:,:,10:].view(-1, C)
#     conf = output[0][:,:,[4,9]].view(-1, 2)
#     score1 = cls*conf[:,0].unsqueeze(1)
#     score2 = cls*conf[:,1].unsqueeze(1)
#     predscore = torch.cat((score1, score2), 0)
#     predconf = torch.cat((conf[:,0], conf[:,1]), 0)

#     lbl = predscore.argmax(1)
#     indsort = predscore[torch.arange(len(predscore)),lbl].argsort(descending=True)
# #     thrscore = torch.where(predscore[torch.arange(len(predscore)),lbl].sort(descending=True)[0]>0.1)[0]
#     thrscore = torch.where(predconf.sort(descending=True)[0]>0.2)[0]
#     if torch.Size(thrscore) == torch.Size([]):
#         predlabel = lbl[indsort][:3]
#         box_nms = predbox[indsort][:3]
#         predconf = predconf[indsort][:3]
#     else:
#         predlabel = lbl[indsort][thrscore]
#         box_nms = predbox[indsort][thrscore]
#         predconf = predconf[indsort][thrscore]
        
#     maxind = torch.arange(len(box_nms))
#     ind_nms = []
#     thr_nms = 0.01
#     box_nms_ = box_nms.clone()
#     while(len(box_nms_)>=1):
#         gt_nms = box_nms_[0]
#         ind_nms.append(maxind[0])
#         iou_nms = my_IoU(box_nms_, gt_nms)
#         box_nms_ = box_nms_[(iou_nms<thr_nms).squeeze()]
#         maxind = maxind[(iou_nms<thr_nms).squeeze()]
#     predbox = box_nms[torch.LongTensor(ind_nms)][:5]*inputsize
#     predlabel = predlabel[torch.LongTensor(ind_nms)][:5]
#     predconf = predconf[torch.LongTensor(ind_nms)][:5]

#     det_boxes.append(predbox.detach())
#     det_labels.append(predlabel.detach())
#     det_scores.append(predconf.detach())


# # Label map
# labels = ('no_mask', 'mask')
# label_map = {k: v + 1 for v, k in enumerate(labels)}
# label_map['background'] = 0
# rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


# # def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
# def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, threshold):
#     """
#     Calculate the Mean Average Precision (mAP) of detected objects.

#     See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

#     :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
#     :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
#     :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
#     :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
#     :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
#     :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
#     :return: list of average precisions for all classes, mean average precision (mAP)
#     """
#     # set_trace()
#     assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
#         true_labels) 
#         #== len(true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
#     n_classes = len(label_map)

#     # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
#     true_images = list()
#     for i in range(len(true_labels)):
#         true_images.extend([i] * true_labels[i].size(0))
#     true_images = torch.LongTensor(true_images).to(
#         device)  # (n_objects), n_objects is the total no. of objects across all images
#     true_boxes = torch.cat(true_boxes, dim=0).to(device)  # (n_objects, 4)
#     true_labels = torch.cat(true_labels, dim=0).to(device)  # (n_objects)
#     # true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

#     assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

#     # Store all detections in a single continuous tensor while keeping track of the image it is from
#     det_images = list()
#     for i in range(len(det_labels)):
#         det_images.extend([i] * det_labels[i].size(0))
#     det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
#     det_boxes = torch.cat(det_boxes, dim=0).to(device)  # (n_detections, 4)
#     det_labels = torch.cat(det_labels, dim=0).to(device)  # (n_detections)
#     det_scores = torch.cat(det_scores, dim=0).to(device)  # (n_detections)

#     assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

#     # Calculate APs for each class (except background)
#     average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
#     precisions_classes = []
#     for c in range(1, n_classes):

#         # Extract only objects with this class
#         true_class_images = true_images[true_labels == c]  # (n_class_objects)
#         true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
#         # true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
#         #n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects
#         n_objects = true_class_images.size(0)
#         # Keep track of which true objects with this class have already been 'detected'
#         # So far, none
#         # set_trace()
#         #true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
#         #    device)  # (n_class_objects)
#         true_class_boxes_detected = torch.zeros((true_class_images.size(0)), dtype=torch.uint8).to(
#             device)  # (n_class_objects)


#         # Extract only detections with this class
#         det_class_images = det_images[det_labels == c]  # (n_class_detections)
#         det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
#         det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
#         n_class_detections = det_class_boxes.size(0)
#         if n_class_detections == 0:
#             print("no detection of class %d is found" %c)
#             precisions_classes.append(torch.zeros((11), dtype=torch.float).to(device))
#             continue

#         # Sort detections in decreasing order of confidence/scores
#         det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
#         det_class_images = det_class_images[sort_ind]  # (n_class_detections)
#         det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

#         # In the order of decreasing scores, check if true or false positive
#         true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
#         false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
#         for d in range(n_class_detections):
#             this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
#             this_image = det_class_images[d]  # (), scalar

#             # Find objects in the same image with this class, their difficulties, and whether they have been detected before
#             object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
#             #object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
#             # If no such object in this image, then the detection is a false positive
#             if object_boxes.size(0) == 0:
#                 false_positives[d] = 1
#                 continue

#             # Find maximum overlap of this detection with objects in this image of this class
#             overlaps = my_IoU(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)

#             max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

#             # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
#             # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
#             original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
#             # We need 'original_ind' to update 'true_class_boxes_detected'

#             # If the maximum overlap is greater than the threshold of 0.5, it's a match
#             if max_overlap.item() > threshold:
#                 # If the object it matched with is 'difficult', ignore it
#                 #if object_difficulties[ind] == 0:
#                     # If this object has already not been detected, it's a true positive
#                     #if true_class_boxes_detected[original_ind] == 0:
#                      #   true_positives[d] = 1
#                      #   true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
#                     # Otherwise, it's a false positive (since this object is already accounted for)
#                     #else:
#                      #   false_positives[d] = 1
#                 if true_class_boxes_detected[original_ind] == 0:
#                     true_positives[d] = 1
#                     true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
#                 # Otherwise, it's a false positive (since this object is already accounted for)
#                 else:
#                     false_positives[d] = 1
#             # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
#             else:
#                 false_positives[d] = 1

#         # Compute cumulative precision and recall at each detection in the order of decreasing scores
#         cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
#         cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
#         cumul_precision = cumul_true_positives / (
#                 cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
#         #cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

#         cumul_recall = cumul_true_positives / n_objects


#         # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
#         recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
#         precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
#         for i, t in enumerate(recall_thresholds):
#             recalls_above_t = cumul_recall >= t
#             if recalls_above_t.any():
#                 precisions[i] = cumul_precision[recalls_above_t].max()
#             else:
#                 precisions[i] = 0.
#         average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
#         precisions_classes.append(precisions)


#     # Calculate Mean Average Precision (mAP)
#     mean_average_precision = average_precisions.mean().item()

#     # Keep class-wise average precisions in a dictionary
#     average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

#     return precisions_classes, average_precisions, mean_average_precision

# # precisions, APs, mAPs = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, 0.5)


# In[ ]:


# mAPs = {}
# APs = {}
# precisions_dict = {}
# for threshold in np.arange(0.5, 0.95, 0.05):  
#     precisions, AP, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, threshold)
# #     print(APs)
#     threshold = "%.2f" %threshold

#     mAPs[threshold] = mAP 
#     APs[threshold] = AP
#     precisions_dict[threshold] = precisions


# print(APs)
# print("\nMean Average Precision (mAP@.5): %.3f" % mAPs["0.50"])
# #set_trace()
# print("\nMean Average Precision (mAP@.7): %.3f" % mAPs["0.70"])
# #set_trace()
# print("\nMean Average Precision (mAP@.9): %.3f" % mAPs["0.90"])
# mean_mAPs = sum(mAPs.values())/len(mAPs)
# print("\nMean Average Precision (mAP@[.5:.95]): %.3f" % mean_mAPs)

# fig = plt.figure(figsize=(10,3))
# i = 1

# for threshold in ["0.50", "0.70", "0.90"]:
#     x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     sub = "13"+str(i)
#     plt.subplot(sub)
#     precisions_dict[threshold][0] = precisions_dict[threshold][0].cpu().numpy()
#     label_ = "threshold_" + threshold
#     plt.step(x, precisions_dict[threshold][0], label=label_)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend()

#     print("plotted figure threshold "+threshold)
#     i = i + 1
# # set_trace()
# figure_name = "P_R_curve_face.png"
# fig.tight_layout()
# plt.show()
# # print("Saving to ", figure_name)
# # fig.savefig(figure_name)
# # print("Saved!")
# plt.close()

# fig = plt.figure(figsize=(10,3))
# i = 1
# #set_trace()
# for threshold in ["0.50", "0.70", "0.90"]:
#     x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     sub = "13"+str(i)
#     plt.subplot(sub)
#     precisions_dict[threshold][1] = precisions_dict[threshold][1].cpu().numpy()
#     label_ = "threshold_" + threshold
#     plt.step(x, precisions_dict[threshold][1], label=label_)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend()

#     print("plotted figure threshold "+threshold)
#     i = i + 1
# # set_trace()
# figure_name = "P_R_curve_facemask.png"
# fig.tight_layout()
# # print("Saving to ", figure_name)
# # fig.savefig(figure_name)
# # print("Saved!")
# plt.show()
# plt.close()
# # set_trace()


# In[ ]:

# if __name__== "__main__":
#     YOLOv1.__module__="my_YOLO"


