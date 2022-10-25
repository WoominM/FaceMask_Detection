#!/usr/bin/env python
# coding: utf-8

# # Lib

# In[1]:


import dataload
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import utils
import matplotlib.pyplot as plt
import cv2
import random
import time
import copy
import os
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Base function

# In[2]:


def my_IoU(anchor, gt):
    if anchor.ndim == 1:
        anchor = anchor.unsqueeze(0)
    if gt.ndim == 1:
        gt = gt.unsqueeze(0)
    anchor = anchor.cpu().numpy()
    IoU = np.zeros((len(anchor),len(gt)))
    for i in range(len(gt)):
        IoU_W = np.maximum(np.min((anchor[:,0], anchor[:,2], int(gt[i,0])*np.ones(len(anchor)), int(gt[i,2])*np.ones(len(anchor))),0) + anchor[:,2]-anchor[:,0] + int(gt[i,2]-gt[i,0]) - np.max((anchor[:,0], anchor[:,2], int(gt[i,0])*np.ones(len(anchor)), int(gt[i,2])*np.ones(len(anchor))), 0), 0)
        IoU_H = np.maximum(np.min((anchor[:,1], anchor[:,3], int(gt[i,1])*np.ones(len(anchor)), int(gt[i,3])*np.ones(len(anchor))),0) + anchor[:,3]-anchor[:,1] + int(gt[i,3]-gt[i,1]) - np.max((anchor[:,1], anchor[:,3], int(gt[i,1])*np.ones(len(anchor)), int(gt[i,3])*np.ones(len(anchor))), 0), 0)
        IoU[:,i] = (IoU_W*IoU_H)/((anchor[:,3]-anchor[:,1])*(anchor[:,2]-anchor[:,0]) + int(gt[i,3]-gt[i,1])*int(gt[i,2]-gt[i,0]) - IoU_W*IoU_H)  
    anchor = torch.Tensor(anchor)
    IoU = torch.Tensor(IoU)
    return IoU


# In[3]:


def xy2delta(anchor, gt):
    anc_w = anchor[:,2]-anchor[:,0]
    anc_h = anchor[:,3]-anchor[:,1]
    anc_ctrx = anchor[:,0]+anc_w/2
    anc_ctry = anchor[:,1]+anc_h/2

    gt_w = gt[:,2]-gt[:,0]
    gt_h = gt[:,3]-gt[:,1]
    gt_ctrx = gt[:,0]+gt_w/2
    gt_ctry = gt[:,1]+gt_h/2

    anc_w[anc_w<=0] = 1e-100
    anc_h[anc_h<=0] = 1e-100

    tx = (gt_ctrx-anc_ctrx)/anc_w
    ty = (gt_ctry-anc_ctry)/anc_h
    tw = torch.log(gt_w/anc_w)
    th = torch.log(gt_h/anc_h)
    
    delta = torch.stack((tx,ty,tw,th),1)    
    return delta


# In[4]:


def delta2xy(anchor, delta):
    anc_w = anchor[:,2]-anchor[:,0]
    anc_h = anchor[:,3]-anchor[:,1]
    anc_ctrx = anchor[:,0]+anc_w/2
    anc_ctry = anchor[:,1]+anc_h/2

    tx = delta[:,0]
    ty = delta[:,1]
    tw = delta[:,2]
    th = delta[:,3]

    ctr_x = tx*anc_w+anc_ctrx
    ctr_y = ty*anc_h+anc_ctry
    w = torch.exp(tw)*anc_w
    h = torch.exp(th)*anc_h
    
    XY = torch.zeros(delta.shape)
    XY[:,0] = ctr_x-w/2
    XY[:,2] = ctr_x+w/2
    XY[:,1] = ctr_y-h/2
    XY[:,3] = ctr_y+h/2  
    return XY


# # RPN

# In[5]:


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.vgg = copy.deepcopy(model.features)
        self.conv_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv_cls = nn.Conv2d(512, 2*k, kernel_size = 1, stride = 1, padding = 0)
        self.conv_reg = nn.Conv2d(512, 4*k, kernel_size = 1, stride = 1, padding = 0)
        self.conv_3.weight.data.normal_(0, 0.01)
        self.conv_3.bias.data.zero_()
        self.conv_cls.weight.data.normal_(0, 0.01)
        self.conv_cls.bias.data.zero_()
        self.conv_reg.weight.data.normal_(0, 0.01)
        self.conv_reg.bias.data.zero_()
        self.topN_nms = 12000
        self.n_nms = 2000
    
    def Evalmode(self):
        self.topN_nms = 2000
        self.n_nms = 300
           
    def nograd(self, TF=False):
        if TF == True:
            self.vgg.requires_grad = False
            
    def GenerateAnchor(self, img_size, anc_ratio, anc_size, sampling):
        Hf = img_size[0]
        Wf = img_size[1]
        anchor = np.zeros((k*Hf*Wf,4))
        ind = 0
        for ii in range(Wf):
            ctrx = (ii+0.5)*sampling
            for jj in range(Hf):
                ctry = (jj+0.5)*sampling
                for i in range(len(anc_ratio)):
                    for j in range(len(anc_size)):
                        w = sampling*anc_size[j]/np.sqrt(anc_ratio[i])
                        h = sampling*anc_size[j]*np.sqrt(anc_ratio[i])
                        anchor[ind,0] = ctrx-w/2
                        anchor[ind,2] = ctrx+w/2
                        anchor[ind,1] = ctry-h/2  
                        anchor[ind,3] = ctry+h/2
                        ind += 1
        anchor = np.round(anchor).astype(int)
        return anchor
    
    def InsideofImage(self, anchor):
        ind_in = torch.where((anchor[:,0]>=0) & (anchor[:,1]>=0) & (anchor[:,2]<=W) & (anchor[:,3]<=H))[0]
        anchor_in = anchor[ind_in]
        return anchor_in, ind_in
    
    def Labeling(self, anchor_in, ind_in, IoU, len_anc):      
        maxIoU_gt = torch.max(IoU, 1)[0]
        maxIoU_ind = torch.where(IoU==torch.max(IoU, 0)[0])[0]
        label = -1*torch.ones((len(ind_in)))
        label[maxIoU_ind] = 1
        label[maxIoU_gt>=thr_pos] = 1
        label[maxIoU_gt<=thr_neg] = 0
        
        if len(label[label==1])>n_pos:
            randind = random.sample(torch.where(label==1)[0].tolist(),(len(label[label==1])-n_pos))
            label[randind] = -1
        n_pos_ = len(label[label==1])
        if len(label[label==0])>n_neg:
            randind = random.sample(torch.where(label==0)[0].tolist(),(len(label[label==0])-n_pos-n_neg+n_pos_))
            label[randind] = -1
            
        labels = -1*torch.ones(len_anc)
        labels[ind_in] = label
        return labels
    
    def Prediction(self, featmap):
        pred_cls = self.conv_cls(featmap)
        pred_reg = self.conv_reg(featmap)
        
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        pred_obj = pred_cls[:,:,1] 
        return pred_cls, pred_reg, pred_obj
    
    def RPNLoss(self, pred_cls, pred_reg, label, delta, crit):
        pred_cls_rpn = pred_cls[0]
        pred_reg_rpn = pred_reg[0]
        labels_rpn = label
        deltas_rpn = delta

        ind_batch = torch.Tensor(range(len(label)))[labels_rpn.ne(-1)].long()
        loss_cls_rpn = crit(pred_cls_rpn[ind_batch], labels_rpn[ind_batch].long())
#         print('cls loss: %.4f' %loss_cls_rpn.data.item())

        x = torch.abs(pred_reg_rpn[labels_rpn>0] - deltas_rpn[labels_rpn>0])
        loss_reg_rpn = (((x<1).float()*0.5*x**2) + ((x>=1).float()*(x-0.5))).sum()
#         print('reg loss: %.4f' %loss_reg_rpn.data.item())

#         N_cls = 256
#         N_reg = len(label)/k
#         N_reg = (labels_rpn>0).sum()+1
        loss_rpn = loss_cls_rpn + lambda_rpn*loss_reg_rpn
        return loss_rpn, loss_cls_rpn.data.item(), loss_reg_rpn.data.item()
    
    def MakeRoI(self, anchor, pred_reg, pred_obj):
        pred_reg_ = pred_reg[0].data        
        pred_obj_ = pred_obj[0].data
        
        roi = delta2xy(anchor, pred_reg_)
        roi[roi<0] = 0
        roi[:,3][roi[:,3]>H] = H-1
        roi[:,2][roi[:,2]>W] = W-1
        ind_roi = (roi[:,2]-roi[:,0]>sampling) & (roi[:,3]-roi[:,1]>sampling)
        roi = roi[ind_roi]
        pred_obj_ = pred_obj_[ind_roi]

        ind_sort = torch.sort(pred_obj_.squeeze(), descending=True)[1][:self.topN_nms]
        roi = roi[ind_sort]
        
        thr_nms = 0.7
        roi_nms = roi.clone()
        ind_nms = torch.Tensor(range(len(roi_nms)))
        ind_nms_ = []
        while(len(roi_nms)>1):
            gt_nms = roi_nms[0].reshape(1,-1)
            ind_nms_.append(ind_nms[0])
            IoU_nms = my_IoU(roi_nms, gt_nms)
            roi_nms = roi_nms[(IoU_nms<thr_nms).squeeze()]
            ind_nms = ind_nms[(IoU_nms<thr_nms).squeeze()]

        ind_nms_ = ind_nms_[:self.n_nms]
        roi = roi[ind_nms_].to(device)
        return roi
    
    def forward(self, img, crit):
        featmap = self.vgg(img)
        featmap = Variable(F.relu(self.conv_3(featmap), inplace=True), requires_grad=True).to(device)
        Hf = featmap.shape[2]
        Wf = featmap.shape[3]  
        anchor = self.GenerateAnchor((Hf,Wf), anc_ratio, anc_size, sampling)
        anchor = torch.from_numpy(anchor).float().to(device)
        
        anchor_in, ind_in = self.InsideofImage(anchor)
        IoU = my_IoU(anchor_in, gt)
        label = self.Labeling(anchor_in, ind_in, IoU, len(anchor)).to(device)
        
        gt_anc = gt[torch.argmax(IoU, 1)]
        delta_in = xy2delta(anchor_in, gt_anc)
        delta = torch.zeros(anchor.shape).to(device)
        delta[ind_in] = delta_in
        delta = delta
        
        pred_cls, pred_reg, pred_obj = self.Prediction(featmap)
        loss, loss_cls, loss_reg = self.RPNLoss(pred_cls, pred_reg, label, delta, crit)
        
        roi = self.MakeRoI(anchor, pred_reg, pred_obj)
        
        return loss, roi, loss_cls, loss_reg


# #  128 Proposal for Fast R-CNN

# In[6]:


def TrainProposal(roi, gt, gt_label, n_sample): 
    n_pos_ = int(n_sample*0.25)
    n_neg_ = n_sample-n_pos_
    thr_pos_ = 0.5
    thr_neg_ = 0

    roi_ = torch.cat((roi, gt), 0)
    IoU = my_IoU(roi_, gt)
    maxIoU_gt, maxIoU_ind = torch.max(IoU, 1)
     
    label_targ = gt_label.float()[maxIoU_ind]

    ind_pos = torch.where(maxIoU_gt>=thr_pos_)[0]
    if len(ind_pos)>n_pos_:
        ind_pos = torch.Tensor(random.sample(ind_pos.tolist(),n_pos_)).long()
    ind_neg = torch.where(maxIoU_gt<=thr_neg_)[0]
    n_neg_r = n_neg_+n_pos_-len(ind_pos)
    if n_neg_r > len(ind_neg): n_neg_r = len(ind_neg)
    if len(ind_neg)>n_neg_:
        ind_neg = torch.Tensor(random.sample(ind_neg.tolist(),n_neg_r)).long()

    ind = torch.cat((ind_pos,ind_neg), 0)
    label_targ[ind_neg] = 0
    label_targ = label_targ[ind]
    roi_targ = roi_[ind]
    
    gt_targ = gt[maxIoU_ind[ind]]
    delta_targ = xy2delta(roi_targ, gt_targ)

    return roi_targ, label_targ, delta_targ


# #  Fast R-CNN

# In[7]:


class FastRCNN(nn.Module):
    def __init__(self, Train=True):
        super(FastRCNN, self).__init__()
        self.vgg = copy.deepcopy(model.features)
        self.fclayer = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.Linear(4096, 4096),
        )
        self.fclayer_cls = nn.Linear(4096,3)
        self.fclayer_reg = nn.Linear(4096,3*4)
        self.fclayer_cls.weight.data.normal_(0, 0.01)
        self.fclayer_cls.bias.data.zero_()
        self.fclayer_reg.weight.data.normal_(0, 0.01)
        self.fclayer_reg.bias.data.zero_()        
        self.RoIpool = nn.AdaptiveMaxPool2d((7,7), return_indices=False)
    
    def nograd(self, TF=False):
        if TF == True:
            self.vgg.requires_grad = False
    
    def RoIPooling(self, featmap, roi):
        roi_f = (roi/sampling).long()
        roi_pool = torch.zeros((len(roi), 512, 7, 7))
        
        for i in range(len(roi)):
            if roi_f[i,2]-roi_f[i,0] <= 1:
                if roi_f[i,2] < featmap.shape[3]:
                    roi_f[i,2] += 1
                else:
                    roi_f[i,0] -= 1
            if roi_f[i,3]-roi_f[i,1] <= 1:
                if roi_f[i,3] < featmap.shape[2]:
                    roi_f[i,3] += 1
                else:
                    roi_f[i,1] -= 1
            roi_feat = featmap[:,:,roi_f[i,1]:roi_f[i,3],roi_f[i,0]:roi_f[i,2]]
            roi_pool[i] = self.RoIpool(roi_feat).data
        roi_pool = roi_pool.contiguous().view(len(roi), -1).to(device)
        return roi_pool
    
    def FastRCNNLoss(self, output_cls, output_reg, label_targ, delta_targ, crit):
        pred_cls_rcnn = output_cls
        pred_reg_rcnn = output_reg.contiguous().view(len(label_targ),-1,4)
        labels_rcnn = label_targ.long()
        deltas_rcnn = delta_targ

        loss_cls_rcnn = crit(pred_cls_rcnn, labels_rcnn)
#         print('cls loss: %.4f' %loss_cls_rcnn.data.item())
        
        pred_reg_rcnn = pred_reg_rcnn[torch.Tensor(range(len(labels_rcnn))).long(),labels_rcnn]
        x = torch.abs(pred_reg_rcnn[labels_rcnn>0] - deltas_rcnn[labels_rcnn>0])
        loss_reg_rcnn = ((x<1).float()*0.5*x**2 + (x>=1).float()*(x-0.5)).sum()
#         print('reg loss: %.4f' %loss_reg_rcnn.data.item())

        N_cls = len(label_targ)
        N_reg = len(label_targ)
#         N_reg = (labels_rcnn>0).sum()+1
        loss_rcnn = loss_cls_rcnn/N_cls + lambda_rcnn*loss_reg_rcnn/N_reg
        return loss_rcnn, loss_cls_rcnn.data.item(), loss_reg_rcnn.data.item()
    
    def forward(self, img, roi, label, delta, crit, Train=True):
        featmap = self.vgg(img)
        roi_pool = self.RoIPooling(featmap, roi)
        output = self.fclayer(roi_pool)
        output_cls = self.fclayer_cls(output)
        output_cls = F.softmax(output_cls, 1) 
        output_reg = self.fclayer_reg(output)
        if Train == True:
            loss_rcnn, loss_cls, loss_reg = self.FastRCNNLoss(output_cls, output_reg, label, delta, crit)
            return loss_rcnn, output_cls, output_reg, loss_cls, loss_reg
        else:
            return output_cls, output_reg


# # Train & Evaluation fuction

# In[8]:


def train_rpn(img, optimizer_rpn, crit_rpn):
    optimizer_rpn.zero_grad()
    loss_rpn, roi, lc_rpn, lr_rpn = rpn.forward(img, crit_rpn)
    print('RPN loss: %.4f (cls loss: %.4f, reg loss: %.4f)' %(loss_rpn.data.item(), lc_rpn, lr_rpn))
    loss_rpn.backward()
    nn.utils.clip_grad_norm_(rpn.parameters(), 1e-3)
    optimizer_rpn.step()
    return loss_rpn, roi

    
def train_rcnn(img, optimizer_rcnn, crit_rcnn, n_sample = 128): 
    optimizer_rcnn.zero_grad()
    loss_rpn, roi, lc_rpn, lr_rpn = rpn.forward(img, crit_rpn)
    roi_targ, label_targ, delta_targ = TrainProposal(roi, gt, gt_label, n_sample)
    loss_rcnn, output_cls, output_reg, lc_rcnn, lr_rcnn = fastrcnn.forward(img, roi_targ, label_targ, delta_targ, crit_rcnn)
    totalloss = loss_rpn+loss_rcnn
    print('FastRCNN loss: %.4f (cls loss: %.4f, reg loss: %.4f)' %(loss_rcnn.data.item(), lc_rcnn, lr_rcnn))
    totalloss.backward()
    #loss_rcnn.backward()
    #nn.utils.clip_grad_norm_(fastrcnn.parameters(), 1e-3)
    optimizer_rcnn.step()
    return loss_rcnn    

def evaluate(img, crit_rpn, crit_rcnn):
    loss_rpn, roi, _, _ = rpn.forward(img, crit_rpn)
    gt_delta = xy2delta(gt, gt)
    out_cls, out_reg = fastrcnn.forward(img, roi, gt_label, gt_delta, crit_rcnn, Train=False)

    blabel = torch.argmax(out_cls,1).to(device)
    box = torch.zeros(len(blabel),4).to(device)
    score = torch.zeros(len(blabel)).to(device)
    for i in range(len(blabel)):
        box[i,:] = out_reg[i,4*blabel[i]:4*blabel[i]+4]
        score[i] = out_cls[i,blabel[i]]
    box = delta2xy(roi[blabel>0], box[blabel>0].data)
    score = score[blabel>0]
    blabel = blabel[blabel>0]
    
    ind = score.argsort().squeeze()
    box = box[ind]
    score = score[ind]
    blabel = blabel[ind]
    
    thr_nms = 0.3
    box_nms = box.clone()
    ind_nms = torch.Tensor(range(len(box_nms)))
    ind_nms_result = []
    while(len(box_nms)>1):
        gt_nms = box_nms[0].reshape(1,-1)
        ind_nms_result.append(ind_nms[0].long().tolist())
        IoU_nms = my_IoU(box_nms, gt_nms)
        box_nms = box_nms[(IoU_nms<thr_nms).squeeze()]
        ind_nms = ind_nms[(IoU_nms<thr_nms).squeeze()]
    
    ind_nms_result = ind_nms_result[:3]
    box = box[ind_nms_result]
    box[:,0][box[:,0]<0] = 0
    box[:,1][box[:,1]<0] = 0
    box[:,2][box[:,2]>img.shape[3]] = img.shape[3]
    box[:,3][box[:,3]>img.shape[2]] = img.shape[2]
    score = score[ind_nms_result]
    blabel = blabel[ind_nms_result]

    return box, score, blabel


# 
# # Main

# In[9]:


if __name__ == '__main__':    
    
    dataloader = dataload.load_data('./data')
    epochs = 5
    train_flag = 300
    
    sampling = 16
    thr_pos = 0.7
    thr_neg = 0.3
    n_pos = 128
    n_neg = 128   
    lambda_rpn = 1
    lambda_rcnn = 1
    anc_ratio = [0.5,1,2]
    anc_size = [2,8,16,32]   
    k = len(anc_ratio)*len(anc_size)
    
    model = torchvision.models.vgg16(pretrained=True)
    model = model
    model.features = model.features[:-1]
    
    rpn = RPN().to(device)
    fastrcnn = FastRCNN().to(device)
    
    lr_rpn = 0.1
    lr_rcnn = 1e-4
    optimizer_rpn = optim.SGD(rpn.parameters(), lr=lr_rpn, momentum=0.9)
    optimizer_rcnn = optim.SGD(fastrcnn.parameters(), lr=lr_rcnn, momentum=0.9)
    
    weight = torch.Tensor([0.95,0.05]).to(device)
    weight = 1/(len(weight)*weight)
    crit_rpn = nn.CrossEntropyLoss(weight)
#     crit_rpn = nn.CrossEntropyLoss()
    crit_rcnn = nn.CrossEntropyLoss()
    
    abnormal = 0
#     randomtrain = random.sample(list(range(train_flag)), 3000)
    randomtrain = list(range(train_flag))


# In[10]:


#     kkk = 3023
#     for epoch in range(1):
#         for iter_ in range(kkk,kkk+1):
#             print('='*100)
#             print('epoch:', epoch)
#             print('iter:',iter_+1)
#             img, bndbox, boxlabel, flag, scale = dataload.Dataloader(dataloader, iter_)
#             if flag == 1:
#                 abnormal += 1
#                 print('Abnormal data!')
#                 continue
#             else:
#                 boxlabel_ = []
#                 for targ in boxlabel:
#                     if targ == 'face':
#                         boxlabel_.append(1)
#                     else:
#                         boxlabel_.append(2)
#                 boxlabel_ = torch.LongTensor(boxlabel_).to(device)
#                 bndbox = bndbox.float().to(device)
#                 img = img.float().to(device)
#                 img_ = dataload.Unnormalize_Orgsizeimg(img, scale)
#                 plt.imshow(img_)

# #             H = img.shape[2]
# #             W = img.shape[3]
# #             gt = bndbox.to(device)
# #             gt_label = boxlabel_.to(device)

# #             start = time.time()
# #             loss_rpn, roi = train_rpn(img, optimizer_rpn, crit_rpn)
# #             end = time.time()
# #             print('time: %.2f s' %(end-start)) 


# Step 1. Train RPN

# In[ ]:


for epoch in range(epochs):
#         for iter_ in range(train_flag):
    for iter_ in randomtrain:
#         for iter_ in range(1,2):
        print('='*100)
        print('epoch:', epoch)
        print('iter:',iter_+1)
        img, bndbox, boxlabel, flag, scale = dataload.Dataloader(dataloader, iter_)
        if flag == 1:
            abnormal += 1
            print('Abnormal data!')
            continue
        else:
            boxlabel_ = []
            for targ in boxlabel:
                if targ == 'face':
                    boxlabel_.append(1)
                else:
                    boxlabel_.append(2)
            boxlabel_ = torch.LongTensor(boxlabel_).to(device)
            bndbox = bndbox.float().to(device)
            img = img.float().to(device)

        H = img.shape[2]
        W = img.shape[3]
        gt = bndbox.to(device)
        gt_label = boxlabel_.to(device)

        start = time.time()
        loss_rpn, roi = train_rpn(img, optimizer_rpn, crit_rpn)
        end = time.time()
        print('time: %.2f s' %(end-start)) 


# Step 2. Train Fast R-CNN

# In[ ]:


for epoch in range(epochs):
#         for iter_ in range(train_flag):
    for iter_ in randomtrain:
#         for iter_ in range(1,2):
        print('='*100)
        print('epoch:', epoch)
        print('iter:', iter_+1)
        img, bndbox, boxlabel, flag, scale = dataload.Dataloader(dataloader, iter_)
        if flag == 1:
            print('Abnormal data!')
            continue
        else:
            boxlabel_ = []
            for targ in boxlabel:
                if targ == 'face':
                    boxlabel_.append(1)
                else:
                    boxlabel_.append(2)
            boxlabel_ = torch.LongTensor(boxlabel_).to(device)
            bndbox = bndbox.float().to(device)
            img = img.float().to(device)
        H = img.shape[2]
        W = img.shape[3]
        gt = bndbox.to(device)
        gt_label = boxlabel_.to(device)
        start = time.time()
        #loss_rpn, roi, _, _ = rpn.forward(img, crit_rpn)
        loss_rcnn = train_rcnn(img, optimizer_rcnn, crit_rcnn)
        end = time.time()
        print('time: %.2f s' %(end-start))


# Step 3. Retrain RPN (Fixed Model)

# In[ ]:


rpn.vgg = copy.deepcopy(fastrcnn.vgg)
rpn.nograd(TF=True)
for epoch in range(epochs):
#         for iter_ in range(train_flag):
    for iter_ in randomtrain:
        print('='*100)
        print('epoch:', epoch)
        print('iter:',iter_+1)
        img, bndbox, boxlabel, flag, scale = dataload.Dataloader(dataloader, iter_)
        if flag == 1:
            print('Abnormal data!')
            continue
        else:
            boxlabel_ = []
            for targ in boxlabel:
                if targ == 'face':
                    boxlabel_.append(1)
                else:
                    boxlabel_.append(2)
            boxlabel_ = torch.LongTensor(boxlabel_).to(device)
            bndbox = bndbox.float().to(device)
            img = img.float().to(device)

        H = img.shape[2]
        W = img.shape[3]
        gt = bndbox.to(device)
        gt_label = boxlabel_.to(device)

        start = time.time()
        loss_rpn, roi = train_rpn(img, optimizer_rpn, crit_rpn)
        end = time.time()
        print('time: %.2f s' %(end-start))


# Step 4. Retrain Fast R-CNN (Fixed Model)

# In[ ]:


fastrcnn.nograd(TF=True)
for epoch in range(epochs):
#         for iter_ in range(train_flag):
    for iter_ in randomtrain:
        print('='*100)
        print('epoch:', epoch)
        print('iter:', iter_+1)
        img, bndbox, boxlabel, flag, scale = dataload.Dataloader(dataloader, iter_)
        if flag == 1:
            print('Abnormal data!')
            continue
        else:
            boxlabel_ = []
            for targ in boxlabel:
                if targ == 'face':
                    boxlabel_.append(1)
                else:
                    boxlabel_.append(2)
            boxlabel_ = torch.LongTensor(boxlabel_).to(device)
            bndbox = bndbox.float().to(device)
            img = img.float().to(device)
#         print(img.shape)
        H = img.shape[2]
        W = img.shape[3]
        gt = bndbox.to(device)
        gt_label = boxlabel_.to(device)
        start = time.time()
        loss_rcnn = train_rcnn(img, optimizer_rcnn, crit_rcnn)
        end = time.time()
        print('time: %.2f s' %(end-start))


# # Evaluation

# In[ ]:


rpn.Evalmode()
print(rpn.topN_nms)

for iter_ in random.sample(list(range(6120)),10):
    print('='*100)
    print('iter:',iter_+1)
    img, bndbox, boxlabel, flag, scale = dataload.Dataloader(dataloader, iter_)
    if flag == 1:
        print('Abnormal data!')
        continue
    else:
        boxlabel_ = []
        for targ in boxlabel:
            if targ == 'face':
                boxlabel_.append(1)
            else:
                boxlabel_.append(2)
        boxlabel_ = torch.LongTensor(boxlabel_).to(device)
        bndbox = bndbox.float().to(device)
        img = img.float().to(device)
    H = img.shape[2]
    W = img.shape[3]
    gt = bndbox.to(device)
    gt_label = boxlabel_.to(device)

    start = time.time()
    box, score, blabel = evaluate(img, crit_rpn, crit_rcnn)
    if not box.tolist():
        continue
    end = time.time()
    print('time: %.2f s' %(end-start))
    
    img_ = dataload.Unnormalize_Orgsizeimg(img, scale)
    Ho = img_.shape[0]
    Wo = img_.shape[1]
    bndbox = dataload.resize_box(bndbox.tolist(), (H,W), (Ho,Wo)).astype(int)
    print(box)
    box = dataload.resize_box(box.long().tolist(), (H,W), (Ho,Wo)).astype(int)
    for j in range(len(bndbox)):
        cv2.rectangle(img_, (bndbox[j][0],bndbox[j][1]), (bndbox[j][2],bndbox[j][3]), (255,0,0), 2)
        if boxlabel_[j] == 1:
            text = 'face'
        elif boxlabel_[j] == 2:
            text = 'face_mask'
        cv2.putText(img_, text, (bndbox[j][0],bndbox[j][1]), 1, 1.5, (255,0,0), 2)
    for j in range(len(box)):
        box_ = box[j].tolist()
        cv2.rectangle(img_, (box_[0],box_[1]), (box_[2],box_[3]), (0,200,255), 2)
        if blabel[j] == 1:
            text = 'face'
        elif blabel[j] == 2:
            text = 'face_mask'
        cv2.putText(img_, text, (box_[0],box_[1]), 1, 1.5, (0,200,255), 2)
    plt.figure(figsize = (10,10))
    plt.imshow(img_)


# In[ ]:


featmap = rpn.vgg(img)
featmap = Variable(F.relu(rpn.conv_3(featmap), inplace=True), requires_grad=True).to(device)
Hf = featmap.shape[2]
Wf = featmap.shape[3]  
anchor = rpn.GenerateAnchor((Hf,Wf), anc_ratio, anc_size, sampling)
anchor = torch.from_numpy(anchor).float().to(device)

pred_cls, pred_reg, pred_obj = rpn.Prediction(featmap)

pred_reg_ = pred_reg[0].data        
pred_obj_ = pred_obj[0].data

roi = delta2xy(anchor, pred_reg_)
roi[roi<0] = 0
roi[:,3][roi[:,3]>H] = H-1
roi[:,2][roi[:,2]>W] = W-1
ind_roi = (roi[:,2]-roi[:,0]>sampling) & (roi[:,3]-roi[:,1]>sampling)
roi = roi[ind_roi]
pred_obj_ = pred_obj_[ind_roi]

ind_sort = torch.sort(pred_obj_.squeeze(), descending=True)[1][:rpn.topN_nms]
roi = roi[ind_sort]

thr_nms = 0.7
roi_nms = roi.clone()
ind_nms = torch.Tensor(range(len(roi_nms)))
ind_nms_ = []
while(len(roi_nms)>1):
    gt_nms = roi_nms[0].reshape(1,-1)
    ind_nms_.append(ind_nms[0])
    IoU_nms = my_IoU(roi_nms, gt_nms)
    roi_nms = roi_nms[(IoU_nms<thr_nms).squeeze()]
    ind_nms = ind_nms[(IoU_nms<thr_nms).squeeze()]

ind_nms_ = ind_nms_[:rpn.n_nms]
roi = roi[ind_nms_].to(device)

