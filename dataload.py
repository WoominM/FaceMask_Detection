from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os

def load_data(data_dir = "./", batch_size = 1):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def loadxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    size = []
    bnd_label = []
    bndbox = []
    for i in range(len(root)):
        if root[i].tag == 'size':
            for ele in root[i]:
                size.append(int(ele.text))
        elif root[i].tag == 'object':
            for ele in root[i]:
                if ele.tag == 'name':
                    bnd_label.append(ele.text)
                elif ele.tag == 'bndbox':
                    bnd = []
                    for ele_ in ele:
                        bnd.append(int(ele_.text))
                    bndbox.append(bnd)
    return [size, bnd_label, bndbox]

def resize_img(img, imgsize, min_size = 600, max_size = 1000):
    W, H = imgsize
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    img = cv2.resize(img.permute(1,2,0).numpy(), (int(W*scale), int(H*scale)))
    return img, scale

def resize_box(bbox, in_size, out_size):
    bbox = np.array(bbox).copy()
    y_scale = float(out_size[0]) / int(in_size[0])
    x_scale = float(out_size[1]) / int(in_size[1])
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    return bbox

def Dataloader(dataloader, i):
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    path = path[:-3]+'xml'
    imgsize, boxlabel, bndbox = loadxml(path)
    if (bndbox == []) | (imgsize == []) | (boxlabel == []) | (0 in imgsize):
        flag = 1
        return [0, 0, 0, flag, 0]
    else : flag = 0             
    img, scale = resize_img(img, imgsize[:-1])
    bndbox = resize_box(bndbox, imgsize[:-1], [scale*ele for ele in imgsize[:-1]])
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0), torch.from_numpy(bndbox), boxlabel, flag, scale

def Unnormalize_Orgsizeimg(img, scale):
    std = np.array([0.485, 0.456, 0.406])
    mean = np.array([0.229, 0.224, 0.225])
    img_ = img[0].permute(1,2,0).cpu().numpy()
    H, W = img_.shape[:2]
    img_ = cv2.resize(img_, (int(np.round(W/scale)), int(np.round(H/scale))))
    img_ = img_*std + mean
    return img_
