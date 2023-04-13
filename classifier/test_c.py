#coding=utf-8

import sys
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(685, 685),
            ToTensorV2()
        ])

        self.samples = []
        with open(args.datapath+'/'+args.list, 'r') as lines:
            for line in lines:
                name, label, box = line.strip().split(',')
                self.samples.append([name, int(label)])

    def __getitem__(self, idx):
        name, label = self.samples[idx]

        image       = cv2.imread(self.args.datapath+'/images/'+name)
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image       = self.transform(image=image)['image']
        return name, image, label

    def __len__(self):
        return len(self.samples)


class Test(object):
    def __init__(self, Data, args):
        ## dataset
        self.args    = args 
        self.data    = Data(args)
        self.loader  = DataLoader(self.data, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.clsnum)
        self.model.load_state_dict(torch.load(args.snapshot))
        self.model.train(False)
        self.model.cuda()

    def cls_save(self):
        top1top5, cnt = {}, 0
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for name, image, label in self.loader:
                image  = image.cuda().float()
                pred   = self.model(image).cpu()
                index  = torch.argsort(pred, dim=-1, descending=True)
                score  = (index==label.unsqueeze(1))
                for n, s in zip(name, score):
                    top1 = True if s[:1].sum().item()==1 else False
                    top5 = True if s[:5].sum().item()==1 else False
                    top1top5[n] = [top1, top5]
                cnt += len(name)
                # print(cnt)

                loss = torch.nn.functional.cross_entropy(pred, label)
                test_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()

                print('Test Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    cnt, test_loss / (cnt + 1), 100. * float(correct) / total, correct, total))

            np.save('cub_test_top1top5', top1top5)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='/data/code/zhaofan/Datasets/CUB/CUB_200_2011')
    parser.add_argument('--snapshot'    ,default='./out/model-30')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--list'        ,default='val.txt')
    parser.add_argument('--clsnum'      ,default=200)
    parser.add_argument('--batch_size'  ,default=64)
    parser.add_argument('--num_workers' ,default=8)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    t = Test(Data, args)
    t.cls_save()