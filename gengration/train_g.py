#coding=utf-8

import os
import sys
import datetime
import argparse
import numpy as np
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from gengration.model_g import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.attention import *


class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])

        self.samples = []
        with open(args.datapath+'/'+args.list, 'r') as lines:
            for line in lines:
                name, label, box = line.strip().split(',')
                self.samples.append([name, int(label)])

    def __getitem__(self, idx):

        name, label = self.samples[idx]
        label = torch.LongTensor([label, label, label, label])
        image       = cv2.imread(self.args.datapath+'/images/'+name)
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rotated_imgs = [
            self.transform(image=image)['image'],
            self.transform(image=self.rotate_img(image, 90))['image'],
            self.transform(image=self.rotate_img(image, 180))['image'],
            self.transform(image=self.rotate_img(image, 270))['image']
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])

        return torch.stack(rotated_imgs, dim=0), rotation_labels, label

    def __len__(self):
        return len(self.samples)

    def rotate_img(self, img, rot):
        if rot == 0:
            return img
        elif rot == 90:
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:
            return np.fliplr(np.flipud(img))
        elif rot == 270:
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class Train(object):
    def __init__(self, Data, Model, args):
        ## dataset
        self.args    = args 
        self.data    = Data(args)
        self.loader  = DataLoader(self.data, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model = Model(args)
        self.model.train(True)
        self.model.cuda()
        ## parameter
        base, head = [], []
        for name, param in self.model.named_parameters():
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)

        self.optimizer = torch.optim.SGD([{'params':base, 'lr':self.args.lr*0.1}, {'params':head, 'lr':self.args.lr}], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O0')
        self.logger = SummaryWriter(args.savepath)

    def cam_reg(self, pred1, pred2, pred3, pred4, outb, outp, label, rota, rotation_labels):
        n = 2
        s = torch.zeros(n, n)
        pred = torch.maximum(pred2, pred4)
        pred = pred.mean(dim=(2, 3))

        pred1, pred2, pred3, pred4 = pred1.mean(dim=(2, 3)), pred2.mean(dim=(2, 3)), pred3.mean(dim=(2, 3)), pred4.mean(dim=(2, 3))
        f1, f2 = outb.mean(dim=(2, 3)), outp.mean(dim=(2, 3))

        f1 = torch.div(f1, f1.norm(dim=1, keepdim=True))
        f2 = torch.div(f2, f2.norm(dim=1, keepdim=True))


        x = [f1, f2]

        for i in range(n):
            for j in range(n):
                s[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    s[i, j] = 1.0 - s[i, j]

        regloss = torch.mul(torch.sum(torch.triu(s)), 0.5)
        loss1 = F.cross_entropy(pred1, label)
        loss2 = -(torch.softmax(pred1.detach(), dim=1) * torch.log_softmax(pred2, dim=1)).sum(dim=1).mean()
        loss5 = F.cross_entropy(pred, label)
        loss6 = F.cross_entropy(rota, rotation_labels)

        loss = (loss1 + loss2)/ 2 + loss5 + regloss + loss6

        return loss


    def train(self):
        global_step = 0

        for epoch in range(self.args.epoch):
            for image, rotation_labels, label in self.loader:
                image, rotation_labels, label= image.cuda().float(),rotation_labels.cuda().long(), label.cuda().long()

                image = torch.reshape(image, (-1, 3, 224, 224))
                rotation_labels = rotation_labels.reshape(-1)
                label = label.reshape(-1)

                ## step1

                pred1, pred2, pred3, pred4, outb, outp, rota = self.model(image, label)

                loss = self.cam_reg(pred1, pred2, pred3, pred4, outb, outp, label, rota, rotation_labels)

                self.optimizer.zero_grad()
                with apex.amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
                self.optimizer.step()

                ## log
                global_step += 1
                self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                self.logger.add_scalars('loss', {'loss':loss.item()}, global_step=global_step)
                if global_step % 10 == 0:
                    print('%s | step:%d/%d/%d | lr=%.6f | loss1=%.6f '%(datetime.datetime.now(), global_step, epoch+1, self.args.epoch, self.optimizer.param_groups[0]['lr'], loss.item()))
            # if epoch>self.args.epoch/2:
            torch.save(self.model.state_dict(), self.args.savepath+'/model-'+str(epoch+1))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='/data/code/zhaofan/Datasets/CUB/CUB_200_2011')
    parser.add_argument('--savepath'    ,default='./out')
    parser.add_argument('--mode'        ,default='train')
    parser.add_argument('--list'        ,default='train.txt')
    parser.add_argument('--clsnum'      ,default=200)
    parser.add_argument('--lr'          ,default=0.02)
    parser.add_argument('--epoch'       ,default=32)
    parser.add_argument('--batch_size'  ,default=64)
    parser.add_argument('--weight_decay',default=1e-4)
    parser.add_argument('--momentum'    ,default=0.9)
    parser.add_argument('--nesterov'    ,default=True)
    parser.add_argument('--num_workers' ,default=0)
    parser.add_argument('--snapshot'    ,default=None)
    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    t = Train(Data, Model, args)
    t.train()
