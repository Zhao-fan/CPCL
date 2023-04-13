#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import ResNet50
from utils import weight_init
from utils.attention import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args    = args
        self.bkbone  = ResNet50()
        self.fc51    = nn.Conv2d(2048,  128, kernel_size=1, stride=1, padding=0)
        self.bn51    = nn.BatchNorm2d(128)
        self.fc52    = nn.Conv2d( 128, 2048, kernel_size=1, stride=1, padding=0)
        self.fc41    = nn.Conv2d(1024,  128, kernel_size=1, stride=1, padding=0)
        self.bn41    = nn.BatchNorm2d(128)
        self.fc42    = nn.Conv2d( 128, 1024, kernel_size=1, stride=1, padding=0)
        self.fc31    = nn.Conv2d( 512,  128, kernel_size=1, stride=1, padding=0)
        self.bn31    = nn.BatchNorm2d(128)
        self.fc32    = nn.Conv2d( 128,  512, kernel_size=1, stride=1, padding=0)

        self.linear5 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn5     = nn.BatchNorm2d(128)
        self.linear4 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn4     = nn.BatchNorm2d(128)
        self.linear3 = nn.Conv2d( 512, 128, kernel_size=1, stride=1, padding=0)
        self.bn3     = nn.BatchNorm2d(128)
        self.linear  = nn.Conv2d(128, self.args.clsnum, kernel_size=1, stride=1, padding=0)


        self.fc51p = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn51p = nn.BatchNorm2d(128)
        self.fc52p = nn.Conv2d(128, 2048, kernel_size=1, stride=1, padding=0)
        self.fc41p = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn41p = nn.BatchNorm2d(128)
        self.fc42p = nn.Conv2d(128, 1024, kernel_size=1, stride=1, padding=0)
        self.fc31p = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn31p = nn.BatchNorm2d(128)
        self.fc32p = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.linear5p = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn5p = nn.BatchNorm2d(128)
        self.linear4p = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn4p = nn.BatchNorm2d(128)
        self.linear3p = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn3p = nn.BatchNorm2d(128)
        self.fc = nn.Linear(2048, 4)

        self.initialize()

    def forward(self, x, label, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)

        out5a  = F.relu(self.bn51(self.fc51(out5.mean(dim=(2,3), keepdim=True))), inplace=True)
        out4a  = F.relu(self.bn41(self.fc41(out4.mean(dim=(2,3), keepdim=True))), inplace=True)
        out3a  = F.relu(self.bn31(self.fc31(out3.mean(dim=(2,3), keepdim=True))), inplace=True)
        vector = out5a*out4a*out3a
        out5b   = torch.sigmoid(self.fc52(vector))*out5
        out4b   = torch.sigmoid(self.fc42(vector))*out4
        out3b   = torch.sigmoid(self.fc32(vector))*out3


        out5b = F.relu(self.bn5(self.linear5(out5b)), inplace=True)
        out5b = F.interpolate(out5b, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4b = F.relu(self.bn4(self.linear4(out4b)), inplace=True)
        out4b = F.interpolate(out4b, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out3b = F.relu(self.bn3(self.linear3(out3b)), inplace=True)
        # print(out5b.shape, out4b.shape, out3b.shape)

        p1 = F.relu(self.linear(out5b), inplace=True)
        p2 = F.relu(self.linear(out4b), inplace=True)
        p3 = F.relu(self.linear(out3b), inplace=True)
        # print(p1.shape, p2.shape, p3.shape)

        if self.args.mode=='train':
            drop5 = attention_drop(p1[:, label, :, :], out5)  # label
            drop4 = attention_drop(p2[:, label, :, :], out4)
            drop3 = attention_drop(p3[:, label, :, :], out3)
        else:
            drop5 = attention_drop_test(p1[:, label, :, :], out5)
            drop4 = attention_drop_test(p2[:, label, :, :], out4)
            drop3 = attention_drop_test(p3[:, label, :, :], out3)


        out5ap = F.relu(self.bn51p(self.fc51p(drop5.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out4ap = F.relu(self.bn41p(self.fc41p(drop4.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out3ap = F.relu(self.bn31p(self.fc31p(drop3.mean(dim=(2, 3), keepdim=True))), inplace=True)
        vectorp = out5ap * out4ap * out3ap
        out5p = torch.sigmoid(self.fc52p(vectorp)) * drop5
        out4p = torch.sigmoid(self.fc42p(vectorp)) * drop4
        out3p = torch.sigmoid(self.fc32p(vectorp)) * drop3

        out5p = F.relu(self.bn5p(self.linear5p(out5p)), inplace=True)
        out5p = F.interpolate(out5p, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4p = F.relu(self.bn4p(self.linear4p(out4p)), inplace=True)
        out4p = F.interpolate(out4p, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out3p = F.relu(self.bn3p(self.linear3p(out3p)), inplace=True)


        outb = out5b * out4b * out3b
        outp = out5p*out4p*out3p

        rota = out5.mean(dim=(2,3), keepdim=False)
        rota = self.fc(rota)

        if self.args.mode=='train':
            pred1 = F.dropout(out5b, p=0.5)
            pred1 = F.relu(self.linear(pred1), inplace=True)

            pred2 = F.dropout(outb, p=0.5)
            pred2 = F.relu(self.linear(pred2), inplace=True)

            pred3 = F.dropout(out5p, p=0.5)
            pred3 = F.relu(self.linear(pred3), inplace=True)

            pred4 = F.dropout(outp, p=0.5)
            pred4 = F.relu(self.linear(pred4), inplace=True)
            return pred1, pred2, pred3, pred4, outb, outp, rota
        else:
            pred1 = F.relu(self.linear(outb), inplace=True)
            pred2 = F.relu(self.linear(outp), inplace=True)
            return pred1, pred2

    def initialize(self):
        if self.args.snapshot:
            print('load model...')
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)