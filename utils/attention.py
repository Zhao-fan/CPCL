#coding=utf-8

import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import time


def attention_drop_test(attention_maps,input_image):
    B,N,W,H = input_image.shape
    input_tensor = input_image
    batch_size, height, width = attention_maps.shape
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(),size=(W,H),mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps,(W,H)).reshape(batch_size,-1)
    part_weights = torch.add(torch.sqrt(part_weights),1e-12)
    part_weights = torch.div(part_weights,torch.sum(part_weights,dim=1).unsqueeze(1)).cpu().numpy()

    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i].detach()
        part_weight = part_weights[i]
        selected_index = np.random.choice(
            np.arange(0, 1), 1, p=part_weight)[0]
        mask = attention_map[selected_index:selected_index + 1, :, :]

        threshold = random.uniform(0.7, 1)  #0.2  0.5
        mask = (mask < threshold * mask.max()).float()
        masks.append(mask)
    masks = torch.stack(masks)
    ret = input_tensor*masks
    return ret

def attention_drop(attention_maps,input_image):
    B,N,W,H = input_image.shape
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(),size=(W,H),mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps,(W,H)).reshape(batch_size,-1)
    part_weights = torch.add(torch.sqrt(part_weights),1e-12)
    part_weights = torch.div(part_weights,torch.sum(part_weights,dim=1).unsqueeze(1)).cpu().numpy()

    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i].detach()
        part_weight = part_weights[i]
        selected_index = np.random.choice(
            np.arange(0, num_parts), 1, p=part_weight)[0]
        mask = attention_map[selected_index:selected_index + 1, :, :]

        threshold = random.uniform(0.5, 0.7)  # 0.5  0.7
        mask = (mask < threshold * mask.max()).float()
        masks.append(mask)
    masks = torch.stack(masks)
    ret = input_tensor*masks
    return ret




if __name__ == '__main__':
    import torch
    a = torch.rand(4*26*26*32).reshape(4, 32, 26, 26)
    # a = torch.Tensor((4, 32, 26, 26))
    img = torch.arange(4*3*448*448.0).reshape(4, 3, 448, 448)

