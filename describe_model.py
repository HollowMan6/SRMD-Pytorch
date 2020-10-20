#!/usr/bin/python3
# -*- coding:utf-8 -*-
# by 'hollowman6' from Lanzhou University(兰州大学)

import re
import os
import torch
from torchsummary import summary
from utils import utils_model
from utils.network_srmd import SRMD as net

n_channels = 3            # fixed
nc = 128                  # fixed, number of channels
nb = 12                   # fixed, number of conv layers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_list = ['srmd_x2', 'srmd_x3', 'srmd_x4', 'srmdnf_x2', 'srmdnf_x3', 'srmdnf_x4']
model_pool = 'model_zoo'


for model_name in model_list:
    sf = [int(s) for s in re.findall(r'\d+', model_name)][0]  # scale factor
    in_nc = 18 if 'nf' in model_name else 19
    model_path = os.path.join(model_pool, model_name+'.pth')

    print(model_name+":")
    
    model = net(in_nc=in_nc, out_nc=n_channels, nc=nc, nb=nb, upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    x = (18, 1, 1)
    if 'nf' not in model_name:  # noise-free SR
        x = (19, 1, 1)
    print(utils_model.describe_model(model))
    summary(model, input_size=x)
    print("")