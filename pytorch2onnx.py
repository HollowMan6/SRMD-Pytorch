import re
import os
import torch
from utils.network_srmd import SRMD as net

n_channels = 3            # fixed
nc = 128                  # fixed, number of channels
nb = 12                   # fixed, number of conv layers

model_list = ['srmd_x2', 'srmd_x3', 'srmd_x4', 'srmdnf_x2', 'srmdnf_x3', 'srmdnf_x4']
model_pool = 'model_zoo'
output_model_pool = 'onnx_models'


for model_name in model_list:
    sf = [int(s) for s in re.findall(r'\d+', model_name)][0]  # scale factor
    in_nc = 18 if 'nf' in model_name else 19
    model_path = os.path.join(model_pool, model_name+'.pth')
    
    model = net(in_nc=in_nc, out_nc=n_channels, nc=nc, nb=nb, upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    x = torch.randn((1, 3, 1, 1))
    k_pca = torch.randn(1, 15, 1, 1)
    m = k_pca.repeat(1, 1, x.size()[-2], x.size()[-1])
    x = torch.cat((x, m), 1)
    if 'nf' not in model_name:  # noise-free SR
        noise_level = torch.zeros(1, 1, 1, 1)
        x = torch.cat((x, noise_level), 1)
    torch_out = torch.onnx._export(model, x, os.path.join(output_model_pool, model_name+".onnx"), export_params=True)
    print("Convert "+ model_name+" success!")