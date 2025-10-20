from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

Align_Corners = False
Align_Corners_Range = False

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, disp_values):
    assert len(x.shape) == 4
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


def get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns=False, max_disp=192.0):
    #shape, (B, H, W)
    #cur_disp: (B, H, W) 是上阶段预测的视差图，已回归
    # disp_inteval_pixel 是视差间隔的倍率，基本上是1
    #return disp_range_samples: (B, D, H, W)
    if not using_ns: #using_ns=True
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel)  # (B, H, W)
        cur_disp_max = (cur_disp + ndisp / 2 * disp_inteval_pixel) 
        # 基本思想是对每个像素的预测视差进行扩张，得到考虑视差的上限和下限，扩展量只和视差级数量有关

        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)
        # 然后按(上限+下限)/视差级数量 的方式得到间隔，根据间隔得到视差级

        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
    else:
        #using neighbor region information to help determine the range.
        #consider the maximum and minimum values ​​in the region.
        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape) #shape=[1,512,256]
        B, H, W = cur_disp.shape
        cur_disp_smooth = F.interpolate((cur_disp / 4.0).unsqueeze(1),
                                        [H // 4, W // 4], mode='bilinear', align_corners=Align_Corners_Range).squeeze(1) #下采样到[1,128,64]
        #get minimum value
        disp_min_ns = torch.abs(F.max_pool2d(-cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2))    # 取负再算max pooling，得到局部最小值，输出size不变
        #get maximum value
        disp_max_ns = F.max_pool2d(cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2) # 同样的方式得到局部最大值

        disp_pred_inter = torch.abs(disp_max_ns - disp_min_ns)    #[1,128,64] 得到预测视差的间隔,这次最大值是45，最小是0.6
        disp_range_comp = (ndisp//4 * disp_inteval_pixel - disp_pred_inter).clamp(min=0) / 2.0  #前项是6，结果大部分是0

        cur_disp_min = (disp_min_ns - disp_range_comp).clamp(min=0, max=max_disp) #基本上没变化
        cur_disp_max = (disp_max_ns + disp_range_comp).clamp(min=0, max=max_disp)

        new_interval = (cur_disp_max - cur_disp_min) / (ndisp//4 - 1) # 新的视差间隔，最大9最小1.2，这个结果是像素级的

        # (B, 1/4D, 1/4H, 1/4W)
        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp//4, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,requires_grad=False).reshape(1,-1,1,1) * new_interval.unsqueeze(1))
        # 按视差间隔和视差级数量分割本阶段考虑的视差范围，生成的数列 [1,6,128,64]，这个结果是像素级的
        disp_range_samples = F.interpolate((disp_range_samples * 4.0).unsqueeze(1),
                                          [ndisp, H, W], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1) # 上采样到(B, D, H, W) [1,24,512,256]
    return disp_range_samples


def get_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, device, dtype, shape, using_ns, ns_size, max_disp=192.0):
    #shape, (B, H, W)
    #cur_disp: (B, H, W) or float
    #return disp_range_values: (B, D, H, W)
    # with torch.no_grad():
    if cur_disp is None:
        cur_disp = torch.tensor(0, device=device, dtype=dtype, requires_grad=False).reshape(1, 1, 1).repeat(*shape) #[1,512,256]，都是0
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)   #(B, H, W) 考虑的最小视差，第一次是0
        cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp) # 考虑的最大视差，第一次是188
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W) 第一次的视差间隔为4

        disp_range_volume = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
        #生成稀疏的视差范围，后一项是从最小视差到最大视差，按视差间隔生成的数列
        # [1，48，512，256]

    else:
        disp_range_volume = get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns, max_disp)

    return disp_range_volume