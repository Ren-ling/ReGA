import os
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, sys, datetime
from torch.autograd import Variable
from utils.convert_3d import AdaBN
from utils.memory_3d import Memory
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

# class Prompt(nn.Module):#E1
    # def __init__(self, prompt_alpha=0.03, image_size=128, in_channels=5):  # 添加 in_channels 参数
    #     super().__init__()
    #     self.prompt_size = int(image_size * prompt_alpha)
    #     if self.prompt_size < 1:
    #         self.prompt_size = 1
    #
    #     self.padding_size = (image_size - self.prompt_size) // 2
    #     self.in_channels = in_channels  # 保存通道数
    #
    #     print(f"Prompt init: image_size={image_size}, prompt_size={self.prompt_size}, in_channels={in_channels}")
    #
    #     # 使用正确的通道数
    #     self.init_para = torch.ones((1, in_channels, self.prompt_size, self.prompt_size, self.prompt_size))
    #     self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
    #     self.pre_prompt = self.data_prompt.detach().cpu().data
    #
    # def update(self, init_data):
    #     with torch.no_grad():
    #         self.data_prompt.copy_(init_data)
class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.03, image_size=128, in_channels=5):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha)
        if self.prompt_size < 1:
            self.prompt_size = 1

        self.padding_size = (image_size - self.prompt_size) // 2
        self.in_channels = in_channels

        print(f"Prompt init: image_size={image_size}, prompt_size={self.prompt_size}, in_channels={in_channels}")

        # 修正：初始化为正确的形状 [1, in_channels, ...]
        self.init_para = torch.ones((1, in_channels, self.prompt_size, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            # 确保 init_data 在正确的设备上
            if init_data.device != self.data_prompt.device:
                init_data = init_data.to(self.data_prompt.device)

            # 如果形状不匹配，尝试调整
            if self.data_prompt.shape != init_data.shape:
                print(f"Adjusting shape in update: {init_data.shape} -> {self.data_prompt.shape}")

                # 如果只是批次维度不同
                if init_data.shape[1:] == self.data_prompt.shape[1:]:
                    if init_data.shape[0] > 1:
                        # 取平均值或第一个
                        init_data = init_data.mean(dim=0, keepdim=True)
                    elif init_data.shape[0] < 1:
                        init_data = init_data.expand(self.data_prompt.shape)

                # 如果需要插值
                if init_data.shape[2:] != self.data_prompt.shape[2:]:
                    init_data = F.interpolate(
                        init_data,
                        size=self.data_prompt.shape[2:],
                        mode='trilinear',
                        align_corners=False
                    )

            self.data_prompt.copy_(init_data)


    def iFFT(self, amp_src_, pha_src, imgD, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifftn(fft_src_, dim=(-3, -2, -1), s=[imgD, imgH, imgW]).real
        return src_in_trg

    # vptta.py 中的 Prompt.forward 方法修改
    def forward(self, x):
        batch_size, C, imgD, imgH, imgW = x.size()

        # print(f"[VPTTA DEBUG] Input shape: {x.shape}")
        # print(f"[VPTTA DEBUG] prompt_size: {self.prompt_size}")
        # print(f"[VPTTA DEBUG] padding_size: {self.padding_size}")
        # print(f"[VPTTA DEBUG] data_prompt shape: {self.data_prompt.shape}")

        # ===== 动态计算填充 =====
        pad_w = (imgW - self.prompt_size) // 2
        pad_h = (imgH - self.prompt_size) // 2
        pad_d = (imgD - self.prompt_size) // 2

        # print(f"[VPTTA DEBUG] Calculated padding: D={pad_d}, H={pad_h}, W={pad_w}")

        # ===== 傅里叶变换 =====
        fft = torch.fft.fftn(x.clone(), dim=(-3, -2, -1))
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src, dim=(-3, -2, -1))

        # print(f"[VPTTA DEBUG] amp_src shape: {amp_src.shape}")

        # ===== 关键修复：正确调整通道数 =====
        # data_prompt 是 [1, 4, 3, 3, 3]
        # 我们需要 [batch_size, 5, 3, 3, 3]

        # 1. 扩展批次维度
        data_prompt_batch = self.data_prompt.repeat(batch_size, 1, 1, 1, 1)
        # print(f"[VPTTA DEBUG] After batch repeat: {data_prompt_batch.shape}")

        # 2. 调整通道数 (4 -> 5)
        current_channels = data_prompt_batch.shape[1]
        target_channels = C

        if current_channels != target_channels:
            # print(f"[VPTTA DEBUG] Adjusting channels: {current_channels} -> {target_channels}")

            # 方法1：如果目标通道数是当前通道数的倍数
            if target_channels % current_channels == 0:
                repeat_factor = target_channels // current_channels
                data_prompt_adj = data_prompt_batch.repeat(1, repeat_factor, 1, 1, 1)
            else:
                # 方法2：创建新的张量并复制权重
                # 创建一个新的张量，形状为 [batch_size, target_channels, prompt_size, prompt_size, prompt_size]
                data_prompt_adj = torch.ones(
                    (batch_size, target_channels, self.prompt_size, self.prompt_size, self.prompt_size),
                    device=data_prompt_batch.device,
                    requires_grad=True
                )

                # 复制已有的权重
                if target_channels > current_channels:
                    # 复制现有通道到新通道
                    for i in range(target_channels):
                        src_channel = i % current_channels
                        data_prompt_adj[:, i] = data_prompt_batch[:, src_channel].clone()
                else:
                    # 如果目标通道数更少，取前 target_channels 个通道
                    data_prompt_adj = data_prompt_batch[:, :target_channels].clone()

                # 如果需要保持可学习，转换为 Parameter
                data_prompt_adj = nn.Parameter(data_prompt_adj, requires_grad=True)

        else:
            data_prompt_adj = data_prompt_batch

        # print(f"[VPTTA DEBUG] After channel adjustment: {data_prompt_adj.shape}")

        # ===== 填充到输入尺寸 =====
        prompt = F.pad(data_prompt_adj,
                       [pad_w, imgW - pad_w - self.prompt_size,
                        pad_h, imgH - pad_h - self.prompt_size,
                        pad_d, imgD - pad_d - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        # print(f"[VPTTA DEBUG] Padded prompt shape: {prompt.shape}")

        # ===== 确保形状完全匹配 =====
        if prompt.shape != amp_src.shape:
            # print(f"[VPTTA WARNING] Shape mismatch: prompt {prompt.shape} vs amp_src {amp_src.shape}")

            # 只调整空间维度，不调整批次和通道
            if prompt.shape[2:] != amp_src.shape[2:]:
                # print(f"[VPTTA DEBUG] Interpolating spatial dimensions...")
                prompt = F.interpolate(
                    prompt,
                    size=amp_src.shape[2:],  # 只包含 D, H, W
                    mode='trilinear',
                    align_corners=False
                )
                # print(f"[VPTTA DEBUG] After spatial interpolation: {prompt.shape}")

        # ===== 应用提示 =====
        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_, dim=(-3, -2, -1))

        # ===== 提取低频部分 =====
        amp_low_ = amp_src[:, :,
        pad_d:pad_d + min(self.prompt_size, imgD - pad_d),
        pad_h:pad_h + min(self.prompt_size, imgH - pad_h),
        pad_w:pad_w + min(self.prompt_size, imgW - pad_w)]

        # ===== 逆傅里叶变换 =====
        src_in_trg = self.iFFT(amp_src_, pha_src, imgD, imgH, imgW)

        return src_in_trg, amp_low_

class VPTTA(nn.Module):
    def __init__(self, model, optimizer, prompt):
        super().__init__()
        # Model
        self.model = model
        # Optimizer
        self.optimizer = optimizer
        # Prompt
        self.prompt = prompt
        self.iters = 1
        # Memory Bank
        self.neighbor = 16
        self.memory_bank = Memory(size=40, dimension=self.prompt.data_prompt.numel())
        self.debug = False
        self.print_prompt()
        print('***' * 20)

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def forward(self, x):
        x_shape = list(x.shape)
        # if (len(x_shape) == 5):
        #     [N, C, D, H, W] = x_shape
        #     new_shape = [N * D, C, H, W]
        #     x = torch.transpose(x, 1, 2)
        #     x = torch.reshape(x, new_shape)

        x = Variable(x)
        self.model.eval()
        self.prompt.train()

        if hasattr(self.model, 'change_BN_status'):
            self.model.change_BN_status(new_sample=True)

        # Initialize Prompt
        init_data = None

        if self.memory_bank is not None and len(self.memory_bank.memory.keys()) >= self.neighbor:
            try:
                _, low_freq = self.prompt(x)
                init_data, score = self.memory_bank.get_neighbours(
                    keys=low_freq.cpu().numpy(),
                    k=min(self.neighbor, len(self.memory_bank.memory.keys()))
                )
                if self.debug:
                    print(f"Memory bank score: {score:.4f}")
            except Exception as e:
                print(f"Memory bank query failed: {e}")
                init_data = torch.ones(
                    (1, 1, self.prompt.prompt_size, self.prompt.prompt_size, self.prompt.prompt_size)).data
        else:
            init_data = torch.ones(
                (1, 1, self.prompt.prompt_size, self.prompt.prompt_size, self.prompt.prompt_size)).data

        self.prompt.update(init_data)

        for tr_iter in range(self.iters):
            prompt_x, _ = self.prompt(x)
            self.model(prompt_x)
            times, bn_loss = 0, 0

            for nm, m in self.model.named_modules():
                if hasattr(m, 'bn_loss'):
                    bn_loss += m.bn_loss
                    times += 1

            if times > 0:
                loss = bn_loss / times
            else:
                loss = torch.tensor(0.001, requires_grad=True, device=x.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if hasattr(self.model, 'change_BN_status'):
                self.model.change_BN_status(new_sample=False)

        # Inference
        self.model.eval()
        self.prompt.eval()
        with torch.no_grad():
            prompt_x, low_freq = self.prompt(x)
            output = self.model(prompt_x)

        # Update the Memory Bank
        if self.memory_bank is not None:
            try:
                self.memory_bank.push(
                    keys=low_freq.cpu().numpy(),
                    logits=self.prompt.data_prompt.detach().cpu().numpy()
                )
            except Exception as e:
                print(f"Memory bank update failed: {e}")

        return output