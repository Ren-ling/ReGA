# code/networks/nnunet_cascade_fullres.py

import pickle
import torch
import torch.nn as nn

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He


class CascadeFullResUNet(Generic_UNet):
    """
    用 nnUNet 的 Generic_UNet 构建 3D cascade fullres 网络，
    并额外提供 TEGDA/TTA 需要的接口：
      - get_feature(x): 返回 encoder 多层特征 [skip1, skip2, ..., bottleneck]
      - get_output(features): 只跑 decoder + seg head
      - forward_no_adapt(x): 显式带梯度的前向，用于 TTA 里的自适应
    """

    def __init__(self, plans_path):
        with open(plans_path, 'rb') as f:
            plans = pickle.load(f)

        # 选 fullres 阶段
        stage = list(plans['plans_per_stage'].keys())[-1]
        stage_plans = plans['plans_per_stage'][stage]

        patch_size = stage_plans['patch_size']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        base_num_features = plans['base_num_features']
        num_modalities = plans['num_modalities']
        num_fg_classes = plans['num_classes']
        num_classes = num_fg_classes + 1  # 背景 + 前景
        # cascade: 输入 = 原始模态 + 上一级预测（不含背景）
        num_input_channels = num_modalities + (num_classes - 1)
        net_numpool = len(pool_op_kernel_sizes)

        # 这里基本照 nnUNet TrainerCascadeFullRes 里的构造参数来
        super().__init__(
            input_channels=num_input_channels,
            base_num_features=base_num_features,
            num_classes=num_classes,
            num_pool=net_numpool,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.2, "inplace": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
            deep_supervision=True,  # 训练时可用；我们前向里会只取主输出
            dropout_in_localization=False,
            final_nonlin=lambda x: x,  # 输出保持 logits，TTA 里自己做 softmax
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=True,  # nnUNet 3D 默认是 conv stride 下采样
            convolutional_upsampling=True,  # conv transpose 上采样
            max_num_features=None,
        )

    # =========================================================
    # 1) 提取 encoder 多层特征：和你 unet_3D 的 get_feature 风格尽量对齐
    #    返回一个 list: [skip1, skip2, ..., skipK, bottleneck]
    # =========================================================
    def get_feature(self, x):
        """
        返回 encoder 的所有 skip + bottleneck，顺序为：
        [skip_level1, skip_level2, ..., skip_levelK, bottleneck]
        其中：
          - skip_level1 分辨率最高（最浅层）
          - skip_levelK 分辨率最低（最靠近 bottleneck 的那层）
          - bottleneck = conv_blocks_context[-1] 的输出
        """
        skips = []

        # encoder 部分：完全照 Generic_UNet.forward
        # 注意：我们这里故意保留和原 forward 一样的逻辑，
        #     只是在最后把中间特征存起来
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            # 我们构建时 convolutional_pooling=True，
            # 所以下面这句不会执行，但保留写法以防你以后改参数
            if not self.convolutional_pooling:
                x = self.td[d](x)
        # print("skips shapes: ", [s.shape for s in skips])  # 打印每一层 skip 的尺寸

        # bottleneck
        x = self.conv_blocks_context[-1](x)
        bottleneck = x

        # 和你 unet_3D 对齐：返回一个 list
        features = skips + [bottleneck]
        return features

    # =========================================================
    # 2) 仅根据 encoder_feature（list）跑 decoder + seg head
    #    用于：TTA 中替换某一层特征后重新 forward
    # =========================================================
    def get_output(self, encoder_features):
        """
        encoder_features: list
          [skip1, skip2, ..., skipK, bottleneck]

        返回：主输出 logits（和 forward() 一致的空间分辨率）
        """
        # 拆出 skip 和 bottleneck
        skips = encoder_features[:-1]
        x = encoder_features[-1]  # bottleneck

        seg_outputs = []

        # decoder：完全照 Generic_UNet.forward 的写法
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            # 与 forward 一致：从后往前取 skip（高层在最后）
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        # 这里我们只要最高分辨的输出（最后一个 seg_outputs）
        # 与 Generic_UNet.forward 中 seg_outputs[-1] 对齐
        return seg_outputs[-1]

    # =========================================================
    # 3) 给 TTA 调用的“带梯度前向”
    #    注意：Generic_UNet.forward 在 deep_supervision=True 时
    #          会返回一个 tuple，我们这里只返回主输出
    # =========================================================
    @torch.enable_grad()
    def forward_no_adapt(self, x):
        """
        显式开启梯度的前向。TTA / MC-dropout 等在测试时需要 backward，
        建议在这些场景用这个接口。
        """
        out = super().forward(x)
        # 如果开启了 deep supervision，这里 out 是 tuple:
        #   (seg_main, seg_ds1, seg_ds2, ...)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    # =========================================================
    # 4) 覆盖 forward：统一只返回主输出（logits）
    #    这样 TTA 里的 self.model(x) 行为明确、简单
    # =========================================================
    def forward(self, x):
        """
        正常 forward：只返回主输出（logits）。
        （如果你想在训练 fullres 阶段用 deep supervision，
          可以不要重写 forward，而是直接用 Generic_UNet 原版；
          这里主要面向 cascade fullres + TTA 的推理场景。）
        """
        return self.forward_no_adapt(x)
