import torch.nn as nn
import numpy as np
    
class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3), keepdims=True)
        cur_std = x.std((2, 3), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return new_x


def replace_bn_with_adabn(model, newBN, warm_n=5):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # 提取原 BN 层的输入通道数
            in_ch = module.num_features
            # 创建新的 AdaBN 层
            new_bn_layer = newBN(in_ch, warm_n)
            # 保留原有参数，确保为可求导参数
            state_dict = module.state_dict()
            new_bn_layer.load_state_dict(state_dict, strict=False)
            new_bn_layer.weight.requires_grad = True
            new_bn_layer.bias.requires_grad = True
            new_bn_layer.running_mean.requires_grad = True
            new_bn_layer.running_var.requires_grad = True
            # new_bn_layer.weight.data.copy_(module.weight.data)
            # new_bn_layer.bias.data.copy_(module.bias.data)
            # new_bn_layer.running_mean.data.copy_(module.running_mean.data)
            # new_bn_layer.running_var.data.copy_(module.running_var.data)
            # new_bn_layer.eps = module.eps  # 复制小常数以避免零除

            # 替换模型中的层
            setattr(model, name, new_bn_layer)

        # 递归地处理子模块
        replace_bn_with_adabn(module, newBN, warm_n)
        
    return model