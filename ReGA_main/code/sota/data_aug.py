import torch
import torch.nn as nn
import torch.nn.functional as F


class CTContrast(nn.Module):
    """
    适用于 CT：对 (B,1,D,H,W) 或 (B,1,H,W) 做强度对比度增强
    这里用“线性对比度”+clamp，避免过强破坏 HU 结构
    """

    def __init__(self, p=0.5, factor_range=(0.85, 1.15), clip=(-1.0, 1.0)):
        super().__init__()
        self.p = p
        self.factor_range = factor_range
        self.clip = clip

    def forward(self, x):
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        # x assumed normalized (e.g., z-score then clamp, or minmax to [-1,1])
        factor = torch.empty(1, device=x.device).uniform_(*self.factor_range).item()
        mean = x.mean(dim=tuple(range(2, x.ndim)), keepdim=True)  # per-sample mean
        y = (x - mean) * factor + mean
        return y.clamp(*self.clip)


class CTSharpness(nn.Module):
    """
    Unsharp mask：y = x + a*(x - blur(x))
    - a 越大越锐，但太大可能放大噪声 → 建议小范围
    - 对 3D：用 3D average blur；对 2D：用 2D average blur
    """

    def __init__(self, p=0.3, amount_range=(0.1, 0.4), blur_ks=3, clip=(-1.0, 1.0)):
        super().__init__()
        self.p = p
        self.amount_range = amount_range
        self.blur_ks = blur_ks
        self.clip = clip

    def forward(self, x):
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        a = torch.empty(1, device=x.device).uniform_(*self.amount_range).item()

        if x.ndim == 5:  # (B,1,D,H,W)
            # average blur 3D
            k = self.blur_ks
            pad = k // 2
            blur = F.avg_pool3d(x, kernel_size=k, stride=1, padding=pad)
        elif x.ndim == 4:  # (B,1,H,W)
            k = self.blur_ks
            pad = k // 2
            blur = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        else:
            raise ValueError("Expected x shape (B,1,H,W) or (B,1,D,H,W).")

        y = x + a * (x - blur)
        return y.clamp(*self.clip)


class CTEqualizeQuantile(nn.Module):
    """
    温和的“均衡”替代：按分位数拉伸到 [-1,1]
    - 对 CT 更稳
    - 不会像强 histogram equalize 那样放大噪声
    """

    def __init__(self, p=0.3, q_low=0.01, q_high=0.99, eps=1e-6):
        super().__init__()
        self.p = p
        self.q_low = q_low
        self.q_high = q_high
        self.eps = eps

    def forward(self, x):
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        # flatten per-sample
        dims = tuple(range(2, x.ndim))
        x_flat = x.flatten(start_dim=2)  # (B,C,N)
        lo = torch.quantile(x_flat, self.q_low, dim=2, keepdim=True)
        hi = torch.quantile(x_flat, self.q_high, dim=2, keepdim=True)
        y = (x_flat - lo) / (hi - lo + self.eps)  # [0,1]
        y = y * 2.0 - 1.0  # [-1,1]
        return y.view_as(x)


class PelvisAugForHD(nn.Module):
    """
    给骨盆CT用：主要强化“边界可见性”，目标是更可能改善 HD95
    """

    def __init__(self):
        super().__init__()
        self.contrast = CTContrast(p=0.5, factor_range=(0.85, 1.15))
        self.sharp = CTSharpness(p=0.35, amount_range=(0.1, 0.35), blur_ks=3)
        self.equalize = CTEqualizeQuantile(p=0.25, q_low=0.01, q_high=0.99)

    def forward(self, x):
        x = self.equalize(x)
        x = self.contrast(x)
        x = self.sharp(x)
        return x
