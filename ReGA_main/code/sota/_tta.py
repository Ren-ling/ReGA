import logging
from statistics import mode
import time
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from . import tent
from . import cotta
from . import ReGA
from . import vptta
from utils.sam import SAM
import SimpleITK as sitk
import math


def setup_source(model, device):
    """Set up the baseline source model without adaptation."""
    model = model.to(device)
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model




def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    if len(params) == 0:
        raise ValueError("No adaptable parameters found! Check if model has BatchNorm3d or affine InstanceNorm3d.")
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer)
    # logging.info(f"model for adaptation: %s", model)
    # logging.info(f"params for adaptation: %s", param_names)
    # logging.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model



def setup_cotta(model, device):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model, device)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer, device=device)
    # logging.info(f"model for adaptation: %s", model)
    # logging.info(f"params for adaptation: %s", param_names)
    # logging.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


def create_ema_model(model):
    ema_model = deepcopy(model)  # get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model




def setup_ReGA(model, device):
    anchor_model = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.5, 0.999))
    mt_model = ReGA.TTA(model.to(device), anchor_model, optimizer, device=device)  # 将主模型也转移到指定设备
    return mt_model




def setup_vptta(model):
    import types
    import torch

    print("=== VPTTA Setup ===")

    from . import vptta

    # 创建 Prompt
    in_channels = 5
    prompt = vptta.Prompt(
        prompt_alpha=0.03,
        image_size=128,
        in_channels=in_channels
    ).to('cuda:0')

    # 计算正确的 Memory 维度
    prompt_size = prompt.prompt_size
    dimension = 1 * in_channels * prompt_size * prompt_size * prompt_size
    print(f"Memory dimension calculated: {dimension}")

    # 导入 Memory
    from utils.memory_3d import Memory

    # 创建 Memory
    memory_bank = Memory(size=40, dimension=dimension)

    optimizer = torch.optim.Adam(
        prompt.parameters(),
        lr=0.05,
        betas=(0.9, 0.99),
        weight_decay=0.0
    )

    # 创建 VPTTA 实例
    vptta_model = vptta.VPTTA(model, optimizer, prompt)

    # 替换 memory_bank
    vptta_model.memory_bank = memory_bank

    return vptta_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    # """
    # if cfg.OPTIM.METHOD == 'Adam':
    return optim.Adam(params,
                      lr=0.00001,
                      betas=(0.9, 0.999),
                      weight_decay=0.9)

