import logging
from statistics import mode
import time
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
# from robustbench.data import dataset_all, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,Scale_img,Scale_imglab,scale,convert_2d,scal_spacing
from robustbench.utils import load_model,setup_source
from robustbench.utils import clean_accuracy as accuracy
from robustbench.metrics import dice_eval,assd_eval, hd95_eval
from robustbench.losses import DiceLoss,DiceCeLoss,WeightedCrossEntropyLoss
import tent
import norm
import cotta
import wjh01
import sitta
import meant
import memo
import upl
import sar
import vida
import svdp
from utils.sam import SAM
from unet import UNet
import SimpleITK as sitk
from conf import cfg, load_cfg_fom_args
import math

def setup_sar(model):
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=0.4*math.log(3))

    return adapt_model

def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    # logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    print(stat_names)
    # logger.info(f"stats for adaptation: %s", stat_names)
   
    return norm_model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return model,tent_model

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def create_ema_model(model):
    ema_model = deepcopy(model) # get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model
def setup_meant(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.OPTIM.LR,betas=(0.5,0.999))
    cotta_model = meant.TTA(model, anchor_model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           )
    return cotta_model

def setup_vsdp(model):
    anchor = deepcopy(model.state_dict())
    anchor_model = deepcopy(model)
    ema_model = create_ema_model(model)

    vsdp_model = svdp.TTA(model,anchor,anchor_model,ema_model)


def setup_sitta(model):
    cotta_model = sitta.TTA(model, 
                            repeat_num = 1,
                            check_p = cfg.MODEL.CKPT_DIR
                           )
    return cotta_model

def setup_vptta(model):
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.OPTIM.LR,betas=(0.5,0.999))

    return model

def setup_memo(model):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    cotta_model = memo.TTA(model,  optimizer )
    return cotta_model

def setup_vida(args, model):
    model = vida.configure_model(model, cfg)
    model_param, vida_param = vida.collect_params(model)
    optimizer = setup_optimizer_vida(model_param, vida_param, 5e-07, 2e-07)
    vida_model = vida.ViDA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = args.unc_thr,
                           ema = cfg.OPTIM.MT,
                           ema_vida = cfg.OPTIM.MT_ViDA,
                           )
    return vida_model

def setup_optimizer_vida(params, params_vida, model_lr, vida_lr):
    # if cfg.OPTIM.METHOD == 'Adam':
    return optim.Adam([{"params": params, "lr": model_lr},
                                {"params": params_vida, "lr": vida_lr}],
                                lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    # elif cfg.OPTIM.METHOD == 'SGD':
    #     return optim.SGD([{"params": params, "lr": model_lr},
    #                               {"params": params_vida, "lr": vida_lr}],
    #                                 momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
    #                                 nesterov=cfg.OPTIM.NESTEROV,
    #                              lr=1e-5,weight_decay=cfg.OPTIM.WD)
    # else:
    #     raise NotImplementedError

def setup_upl(model):
    model.train()
    num_dec = cfg.MODEL.NUM_DEC
    dec_list = []
    for i in range(1, num_dec+1):
        print(model)
        dec_i = deepcopy(model.dec1)
        setattr(dec_i, 'name', f'dec_{i}')  # 修改属性或添加其他标识符
        dec_list.append(dec_i)
    optimizer_params = []
    # for dec_i in dec_list:
    #     optimizer_params.extend(dec_i.parameters())
    optimizer_params.extend(model.enc.parameters())
    optimizer = setup_optimizer(optimizer_params)
    upl_model = upl.TTA(model.enc, dec_list, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           )
    return upl_model
# def setup_wjh01(model):
#     anchor_model = deepcopy(model)
#     # model = tent.configure_model(model)
#     # params, param_names = tent.collect_params(model)
#     # optimizer = setup_optimizer(params)
#     # tent_model = tent.Tent(model, optimizer,
#     #                        steps=cfg.OPTIM.STEPS,
#     #                        episodic=cfg.MODEL.EPISODIC)
#     # model.eval()
#     # model = wjh01.configure_model(model, eps=1e-5, momentum=0.1,
#     #              reset_stats=False, no_stats=False)
#     model.train()
#     anchor_model.eval()
#     optimizer = 0
#     cotta_model = wjh01.TTA(model,anchor_model,
#                            steps=cfg.OPTIM.STEPS,
#                            episodic=cfg.MODEL.EPISODIC, 
#                            mt_alpha=cfg.OPTIM.MT, 
#                            rst_m=cfg.OPTIM.RST, 
#                            )
#     return cotta_model
def setup_wjh01(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR, betas=(0.5, 0.999))
    
    # 每经过 cfg.OPTIM.STEP_SIZE 步，将学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # cotta_model = wjh01.TTA(model, anchor_model, optimizer,scheduler=scheduler,
    #                         steps=cfg.OPTIM.STEPS,
    #                         episodic=cfg.MODEL.EPISODIC, 
    #                         mt_alpha=cfg.OPTIM.MT, 
    #                         rst_m=cfg.OPTIM.RST, 
    #                           # 将 scheduler 传递给模型，或者在训练循环中调用
    #                         )
    # cotta_model = wjh01.TTA(model, anchor_model, optimizer,
    #                     steps=cfg.OPTIM.STEPS,
    #                     episodic=cfg.MODEL.EPISODIC, 
    #                     mt_alpha=cfg.OPTIM.MT, 
    #                     rst_m=cfg.OPTIM.RST, 
    #                         # 将 scheduler 传递给模型，或者在训练循环中调用
    #                     )
    cotta_model = wjh01.TTA(model, anchor_model
                    )
    return cotta_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError