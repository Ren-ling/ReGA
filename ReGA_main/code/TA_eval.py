import argparse
import logging
import os
import random
import shutil
import sys
import time
import types
import numpy as np
import torch
import pickle
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import SimpleITK as sitk
import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders import utils
from dataloaders.brats2023 import BraTS2023
from dataloaders.ctpelvic_cascade_tta1 import CTPelvicCascadeTTADataset
from networks.net_factory_3d import net_factory_3d
from test_single_case import test_single_case
from utils import losses, metrics, ramps
from utils.calculate_metrics import evaluate_predictions
from val_3D import test_all_case
import csv
import math
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast  # 引入半精度加速
# 把 code/pymic 这个目录加到 sys.path，让里面那个真正的 pymic 包可以被 import 到
this_dir = os.path.dirname(__file__)
pymic_root = os.path.join(this_dir, "pymic")
if pymic_root not in sys.path:
    sys.path.append(pymic_root)
from pymic.util.evaluation_seg import get_multi_class_evaluation_score, post_process_sdf
from sota.metric import *
from sota._tta import *
from sota.load_full_volume import *
# from sota.adic import ADIC
import pandas as pd
import gc

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r"E:\code_xx\ReGA-main\ReGA-main\data\CTPelvic1k", help='Name of Experiment')
parser.add_argument('--source_domain', type=str,
                    default='dataset6', help='The source domain')
parser.add_argument('--source_checkpoint', type=str,
                    default=r"E:\code_xx\ReGA-main\ReGA-main\data\CTPelvic1k\source_check_point"
                            r"/cascade_fullres_CTpelvic_fold0"
                            r".pth", help='The source domain checkpoint')
parser.add_argument('--target_domain', type=str,
                    default='dataset3', help='The target domain')
parser.add_argument('--TTA_method', type=str,
                    default='ReGA', help='The TTA methods')
parser.add_argument('--num_class', type=int,
                    default='4', help='The number of class')
parser.add_argument('--exp', type=str,
                    default='CTPelvic1k', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='cascade_fullres', help='model_name')
parser.add_argument('--iterations', type=int,
                    default=2, help='maximum epoch number to test')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--patch_size', type=list, default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=9999,
                    help='labeled data')

# 在这里下面加：
parser.add_argument('--plans_path', type=str, default=r"E:\code_xx\ReGA-main\ReGA-main-dual\data\CTPelvic1k"
                                                      r"\source_check_point"
                                                      r"\plans.pkl",
                    help='path to nnUNet plans.pkl for cascade_fullres')
parser.add_argument('--img_dir', type=str, default=r"E:\code_xx\ReGA-main\ReGA-main-dual\data\CTPelvic1k\img3",
                    help='CTPelvic imagesTs folder (e.g. .../Task11_CTPelvic1K/imagesTs)')
parser.add_argument('--lowres_dir', type=str, default=r"E:\code_xx\ReGA-main\ReGA-main-dual\data\CTPelvic1k\lowres",
                    help='lowres prediction folder for cascade_fullres (3dlowres_pred)')
parser.add_argument('--device', type=str, default='cuda',
                    help='device')
parser.add_argument('--mode', type=str, default='train_test',
                    choices=['train_only', 'test_only', 'train_test'],
                    help='运行模式: train_only(仅TTA训练), test_only(仅测试), train_test(训练+测试)')
parser.add_argument('--adapted_checkpoint', type=str, default=r"E:\code_xx\ReGA-main\1ReGA_xx\model\CTPelvic1k\cascade_fullres\final_adapted_model.pth",
                    help='已适应的模型权重路径，test_only模式时需要')

args = parser.parse_args()


def setup_TTA_model(base_model, TTA_method):
    if TTA_method == "source_test":
        logging.info("test-time adaptation: NONE")
        model = setup_source(base_model, args.device)
    elif TTA_method == "tent":
        logging.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    elif TTA_method == "cotta":
        logging.info("test-time adaptation: CoTTA")
        device = args.device  # 从命令行获取设备信息
        model = setup_cotta(base_model, device=device)  # 传递设备信息
    elif TTA_method == "ReGA":
        logging.info("test-time adaptation: ReGA")
        device = args.device  # 从命令行获取设备信息
        # 在train函数中，创建TTA时
        model = setup_ReGA(base_model, device=device)
        print(f"传递给TTA的model类型: {type(model)}")
        print(f"model是否包含forward方法: {hasattr(model, 'forward')}")
        print(f"model是否包含__call__方法: {hasattr(model, '__call__')}")
    elif TTA_method == "vptta":
        logging.info("test-time adaptation: VPTTA")
        model = setup_vptta(base_model)
    else:
        raise "no specific method of {}".format(TTA_method)
    return model


def get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1

    # 生成高斯模糊
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # 归一化，让最高点为 1
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1

    # 转为 float32 并防止除零
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def load_full_volume(name, args, num_classes):
    """加载整图输入：CT + lowres one-hot（不 crop/pad）"""
    img_path = os.path.join(args.img_dir, name + "_0000.nii.gz")
    lowres_path = os.path.join(args.lowres_dir, name + ".nii.gz")

    # 读高分辨图
    img_itk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # [D, H, W]
    img = img[None]  # [1, D, H, W]

    # 读 lowres
    lowres_itk = sitk.ReadImage(lowres_path)
    lowres = sitk.GetArrayFromImage(lowres_itk).astype(np.int16)

    # one-hot
    lowres_oh = []
    for c in range(1, num_classes + 1):
        lowres_oh.append((lowres == c).astype(np.float32))
    lowres_oh = np.stack(lowres_oh, axis=0)  # [4, D, H, W]

    # 拼接
    full_inp = np.concatenate([img, lowres_oh], axis=0)  # [5, D, H, W]


    return full_inp

def swap_lr_channels(probs, right_idx, left_idx):
    """
    probs: torch.Tensor, shape (1, C, D, H, W)
    """
    probs = probs.clone()
    tmp = probs[:, right_idx].clone()
    probs[:, right_idx] = probs[:, left_idx]
    probs[:, left_idx] = tmp
    return probs

def load_full_gt(name, args):
    """加载整图 GT 和 spacing"""
    label_path = os.path.join(args.img_dir, name + "_mask_4label.nii.gz")

    if os.path.isfile(label_path):
        lab_itk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(lab_itk).astype(np.int16)
        spacing = lab_itk.GetSpacing()  # 获取 spacing 信息
    else:
        label = np.zeros((1, 1, 1))  # dummy
        spacing = (1.0, 1.0, 1.0)  # 使用默认的 spacing

    return label, spacing  # 返回标签和 spacing


import torch
import numpy as np
import math
from scipy.ndimage import gaussian_filter


def get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # 归一化：最高点为 1
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # 避免边缘数值过低导致除法不稳定
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


import torch
from torch.cuda.amp import autocast

def test_single_case_refined(net, image, patch_size, stride_step=0.33, num_classes=5, device=None, batch_size=1,
                             z_ceiling=None):
    import time
    import numpy as np
    import torch

    from scipy.ndimage import gaussian_filter
    start_time_total = time.perf_counter()  # 总计时开始
    net.eval()
    c, d, h, w = image.shape
    pd, ph, pw = patch_size

    # 1. Padding 逻辑
    tick = time.perf_counter()
    pad_d, pad_h, pad_w = max(pd - d, 0), max(ph - h, 0), max(pw - w, 0)
    pad_params = [(0, 0), (pad_d // 2, pad_d - pad_d // 2), (pad_h // 2, pad_h - pad_h // 2),
                  (pad_w // 2, pad_w - pad_w // 2)]
    image_padded = np.pad(image, pad_params, mode='constant', constant_values=0)
    _, d_pad, h_pad, w_pad = image_padded.shape

    # 2. 准备数据
    img_tensor = torch.from_numpy(image_padded).float().to(device)
    gaussian = torch.from_numpy(get_gaussian(patch_size)).to(device)
    score_map = torch.zeros((num_classes, d_pad, h_pad, w_pad), device=device)
    cnt_map = torch.zeros((d_pad, h_pad, w_pad), device=device)
    add_pad = True  # 修正：原代码中若有padding应设为True

    prep_time = time.perf_counter() - tick
    print(f"[Timer] Data Preparation: {prep_time:.3f}s")

    # 3. 计算步长与坐标
    stride_d, stride_h, stride_w = int(pd * stride_step), int(ph * stride_step), int(pw * stride_step)

    def get_steps(data_len, patch_len, stride):
        steps = list(range(0, data_len - patch_len + 1, stride))
        if steps[-1] != data_len - patch_len: steps.append(data_len - patch_len)
        return steps

    z_steps, y_steps, x_steps = get_steps(d_pad, pd, stride_d), get_steps(h_pad, ph, stride_h), get_steps(w_pad, pw,
                                                                                                          stride_w)

    all_coords = []
    for z in z_steps:
        for y in y_steps:
            for x in x_steps:
                all_coords.append((z, y, x))

    # 5. 核心：Batch 处理循环
    print(f"[Timer] Starting Inference for {len(all_coords)} patches...")
    inference_start_time = time.perf_counter()
    flip_dims = [None, (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    # # ===== 你需要按真实标签定义填写这个映射 =====
    # # 例子：如果 类别1=Right，类别2=Left，就写 (1,2)
    # # 如果 类别3=Right，类别4=Left，就写 (3,4)
    lr_swap_pairs = [(2, 3)]  # <<< 改成你数据的左右对应类别
    for i in range(0, len(all_coords), batch_size):
        coords_batch = all_coords[i: i + batch_size]
        patches = [img_tensor[:, z:z + pd, y:y + ph, x:x + pw] for z, y, x in coords_batch]
        batch_t = torch.stack(patches)

        b_actual = batch_t.shape[0]
        batch_probs = torch.zeros((b_actual, num_classes, pd, ph, pw), device=device)

        with torch.no_grad():
            with autocast():
                for axes in flip_dims:
                    inp = torch.flip(batch_t, dims=axes) if axes else batch_t
                    out = net(inp)
                    prob = torch.softmax(out, dim=1)

                    if axes:
                        # 1) 先把空间翻回来
                        prob = torch.flip(prob, dims=axes)

                        # 2) 如果包含 X 轴翻转（dim=4，对应 W 方向）
                        if 4 in axes:
                            for r, l in lr_swap_pairs:
                                prob = swap_lr_channels(prob, r, l)

                    batch_probs += prob
                    del out, prob, inp

        batch_probs /= len(flip_dims)

        # 6. 将 Batch 结果放回全图
        for idx, (z, y, x) in enumerate(coords_batch):
            score_map[:, z:z + pd, y:y + ph, x:x + pw] += batch_probs[idx] * gaussian
            cnt_map[z:z + pd, y:y + ph, x:x + pw] += gaussian

    inference_time = time.perf_counter() - inference_start_time
    print(
        f"[Timer] Inference & Aggregation: {inference_time:.3f}s (Avg: {inference_time / len(all_coords):.4f}s/patch)")

    # 7. 后处理与还原
    post_start_time = time.perf_counter()
    if z_ceiling is not None:
        score_map[1:, :z_ceiling, :, :] = 0
        score_map[0, :z_ceiling, :, :] = 1

    final_map = score_map / cnt_map.unsqueeze(0)
    # ============ 方案1：分块处理argmax以避免大内存分配 ============
    d_pad, h_pad, w_pad = final_map.shape[1], final_map.shape[2], final_map.shape[3]
    pred_label = np.zeros((d_pad, h_pad, w_pad), dtype=np.uint8)

    pred_label = torch.argmax(final_map, dim=0).cpu().numpy()


    if add_pad:
        zs, ys, xs = pad_params[1][0], pad_params[2][0], pad_params[3][0]
        pred_label = pred_label[zs:zs + d, ys:ys + h, xs:xs + w]

    post_time = time.perf_counter() - post_start_time
    total_time = time.perf_counter() - start_time_total

    print(f"[Timer] Post-processing: {post_time:.3f}s")
    print(f"[Timer] Total time for single case: {total_time:.3f}s")
    print("-" * 30)
    # ============ 【硬编码保存学生模型输出】 ============
    student_save_dir = r"E:\code_xx\ReGA-main\1ReGA_xx\model\CTPelvic1k\cascade_fullres\predictionReGA\student_test"
    os.makedirs(student_save_dir, exist_ok=True)

    # 生成保存文件名（根据输入形状或时间戳）
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(student_save_dir, f"student_pred_{timestamp}.nii.gz")

    pred_itk = sitk.GetImageFromArray(pred_label.astype(np.float32))
    sitk.WriteImage(pred_itk, save_path)
    print(f"学生模型输出已保存到: {save_path}")


    return pred_label

def online_evaluation(name, label, pred, spacing):
    """
    name:  case 名称
    label: GT 体素标注，shape=(D,H,W) 或 (1,D,H,W)（你前面是 label_batch[0][0] 喂进来的）
    pred:  模型预测标签，shape=(D,H,W)

    这里我们假设 CTPelvic1k 有 4 个前景类别，标签分别为 1,2,3,4：
      - WT_dice 对应 class 1
      - TC_dice 对应 class 2
      - ET_dice 对应 class 3
      - EC_dice 对应 class 4
    """
    WT_hd95 = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[1], fuse_label=True,
        spacing=spacing, metric='hd95'
    )[0]

    WT_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[1], fuse_label=True,
        spacing=spacing, metric='dice'
    )[0]

    TC_hd95 = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[2], fuse_label=True,
        spacing=spacing, metric='hd95'
    )[0]

    TC_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[2], fuse_label=True,
        spacing=spacing, metric='dice'
    )[0]

    ET_hd95 = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[3], fuse_label=True,
        spacing=spacing, metric='hd95'
    )[0]

    ET_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[3], fuse_label=True,
        spacing=spacing, metric='dice'
    )[0]

    EC_hd95 = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[4], fuse_label=True,
        spacing=spacing, metric='hd95'
    )[0]

    EC_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[4], fuse_label=True,
        spacing=spacing, metric='dice'
    )[0]
    # ============ 新增：ASD (Average Symmetric Surface Distance) 计算 ============
    WT_assd = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[1], fuse_label=True,
        spacing=spacing, metric='assd'
    )[0]

    TC_assd = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[2], fuse_label=True,
        spacing=spacing, metric='assd'
    )[0]

    ET_assd = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[3], fuse_label=True,
        spacing=spacing, metric='assd'
    )[0]

    EC_assd = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[4], fuse_label=True,
        spacing=spacing, metric='assd'
    )[0]

    Average_dice = (WT_dice + TC_dice + ET_dice + EC_dice) / 4.0
    Average_hd95 = (WT_hd95 + TC_hd95 + ET_hd95 + EC_hd95) / 4.0
    Average_assd = (WT_assd + TC_assd + ET_assd + EC_assd) / 4.0

    WT_dice = round(WT_dice * 100, 2)
    TC_dice = round(TC_dice * 100, 2)
    ET_dice = round(ET_dice * 100, 2)
    EC_dice = round(EC_dice * 100, 2)
    Average_dice = round(Average_dice * 100, 2)

    print("WT_hd95:", WT_hd95)
    print("TC_hd95:", TC_hd95)
    WT_hd95 = round(WT_hd95, 2)
    TC_hd95 = round(TC_hd95, 2)
    ET_hd95 = round(ET_hd95, 2)
    EC_hd95 = round(EC_hd95, 2)
    Average_hd95 = round(Average_hd95, 2)

    WT_assd = round(WT_assd, 2)
    TC_assd = round(TC_assd, 2)
    ET_assd = round(ET_assd, 2)
    EC_assd = round(EC_assd, 2)
    Average_assd = round(Average_assd, 2)

    # 打印所有指标
    print(
        f'Ground Truth Dice: '
        f'WT-{WT_dice}, TC-{TC_dice}, ET-{ET_dice}, EC-{EC_dice}, Avg-{Average_dice}\n'
        f'Ground Truth HD95: '
        f'WT-{WT_hd95}, TC-{TC_hd95}, ET-{ET_hd95}, EC-{EC_hd95}, Avg-{Average_hd95}\n'
        f'Ground Truth ASSD: '
        f'WT-{WT_assd}, TC-{TC_assd}, ET-{ET_assd}, EC-{EC_assd}, Avg-{Average_assd}'
    )

    case_result = {
        'name': name,
        'WT_dice': WT_dice,
        'TC_dice': TC_dice,
        'ET_dice': ET_dice,
        'EC_dice': EC_dice,
        'Avg_dice': Average_dice,
        'WT_hd95': WT_hd95,
        'TC_hd95': TC_hd95,
        'ET_hd95': ET_hd95,
        'EC_hd95': EC_hd95,
        'Avg_hd95': Average_hd95,
        'WT_assd': WT_assd,  # 新增ASSD字段
        'TC_assd': TC_assd,  # 新增ASSD字段
        'ET_assd': ET_assd,  # 新增ASSD字段
        'EC_assd': EC_assd,  # 新增ASSD字段
        'Avg_assd': Average_assd  # 新增ASSD字段
    }

    return case_result


def test_single_case1(net, image, num_classes, device=None):
    image = image.to(device).float()  # Move the image to the specified device
    with torch.no_grad():
        y1 = net(image)  # Perform inference on the specified device
        y = torch.argmax(y1, dim=1)
        label = y.cpu().numpy()[0]  # Move result back to CPU and convert to numpy

    return label


def train(args, snapshot_path):
    import os
    import SimpleITK as sitk  # ← 确保这里有
    print(f"命令行指定的patch_size: {args.patch_size}")

    with open(args.plans_path, 'rb') as f:
        plans = pickle.load(f)

    stage = list(plans['plans_per_stage'].keys())[-1]
    nnunet_patch = plans['plans_per_stage'][stage]['patch_size']

    print(f"nnUNet plans中的patch_size: {nnunet_patch}")

    train_data_path = args.root_path + '/' + args.target_domain
    batch_size = args.batch_size
    num_classes = args.num_class
    save_mode_path = "E:\code_xx\-main\1_xx\model\CTPelvic1k\cascade_fullres"  # 或者设置一个默认值
    final_save_path = "E:\code_xx\-main\1_xx\model\CTPelvic1k\cascade_fullres"

    with open(args.plans_path, 'rb') as f:
        plans = pickle.load(f)

    stage = list(plans['plans_per_stage'].keys())[-1]
    nnunet_patch = plans['plans_per_stage'][stage]['patch_size']

    args.patch_size = nnunet_patch
    print("Patch size automatically set to nnUNet:", args.patch_size)
    print(f"运行模式: {args.mode}")  # 新增

    # 1. 构建模型 & dataloader
    if args.model == "cascade_fullres":
        # ---- CTPelvic + cascade_fullres 路线 ----
        assert args.plans_path != "", "cascade_fullres 必须提供 --plans_path 指向 Task11 的 plans.pkl"
        assert args.img_dir != "" and args.lowres_dir != "", "cascade_fullres 必须提供 --img_dir 和 --lowres_dir"

        # 构建 nnUNet 的 cascade_fullres 网络（内部自己根据 plans.pkl 设置 in_channels 和 num_classes）
        model = net_factory_3d(
            net_type="cascade_fullres",
            in_chns=1,  # 对这个分支其实没用，会被忽略
            class_num=num_classes,  # 这里一般 = 背景 + 前景类数（比如 1+4=5）
            plans_path=args.plans_path
        )

        # 构建 CT Pelvic + lowres 的 Dataset
        db_train = CTPelvicCascadeTTADataset(
            img_dir=args.img_dir,
            lowres_dir=args.lowres_dir,
            split='all',
            num=args.labeled_num,
            num_classes=num_classes,
            patch_size=args.patch_size  # <<< 必须传入 !!!
        )


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=1, pin_memory=False)

    model.train()

    # 3. 根据模式加载权重
    if args.mode == 'test_only':
        # test_only模式：直接加载已适应的模型
        if not args.adapted_checkpoint:
            raise ValueError("test_only模式需要提供--adapted_checkpoint参数")

        # 1. 加载权重
        checkpoint = torch.load(args.adapted_checkpoint, map_location='cpu')

        # 2. 应用TTA包装（与训练时相同）
        model = setup_TTA_model(model, args.TTA_method)
        print(f"已应用TTA包装: {args.TTA_method}")

        # 3. 加载已适应权重
        model.load_state_dict(checkpoint, strict=False)
        print(f"已加载适应后的TTA模型权重: {args.adapted_checkpoint}")
        logging.info(f"已加载适应后的TTA模型权重: {args.adapted_checkpoint}")

        # 4. 设置为测试模式
        model.set_adapt(False) if hasattr(model, 'set_adapt') else None
        model.eval()
    else:
        # train_only 或 train_test 模式：加载源域权重，然后应用TTA包装
        checkpoint = torch.load(args.source_checkpoint, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items()
                           if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(
            f"Successfully loaded {len(pretrained_dict)} layers from source checkpoint, "
            f"ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")
        logging.info(
            f"Successfully loaded {len(pretrained_dict)} layers from source checkpoint, "
            f"ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")

        # 包一层 TTA（这里可以选 tegda）
        model = setup_TTA_model(model, args.TTA_method)

    # 教师学生模型的参数
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    save_output_dir = os.path.join(snapshot_path, 'prediction' + args.TTA_method)
    os.makedirs(save_output_dir, exist_ok=True)
    iter_num = 0
    results = []

    # 5. 循环目标域样本做 online TTA + eval
    for i_batch, sampled_batch in enumerate(trainloader):
        torch.cuda.empty_cache()
        # 如果是 cascade_fullres 分支，使用 GT 做评估
        if args.model == "cascade_fullres":
            name = sampled_batch['name'][0]
            print(f"sampled_batch['image'] 原始形状: {sampled_batch['image'].shape}")
            volume_batch = sampled_batch['image'].squeeze(0).to(args.device)  # 变成 [4, C, pd, ph, pw]
            print(f"volume_batch squeeze后形状: {volume_batch.shape}")
            label_batch = sampled_batch['label'].squeeze(0)  # 变成 [4, 1, pd, ph, pw]

            print(f"--- Processing Case: {name} ---")
            print("Input Batch Shape for Adaptation:", volume_batch.shape)

            # ============ TTA训练阶段 ============
            if args.mode in ['train_only', 'train_test']:
                print(">>> 进行TTA训练适应...")
                # 只有TEGDA需要调用set_adapt
                if args.TTA_method == 'tegda' and hasattr(model, 'set_adapt'):
                    model.set_adapt(True)

                # 所有TTA方法都需要设为训练模式
                model.train()

                # with torch.no_grad():  # 暂时保持与原来一致，但可能需要修改
                output = test_single_case1(model, volume_batch, num_classes=args.num_class, device=args.device)
                # 保存训练预测，使用原始文件名
                import os
                import SimpleITK as sitk

                train_save_dir = r"E:\code_xx\TEGDA-main\1TEGDA_xx\model\CTPelvic1k\cascade_fullres\predictiontegda\train"
                os.makedirs(train_save_dir, exist_ok=True)

                # 使用原始文件名
                save_path = os.path.join(train_save_dir, f"{name}_train.nii.gz")
                pred_itk = sitk.GetImageFromArray(output.astype(np.float32))
                sitk.WriteImage(pred_itk, save_path)
                print(f"训练模式预测已保存到: {save_path}")

                # ============ 结束保存 ============
                print(f"Case {name}: TTA训练完成")
                gc.collect()
                torch.cuda.empty_cache()

            # ============ 推理测试阶段 ============
            if args.mode in ['test_only', 'train_test']:
                print(">>> 进行推理测试...")
                if args.TTA_method == 'tegda' and hasattr(model, 'set_adapt'):
                    model.set_adapt(False)
                elif args.TTA_method == 'tent':
                    # 对于TENT，我们需要特殊处理
                    if hasattr(model, 'forward_no_adapt'):
                        # 保存原来的forward方法
                        original_forward = model.forward
                        # 临时替换为forward_no_adapt
                        model.forward = model.forward_no_adapt
                    else:
                        # 如果没有forward_no_adapt，设为eval并冻结
                        model.eval()
                        model.requires_grad_(False)

                model.eval()  # 设置为评估模式

                # 加载完整体积
                full_inp, properties = load_full_volume_cascade_final(name, args)

                # 清理缓存
                torch.cuda.empty_cache()

                with torch.no_grad():
                    pred_full = test_single_case_refined(
                        net=model,
                        image=full_inp,  # 整图 numpy (C, D, H, W)
                        patch_size=args.patch_size,
                        num_classes=5,
                        device=args.device
                    )  # pred_full: [D_full, H_full, W_full] numpy label map

                # ================= Phase 3: Online Evaluation & Save =================
                pred_full_post = post_process_sdf(
                    pred_full,
                    num_classes=4,  # CTPelvic1K 有 4 个前景类
                    sdf_th=35,  # 推荐从 30~40 开始尝试
                    region_th=1000  # 推荐 1000~3000
                )
                original_size = properties['original_size_of_raw_data']
                pred_final = resize_segmentation(pred_full_post, original_size, order=0)
                # 3. 加载地面真值 (Ground Truth) 用于指标计算
                full_gt, _ = load_full_gt(name, args)

                # 4. HD95 计算所需的最终间距 (Z, Y, X)
                eval_spacing = tuple(properties['original_spacing'])

                case_result = online_evaluation(name, full_gt, pred_final, eval_spacing)
                print(f"DEBUG - 评估结果: {case_result}")  # 添加这行

                results.append(case_result)
                print(f"DEBUG - results列表长度: {len(results)}")  # 添加这行

                # 保存整图预测
                test_save_path = os.path.join(save_output_dir, name + ".nii.gz")
                prd_itk = sitk.GetImageFromArray(pred_final.astype(np.float32))
                sitk.WriteImage(prd_itk, test_save_path)
                del full_inp, pred_full, pred_final, pred_full_post
                torch.cuda.empty_cache()

        else:
            name, volume_batch, label_batch = sampled_batch['name'][0], sampled_batch['image'], sampled_batch['label']
            # volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            output = test_single_case(model, volume_batch, num_classes=args.num_class)
            print("Adaptated Case:", name)
            # print(model.state_dict())
            case_result = online_evaluation(name, label_batch[0][0], output)
            results.append(case_result)
            ###################### save output ##################
            test_save_path = os.path.join(save_output_dir, sampled_batch['name'][0])
            prd_itk = sitk.GetImageFromArray(output.astype(np.float32))
            sitk.WriteImage(prd_itk, test_save_path)

        iter_num = iter_num + 1
        # 如果是train_only或train_test模式，定期保存中间检查点
        if args.mode in ['train_only', 'train_test'] and iter_num % 5 == 0:
            save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_adapted.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"中间模型已保存到: {save_mode_path}")

        # 训练完成后保存最终模型（仅训练模式）
        if args.mode in ['train_only', 'train_test']:
            final_save_path = os.path.join(snapshot_path, 'final_adapted_model.pth')
            torch.save(model.state_dict(), final_save_path)
            logging.info(f"最终适应模型已保存到: {final_save_path}")

        # 只在需要评估的模式下保存结果
        if args.mode in ['test_only', 'train_test']:
            if not results:
                logging.warning("没有评估结果可保存！")
                return "测试完成，但无结果"

            result_df = pd.DataFrame(results)

            # 计算Dice统计量
            WT_mean = round(result_df['WT_dice'].mean(), 2)
            WT_std = round(result_df['WT_dice'].std(), 2)
            TC_mean = round(result_df['TC_dice'].mean(), 2)
            TC_std = round(result_df['TC_dice'].std(), 2)
            ET_mean = round(result_df['ET_dice'].mean(), 2)
            ET_std = round(result_df['ET_dice'].std(), 2)
            EC_mean = round(result_df['EC_dice'].mean(), 2)
            EC_std = round(result_df['EC_dice'].std(), 2)
            Avg_mean = round(result_df['Avg_dice'].mean(), 2)
            Avg_std = round(result_df['Avg_dice'].std(), 2)

            # 计算HD95统计量
            WT_hd_mean = round(result_df['WT_hd95'].mean(), 2)
            WT_hd_std = round(result_df['WT_hd95'].std(), 2)
            TC_hd_mean = round(result_df['TC_hd95'].mean(), 2)
            TC_hd_std = round(result_df['TC_hd95'].std(), 2)
            ET_hd_mean = round(result_df['ET_hd95'].mean(), 2)
            ET_hd_std = round(result_df['ET_hd95'].std(), 2)
            EC_hd_mean = round(result_df['EC_hd95'].mean(), 2)
            EC_hd_std = round(result_df['EC_hd95'].std(), 2)
            Avg_hd_mean = round(result_df['Avg_hd95'].mean(), 2)
            Avg_hd_std = round(result_df['Avg_hd95'].std(), 2)

            # ===== 新增：ASSD统计量 =====
            WT_assd_mean = round(result_df['WT_assd'].mean(), 2)
            WT_assd_std = round(result_df['WT_assd'].std(), 2)
            TC_assd_mean = round(result_df['TC_assd'].mean(), 2)
            TC_assd_std = round(result_df['TC_assd'].std(), 2)
            ET_assd_mean = round(result_df['ET_assd'].mean(), 2)
            ET_assd_std = round(result_df['ET_assd'].std(), 2)
            EC_assd_mean = round(result_df['EC_assd'].mean(), 2)
            EC_assd_std = round(result_df['EC_assd'].std(), 2)
            Avg_assd_mean = round(result_df['Avg_assd'].mean(), 2)
            Avg_assd_std = round(result_df['Avg_assd'].std(), 2)

            # ===== 构造 mean 行，包含 Dice、HD95 和 ASSD 的 mean ± std =====
            mean_std_row = pd.DataFrame({
                'name': ['mean'],
                'WT_dice': [f'{WT_mean}±{WT_std}'],
                'TC_dice': [f'{TC_mean}±{TC_std}'],
                'ET_dice': [f'{ET_mean}±{ET_std}'],
                'EC_dice': [f'{EC_mean}±{EC_std}'],
                'Avg_dice': [f'{Avg_mean}±{Avg_std}'],

                'WT_hd95': [f'{WT_hd_mean}±{WT_hd_std}'],
                'TC_hd95': [f'{TC_hd_mean}±{TC_hd_std}'],
                'ET_hd95': [f'{ET_hd_mean}±{ET_hd_std}'],
                'EC_hd95': [f'{EC_hd_mean}±{EC_hd_std}'],
                'Avg_hd95': [f'{Avg_hd_mean}±{Avg_hd_std}'],

                # 新增 ASSD 列
                'WT_assd': [f'{WT_assd_mean}±{WT_assd_std}'],
                'TC_assd': [f'{TC_assd_mean}±{TC_assd_std}'],
                'ET_assd': [f'{ET_assd_mean}±{ET_assd_std}'],
                'EC_assd': [f'{EC_assd_mean}±{EC_assd_std}'],
                'Avg_assd': [f'{Avg_assd_mean}±{Avg_assd_std}'],
            })

            # 添加平均值和标准差到结果 DataFrame（放在最上面）
            result_df = pd.concat([mean_std_row, result_df], ignore_index=True)

            # 写 CSV
            output_dir = snapshot_path + "/final_result.csv"
            result_df.to_csv(output_dir, index=False)

    logging.info("save model to {}".format(save_mode_path))
    writer.close()
    # del volume_batch, output, pred_full, full_inp  # 明确删除大变量
    # torch.cuda.empty_cache()
    # gc.collect()
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 根据实验名称（args.exp）和模型名称（args.model）创建一个文件夹，用来存放这次训练的模型权重
    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    # 把当前目录下所有的代码文件复制到实验文件夹里的 /code 子目录中

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
