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
import copy
import pickle
from nnunet.preprocessing.preprocessing import GenericPreprocessor
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.preprocessing.cropping import ImageCropper
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2023 import BraTS2023
from dataloaders.dataet_for_da import *
from networks.net_factory_3d import net_factory_3d
from test_single_case import test_single_case
from utils import losses, metrics, ramps
from utils.calculate_metrics import evaluate_predictions
from val_3D import test_all_case
import csv
import math
import SimpleITK as sitk

# 把 code/pymic 这个目录加到 sys.path，让里面那个真正的 pymic 包可以被 import 到
this_dir = os.path.dirname(__file__)
pymic_root = os.path.join(this_dir, "pymic")
if pymic_root not in sys.path:
    sys.path.append(pymic_root)
from pymic.util.evaluation_seg import get_multi_class_evaluation_score

from sota._tta import *
# from sota.adic import ADIC
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2023', help='Name of Experiment')
parser.add_argument('--source_domain', type=str,
                    default='dataset6', help='The source domain')
parser.add_argument('--source_checkpoint', type=str,
                    default='', help='The source domain checkpoint')
parser.add_argument('--target_domain', type=str,
                    default='dataset3', help='The target domain')
parser.add_argument('--TTA_method', type=str,
                    default='source_test', help='The TTA methods')
parser.add_argument('--num_class', type=int,
                    default='5', help='The number of class')
parser.add_argument('--exp', type=str,
                    default='CTPelvic1k', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--iterations', type=int,
                    default=2, help='maximum epoch number to test')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--patch_size', type=list, default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=100,
                    help='labeled data')

# 在这里下面加：
parser.add_argument('--plans_path', type=str, default='',
                    help='path to nnUNet plans.pkl for cascade_fullres')
parser.add_argument('--img_dir', type=str, default='',
                    help='CTPelvic imagesTs folder (e.g. .../Task11_CTPelvic1K/imagesTs)')
parser.add_argument('--lowres_dir', type=str, default='',
                    help='lowres prediction folder for cascade_fullres (3dlowres_pred)')

args = parser.parse_args()


def nnunet_predict_full_volume(
        tta_model,  # ← TTA 封装，例如 tegda.TTA(...)
        base_model,  # ← 里面真正的 CascadeFullResUNet (Generic_UNet 子类)
        image_nii_path,
        lowres_nii_path,
        plans_path,
        do_mirroring=False,
        save_softmax=False,
        out_nifti_path=None,
):
    """
    用 nnUNet 的预处理 + predict_3D + 反 resample/反 crop，
    得到和 baseline 完全一样的 NIfTI 分割。

    和原版相比，多了一步：
      - 在 nnUNet 预处理空间，对 [CT + lowres one-hot] 做一次 TTA.forward()
        从而更新 base_model 的权重，然后再用 base_model.predict_3D 做正式推理。
    """

    # ----------------------------
    # 1) 读 plans 和 nnUNet 参数
    # ----------------------------
    with open(plans_path, 'rb') as f:
        plans = pickle.load(f)

    stage = list(plans['plans_per_stage'].keys())[-1]
    stage_plans = plans['plans_per_stage'][stage]
    target_spacing = stage_plans['current_spacing']
    transpose_forward = plans['transpose_forward']

    # ----------------------------
    # 2) nnUNet 风格预处理 (crop + resample + normalize)
    # ----------------------------
    # data: (C, D, H, W)  （这里只读 CT 一个模态）
    data, seg_dummy, properties = ImageCropper.crop_from_list_of_files([image_nii_path], None)

    # 按 transpose_forward 置换维度（nnUNet 的内部约定）
    data = data.transpose((0, *[i + 1 for i in transpose_forward]))

    preprocessor = GenericPreprocessor(
        plans['normalization_schemes'],
        plans['use_mask_for_norm'],
        transpose_forward,
        plans['dataset_properties']['intensityproperties']
    )
    data, seg_dummy, properties = preprocessor.resample_and_normalize(
        data, target_spacing, properties, seg_dummy, force_separate_z=None
    )
    # 此时 data 形状大概是 (1, D', H', W')

    # ----------------------------
    # 3) 读 lowres segmentation，resize + one-hot
    # ----------------------------
    lowres_itk = sitk.ReadImage(lowres_nii_path)
    lowres = sitk.GetArrayFromImage(lowres_itk).astype(np.int16)  # (D_orig, H_orig, W_orig)

    # 如果尺寸不一致，先 resize 到和 data 匹配
    if lowres.shape != data.shape[1:]:
        lowres = resize_segmentation(lowres, data.shape[1:], order=1)

    num_fg = plans['num_classes']  # 前景类别数，比如 4
    onehot = []
    for c in range(1, num_fg + 1):
        onehot.append((lowres == c).astype(np.float32))
    onehot = np.stack(onehot, 0)  # (num_fg, D', H', W')

    # cascade_fullres 的输入: 原始模态 + 上一级预测（不含背景）
    # data:   (1, D',H',W')
    # onehot: (num_fg, D',H',W')
    data_for_net = np.concatenate([data, onehot], axis=0)  # (1 + num_fg, D',H',W')

    # ----------------------------
    # 4) TTA 适应：在 nnUNet 预处理空间上，用真正的 cascade 输入做一次 forward_and_adapt
    # ----------------------------
    # 拿到 device（一般是 cuda:0）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    tta_model.to(device)

    if hasattr(tta_model, "reset"):
        tta_model.reset()

    # 必须手动 train 模式
    base_model.train()
    tta_model.train()

    # 从 plans 取参数
    patch_size = stage_plans['patch_size']  # 例如 (128, 128, 128)
    c, d, h, w = data_for_net.shape
    pd, ph, pw = patch_size
    mirror_axes = tuple(stage_plans.get('mirroring_axes', (0, 1)))
    step = 2  # patch_size // 2

    print("Start test-time adaptation (patch-wise, no monkey-patch)...")

    # # 直接使用 nnUNet 官方的 patch generator（最稳）
    # from nnunet.inference.predict import predict_3D_get_patches
    #
    # # 获取所有 patch 的坐标和权重（高斯权重）
    # patches, coordinates, gaussian_importance_map = predict_3D_get_patches(
    #     data_for_net,
    #     patch_size,
    #     step_size,
    #     use_gaussian=use_gaussian,
    #     pad_border_mode="edge",
    #     pad_kwargs={'constant_values': 0}
    # )

    adapt_steps = 3
    for step_idx in range(adapt_steps):
        print(f"  → Adaptation pass {step_idx + 1}/{adapt_steps}")

        # 手写一个极简 3D sliding window generator
        for z in range(0, d - pd + 1, pd // step):
            for y in range(0, h - ph + 1, ph // step):
                for x in range(0, w - pw + 1, pw // step):
                    # 裁出 patch
                    patch = data_for_net[:,
                            z:z + pd,
                            y:y + ph,
                            x:x + pw]

                    # 转 tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)  # (1, C, pd, ph, pw)

                    # 关键！直接调用 TEGDA 的核心函数，绝不走 forward！
                    tta_model.forward_and_adapt(patch_tensor, base_model, tta_model.optimizer)

                    # 可选：每 100 个 patch 打印一次进度
                    # if (z * h * w + y * w + x) % 100 == 0:
                    #     print(f"    processed patch at ({z},{y},{x})")

    print("TTA adaptation finished (hand-crafted, no monkey patch, no recursion).")

    # 适应完切回 eval
    base_model.eval()
    tta_model.eval()

    # 注意：predict_3D 期望 numpy 的 (C, D, H, W)，不是 tensor
    pred_seg, _, softmax_pred, _ = base_model.predict_3D(
        x=data_for_net,
        do_mirroring=do_mirroring,  # 这里可以开（TTA 常用）
        num_repeats=1,
        use_train_mode=False,
        batch_size=1,
        mirror_axes=mirror_axes if do_mirroring else tuple(),
        tiled=True,  # 必须开！
        tile_in_z=True,
        step=2,
        patch_size=stage_plans['patch_size'],
        regions_class_order=None,
        use_gaussian=True,
        pad_border_mode="edge",
        pad_kwargs={'constant_values': 0},
        all_in_gpu=False,
    )
    # ----------------------------
    # 6) 把 softmax/seg 从 resampled+cropped 空间，恢复回原图空间 → NIfTI
    # ----------------------------
    if out_nifti_path is not None:
        # softmax_pred: (C, D',H',W')  → 加个 batch 维度
        save_segmentation_nifti_from_softmax(
            softmax_pred[None],  # (1, C, D',H',W')
            out_nifti_path,
            properties,
            1,
            None,
            None,
            None,
            None,
            force_separate_z=None,
            interpolation_order=1
        )

    # pred_seg 是在 resampled+cropped 空间的标签图，一般这里只用来 debug
    return pred_seg


def setup_TTA_model(base_model, TTA_method):
    if TTA_method == "source_test":
        logging.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    elif TTA_method == "norm":
        logging.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    elif TTA_method == "tent":
        logging.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    elif TTA_method == "cotta":
        logging.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    elif TTA_method == "sar":
        logging.info("test-time adaptation: SAR")
        model = setup_sar(base_model)
    elif TTA_method == "meant":
        logging.info("test-time adaptation: meant")
        model = setup_meant(base_model)
    elif TTA_method == "tegda":
        logging.info("test-time adaptation: tegda")
        model = setup_ReGA(base_model)
    elif TTA_method == "sitta":
        logging.info("test-time adaptation: sitta")
        model = setup_sitta(base_model)
    elif TTA_method == "vptta":
        logging.info("test-time adaptation: VPTTA")
        model = setup_vptta(base_model)
    else:
        raise "no specific method of {}".format(TTA_method)
    return model


def online_evaluation(name, label, pred):
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

    # 确保是 numpy array / 去掉多余通道维度，这一步如果前面已经处理好了可以省略
    # label = np.asarray(label)
    print("pred shape:", pred.shape, "gt shape:", label.shape)

    WT_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[1], fuse_label=True,
        spacing=[1.0, 1.0, 1.0], metric='dice'
    )[0]

    TC_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[2], fuse_label=True,
        spacing=[1.0, 1.0, 1.0], metric='dice'
    )[0]

    ET_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[3], fuse_label=True,
        spacing=[1.0, 1.0, 1.0], metric='dice'
    )[0]

    EC_dice = get_multi_class_evaluation_score(
        s_volume=pred, g_volume=label,
        label_list=[4], fuse_label=True,
        spacing=[1.0, 1.0, 1.0], metric='dice'
    )[0]

    Average_dice = (WT_dice + TC_dice + ET_dice + EC_dice) / 4.0

    WT_dice = round(WT_dice * 100, 2)
    TC_dice = round(TC_dice * 100, 2)
    ET_dice = round(ET_dice * 100, 2)
    EC_dice = round(EC_dice * 100, 2)
    Average_dice = round(Average_dice * 100, 2)

    print(
        f'Ground Truth Dice: '
        f'WT-{WT_dice}, TC-{TC_dice}, ET-{ET_dice}, EC-{EC_dice}, Avg-{Average_dice}'
    )

    case_result = {
        'name': name,
        'WT_dice': WT_dice,
        'TC_dice': TC_dice,
        'ET_dice': ET_dice,
        'EC_dice': EC_dice,
        'Avg_dice': Average_dice
    }

    return case_result


def test_single_case1(net, image, num_classes):
    image = image.cuda().float()
    # with torch.no_grad():
    #     y1 = net(image)
    #     y = torch.argmax(y1, dim=1)
    y1 = net(image)
    print(y1)
    y = torch.argmax(y1, dim=1)
    label = y.cpu().numpy()[0]
    # print(label.shape)
    # score_map /= np.expand_dims(cnt, axis=0)  # 平均化
    # label_map = np.argmax(score_map, axis=0)  # 取最大值对应的类别

    return label


def train(args, snapshot_path):
    train_data_path = args.root_path + '/' + args.target_domain
    batch_size = args.batch_size
    num_classes = args.num_class

    with open(args.plans_path, 'rb') as f:
        plans = pickle.load(f)

    stage = list(plans['plans_per_stage'].keys())[-1]
    nnunet_patch = plans['plans_per_stage'][stage]['patch_size']

    args.patch_size = nnunet_patch
    print("Patch size automatically set to nnUNet:", args.patch_size)

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
        )
    else:
        # ---- 原来的 BraTS 路线 ----
        train_data_path = args.root_path + '/' + args.target_domain

        # 比如 unet_3D / vnet 等
        model = net_factory_3d(
            net_type=args.model,
            in_chns=4,  # BraTS2023 是 4 通道
            class_num=num_classes
        )

        db_train = BraTS2023(
            base_dir=train_data_path,
            split='all',
            num=args.labeled_num
        )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    # worker_init_fn=worker_init_fn)

    model.train()

    # 3. 加载源域权重
    checkpoint = torch.load(args.source_checkpoint, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(
        f"Successfully loaded {len(pretrained_dict)} layers from checkpoint, ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")
    logging.info(
        f"Successfully loaded {len(pretrained_dict)} layers from checkpoint, ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")

    # 4. 包一层 TTA（这里可以选 tegda）
    model = setup_TTA_model(model, args.TTA_method)  # （1-x)*model + x*model
    # 教师学生模型的参数
    # 拿到真正的 UNet（Generic_UNet / CascadeFullResUNet）
    if hasattr(model, 'model'):  # 比如 tegda / tent / cotta 等都有 self.model
        base_model = model.model
    else:
        base_model = model

    print("model type:", type(model))  # TTA
    print("base_model type:", type(base_model))  # CascadeFullResUNet

    print("has predict_3D on base_model?", hasattr(base_model, "predict_3D"))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    save_output_dir = os.path.join(snapshot_path, 'prediction' + args.TTA_method)
    os.makedirs(save_output_dir, exist_ok=True)
    iter_num = 0
    results = []

    # 5. 循环目标域样本做 online TTA + eval
    for i_batch, sampled_batch in enumerate(trainloader):
        # 如果是 cascade_fullres 分支，使用 GT 做评估
        if args.model == "cascade_fullres":
            name = sampled_batch['name'][0]
            img_path = sampled_batch['img_path'][0]
            lowres_path = sampled_batch['lowres_path'][0]
            label_path = sampled_batch['label_path'][0]
            # # model 是 TTA wrapper，里面会调用 forward_and_adapt，并 update base_model 参数
            # img_itk = sitk.ReadImage(img_path)
            # img_np = sitk.GetArrayFromImage(img_itk).astype(np.float32)[None][None]  # [1,1,D,H,W]
            # img_t = torch.from_numpy(img_np).cuda()
            # _ = model(img_t)  # 🔴 这里会触发 TTA.forward → forward_and_adapt，对 base_model 权重做一次更新

            pred_nifti_path = os.path.join(save_output_dir, name + ".nii.gz")
            pred_seg_resampled = nnunet_predict_full_volume(
                tta_model=model,  # ← TTA 封装
                base_model=base_model,  # ← 里面的 CascadeFullResUNet
                image_nii_path=img_path,
                lowres_nii_path=lowres_path,
                plans_path=args.plans_path,
                do_mirroring=False,
                save_softmax=False,
                out_nifti_path=pred_nifti_path,  # 真正保存整幅预测（原图空间）
            )

            print("Adaptated Case:", name)

            # 计算评估（注意把 label 转成 numpy，且去掉多余维度）
            gt_itk = sitk.ReadImage(label_path)
            gt_np = sitk.GetArrayFromImage(gt_itk).astype(np.int16)

            pred_itk = sitk.ReadImage(pred_nifti_path)
            pred_np = sitk.GetArrayFromImage(pred_itk).astype(np.int16)

            print("pred full shape:", pred_np.shape, "gt full shape:", gt_np.shape)
            case_result = online_evaluation(name, gt_np, pred_np)
            results.append(case_result)

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

    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(iter_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)

    result_df = pd.DataFrame(results)

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

    mean_std_row = pd.DataFrame({
        'name': ['mean'],
        'WT_dice': [f'{WT_mean}±{WT_std}'],
        'TC_dice': [f'{TC_mean}±{TC_std}'],
        'ET_dice': [f'{ET_mean}±{ET_std}'],
        'EC_dice': [f'{EC_mean}±{EC_std}'],
        'Avg_dice': [f'{Avg_mean}±{Avg_std}'],
    })

    # 添加平均值和标准差到结果 DataFrame
    result_df = pd.concat([mean_std_row, result_df], ignore_index=True)

    # 写 CSV
    output_dir = snapshot_path + "/final_result.csv"
    result_df.to_csv(output_dir, index=False)

    logging.info("save model to {}".format(save_mode_path))
    writer.close()
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

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
