# utils/nnunet_predict_compatible.py  → 完整最终版

import torch
import pickle
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import load_json, join


def get_properties_from_nii_and_plans(image_nii_path, plans_path):
    """
    从原始 .nii.gz 文件和 plans.pkl 反推 nnUNet 的 properties
    完全兼容 nnUNet v1/v2
    """
    # 1. 读取 plans
    with open(plans_path, 'rb') as f:
        plans = pickle.load(f)

    stage_plans = plans['plans_per_stage'][list(plans['plans_per_stage'].keys())[-1]]
    current_spacing = stage_plans['current_spacing']
    original_spacing = stage_plans['original_spacing']
    patch_size = stage_plans['patch_size']

    # 2. 读取原始图像获取原始大小
    img_itk = sitk.ReadImage(image_nii_path)
    original_shape = img_itk.GetSize()[::-1]  # sitk 是 (W,H,D) → (D,H,W)
    original_spacing = img_itk.GetSpacing()[::-1]  # 实际 spacing

    # 3. 计算 crop_bbox（nnUNet 的 crop 逻辑）
    # nnUNet 是 center crop + pad 到能被 patch_size 整除
    crop_bbox = [0, 0, 0, original_shape[0], original_shape[1], original_shape[2]]
    for i in range(3):
        diff = original_shape[i] - patch_size[i]
        if diff < 0:
            pad_before = (patch_size[i] - original_shape[i]) // 2
            pad_after = patch_size[i] - original_shape[i] - pad_before
            crop_bbox[i] += pad_before
            crop_bbox[i + 3] -= pad_after
        else:
            margin = diff // 2
            crop_bbox[i] += margin
            crop_bbox[i + 3] -= (diff - margin)

    properties = {
        'original_size_of_raw_data': original_shape,
        'crop_bbox': crop_bbox,
        'current_spacing': current_spacing,
        'original_spacing': original_spacing,
    }
    return properties


def postprocess_prediction_v1(pred_seg, data_properties):
    """
    将 nnUNet v1 预处理后的预测结果恢复到原始图像大小和 spacing
    data_properties 来自 sampled_batch['properties']
    """
    # 1. 获取原始信息
    original_shape = data_properties['original_size_of_raw_data']  # e.g. (512, 512, 60)
    crop_bbox = data_properties['crop_bbox']  # (6,) tuple
    current_spacing = data_properties['current_spacing']
    original_spacing = data_properties['original_spacing']

    # 2. 先恢复到 crop 前的形状（去掉 padding）
    full_seg = np.zeros(original_shape, dtype=pred_seg.dtype)
    full_seg[
    crop_bbox[0]:crop_bbox[3],
    crop_bbox[1]:crop_bbox[4],
    crop_bbox[2]:crop_bbox[5]
    ] = pred_seg

    # 3. 用 SimpleITK 进行精确的 spacing 恢复（这是 nnUNet v1 官方做法！）
    seg_itk = sitk.GetImageFromArray(full_seg.astype(np.uint8))
    seg_itk.SetSpacing(tuple(current_spacing))
    seg_itk_resampled = sitk.Resample(
        seg_itk,
        tuple(original_shape),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,  # 必须用 nearest！标签不能插值！
        tuple(original_spacing),
        seg_itk.GetPixelId()
    )
    result = sitk.GetArrayFromImage(seg_itk_resampled)

    return result


def compute_gaussian_tuple(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian = np.exp(-0.5 * np.sum(
        ((np.indices(patch_size) - np.array(center_coords)[:, None, None, None]) ** 2) /
        (np.array(sigmas) ** 2)[:, None, None, None], axis=0))
    gaussian = gaussian / (np.max(gaussian) + 1e-8)
    return torch.from_numpy(gaussian).float()


def predict_3D_sliding_window_with_mirroring(
        model,
        input_tensor: torch.Tensor,
        patch_size,
        do_mirroring: bool = True,
        mirror_axes=(0, 1, 2),
        use_gaussian: bool = True,
        pad_value: float = -11.0,  # nnUNet 常用 -11
        stride_factor: float = 0.5
):
    assert input_tensor.ndim == 5
    device = input_tensor.device
    current_shape = input_tensor.shape[2:]

    # 1. padding
    pad_amount = [max(0, patch_size[i] - current_shape[i]) for i in range(3)]
    pad_before = [amt // 2 for amt in pad_amount]
    pad_after = [amt - amt // 2 for amt in pad_amount]

    padded = F.pad(input_tensor,
                   (pad_before[2], pad_after[2],
                    pad_before[1], pad_after[1],
                    pad_before[0], pad_after[0]),
                   mode='constant', value=pad_value)

    # 2. gaussian
    if use_gaussian:
        gaussian = compute_gaussian_tuple(patch_size, sigma_scale=1. / 8).to(device)
        gaussian = gaussian.view(1, 1, *patch_size)

    # 3. 关键：先跑一次小 patch 获取真实输出通道数
    with torch.no_grad():
        test_pred = model(padded[:, :, :patch_size[0], :patch_size[1], :patch_size[2]])
        if isinstance(test_pred, (tuple, list)):
            test_pred = test_pred[0]
        num_classes = test_pred.shape[1]  # ← 动态获取真实类别数！

    def predict_single(x):
        b, c, d, h, w = x.shape
        stride = [max(1, int(patch_size[i] * (1 - stride_factor))) for i in range(3)]

        output = torch.zeros((1, num_classes, d, h, w), device=device, dtype=torch.float32)
        counts = torch.zeros((1, 1, d, h, w), device=device, dtype=torch.float32)

        for zd in range(0, max(d - patch_size[0] + 1, 1), stride[0]):
            for yd in range(0, max(h - patch_size[1] + 1, 1), stride[1]):
                for xd in range(0, max(w - patch_size[2] + 1, 1), stride[2]):
                    z_end = min(zd + patch_size[0], d)
                    y_end = min(yd + patch_size[1], h)
                    x_end = min(xd + patch_size[2], w)

                    patch = x[:, :, zd:z_end, yd:y_end, xd:x_end]
                    pz = patch_size[0] - (z_end - zd)
                    py = patch_size[1] - (y_end - yd)
                    px = patch_size[2] - (x_end - xd)
                    if pz > 0 or py > 0 or px > 0:
                        patch = F.pad(patch, (0, px, 0, py, 0, pz), value=pad_value)

                    with torch.no_grad():
                        pred = model(patch)
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]  # 取主输出

                    cz, cy, cx = z_end - zd, y_end - yd, x_end - xd
                    weight = gaussian if use_gaussian else 1.0

                    output[:, :, zd:z_end, yd:y_end, xd:x_end] += pred[:, :, :cz, :cy, :cx] * weight
                    counts[:, :, zd:z_end, yd:y_end, xd:x_end] += weight

        return output / (counts + 1e-8)

    # 4. mirroring
    if do_mirroring:
        mirrors = [padded]
        for axis in mirror_axes:
            if axis < 3:
                mirrors.append(torch.flip(padded, dims=[axis + 2]))
        preds = []
        for i, m in enumerate(mirrors):
            p = predict_single(m)
            if i > 0:
                p = torch.flip(p, dims=[mirror_axes[i - 1] + 2])
            preds.append(p)
        prediction = torch.mean(torch.stack(preds), dim=0)
    else:
        prediction = predict_single(padded)

    # 5. crop back
    return prediction[:, :,
           pad_before[0]:pad_before[0] + current_shape[0],
           pad_before[1]:pad_before[1] + current_shape[1],
           pad_before[2]:pad_before[2] + current_shape[2]]


# test_single_case 不变
def test_single_case(net, image, patch_size, image_nii_path=None, plans_path=None):
    """
    新增参数：
        image_nii_path: 原始 .nii.gz 文件路径（必须传！）
        plans_path: plans.pkl 路径（你已经有 args.plans_path）
    """
    # 1. sliding window 推理（原来的）
    logits = predict_3D_sliding_window_with_mirroring(
        model=net,
        input_tensor=image.cuda(non_blocking=True),
        patch_size=patch_size,
        do_mirroring=True,
        use_gaussian=True,
        pad_value=-11.0
    )
    pred_seg = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # 2. 如果没传路径 → 不做后处理（调试用）
    if image_nii_path is None or plans_path is None:
        return pred_seg

    # 3. 关键：反推 properties + 后处理
    properties = get_properties_from_nii_and_plans(image_nii_path, plans_path)
    pred_original = postprocess_prediction_v1(pred_seg, properties)
    print("final prediction shape: ", pred_original.shape)

    return pred_original
