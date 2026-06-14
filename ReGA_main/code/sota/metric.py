import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
from skimage.measure import label
import math
from skimage.measure import label
from scipy.spatial import Delaunay
import nibabel as nb
import os
import torch.nn as nn
from pymic.util.evaluation_seg import get_multi_class_evaluation_score, post_process_sdf, binary_dice
from skimage.metrics import hausdorff_distance

def cnh_loss_per_class(probs, num_classes=4):
    # probs: (1, num_classes, D, H, W)
    global loss_tu
    device = probs.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for class_id in range(1, num_classes):  # 跳过背景（class 0）
        # 每个类别的二值掩码（只考虑 argmax == class_id 的体素）
        pred_labels = torch.argmax(probs, dim=1)  # (1, D, H, W)
        binary_mask = (pred_labels == class_id).float()  # (1, D, H, W)

        # 连通域分析（numpy 辅助）
        mask_np = binary_mask[0].cpu().numpy().astype(np.int32)
        labeled_mask_np, num_regions = label(mask_np, connectivity=3, return_num=True)

        if num_regions == 0:
            continue  # 该类无前景，跳过

        all_regions = []
        region_sizes = []

        for region_id in range(1, num_regions + 1):
            coords = np.where(labeled_mask_np == region_id)
            coords_tensor = tuple(torch.tensor(c, device=device, dtype=torch.long) for c in coords)

            # 只取该类别的概率（probs[0, class_id, ...]）
            probs_in_region = probs[0, class_id, coords_tensor[0], coords_tensor[1], coords_tensor[2]]  # (N,)
            avg_prob = torch.mean(probs_in_region)  # 可导
            size = probs_in_region.numel()

            region_sizes.append(size)
            all_regions.append({'coords': coords_tensor, 'avg_prob': avg_prob, 'size': size})

        # 动态 alpha
        total_size = sum(region_sizes)
        largest_size = max(region_sizes)
        largest_ratio = largest_size / (total_size + 1e-6)
        alpha = 0.01 + 0.06 * largest_ratio

        # 可信度
        for r in all_regions:
            r['credibility'] = r['avg_prob'] * (r['size'] ** alpha)

        # 中心区域（最高可信度）
        center_region = max(all_regions, key=lambda x: x['credibility'])

        # 损失计算（只针对该类）
        total_size_tensor = torch.tensor(total_size, dtype=torch.float32, device=device)
        total_credibility_tensor = torch.tensor(0.0, device=device)
        for r in all_regions:
            if r is not center_region:
                total_credibility_tensor += r['credibility']

        center_size_tensor = torch.tensor(center_region['size'], dtype=torch.float32, device=device)
        loss_class = (1 - center_size_tensor / total_size_tensor) * total_credibility_tensor

        total_loss = total_loss + loss_class  # 累加所有类别的损失
        total_loss = total_loss / (num_classes - 1)
        loss_tu = total_loss / (total_size_tensor + 1e-6)

    return loss_tu  # 平均（或直接 sum，根据需求）


def intra_organ_homogeneity_loss(logits: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """
    鼓励每个器官内部概率局部一致性
    """
    assert logits.dim() == 5
    B, C, D, H, W = logits.shape
    pad = window_size // 2

    probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
    pred_label = torch.argmax(probs, dim=1)  # (B, D, H, W)

    loss = 0.0
    num_classes = 0

    # 正确的平均池化核：输入通道=1, 输出通道=1
    kernel = torch.ones((1, 1, window_size, window_size, window_size), device=logits.device)
    kernel = kernel / kernel.sum()  # 归一化

    for c in range(1, C):  # 跳过背景类 (c=0)
        mask_c = (pred_label == c).float().unsqueeze(1)  # (B, 1, D, H, W)
        if mask_c.sum() == 0:
            continue

        # 取出当前类的概率图：(B, 1, D, H, W)
        prob_c = probs[:, c:c+1]  # 输入通道数=1

        # 局部平均（平均池化）
        prob_c_local = F.conv3d(prob_c, kernel, padding=pad)  # groups 默认=1，正确！

        # 方差（绝对差）
        variance = torch.abs(prob_c - prob_c_local)

        # 只在该器官区域内计算损失
        loss_c = (variance * mask_c).sum() / (mask_c.sum() + 1e-6)
        loss += loss_c
        num_classes += 1

    if num_classes == 0:
        return torch.tensor(0.0, device=logits.device)

    return loss / num_classes

def ih_loss(
        logits: torch.Tensor,
        window_size: int = 3,
        reduction: str = "mean",
        eps: float = 1e-8,
) -> torch.Tensor:
    assert logits.dim() == 5, "logits must be (B, C=4, D, H, W)"
    B, C, D, H, W = logits.shape
    # assert C == 4, "C must be 4 with channel 0 as background"
    assert window_size % 2 == 1, "window_size must be an odd number"

    # 1) softmax 概率
    probs = F.softmax(logits, dim=1)
    bg = probs[:, 0:1]  # (B,1,D,H,W)
    fg = probs[:, 1:]  # (B,3,D,H,W)
    fg_max, _ = fg.max(dim=1, keepdim=True)  # (B,1,D,H,W)

    # 2) 只惩罚背景占优（ReLU 截断）
    diff_pos = F.relu(bg - fg_max)  # (B,1,D,H,W)

    # 3) 邻域内累积：用 3D 卷积（核=全1），等价于在局部窗口求和
    k = window_size
    pad = k // 2
    kernel = torch.ones((1, 1, k, k, k), device=logits.device, dtype=logits.dtype)
    # groups=1，stride=1，same padding
    neighborhood_sum = F.conv3d(diff_pos, kernel, bias=None, stride=1, padding=pad)
    # (B,1,D,H,W)

    # 4) 全图聚合为标量损失
    if reduction == "mean":
        loss = neighborhood_sum.mean()
    elif reduction == "sum":
        loss = neighborhood_sum.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    return loss / (window_size ** 3)


import numpy as np
from skimage.metrics import hausdorff_distance


def find_optimal_params(pred, ground_truth):
    """通过网格搜索找到最优参数"""
    best_params = None
    best_score = -np.inf

    # 参数网格
    sdf_ths = [2, 5, 10, 15, 20, 25, 30]
    region_ths = [20, 50, 100, 200, 500]
    main_region_ths = [1000, 3000, 5000, 10000]

    for sdf_th in sdf_ths:
        for region_th in region_ths:
            for main_region_th in main_region_ths:
                pred_post = post_process_sdf(
                    pred,
                    num_classes=4,
                    sdf_th=sdf_th,
                    region_th=region_th
                )

                # 计算分数：Dice提升 - 0.1 * HD95降低
                scores = []
                for c in [1, 2, 3, 4]:
                    dice_before = binary_dice((ground_truth == c), (pred == c))
                    dice_after = binary_dice((ground_truth == c), (pred_post == c))

                    # Hausdorff距离
                    if (pred == c).sum() > 0 and (pred_post == c).sum() > 0:
                        hd_before = hausdorff_distance(
                            np.argwhere(pred == c),
                            np.argwhere(ground_truth == c)
                        )
                        hd_after = hausdorff_distance(
                            np.argwhere(pred_post == c),
                            np.argwhere(ground_truth == c)
                        )
                    else:
                        hd_before = hd_after = 1000

                    score = (dice_after - dice_before) - 0.1 * (hd_after - hd_before)
                    scores.append(score)

                avg_score = np.mean(scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = (sdf_th, region_th, main_region_th)
                    print(f"New best: sdf_th={sdf_th}, region_th={region_th}, "
                          f"main_region_th={main_region_th}, score={avg_score:.4f}")

    return best_params, best_score


import torch
import torch.nn.functional as F


# def compute_region_contrastive_loss(feature_map, prediction_map, entropy_map, num_classes=4):
#     print("=== Debug types & shapes in contrastive loss ===")
#     print("feature_map :", type(feature_map), getattr(feature_map, 'shape', 'no shape'),
#           getattr(feature_map, 'device', 'no device'))
#     print("prediction_map:", type(prediction_map), getattr(prediction_map, 'shape', 'no shape'))
#     print("entropy_map  :", type(entropy_map), getattr(entropy_map, 'shape', 'no shape'))
#     print("=======================================")
#
#     # 自動轉換 numpy → tensor（保險用）
#     if isinstance(prediction_map, np.ndarray):
#         print("WARNING: prediction_map is numpy → converting to tensor")
#         prediction_map = torch.from_numpy(prediction_map).to(feature_map.device).long()
#
#     if isinstance(entropy_map, np.ndarray):
#         print("WARNING: entropy_map is numpy → converting to tensor")
#         entropy_map = torch.from_numpy(entropy_map).to(feature_map.device)    # 判斷是 2D 還是 3D
#     ndim = feature_map.ndim - 2  # 減掉 B 和 C，剩下空間維度數
#
#     if ndim == 2:  # 2D
#         target_spatial = feature_map.shape[2:]  # (H, W)
#     elif ndim == 3:  # 3D
#         target_spatial = feature_map.shape[2:]  # (D, H, W)
#     else:
#         raise ValueError(f"Unsupported feature_map dimension: {feature_map.shape}")
#
#         # 確保 prediction_map 有 batch 維度，並轉成 float 才能 interpolate
#     if prediction_map.ndim == 3:
#             prediction_map = prediction_map.unsqueeze(0)  # [B,D,H,W]
#     prediction_map = F.interpolate(
#             prediction_map.float().unsqueeze(1),  # → [B,1,D,H,W]
#             size=target_spatial,
#             mode='nearest'  # 類別標籤用 nearest
#     ).squeeze(1).long()  # 回 [B,D',H',W']
#
#     # entropy 用 area 或 bilinear，比較平滑
#     if entropy_map.ndim == 3:
#             entropy_map = entropy_map.unsqueeze(0)
#     entropy_map = F.interpolate(
#             entropy_map.unsqueeze(1),
#             size=target_spatial,
#             mode='area'  # 或 'bilinear'
#         ).squeeze(1)
#
#     # 現在形狀應該一致了
#     print("After downsample:")
#     print("  pred shape:", prediction_map.shape)
#     print("  entropy shape:", entropy_map.shape)
#     print("  feature shape:", feature_map.shape)
#
#
#     B, C = feature_map.shape[:2]
#
#     # 展平特徵、預測、熵
#     # 3D: (B, C, D*H*W)   或  2D: (B, C, H*W)
#     feature_map_flat = feature_map.view(B, C, -1)  # (B, C, N)
#     prediction_map_flat = prediction_map.view(B, -1)  # (B, N)
#     entropy_map_flat = entropy_map.view(B, -1)  # (B, N)
#
#     # 計算每個類別的區域質心
#     region_centroids = torch.zeros(B, num_classes, C, device=feature_map.device)
#
#     for k in range(num_classes):
#         mask = (prediction_map_flat == k).float()  # (B, N)
#         if mask.sum() == 0:
#             continue  # 這個類別在本 batch 沒出現，跳過
#
#         weighted_feature = feature_map_flat * mask.unsqueeze(1)  # (B, C, N)
#         weighted_feature *= (1 - entropy_map_flat).unsqueeze(1)  # 信心度加權
#
#         # 質心 = 加權平均
#         centroid = weighted_feature.sum(dim=-1) / (mask.sum(dim=-1, keepdim=True) + 1e-8)
#         region_centroids[:, k] = centroid
#
#     # ------------------ 對比損失計算 ------------------
#     # (B, num_classes, num_classes)
#     similarity_matrix = torch.matmul(
#         region_centroids,
#         region_centroids.transpose(1, 2)
#     )
#
#     # 正樣本：相同類別，負樣本：不同類別
#     pos_mask = torch.eye(num_classes, device=feature_map.device).unsqueeze(0).expand(B, -1, -1)
#     neg_mask = 1 - pos_mask
#
#     pos_sim = (similarity_matrix * pos_mask).sum(dim=-1)  # (B, num_classes)
#     neg_sim = (similarity_matrix * neg_mask).sum(dim=-1)
#
#     # 簡單的對比損失（可再改成 InfoNCE 等更穩定的形式）
#     loss = (neg_sim - pos_sim).mean()
#
#     return loss

# import torch
# import torch.nn.functional as F
# import numpy as np


def compute_region_contrastive_loss(feature_map, prediction_map, entropy_map, num_classes=4, temperature=0.07):
    print("=== Debug types & shapes in contrastive loss ===")
    print("feature_map :", type(feature_map), getattr(feature_map, 'shape', 'no shape'),
          getattr(feature_map, 'device', 'no device'))
    print("prediction_map:", type(prediction_map), getattr(prediction_map, 'shape', 'no shape'))
    print("entropy_map :", type(entropy_map), getattr(entropy_map, 'shape', 'no shape'))
    print("=======================================")

    # 自動轉換 numpy → tensor（保險用）
    if isinstance(prediction_map, np.ndarray):
        print("WARNING: prediction_map is numpy → converting to tensor")
        prediction_map = torch.from_numpy(prediction_map).to(feature_map.device).long()
    if isinstance(entropy_map, np.ndarray):
        print("WARNING: entropy_map is numpy → converting to tensor")
        entropy_map = torch.from_numpy(entropy_map).to(feature_map.device)

    # 判斷是 2D 還是 3D
    ndim = feature_map.ndim - 2  # 減掉 B 和 C，剩下空間維度數
    if ndim == 2:  # 2D
        target_spatial = feature_map.shape[2:]  # (H, W)
    elif ndim == 3:  # 3D
        target_spatial = feature_map.shape[2:]  # (D, H, W)
    else:
        raise ValueError(f"Unsupported feature_map dimension: {feature_map.shape}")

    # 確保 prediction_map 有 batch 維度，並轉成 float 才能 interpolate
    if prediction_map.ndim == 3:
        prediction_map = prediction_map.unsqueeze(0)  # [B,D,H,W]
    prediction_map = F.interpolate(
        prediction_map.float().unsqueeze(1),  # → [B,1,D,H,W]
        size=target_spatial,
        mode='nearest'  # 類別標籤用 nearest
    ).squeeze(1).long()  # 回 [B,D',H',W']

    # entropy 用 area 或 bilinear，比較平滑
    if entropy_map.ndim == 3:
        entropy_map = entropy_map.unsqueeze(0)
    entropy_map = F.interpolate(
        entropy_map.unsqueeze(1),
        size=target_spatial,
        mode='area'  # 或 'bilinear'
    ).squeeze(1)

    # 現在形狀應該一致了
    print("After downsample:")
    print(" pred shape:", prediction_map.shape)
    print(" entropy shape:", entropy_map.shape)
    print(" feature shape:", feature_map.shape)

    B, C = feature_map.shape[:2]
    # 展平特徵、預測、熵
    # 3D: (B, C, D*H*W) 或 2D: (B, C, H*W)
    feature_map_flat = feature_map.view(B, C, -1)  # (B, C, N)
    prediction_map_flat = prediction_map.view(B, -1)  # (B, N)
    entropy_map_flat = entropy_map.view(B, -1)  # (B, N)

    # 計算每個類別的區域質心
    region_centroids = torch.zeros(B, num_classes, C, device=feature_map.device)
    valid_classes = []  # 追蹤哪些類別有有效質心
    for k in range(num_classes):
        mask = (prediction_map_flat == k).float()  # (B, N)
        mask_sum = mask.sum(dim=-1)  # (B,)
        if (mask_sum > 0).any():  # 至少有一個 batch 有這個類別
            weighted_feature = feature_map_flat * mask.unsqueeze(1)  # (B, C, N)
            weighted_feature *= (1 - entropy_map_flat).unsqueeze(1)  # 信心度加權
            # 質心 = 加權平均
            centroid = weighted_feature.sum(dim=-1) / (mask_sum.unsqueeze(1) + 1e-8)
            region_centroids[:, k] = centroid
            valid_classes.append(k)

    if len(valid_classes) < 2:
        # 如果少於 2 個類別，無法計算對比損失，返回 0
        return torch.tensor(0.0, device=feature_map.device)

    # 只使用有效類別的質心
    region_centroids = region_centroids[:, valid_classes]  # (B, num_valid, C)
    num_valid = len(valid_classes)

    # 計算相似度矩陣 (B, num_valid, num_valid)
    # 先歸一化質心以計算 cosine similarity（InfoNCE 常用）
    region_centroids_norm = F.normalize(region_centroids, p=2, dim=-1)
    similarity_matrix = torch.bmm(
        region_centroids_norm,
        region_centroids_norm.transpose(1, 2)
    ) / temperature  # 加入溫度 tau

    # InfoNCE 損失
    loss = torch.tensor(0.0, device=feature_map.device)
    for i in range(num_valid):
        # 對每個 anchor i，正樣本是自己，負樣本是其他
        logits = similarity_matrix[:, i]  # (B, num_valid)
        labels = torch.full((B,), i, dtype=torch.long, device=feature_map.device)
        loss += F.cross_entropy(logits, labels, reduction='mean')

    loss /= num_valid  # 平均每個 anchor 的損失

    return loss




