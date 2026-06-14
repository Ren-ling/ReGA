

import torch.nn as nn
import torch.utils.data
from torchvision.transforms.functional import to_pil_image, rotate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from pymic.util.evaluation_seg import get_multi_class_evaluation_score
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from skimage import feature, morphology
import os
import torch
import torch.nn.functional as F


class CSCS(nn.Module):
    def __init__(self):
        super(CSCS, self).__init__()
        self.class_labels = [1, 2, 3, 4]

    def extract_edges_canny(self, volume, sigma=1.0):
        """
        使用Canny边缘检测提取3D体积的边缘

        参数:
        volume: numpy array, 形状 (D, H, W)
        sigma: Canny边缘检测的高斯平滑参数

        返回:
        edges: numpy array, 形状 (D, H, W), 边缘位置为1，非边缘为0
        """
        edges = np.zeros_like(volume, dtype=np.float32)

        for d in range(volume.shape[0]):
            slice_data = volume[d]
            if np.any(slice_data > 0):
                # 对每个类别分别提取边缘
                slice_edges = np.zeros_like(slice_data, dtype=np.float32)

                for label in self.class_labels:
                    label_mask = (slice_data == label).astype(np.uint8)
                    if np.any(label_mask):
                        label_edges = feature.canny(label_mask.astype(np.float64), sigma=sigma)
                        slice_edges[label_edges] = label  # 使用标签值标记边缘

                edges[d] = slice_edges

        return edges

    def calculate_boundary_hd95(self, pred_stack, spacing=[1.0, 1.0, 1.0], target_label=None):
        """
        计算dropout预测结果之间的边界Hausdorff距离

        参数:
        pred_stack: numpy array, 形状 (n_iter, D, H, W), dropout预测结果
        spacing: 体素间距
        target_label: 目标类别标签 (可选)

        返回:
        boundary_hd_score: float, 边界一致性得分 (0-1之间，越高表示边界越一致)
        """
        n_iter = pred_stack.shape[0]

        if n_iter < 2:
            return 0.0

        # 提取所有dropout预测的边缘
        edge_stack = []
        for i in range(n_iter):
            edges = self.extract_edges_canny(pred_stack[i], sigma=1.0)
            edge_stack.append(edges)

        # 计算所有边缘对之间的HD95距离
        hd_scores = []
        for i in range(n_iter):
            for j in range(i + 1, n_iter):
                if target_label is not None:
                    # 计算特定类别的边界HD95
                    edge_i = (edge_stack[i] == target_label).astype(np.uint8)
                    edge_j = (edge_stack[j] == target_label).astype(np.uint8)

                    if np.any(edge_i) and np.any(edge_j):
                        try:
                            hd95 = get_multi_class_evaluation_score(
                                s_volume=edge_i, g_volume=edge_j,
                                label_list=[1], fuse_label=True,
                                spacing=spacing, metric='hd95')[0]

                            if hd95 is not None and not np.isinf(hd95):
                                hd_score = np.exp(-float(hd95) / 25.0)
                                hd_scores.append(hd_score)
                        except:
                            continue
                else:
                    # 计算所有类别的平均边界HD95
                    class_hd_scores = []
                    for label in self.class_labels:
                        edge_i = (edge_stack[i] == label).astype(np.uint8)
                        edge_j = (edge_stack[j] == label).astype(np.uint8)

                        if np.any(edge_i) and np.any(edge_j):
                            try:
                                hd95 = get_multi_class_evaluation_score(
                                    s_volume=edge_i, g_volume=edge_j,
                                    label_list=[1], fuse_label=True,
                                    spacing=spacing, metric='hd95')[0]

                                if hd95 is not None and not np.isinf(hd95):
                                    hd_score = np.exp(-float(hd95) / 25.0)
                                    class_hd_scores.append(hd_score)
                            except:
                                continue

                    if class_hd_scores:
                        hd_scores.append(np.mean(class_hd_scores))

        # 如果没有有效的HD95值，返回0
        if not hd_scores:
            return 0.0

        return np.mean(hd_scores)

    def calculate_ent_region(self, label_list, pred_stack, ent_threshold):
        # pred_stack 形状: (n_iter, D, H, W)
        pred_sub = np.zeros_like(pred_stack, dtype=np.float32)
        for lab in label_list:
            pred_sub += (pred_stack == lab)

        soft_pred = np.mean(pred_sub, axis=0)  # (D, H, W)
        # 计算熵
        ent_pred = (-soft_pred * np.log(soft_pred + 1e-6))

        # 【关键修改】：直接返回布尔矩阵 (True/False)
        return ent_pred > ent_threshold

    def calculate_ent_weight(self, label_list, pred, alpha=1.0):
        pred_sub = np.zeros_like(pred)
        for lab in label_list:
            pred_sub = pred_sub + np.asarray(pred == lab, np.uint8)
        soft_pred = np.mean(pred_sub, axis=0)
        uni_pred = np.max(pred_sub, axis=0)
        ent_pred = -(soft_pred * np.log(soft_pred + 1e-6) + (1 - soft_pred) * np.log(1 - soft_pred + 1e-6))
        ent_pred = (ent_pred - ent_pred.min() + 1e-6) / (ent_pred.max() - ent_pred.min() + 2e-6)
        ent_weight = ((1 - ent_pred)[uni_pred == 1].sum() + 1e-6) / (uni_pred.sum() + 1e-6)
        return ent_weight ** alpha

    def compute_adaptive_hd_score(self, hd_val, sigma=25.0):
        if hd_val is None or np.isinf(hd_val) or np.isnan(hd_val):
            return 0.0
        # 使用高斯衰减，对边缘微小退化更敏感
        return np.exp(-(hd_val ** 2) / (2 * sigma ** 2))

    def calculate_boundary_consistency(self, pred_stack, curr_pred, n_iter, target_label=None):
        """
        计算边界一致性
        如果 target_label 为 None，返回所有类别的平均边界一致性
        如果 target_label 指定，返回特定类别的边界一致性

        参数:
        pred_stack: (n_iter, D, H, W) dropout预测堆栈
        curr_pred: (D, H, W) 当前预测
        n_iter: dropout迭代次数
        target_label: 目标类别标签 (可选)

        返回:
        如果 target_label=None: 所有类别的平均边界一致性
        如果 target_label指定: 特定类别的边界一致性
        """
        # 1. 提取当前预测的边界
        curr_edges = self.extract_edges_canny(curr_pred, sigma=1.0)

        # 2. 计算与每个dropout预测的边界HD95
        boundary_scores = []

        for i in range(n_iter):
            # 提取dropout预测的边界
            dropout_edges = self.extract_edges_canny(pred_stack[i], sigma=1.0)

            if target_label is not None:
                # 计算特定类别的边界一致性
                curr_edge_mask = (curr_edges == target_label).astype(np.uint8)
                dropout_edge_mask = (dropout_edges == target_label).astype(np.uint8)

                if np.any(curr_edge_mask) and np.any(dropout_edge_mask):
                    try:
                        hd95 = get_multi_class_evaluation_score(
                            s_volume=curr_edge_mask, g_volume=dropout_edge_mask,
                            label_list=[1], fuse_label=True,
                            spacing=[1.0, 1.0, 1.0], metric='hd95')[0]

                        if hd95 is not None and not np.isinf(hd95):
                            # 转换为一致性得分
                            hd_score = np.exp(-float(hd95) / 25.0)
                            boundary_scores.append(hd_score)
                    except:
                        continue
            else:
                # 计算所有类别的平均边界一致性（保持向后兼容）
                class_scores = []
                for label in self.class_labels:
                    # 提取当前类别的边界掩码
                    curr_edge_mask = (curr_edges == label).astype(np.uint8)
                    dropout_edge_mask = (dropout_edges == label).astype(np.uint8)

                    # 如果两个边界都有像素，计算HD95
                    if np.any(curr_edge_mask) and np.any(dropout_edge_mask):
                        try:
                            hd95 = get_multi_class_evaluation_score(
                                s_volume=curr_edge_mask, g_volume=dropout_edge_mask,
                                label_list=[1], fuse_label=True,
                                spacing=[1.0, 1.0, 1.0], metric='hd95')[0]

                            if hd95 is not None and not np.isinf(hd95):
                                # 转换为一致性得分
                                hd_score = np.exp(-float(hd95) / 25.0)
                                class_scores.append(hd_score)
                        except:
                            continue

                # 计算当前dropout预测的边界一致性得分
                if class_scores:
                    boundary_scores.append(np.mean(class_scores))

        # 计算平均边界一致性
        if boundary_scores:
            return np.mean(boundary_scores)#只有一个值时就是本身
        else:
            return 0.0

    # 在 CSCS.py 的 evaluate_dropout 方法中修改返回部分：

    def evaluate_dropout(self, input, curr_pred, net, n_iter=5, dropout=0.5, ent_threshold=0.3,
                         boundary_weight=0.1):
        """
        修改目标：同时获取dropout特征、预测和评分
        返回：
        - batch_est_scores: (B, n_iter, 4) 每个dropout的评分
        - dropout_features: (n_iter, B, C_feat, D, H, W) 每个dropout的特征
        - dropout_predictions: (n_iter, B, D, H, W) 每个dropout的硬标签预测
        """
        import time
        device = next(net.parameters()).device
        B, _, D, H, W = input.shape
        input = input.to(device).float()
        net = net.to(device)

        predictions = []  # 存储softmax预测
        features_list = []  # 存储每个dropout倒数第一个特征
        features_list2 = [] # 存储每个dropout倒数第二个特征
        hard_predictions = []  # 存储每个dropout的硬标签

        # 开启 dropout
        for module in net.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout3d):
                module.train()
        # ============ 【硬编码创建dropout保存目录】 ============
        # dropout标签保存路径（新增）
        label_save_dir = r"E:\code_xx\TEGDA-main\1TEGDA_xxx\model\CTPelvic1k\cascade_fullres\predictiontegda\label"
        dropout_save_dir = r"E:\code_xx\TEGDA-main\1TEGDA_xxx\model\CTPelvic1k\cascade_fullres\predictiontegda\dropout"
        os.makedirs(dropout_save_dir, exist_ok=True)
        print(f"Dropout输出将保存到: {dropout_save_dir}")

        with torch.no_grad():
            for iter_idx in range(n_iter):
                # 关键修改：获取特征和预测
                if hasattr(net, 'get_feature'):
                    print("！！！！！get_feature is exist ^v^")
                    # 获取特征
                    features = net.get_feature(input)

                    # 假设取最后一个特征层（与tegda一致）
                    feature_last = features[-1] if isinstance(features, list) else features
                    feature_last2 = features[-3]

                    features_list.append(feature_last.detach().clone())
                    features_list2.append(feature_last2.detach().clone())

                    # 获取输出
                    if hasattr(net, 'get_output'):
                        pred = net.get_output(features)
                    else:
                        pred = net.forward_no_adapt(input)
                else:
                    # 如果没有get_feature方法
                    print("！！！！！get_feature is not exist T^T")
                    pred = net.forward_no_adapt(input)
                    features_list.append(None)

                pred_softmax = F.softmax(pred, dim=1)
                predictions.append(pred_softmax)

                # 存储硬标签预测
                pred_hard = torch.argmax(pred, dim=1).cpu().numpy()
                hard_predictions.append(pred_hard)

                # ============ 【保存dropout预测和标签】 ============
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                for b in range(B):
                    # 生成文件名
                    if B > 1:
                        base_name = f"dropout_batch{b}_iter{iter_idx}_{timestamp}"
                    else:
                        base_name = f"dropout_iter{iter_idx}_{timestamp}"

                    # 1. 保存dropout预测（原始路径）
                    dropout_save_path = os.path.join(dropout_save_dir, f"{base_name}.nii.gz")
                    pred_slice = pred_hard[b] if B > 1 else pred_hard[0]

                    # 保存为NIfTI格式
                    pred_itk = sitk.GetImageFromArray(pred_slice.astype(np.float32))
                    sitk.WriteImage(pred_itk, dropout_save_path)

                    # # 2. 保存dropout标签（新增的label文件夹）
                    # label_save_path = os.path.join(label_save_dir, f"{base_name}.nii.gz")
                    # sitk.WriteImage(pred_itk, label_save_path)  # 保存相同的预测作为标签

                    # # 3. 可选：保存softmax概率
                    # prob_save_dir = os.path.join(dropout_save_dir, 'probabilities')
                    # os.makedirs(prob_save_dir, exist_ok=True)

                    # for cls_idx in range(pred_softmax.shape[1]):
                    #     prob_save_name = f"{base_name}_prob_class{cls_idx}.nii.gz"
                    #     prob_save_path = os.path.join(prob_save_dir, prob_save_name)
                    #
                    #     prob_slice = pred_softmax[b, cls_idx].cpu().numpy()
                    #     prob_itk = sitk.GetImageFromArray(prob_slice.astype(np.float32))
                    #     sitk.WriteImage(prob_itk, prob_save_path)

                print(f"Dropout迭代 {iter_idx} 预测已保存")


        # 堆叠结果
        predictions = torch.stack(predictions, dim=0)  # (n_iter, B, C, D, H, W)

        dropout_features = None
        dropout_features2= None
        if all(f is not None for f in features_list):
            dropout_features = torch.stack(features_list, dim=0)  # (n_iter, B, C_feat, D, H, W)
            dropout_features2 = torch.stack(features_list2, dim=0)

        dropout_hard_preds = np.stack(hard_predictions, axis=0) if hard_predictions else None  # (n_iter, B, D, H, W)

        # 计算平均概率和平均预测
        mean_prob_batch = torch.mean(predictions, dim=0)  # (B, C, D, H, W)
        avg_pred_batch = torch.argmax(mean_prob_batch, dim=1).cpu().numpy()  # (B, D, H, W)

        # 计算不一致性掩码
        mismatch_mask_batch = (curr_pred != avg_pred_batch).astype(np.float32)

        # 计算每个dropout的评分
        batch_est_scores = []  # (B, n_iter, 4)
        batch_uncertain_masks = []

        for b in range(B):
            sample_scores = []
            curr_sample_pred = curr_pred[b]

            # 计算每个dropout的评分
            for i in range(n_iter):
                dropout_pred = hard_predictions[i][b] if hard_predictions else None

                cls_scores = []
                for cls_id in self.class_labels:
                    w_cls = self.calculate_ent_weight_single_dropout(
                        label_list=[cls_id],
                        dropout_pred=dropout_pred,
                        curr_pred=curr_sample_pred)

                    dice_score = self.calculate_dice_score(dropout_pred, curr_sample_pred, cls_id)
                    hd_score = self.calculate_hd95_score(dropout_pred, curr_sample_pred, cls_id)
                    boundary_score = self.calculate_boundary_consistency_single(
                        dropout_pred, curr_sample_pred, cls_id)

                    consistency = (0.6 * dice_score +
                                   0 * hd_score +
                                   0.4 * boundary_score)
                    cls_score = w_cls * consistency
                    cls_scores.append(cls_score)

                sample_scores.append(cls_scores)

            batch_est_scores.append(sample_scores)

            # 计算不确定性区域
            sample_uncertain = np.zeros((D, H, W), dtype=bool)
            if hard_predictions:
                pred_hard_stack = np.stack([hard_predictions[i][b] for i in range(n_iter)], axis=0)
                for cls_id in self.class_labels:
                    reg = self.calculate_ent_region([cls_id], pred_hard_stack, ent_threshold)
                    sample_uncertain = np.logical_or(sample_uncertain, reg)

            batch_uncertain_masks.append(sample_uncertain.astype(np.float32))

        # 计算平均分数
        uncertain_mask_batch = np.stack(batch_uncertain_masks, axis=0)

        per_sample_avg_scores = []
        for sample_scores in batch_est_scores:
            avg_scores = np.mean(sample_scores, axis=0)  # (4,)
            per_sample_avg_scores.append(avg_scores)

        avg_scores_by_class = np.mean(per_sample_avg_scores, axis=0)  # (4,)

        # 计算边界一致性
        overall_boundary_consistency = 0.0
        boundary_scores_all = []
        for b in range(B):
            if hard_predictions:
                pred_hard_stack = np.stack([hard_predictions[i][b] for i in range(n_iter)], axis=0)
                curr_sample_pred = curr_pred[b]
                score = self.calculate_boundary_consistency(
                    pred_hard_stack, curr_sample_pred, n_iter, target_label=None)
                boundary_scores_all.append(score)

        if boundary_scores_all:
            overall_boundary_consistency = np.mean(boundary_scores_all)

        boundary_info = {
            'boundary_consistency': overall_boundary_consistency,
            'boundary_weight': boundary_weight,
            'per_class_boundary': np.mean(batch_est_scores, axis=0)
        }

        # 返回所有信息
        return (avg_scores_by_class[0], avg_scores_by_class[1], avg_scores_by_class[2], avg_scores_by_class[3],
                mismatch_mask_batch, uncertain_mask_batch, mean_prob_batch,
                batch_est_scores, dropout_features,dropout_features2, dropout_hard_preds, boundary_info)

    def CSCS(self, input, pred, model, boundary_weight=0.1):
        """
        返回包含dropout特征和评分的完整结果
        """
        results = self.evaluate_dropout(input, pred, model, dropout=0.5, boundary_weight=boundary_weight)

        # 解包所有结果
        (est_WT, est_TC, est_ET, est_EC, mismatch_mask, uncertain_mask,
         pred_mean, batch_est_scores, dropout_features,dropout_features2, dropout_hard_preds, boundary_info) = results

        # 计算平均分数
        per_sample_avg_scores = []
        for sample_scores in batch_est_scores:
            avg_score = np.mean(sample_scores)
            per_sample_avg_scores.append(avg_score)

        est_avg = np.mean(per_sample_avg_scores) if per_sample_avg_scores else 0.0

        # 计算每个类别的平均分数（用于显示）
        res_avg_classes = [round(float(x) * 100, 2) for x in [est_WT, est_TC, est_ET, est_EC]]
        batch_total_avg = round(sum(res_avg_classes) / 4.0, 2)

        # 计算熵图
        entropy_map = -torch.sum(pred_mean * torch.log(pred_mean + 1e-12), dim=1)
        entropy_map = entropy_map / np.log(pred_mean.shape[1])
        avg_entropy = entropy_map.mean().item()


        # 返回完整结果
        return (res_avg_classes[0], res_avg_classes[1], res_avg_classes[2], res_avg_classes[3],
                est_avg, mismatch_mask, avg_entropy, entropy_map, pred_mean,
                batch_est_scores, dropout_features,dropout_features2, dropout_hard_preds, boundary_info)

    # 在 CSCS 类中添加以下方法：

    def calculate_ent_weight_single_dropout(self, label_list, dropout_pred, curr_pred):
        """
        计算单次dropout预测的熵权重

        参数:
        label_list: 类别列表，如 [1, 2, 3, 4]
        dropout_pred: 单次dropout预测结果，形状 (D, H, W)
        curr_pred: 当前预测结果，形状 (D, H, W)

        返回:
        ent_weight: 熵权重
        """
        # 将dropout_pred和curr_pred转换为numpy数组
        if isinstance(dropout_pred, torch.Tensor):
            dropout_pred = dropout_pred.cpu().numpy()
        if isinstance(curr_pred, torch.Tensor):
            curr_pred = curr_pred.cpu().numpy()

        # 创建二值掩码
        dropout_mask = np.zeros_like(dropout_pred)
        curr_mask = np.zeros_like(curr_pred)

        for lab in label_list:
            dropout_mask = dropout_mask + np.asarray(dropout_pred == lab, np.uint8)
            curr_mask = curr_mask + np.asarray(curr_pred == lab, np.uint8)

        # 计算soft预测（这里用0/1表示）
        soft_pred = dropout_mask.astype(np.float32)  # 对于单次dropout，就是0或1

        # 计算联合掩码
        union_mask = np.maximum(dropout_mask, curr_mask)

        # 计算熵
        # 注意：当soft_pred为0或1时，log会有问题，需要加小量
        epsilon = 1e-6
        ent_pred = -(soft_pred * np.log(soft_pred + epsilon) +
                     (1 - soft_pred) * np.log(1 - soft_pred + epsilon))

        # 归一化熵
        if ent_pred.max() > ent_pred.min():
            ent_pred = (ent_pred - ent_pred.min() + epsilon) / (ent_pred.max() - ent_pred.min() + 2 * epsilon)

        # 计算权重
        if union_mask.sum() > 0:
            # 在一致区域(union_mask==1)中，低熵区域的比例
            low_entropy_ratio = ((1 - ent_pred)[union_mask == 1].sum() + epsilon) / (union_mask.sum() + epsilon)
        else:
            low_entropy_ratio = 1.0  # 如果没有前景区域，权重为1

        return low_entropy_ratio

    def calculate_dice_score(self, pred1, pred2, target_label):
        """
        计算两个预测之间特定类别的Dice分数

        参数:
        pred1: 预测1，形状 (D, H, W)
        pred2: 预测2，形状 (D, H, W)
        target_label: 目标类别标签

        返回:
        dice_score: Dice分数
        """
        mask1 = (pred1 == target_label).astype(np.float32)
        mask2 = (pred2 == target_label).astype(np.float32)

        if mask1.sum() == 0 and mask2.sum() == 0:
            return 1.0  # 如果两个都没有预测到该类，认为是完美一致

        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum()

        if union > 0:
            dice_score = 2.0 * intersection / union
        else:
            dice_score = 0.0

        return dice_score

    def calculate_hd95_score(self, pred1, pred2, target_label, spacing=[1.0, 1.0, 1.0]):
        """
        计算两个预测之间特定类别的HD95距离并转换为分数

        参数:
        pred1: 预测1，形状 (D, H, W)
        pred2: 预测2，形状 (D, H, W)
        target_label: 目标类别标签
        spacing: 体素间距

        返回:
        hd_score: 转换为0-1之间的分数
        """
        mask1 = (pred1 == target_label).astype(np.uint8)
        mask2 = (pred2 == target_label).astype(np.uint8)

        if np.any(mask1) and np.any(mask2):
            try:
                hd95 = get_multi_class_evaluation_score(
                    s_volume=mask1, g_volume=mask2,
                    label_list=[1], fuse_label=True,
                    spacing=spacing, metric='hd95')[0]

                if hd95 is not None and not np.isinf(hd95) and not np.isnan(hd95):
                    # 将HD95转换为一致性分数 (HD95越小，分数越高)
                    hd_score = np.exp(-float(hd95) / 25.0)
                    return hd_score
            except:
                pass

        # 如果无法计算HD95，检查是否都包含或不包含该类
        if (np.any(mask1) and not np.any(mask2)) or (not np.any(mask1) and np.any(mask2)):
            return 0.0  # 一个包含一个不包含，完全不一致
        else:
            return 1.0  # 都不包含或计算失败，认为一致

    def calculate_boundary_consistency_single(self, dropout_pred, curr_pred, target_label, spacing=[1.0, 1.0, 1.0]):
        """
        计算单次dropout预测与当前预测的边界一致性

        参数:
        dropout_pred: 单次dropout预测，形状 (D, H, W)
        curr_pred: 当前预测，形状 (D, H, W)
        target_label: 目标类别标签

        返回:
        boundary_score: 边界一致性分数
        """
        # 提取边界
        dropout_edges = self.extract_edges_canny_single(dropout_pred, target_label)
        curr_edges = self.extract_edges_canny_single(curr_pred, target_label)

        if np.any(dropout_edges) and np.any(curr_edges):
            try:
                hd95 = get_multi_class_evaluation_score(
                    s_volume=dropout_edges, g_volume=curr_edges,
                    label_list=[1], fuse_label=True,
                    spacing=spacing, metric='hd95')[0]

                if hd95 is not None and not np.isinf(hd95) and not np.isnan(hd95):
                    # 转换为一致性得分
                    boundary_score = np.exp(-float(hd95) / 25.0)
                    return boundary_score
            except:
                pass

        # 检查是否都有边界或都没有边界
        if (np.any(dropout_edges) and not np.any(curr_edges)) or (not np.any(dropout_edges) and np.any(curr_edges)):
            return 0.0  # 一个有边界一个没有，完全不一致
        else:
            return 1.0  # 都没有边界或计算失败，认为一致

    def extract_edges_canny_single(self, volume, target_label, sigma=1.0):
        """
        提取单个类别的Canny边缘

        参数:
        volume: 预测体积，形状 (D, H, W)
        target_label: 目标类别标签
        sigma: Canny边缘检测的高斯平滑参数

        返回:
        edges: 边缘掩码，形状 (D, H, W)
        """
        edges = np.zeros_like(volume, dtype=np.float32)

        for d in range(volume.shape[0]):
            slice_data = volume[d]
            label_mask = (slice_data == target_label).astype(np.uint8)

            if np.any(label_mask):
                label_edges = feature.canny(label_mask.astype(np.float64), sigma=sigma)
                edges[d][label_edges] = 1

        return edges

    def compute_curr_pred_entropy_map(self, pred_mean_prob):
        """
        计算 curr_pred 的熵图（基于平均预测概率）

        参数:
        pred_mean_prob: torch.Tensor, 形状 (B, C, D, H, W)
                        从 dropout 预测得到的平均概率分布

        返回:
        entropy_map: torch.Tensor, 形状 (B, D, H, W)
                    每个体素的熵值（已归一化到 0-1 之间）
        """
        # 计算信息熵: -sum(p * log(p))
        entropy_map = -torch.sum(pred_mean_prob * torch.log(pred_mean_prob + 1e-12), dim=1)

        # 归一化：除以 log(C) 使其在 0-1 之间
        entropy_map = entropy_map / np.log(pred_mean_prob.shape[1])

        return entropy_map

    def save_entropy_map_as_nii(self, entropy_map: object, save_path: object, spacing: object = (1.0, 1.0, 1.0), origin: object = (0, 0, 0)) -> None:
        """
        将熵图保存为 NIfTI 格式

        参数:
        entropy_map: numpy array 或 torch.Tensor, 形状 (D, H, W) 或 (B, D, H, W)
                     如果有多批次，只保存第一个
        save_path: str, 保存路径
        spacing: tuple, 体素间距 (x, y, z)
        origin: tuple, 原点坐标
        """
        # 转换为 numpy array
        if isinstance(entropy_map, torch.Tensor):
            entropy_map = entropy_map.detach().cpu().numpy()

        # 如果有多批次，取第一个
        if entropy_map.ndim == 4:
            entropy_map = entropy_map[0]

        # 确保是 3D 数据
        if entropy_map.ndim != 3:
            raise ValueError(f"熵图应该是3D数据，但得到形状: {entropy_map.shape}")

        # 创建 SimpleITK 图像
        entropy_sitk = sitk.GetImageFromArray(entropy_map.astype(np.float32))

        # 设置元数据
        entropy_sitk.SetSpacing(spacing)
        entropy_sitk.SetOrigin(origin)

        # 保存为 NIfTI
        sitk.WriteImage(entropy_sitk, save_path)
        print(f"熵图已保存到: {save_path}")
