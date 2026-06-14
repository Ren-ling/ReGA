# hd_calculator.py
# 从你的评估脚本中抽离出的 HD 计算模块（使用 SimpleITK）

import numpy as np
import SimpleITK as sitk
from collections import OrderedDict


def compute_hd_binary_safe(pred_binary: np.ndarray, gt_binary: np.ndarray):
    """
    安全计算单个类别的 HD 和 AvgHD，处理空预测/GT情况
    """
    pred_binary = pred_binary.astype(np.float32)
    gt_binary = gt_binary.astype(np.float32)

    # 检查是否为空
    if pred_binary.sum() == 0 or gt_binary.sum() == 0:
        # 空类返回一个大惩罚值（你可以调整）
        return {"Hausdorff": 400.0, "avgHausdorff": 400.0}  # 400mm 作为惩罚，够大但不爆炸

    try:
        labelPred = sitk.GetImageFromArray(pred_binary, isVector=False)
        labelTrue = sitk.GetImageFromArray(gt_binary, isVector=False)

        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(labelTrue > 0.5, labelPred > 0.5)

        return {
            "Hausdorff": hausdorff_filter.GetHausdorffDistance(),
            "avgHausdorff": hausdorff_filter.GetAverageHausdorffDistance()
        }
    except Exception as e:
        print(f"HD computation exception: {e}")
        return {"Hausdorff": 400.0, "avgHausdorff": 400.0}


def compute_hd_multi_class(pred: np.ndarray, gt: np.ndarray, num_classes=4):
    """
    返回每个类和平均的 HD 值（float），方便 online_evaluation 使用
    """
    hd_values = []

    for i in range(1, num_classes + 1):
        class_pred = (pred == i)
        class_gt = (gt == i)
        result = compute_hd_binary_safe(class_pred, class_gt)
        hd_values.append(result["Hausdorff"])
        print(f"Class {i} HD: {result['Hausdorff']:.2f}")

    # 平均 HD
    avg_hd = np.mean(hd_values)
    hd_values.append(avg_hd)
    print(f"Average HD: {avg_hd:.2f}")

    return hd_values
