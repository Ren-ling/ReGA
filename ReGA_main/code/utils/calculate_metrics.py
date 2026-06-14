import os
import numpy as np
import csv
import SimpleITK as sitk
from medpy.metric.binary import dc, assd, hd95

# 计算指标函数
def calculate_metrics(pred, gt, num_classes):
    dice_scores = []
    assd_scores = []
    hd95_scores = []
    
    for class_idx in range(1, num_classes):
        pred_class = (pred == class_idx).astype(np.uint8)
        gt_class = (gt == class_idx).astype(np.uint8)

        if np.sum(gt_class) > 0:  # 如果GT中有该类，则计算指标
            dice = dc(pred_class, gt_class)
            assd_score = assd(pred_class, gt_class)
            hd95_score = hd95(pred_class, gt_class)
        else:
            dice, assd_score, hd95_score = 1.0, 0.0, 0.0  # 当GT没有该类时

        dice_scores.append(dice)
        assd_scores.append(assd_score)
        hd95_scores.append(hd95_score)

    return dice_scores, assd_scores, hd95_scores

# 主函数
def evaluate_predictions(pred_dir, gt_dir, pred_naming_feature, gt_naming_feature, num_classes=4, output_csv_name='evaluation_metrics.csv'):
    output_csv = os.path.join(pred_dir,output_csv_name) 
    score_all_data_dice = []
    score_all_data_assd = []
    score_all_data_hd95 = []
    name_score_list_dice = []
    name_score_list_assd = []
    name_score_list_hd95 = []

    pred_files = os.listdir(pred_dir)

    for pred_file in pred_files:
        # 获取预测文件和GT文件的路径
        gt_file = pred_file.replace(pred_naming_feature, gt_naming_feature)
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        if not os.path.exists(gt_path):
            print(f"GT file for {pred_file} not found, skipping...")
            continue

        # 读取预测和GT
        pred_image = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        gt_image = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

        # 计算三个指标
        dice, assd, hd95 = calculate_metrics(pred_image, gt_image, num_classes)

        score_vector_dice = dice + [np.mean(dice)]
        score_vector_assd = assd + [np.mean(assd)]
        score_vector_hd95 = hd95 + [np.mean(hd95)]

        # 保存每个样本的结果
        name_score_list_dice.append([pred_file] + score_vector_dice)
        name_score_list_assd.append([pred_file] + score_vector_assd)
        name_score_list_hd95.append([pred_file] + score_vector_hd95)

        score_all_data_dice.append(score_vector_dice)
        score_all_data_assd.append(score_vector_assd)
        score_all_data_hd95.append(score_vector_hd95)

    # 计算总体均值和标准差
    score_all_data_dice = np.asarray(score_all_data_dice)
    score_mean_dice = score_all_data_dice.mean(axis=0)
    score_std_dice = score_all_data_dice.std(axis=0)

    score_all_data_assd = np.asarray(score_all_data_assd)
    score_mean_assd = score_all_data_assd.mean(axis=0)
    score_std_assd = score_all_data_assd.std(axis=0)

    score_all_data_hd95 = np.asarray(score_all_data_hd95)
    score_mean_hd95 = score_all_data_hd95.mean(axis=0)
    score_std_hd95 = score_all_data_hd95.std(axis=0)

    # 将结果保存为CSV
    with open(output_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # 写入标题
        head = ['image'] + [f"class_{i}" for i in range(1, num_classes)] + ["average"]
        csv_writer.writerow(['Metric'] + head)

        # 写入各个样本的结果
        for item in name_score_list_dice:
            csv_writer.writerow(['Dice'] + item)
        for item in name_score_list_assd:
            csv_writer.writerow(['ASSD'] + item)
        for item in name_score_list_hd95:
            csv_writer.writerow(['Hd95'] + item)

        # 写入总体均值和标准差
        csv_writer.writerow(['Dice_mean'] + list(score_mean_dice))
        csv_writer.writerow(['Dice_std'] + list(score_std_dice))
        csv_writer.writerow(['ASSD_mean'] + list(score_mean_assd))
        csv_writer.writerow(['ASSD_std'] + list(score_std_assd))
        csv_writer.writerow(['Hd95_mean'] + list(score_mean_hd95))
        csv_writer.writerow(['Hd95_std'] + list(score_std_hd95))

    print(f"Metrics saved to {output_csv}")

# 示例调用
# evaluate_predictions('path_to_predictions', 'path_to_gt', 'pred_feature', 'gt_feature', num_classes=4, output_csv='metrics_results.csv')
