import os
import nibabel as nib
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import subfiles, join


def resample_label_to_image(image_path, label_path, output_label_path):
    """
    重采样标签以匹配图像形状，使用最近邻插值。
    返回 True 表示重采样成功，False 表示形状已匹配或失败。
    """
    try:
        # 读取图像和标签
        img = sitk.ReadImage(image_path)
        lbl = sitk.ReadImage(label_path)

        img_size = img.GetSize()  # (x, y, z)
        lbl_size = lbl.GetSize()

        if img_size != lbl_size:
            print(f"Resampling {os.path.basename(label_path)}: Image size {img_size} -> Label size {lbl_size}")
            # 设置重采样器
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)  # 参考图像的元信息（大小、间距、方向）
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 最近邻插值，适合标签
            resampler.SetDefaultPixelValue(0)  # 填充值为背景 (0)
            resampled_lbl = resampler.Execute(lbl)

            # 保存重采样后的标签
            sitk.WriteImage(resampled_lbl, output_label_path)
            print(f"  -> Saved resampled label: {output_label_path}")

            # 验证重采样结果
            resampled_data = nib.load(output_label_path).get_fdata()
            img_data = nib.load(image_path).get_fdata()
            if resampled_data.shape != img_data.shape:
                print(f"  -> Error: Resampled label shape {resampled_data.shape} does not match image {img_data.shape}")
                return False
            return True
        else:
            print(f"  -> Shapes already match for {os.path.basename(label_path)}")
            return False
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return False


# 定义目录
task_dir = "/home/hello/code_rl/CTPelvic1K-main/all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K"
images_tr = join(task_dir, "imagesTr")
labels_tr = join(task_dir, "labelsTr")

# 检查目录权限
if not os.access(images_tr, os.R_OK) or not os.access(labels_tr, os.W_OK):
    print(f"Error: No read permission for {images_tr} or write permission for {labels_tr}")
    exit(1)

# 获取图像文件列表
image_files = subfiles(images_tr, suffix=".nii.gz", join=False)
fixed_count = 0
failed_cases = []

for img_file in image_files:
    case_id = img_file[:-7]  # 移除 ".nii.gz"，匹配文件名格式
    img_path = join(images_tr, img_file)
    lbl_path = join(labels_tr, f"{case_id}.nii.gz")

    if not os.path.exists(lbl_path):
        print(f"Missing label for {case_id}")
        failed_cases.append(case_id)
        continue

    # 检查形状
    try:
        img_data = nib.load(img_path).get_fdata()
        lbl_data = nib.load(lbl_path).get_fdata()
    except Exception as e:
        print(f"Error loading {case_id}: {e}")
        failed_cases.append(case_id)
        continue

    if img_data.shape != lbl_data.shape:
        print(f"Fixing shape mismatch for {case_id}: Image {img_data.shape}, Label {lbl_data.shape}")
        output_label_path = lbl_path.replace(".nii.gz", "_resampled.nii.gz")
        if resample_label_to_image(img_path, lbl_path, output_label_path):
            try:
                # 替换原始标签文件
                os.rename(output_label_path, lbl_path)
                fixed_count += 1
                print(f"  -> Fixed {case_id}")
            except Exception as e:
                print(f"  -> Error replacing {lbl_path}: {e}")
                failed_cases.append(case_id)
        else:
            print(f"  -> Failed to resample {case_id}")
            failed_cases.append(case_id)
    else:
        print(f"{case_id}: Already matching ({img_data.shape})")

print(f"\nTotal fixed cases: {fixed_count}")
if failed_cases:
    print(f"Failed cases ({len(failed_cases)}): {', '.join(failed_cases)}")