# import os
# import nibabel as nib
# from batchgenerators.utilities.file_and_folder_operations import subfiles, join
#
# # 定义任务目录
# task_dir = "/home/hello/code_rl/CTPelvic1K-main/all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K"
# images_tr = join(task_dir, "imagesTr")
# labels_tr = join(task_dir, "labelsTr")
#
# # 获取图像文件列表
# image_files = subfiles(images_tr, suffix=".nii.gz", join=False)
#
# for img_file in image_files:
#     # 假设图像文件名格式为 case_id.nii.gz（不含 _0000）
#     case_id = img_file[:-7]  # 移除 ".nii.gz"
#     img_path = join(images_tr, img_file)
#     lbl_path = join(labels_tr, f"{case_id}.nii.gz")
#
#     # 检查分割掩码文件是否存在
#     if not os.path.exists(lbl_path):
#         print(f"Missing label for {case_id}")
#         continue
#
#     # 加载图像和分割掩码
#     img_data = nib.load(img_path).get_fdata()
#     lbl_data = nib.load(lbl_path).get_fdata()
#
#     # 比较形状
#     if img_data.shape != lbl_data.shape:
#         print(f"Shape mismatch for {case_id}: Image {img_data.shape}, Label {lbl_data.shape}")
#     else:
#         print(f"{case_id}: Shapes match ({img_data.shape})")

import os, SimpleITK as sitk

img_dir = "/mnt/newdisk/CTPelvic1k/all_data/nnUNet/nnUNet_raw_splitted/Task11_CTPelvic1K/imagesTr"
lab_dir = "/mnt/newdisk/CTPelvic1k/all_data/nnUNet/nnUNet_raw_splitted/Task11_CTPelvic1K/labelsTr"

bad = []
for f in sorted(os.listdir(img_dir)):
    if not f.endswith(".nii.gz"):
        continue
    # nnU-Net 约定：图像名 like <ID>_0000.nii.gz；标签名 <ID>.nii.gz
    ID = f[:-12]  # 去掉 "_0000.nii.gz"
    img_p = os.path.join(img_dir, f)
    lab_p = os.path.join(lab_dir, ID + ".nii.gz")
    if not os.path.exists(lab_p):
        print("[MISS LABEL]", ID, lab_p);
        bad.append((ID, "no_label"));
        continue
    img = sitk.ReadImage(img_p);
    lab = sitk.ReadImage(lab_p)
    if (img.GetSize() != lab.GetSize() or
            img.GetSpacing() != lab.GetSpacing() or
            img.GetOrigin() != lab.GetOrigin() or
            img.GetDirection() != lab.GetDirection()):
        print("\n[ MISMATCH ]", ID)
        print(" image size/spacing:", img.GetSize(), img.GetSpacing())
        print(" label size/spacing:", lab.GetSize(), lab.GetSpacing())
        bad.append((ID, "mismatch"))
print("\nTotal bad:", len(bad))
# import SimpleITK as sitk
# import os
#
# img_dir = "/mnt/newdisk/CTPelvic1k/all_data/nnUNet/nnUNet_raw_splitted/Task11_CTPelvic1K/imagesTr"
# lab_dir = "/mnt/newdisk/CTPelvic1k/all_data/nnUNet/nnUNet_raw_splitted/Task11_CTPelvic1K/labelsTr"
# out_dir = lab_dir   # 输出目录
# os.makedirs(out_dir, exist_ok=True)
#
# for f in sorted(os.listdir(img_dir)):
#     if not f.endswith("_0000.nii.gz"):
#         continue
#     ID = f[:-12]
#     img_path = os.path.join(img_dir, f)
#     lab_path = os.path.join(lab_dir, ID + ".nii.gz")
#     out_path = os.path.join(out_dir, ID + ".nii.gz")
#
#     if not os.path.exists(lab_path):
#         print(f"[MISSING] {ID}")
#         continue
#
#     img = sitk.ReadImage(img_path)
#     lab = sitk.ReadImage(lab_path)
#
#     # 重采样 label 到 image 的空间
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(img)  # 以 image 为参考
#     resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 标签用最近邻
#     resampler.SetOutputPixelType(sitk.sitkUInt8)
#
#     lab_resampled = resampler.Execute(lab)
#     sitk.WriteImage(lab_resampled, out_path)
#     print(f"Resampled {ID}")