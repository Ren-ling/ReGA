import os
import nibabel as nib
import numpy as np
from pathlib import Path

# 原始标签文件目录
raw_label_dir = "/home/hello/code_rl/CTPelvic1K-main/all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K/labelsTr"
# 预处理后的数据目录
processed_dir = "/home/hello/code_rl/CTPelvic1K-main/all_data/nnUNet/nnUNet_processed/Task11_CTPelvic1K/nnUNet_stage0"

# 预期类别（0-4，共5个）
expected_classes = {0, 1, 2, 3, 4}

def check_raw_labels():
    print("Checking raw label files...")
    invalid_files = []
    for label_file in os.listdir(raw_label_dir):
        if label_file.endswith('.nii.gz'):
            file_path = os.path.join(raw_label_dir, label_file)
            try:
                label = nib.load(file_path).get_fdata()
                unique_labels = set(np.unique(label).astype(int))
                print(f"{label_file}: Unique labels = {unique_labels}")
                if not unique_labels.issubset(expected_classes):
                    invalid_values = unique_labels - expected_classes
                    print(f"Warning: {label_file} contains invalid labels: {invalid_values}")
                    invalid_files.append((file_path, invalid_values))
                if label.size == 0 or np.all(label == 0):
                    print(f"Warning: {label_file} is empty or all zeros!")
            except Exception as e:
                print(f"Error loading {label_file}: {e}")
    if not invalid_files:
        print("All raw label files contain only expected classes (0-4).")
    else:
        print(f"Found {len(invalid_files)} files with invalid labels.")

def check_processed_npz():
    print("\nChecking processed .npz files...")
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} does not exist. Please run preprocessing first.")
        return
    for npz_file in os.listdir(processed_dir):
        if npz_file.endswith('.npz'):
            file_path = os.path.join(processed_dir, npz_file)
            try:
                data = np.load(file_path)
                print(f"{npz_file}: Available keys = {data.files}")
                if 'seg' in data.files:
                    seg = data['seg']
                    unique_labels = set(np.unique(seg).astype(int))
                    print(f"{npz_file}: Unique labels = {unique_labels}")
                    if not unique_labels.issubset(expected_classes):
                        invalid_values = unique_labels - expected_classes
                        print(f"Warning: {npz_file} contains invalid labels: {invalid_values}")
                else:
                    print(f"Warning: {npz_file} does not contain 'seg' key!")
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")

if __name__ == "__main__":
    check_raw_labels()
    check_processed_npz()