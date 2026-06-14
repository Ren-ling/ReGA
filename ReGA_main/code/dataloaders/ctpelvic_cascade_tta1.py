import os
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.transform import resize
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage.measurements import sum as sum_labels
import pickle
from multiprocessing import Pool
import random  # 添加这行

RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minyidx = int(np.min(mask_voxel_coords[1]))
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1
    minxidx = int(np.min(mask_voxel_coords[2]))
    maxxidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minyidx, maxyidx], [minxidx, maxxidx]]

class ImageCropper(object):
    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data = np.vstack([sitk.GetArrayFromImage(sitk.ReadImage(f))[None] for f in data_files])
        if seg_file is not None:
            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))[None]
        else:
            seg = None
        properties = {'original_size_of_raw_data': data[0].shape, 'size_after_cropping': data[0].shape, 'crop_bbox': None}
        # Optional: Add crop to non-zero for better processing, but skipped for simplicity
        return data, seg, properties

def resample_data_or_seg(data, new_shape, order, mode='edge', cval=0, clip=True, anti_aliasing=False):
    return resize(data, new_shape, order=order, mode=mode, cval=cval, clip=clip, anti_aliasing=anti_aliasing)

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=1, force_separate_z=None, order_z_data=0, order_z_seg=0):
    """
    Simplified version based on nnU-Net logic.
    """
    shape = data[0].shape
    new_shape = np.round(original_spacing / np.array(target_spacing) * np.array(shape)).astype(int)
    max_spacing = np.max(original_spacing)
    min_spacing = np.min(original_spacing)
    if force_separate_z is None:
        force_separate_z = (max_spacing / min_spacing > RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD)
    if force_separate_z:
        axis = np.argmax(original_spacing)
        if axis == 0:
            new_shape_2d = new_shape[1:]
            old_shape_2d = shape[1:]
        elif axis == 1:
            new_shape_2d = new_shape[0,2]
            old_shape_2d = shape[0,2]
        else:
            new_shape_2d = new_shape[0:2]
            old_shape_2d = shape[0:2]
        data_resampled = np.zeros([data.shape[0]] + list(new_shape), dtype=data.dtype)
        seg_resampled = np.zeros([seg.shape[0]] + list(new_shape), dtype=seg.dtype) if seg is not None else None
        for c in range(data.shape[0]):
            for slice_id in range(shape[axis]):
                if axis == 0:
                    data_2d = data[c, slice_id]
                elif axis == 1:
                    data_2d = data[c, :, slice_id]
                else:
                    data_2d = data[c, :, :, slice_id]
                resampled_2d = resample_data_or_seg(data_2d, new_shape_2d, order_z_data)
                if axis == 0:
                    data_resampled[c, slice_id] = resampled_2d
                elif axis == 1:
                    data_resampled[c, :, slice_id] = resampled_2d
                else:
                    data_resampled[c, :, :, slice_id] = resampled_2d
        if seg is not None:
            for slice_id in range(shape[axis]):
                if axis == 0:
                    seg_2d = seg[0, slice_id]
                elif axis == 1:
                    seg_2d = seg[0, :, slice_id]
                else:
                    seg_2d = seg[0, :, :, slice_id]
                resampled_2d = resample_data_or_seg(seg_2d, new_shape_2d, order_z_seg)
                if axis == 0:
                    seg_resampled[0, slice_id] = resampled_2d
                elif axis == 1:
                    seg_resampled[0, :, slice_id] = resampled_2d
                else:
                    seg_resampled[0, :, :, slice_id] = resampled_2d
        # now resample along the axis
        old_spacing_axis = original_spacing[axis]
        target_spacing_axis = target_spacing[axis]
        new_num_slices = np.round((old_spacing_axis / target_spacing_axis) * shape[axis]).astype(int)
        final_new_shape = list(new_shape)
        final_new_shape[axis] = new_num_slices
        for c in range(data.shape[0]):
            data_resampled[c] = resample_data_or_seg(data_resampled[c], final_new_shape, order_data)
        if seg is not None:
            seg_resampled[0] = resize_segmentation(seg_resampled[0], final_new_shape, order_seg)
    else:
        new_shape = np.round(original_spacing / np.array(target_spacing) * np.array(shape)).astype(int)
        data_resampled = np.zeros([data.shape[0]] + list(new_shape), dtype=data.dtype)
        for c in range(data.shape[0]):
            data_resampled[c] = resample_data_or_seg(data[c], new_shape, order_data, anti_aliasing=order_data > 0)
        if seg is not None:
            seg_resampled = resize_segmentation(seg[0], new_shape, order_seg)
            seg_resampled = seg_resampled[None]
        else:
            seg_resampled = None
    # properties["size_after_resampling"] = data_resampled[0].shape
    # properties["spacing_after_resampling"] = target_spacing
    return data_resampled, seg_resampled

class GenericPreprocessor(object):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list), intensityproperties=None):
        """
        :param normalization_scheme_per_modality: dict {0:'nonCT'}
        :param use_nonzero_mask: {0:False}
        :param intensityproperties:
        """
        self.transpose_forward = transpose_forward
        self.intensityproperties = intensityproperties
        self.normalization_scheme_per_modality = normalization_scheme_per_modality
        self.use_nonzero_mask = use_nonzero_mask
    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        all_data = np.load(os.path.join(cropped_output_dir, "%s.npz" % case_identifier))['data']
        data = all_data[:-1].astype(np.float32)
        seg = all_data[-1:]
        with open(os.path.join(cropped_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """
        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        data, seg = resample_patient(data, seg, original_spacing_transposed, target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")
        if seg is not None: # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0
        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask
        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"
        print("normalization...")
        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        print("normalization done")
        return data, seg, properties
    def preprocess_test_case(self, data_files, target_spacing, seg_file=None, force_separate_z=None):
        data, seg, properties = ImageCropper.crop_from_list_of_files(data_files, seg_file)
        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))
        data, seg, properties = self.resample_and_normalize(data, target_spacing, properties, seg,
                                                            force_separate_z=force_separate_z)
        return data.astype(np.float32), seg, properties
    def _run_star(self, args):
        target_spacing, case_identifier, output_folder_stage, cropped_output_dir, force_separate_z = args
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier)
        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))
        data, seg, properties = self.resample_and_normalize(data, target_spacing, properties, seg, force_separate_z)
        all_data = np.vstack((data, seg)).astype(np.float32)
        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier), data=all_data.astype(np.float32))
        with open(os.path.join(output_folder_stage, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=8, force_separate_z=None):
        """
        :param target_spacings: list of lists [[1.25, 1.25, 5]]
        :param input_folder_with_cropped_npz: dim: c, x, y, z | npz_file['data'] np.savez_compressed(fname.npz, data=arr)
        :param output_folder:
        :param num_threads:
        :param force_separate_z: None
        :return:
        """
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        maybe_mkdir_p(output_folder)
        num_stages = len(target_spacings)
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages
        assert len(num_threads) == num_stages
        for i in range(num_stages):
            all_args = []
            output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i)
            maybe_mkdir_p(output_folder_stage)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z
                all_args.append(args)
            p = Pool(num_threads[i])
            p.map(self._run_star, all_args)
            p.close()
            p.join()

# Helper for subfiles and get_case_identifier_from_npz
def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def get_case_identifier_from_npz(file):
    case_identifier = os.path.basename(file)[:-4]
    return case_identifier

def maybe_mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_nifti_for_debug(inp, label, sample_name, output_dir="./debug_viz", ):
    """
    将 inp[0]（原始图像）和 label 保存为 NIfTI 文件，供 3D Slicer 查看。
    Parameters:
        inp: np.ndarray, shape [C, D, H, W]
        label: np.ndarray, shape [D, H, W] (integer labels)
        output_dir: 保存路径
        sample_name: 样本名称（如 case_id）
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1. 保存原始图像（通道0）
    image_data = inp[0] # [D, H, W]
    img_itk = sitk.GetImageFromArray(image_data.astype(np.float32))
    sitk.WriteImage(img_itk, os.path.join(output_dir, f"{sample_name}_image_patch.nii.gz"))
    # 2. 保存标签（必须是整数类型）
    label_itk = sitk.GetImageFromArray(label.astype(np.uint8)) # 或 np.int16
    sitk.WriteImage(label_itk, os.path.join(output_dir, f"{sample_name}_label_patch.nii.gz"))
    print(f"✅ 已保存到 {output_dir}:")
    print(f" - {sample_name}_image.nii.gz")
    print(f" - {sample_name}_label.nii.gz")

def visualize_inp(inp, title="inp visualization"):
    C, D, H, W = inp.shape
    mid_slice = D // 2 # 取中间深度的 slice
    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
    if C == 1:
        axes = [axes]
    for c in range(C):
        img_slice = inp[c, mid_slice] # [H, W]
        axes[c].imshow(img_slice, cmap='gray')
        axes[c].set_title(f"Channel {c}")
        axes[c].axis('off')
        # 打印该通道是否全为0
        is_all_zero = np.allclose(img_slice, 0)
        print(f"Channel {c}: all zero? {is_all_zero} | min={img_slice.min():.3f}, max={img_slice.max():.3f}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
def resize_segmentation(segmentation, new_shape, order=1, cval=0):
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    reshaped = np.zeros(new_shape, dtype=tpe)
    for c in unique_labels:
        mask = segmentation == c
        resized = resize(mask.astype(np.float32), new_shape, order=order,
                         mode="edge", clip=True, anti_aliasing=False)
        reshaped[resized >= 0.5] = c
    return reshaped
# ===========================
# 关键：nnUNet patch 修正函数
# 要参照 from nnunet.training.data_augmentation.
# default_data_augmentation import get_default_augmentation
# 进行修改
# ===========================
def crop_pad_to_patch(data, patch_size):
    """
    data: numpy array [C, D, H, W] 或 [D, H, W]
    返回: 裁剪/填充后的相同 shape
    """
    if data.ndim == 3:
        data = data[None]
    C, D, H, W = data.shape
    pd, ph, pw = patch_size
    # ---------- pad ----------
    pad_d_before = max((pd - D) // 2, 0)
    pad_d_after = max(pd - D - pad_d_before, 0)
    pad_h_before = max((ph - H) // 2, 0)
    pad_h_after = max(ph - H - pad_h_before, 0)
    pad_w_before = max((pw - W) // 2, 0)
    pad_w_after = max(pw - W - pad_w_before, 0)
    data = np.pad(
        data,
        ((0, 0),
         (pad_d_before, pad_d_after),
         (pad_h_before, pad_h_after),
         (pad_w_before, pad_w_after)),
        mode="constant"
    )
    # ---------- crop center ----------
    D2, H2, W2 = data.shape[1:]
    sd = (D2 - pd) // 2
    sh = (H2 - ph) // 2
    sw = (W2 - pw) // 2
    data = data[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
    return data


import torch.nn.functional as F


def get_tta_patches_internal(patch_size, inp, label, num_patches=3):
    """
    使用torch的3D仿射变换进行旋转，保持形状不变
    """
    C, D, H, W = inp.shape
    pd, ph, pw = patch_size

    # 预先 Pad
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        inp = np.pad(inp, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
        label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
        D, H, W = inp.shape[1:]

    # 获取中心patch的坐标
    sd = (D - pd) // 2
    sh = (H - ph) // 2
    sw = (W - pw) // 2

    # 提取中心patch并转换为torch tensor
    p_inp_base = torch.from_numpy(inp[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]).float()
    p_lbl_base = torch.from_numpy(label[sd:sd + pd, sh:sh + ph, sw:sw + pw]).long()

    patches_inp = []
    patches_lbl = []

    # 3D仿射变换的网格
    grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, C, pd, ph, pw), align_corners=False)

    for i in range(num_patches):
        if i == 0:
            # 不旋转
            p_inp = p_inp_base.clone()
            p_lbl = p_lbl_base.clone()

        elif i == 1:
            # 绕Z轴旋转90度（XY平面）
            # 创建绕Z轴旋转90度的变换矩阵
            angle = torch.tensor([90.0]) * torch.pi / 180.0
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # 3D旋转矩阵 (绕Z轴)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0, 0],
                [sin_a, cos_a, 0, 0],
                [0, 0, 1, 0]
            ]).squeeze(1)

            grid_rotated = F.affine_grid(rotation_matrix.unsqueeze(0), (1, C, pd, ph, pw), align_corners=False)
            p_inp = F.grid_sample(p_inp_base.unsqueeze(0), grid_rotated, mode='bilinear', align_corners=False).squeeze(
                0)
            p_lbl = F.grid_sample(p_lbl_base.float().unsqueeze(0).unsqueeze(0),
                                  grid_rotated, mode='nearest', align_corners=False).squeeze(0).squeeze(0).long()

        elif i == 2:
            # 绕Y轴旋转90度（XZ平面）
            # 创建绕Y轴旋转90度的变换矩阵
            angle = torch.tensor([90.0]) * torch.pi / 180.0
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # 3D旋转矩阵 (绕Y轴)
            rotation_matrix = torch.tensor([
                [cos_a, 0, sin_a, 0],
                [0, 1, 0, 0],
                [-sin_a, 0, cos_a, 0]
            ]).squeeze(1)

            grid_rotated = F.affine_grid(rotation_matrix.unsqueeze(0), (1, C, pd, ph, pw), align_corners=False)
            p_inp = F.grid_sample(p_inp_base.unsqueeze(0), grid_rotated, mode='bilinear', align_corners=False).squeeze(
                0)
            p_lbl = F.grid_sample(p_lbl_base.float().unsqueeze(0).unsqueeze(0),
                                  grid_rotated, mode='nearest', align_corners=False).squeeze(0).squeeze(0).long()

        patches_inp.append(p_inp)
        patches_lbl.append(p_lbl.unsqueeze(0))

    inp_batch = torch.stack(patches_inp)
    lbl_batch = torch.stack(patches_lbl)

    return inp_batch.contiguous(), lbl_batch.contiguous()


# def get_tta_patches_internal(patch_size, inp, label, num_patches=3):
#     """
#     修复版：结合原版稳定性和CT值采样功能，确保patch尺寸严格一致
#     """
#     C, D, H, W = inp.shape
#     pd, ph, pw = patch_size
#
#     # ==================== 1. 确保输入数据有效性 ====================
#     # 检查NaN/Inf并清理
#     if np.isnan(inp).any() or np.isinf(inp).any():
#         print(f"警告：输入数据包含NaN/Inf，进行清理...")
#         inp = np.nan_to_num(inp, nan=-1000.0)
#
#     if np.isnan(label).any() or np.isinf(label).any():
#         print(f"警告：标签数据包含NaN/Inf，进行清理...")
#         label = np.nan_to_num(label, nan=0.0)
#
#     # ==================== 2. 前置padding（确保图像不小于patch） ====================
#     # 如果图像尺寸小于patch，先整体padding
#     if D < pd or H < ph or W < pw:
#         pad_d_before = max(0, (pd - D) // 2)
#         pad_d_after = max(0, pd - D - pad_d_before)
#         pad_h_before = max(0, (ph - H) // 2)
#         pad_h_after = max(0, ph - H - pad_h_before)
#         pad_w_before = max(0, (pw - W) // 2)
#         pad_w_after = max(0, pw - W - pad_w_before)
#
#         inp = np.pad(inp, ((0, 0),
#                            (pad_d_before, pad_d_after),
#                            (pad_h_before, pad_h_after),
#                            (pad_w_before, pad_w_after)),
#                      mode='constant', constant_values=-1000)
#         label = np.pad(label, ((pad_d_before, pad_d_after),
#                                (pad_h_before, pad_h_after),
#                                (pad_w_before, pad_w_after)),
#                        mode='constant', constant_values=0)
#         D, H, W = inp.shape[1:]
#         print(f"前置padding后尺寸: {D}x{H}x{W}")
#
#     # ==================== 3. CT值采样选择起始位置 ====================
#     ct_data = inp[0]  # CT通道
#
#     # 降采样加速计算
#     downsample_factor = 4
#     D_small = D // downsample_factor
#     H_small = H // downsample_factor
#     W_small = W // downsample_factor
#
#     # 确保降采样后至少有一个体素
#     if D_small < 1: D_small = 1
#     if H_small < 1: H_small = 1
#     if W_small < 1: W_small = 1
#
#     # 安全的降采样
#     ct_small = resize(ct_data, (D_small, H_small, W_small),
#                       order=1, mode='edge', anti_aliasing=False)
#
#     # 软组织检测（CT值范围）
#     soft_min, soft_max = -200, 500
#     soft_mask_small = (ct_small > soft_min) & (ct_small < soft_max)
#
#     soft_coords = np.where(soft_mask_small)
#
#     # 清理中间变量
#     del ct_small, soft_mask_small
#
#     # ==================== 4. 确定起始位置（确保不越界） ====================
#     if len(soft_coords[0]) < 10:
#         # 软组织点太少，使用中心位置
#         print(f"软组织点不足 ({len(soft_coords[0])})，使用中心位置")
#         z_start = max(0, min((D - pd) // 2, D - pd))
#         y_start = max(0, min((H - ph) // 2, H - ph))
#         x_start = max(0, min((W - pw) // 2, W - pw))
#     else:
#         # 从软组织点中采样
#         max_candidates = min(50, len(soft_coords[0]))
#         indices = np.random.choice(len(soft_coords[0]), max_candidates, replace=False)
#
#         # 映射回原始分辨率并限制在有效范围内
#         candidates_z = soft_coords[0][indices] * downsample_factor
#         candidates_y = soft_coords[1][indices] * downsample_factor
#         candidates_x = soft_coords[2][indices] * downsample_factor
#
#         # 限制在有效范围内 [0, D-pd-1] 等
#         candidates_z = np.clip(candidates_z, 0, max(0, D - pd - 1))
#         candidates_y = np.clip(candidates_y, 0, max(0, H - ph - 1))
#         candidates_x = np.clip(candidates_x, 0, max(0, W - pw - 1))
#
#         # 清理坐标数据
#         del soft_coords
#
#         # 选择到图像中心距离最近的点
#         center_z, center_y, center_x = D // 2, H // 2, W // 2
#         best_idx = 0
#         best_dist = float('inf')
#
#         for idx, (z, y, x) in enumerate(zip(candidates_z, candidates_y, candidates_x)):
#             # 计算patch中心
#             patch_center_z = z + pd // 2
#             patch_center_y = y + ph // 2
#             patch_center_x = x + pw // 2
#
#             # 距离（平方和，不开方节省计算）
#             dist = (patch_center_z - center_z) ** 2 + \
#                    (patch_center_y - center_y) ** 2 + \
#                    (patch_center_x - center_x) ** 2
#
#             if dist < best_dist:
#                 best_dist = dist
#                 best_idx = idx
#
#         z_start = int(candidates_z[best_idx])
#         y_start = int(candidates_y[best_idx])
#         x_start = int(candidates_x[best_idx])
#
#         # 最终范围检查
#         z_start = max(0, min(z_start, D - pd))
#         y_start = max(0, min(y_start, H - ph))
#         x_start = max(0, min(x_start, W - pw))
#
#     # ==================== 5. 提取patch（确保尺寸正确） ====================
#     # 验证起始位置有效性
#     assert 0 <= z_start <= D - pd, f"z_start {z_start} 超出范围 [0, {D - pd}]"
#     assert 0 <= y_start <= H - ph, f"y_start {y_start} 超出范围 [0, {H - ph}]"
#     assert 0 <= x_start <= W - pw, f"x_start {x_start} 超出范围 [0, {W - pw}]"
#
#     # 提取patch
#     base_patch_inp = inp[:, z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
#     base_patch_lbl = label[z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
#
#     # 验证尺寸
#     assert base_patch_inp.shape == (C, pd, ph, pw), \
#         f"Patch尺寸错误: {base_patch_inp.shape} != ({C}, {pd}, {ph}, {pw})"
#     assert base_patch_lbl.shape == (pd, ph, pw), \
#         f"标签尺寸错误: {base_patch_lbl.shape} != ({pd}, {ph}, {pw})"
#
#     # ==================== 6. 转换为tensor并数值裁剪 ====================
#     p_inp_base = torch.from_numpy(base_patch_inp).float()
#     p_lbl_base = torch.from_numpy(base_patch_lbl).long()
#
#     # CT值裁剪（防止异常值）
#     if C > 0:  # 如果有CT通道
#         p_inp_base = torch.clamp(p_inp_base, min=-1024, max=3071)
#
#     # ==================== 7. TTA变换生成3个patch ====================
#     patches_inp = []
#     patches_lbl = []
#
#     # Patch 0: 原始方向
#     patches_inp.append(p_inp_base.clone())
#     patches_lbl.append(p_lbl_base.clone().unsqueeze(0))
#
#     # 准备旋转角度（90度）
#     angle = torch.tensor([90.0]) * torch.pi / 180.0
#     cos_a = torch.cos(angle)
#     sin_a = torch.sin(angle)
#
#     # Patch 1: 绕Z轴旋转90度
#     try:
#         rotation_matrix_z = torch.tensor([
#             [cos_a, -sin_a, 0, 0],
#             [sin_a, cos_a, 0, 0],
#             [0, 0, 1, 0]
#         ]).squeeze(1)
#
#         grid_rotated_z = F.affine_grid(rotation_matrix_z.unsqueeze(0),
#                                        (1, C, pd, ph, pw),
#                                        align_corners=False)
#
#         p_inp_z = F.grid_sample(p_inp_base.unsqueeze(0), grid_rotated_z,
#                                 mode='bilinear', align_corners=False).squeeze(0)
#         p_inp_z = torch.clamp(p_inp_z, min=-1024, max=3071)  # 数值裁剪
#
#         p_lbl_z = F.grid_sample(p_lbl_base.float().unsqueeze(0).unsqueeze(0),
#                                 grid_rotated_z, mode='nearest', align_corners=False)
#         p_lbl_z = p_lbl_z.squeeze(0).squeeze(0).long()
#
#         patches_inp.append(p_inp_z)
#         patches_lbl.append(p_lbl_z.unsqueeze(0))
#     except Exception as e:
#         print(f"Z轴旋转失败，使用原始patch: {e}")
#         patches_inp.append(p_inp_base.clone())
#         patches_lbl.append(p_lbl_base.clone().unsqueeze(0))
#
#     # Patch 2: 绕Y轴旋转90度
#     try:
#         rotation_matrix_y = torch.tensor([
#             [cos_a, 0, sin_a, 0],
#             [0, 1, 0, 0],
#             [-sin_a, 0, cos_a, 0]
#         ]).squeeze(1)
#
#         grid_rotated_y = F.affine_grid(rotation_matrix_y.unsqueeze(0),
#                                        (1, C, pd, ph, pw),
#                                        align_corners=False)
#
#         p_inp_y = F.grid_sample(p_inp_base.unsqueeze(0), grid_rotated_y,
#                                 mode='bilinear', align_corners=False).squeeze(0)
#         p_inp_y = torch.clamp(p_inp_y, min=-1024, max=3071)  # 数值裁剪
#
#         p_lbl_y = F.grid_sample(p_lbl_base.float().unsqueeze(0).unsqueeze(0),
#                                 grid_rotated_y, mode='nearest', align_corners=False)
#         p_lbl_y = p_lbl_y.squeeze(0).squeeze(0).long()
#
#         patches_inp.append(p_inp_y)
#         patches_lbl.append(p_lbl_y.unsqueeze(0))
#     except Exception as e:
#         print(f"Y轴旋转失败，使用原始patch: {e}")
#         patches_inp.append(p_inp_base.clone())
#         patches_lbl.append(p_lbl_base.clone().unsqueeze(0))
#
#     # ==================== 8. 堆叠batch并最终检查 ====================
#     inp_batch = torch.stack(patches_inp)
#     lbl_batch = torch.stack(patches_lbl)
#
#     # 最终尺寸验证
#     assert inp_batch.shape == (num_patches, C, pd, ph, pw), \
#         f"输入batch尺寸错误: {inp_batch.shape} != ({num_patches}, {C}, {pd}, {ph}, {pw})"
#     assert lbl_batch.shape == (num_patches, 1, pd, ph, pw), \
#         f"标签batch尺寸错误: {lbl_batch.shape} != ({num_patches}, 1, {pd}, {ph}, {pw})"
#
#     # 检查NaN
#     if torch.isnan(inp_batch).any():
#         print(f"警告：最终batch包含NaN，进行清理...")
#         inp_batch = torch.nan_to_num(inp_batch, nan=-1000.0)
#
#     # 输出采样信息
#     print(f"✅ Patch采样完成:")
#     print(f"  位置: [{z_start}:{z_start + pd}, {y_start}:{y_start + ph}, {x_start}:{x_start + pw}]")
#     print(f"  CT值范围: {ct_data[z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw].min():.1f} to "
#           f"{ct_data[z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw].max():.1f} HU")
#     print(f"  输出尺寸: {inp_batch.shape}, 值范围: [{inp_batch.min():.1f}, {inp_batch.max():.1f}]")
#     print(f"  包含NaN: {torch.isnan(inp_batch).any().item()}")
#
#     return inp_batch.contiguous(), lbl_batch.contiguous()

class CTPelvicCascadeTTADataset(Dataset):
    def __init__(self,
                 img_dir,
                 lowres_dir,
                 split='all',
                 num=None,
                 num_classes=4,
                 patch_size=None,
                 target_spacing=(0.79998779, 0.854, 0.854),  # Transposed order (z, y, x)
                 intensityproperties=None):

        """
        patch_size: 由 train() 中 args.patch_size 提供，必须是 nnUNet 的 patch
        target_spacing: 目标 spacing，transposed order
        intensity properties: 强度统计，从训练集
        """
        if intensityproperties is None:
            intensityproperties = {
                0: {
                    'mean': 375.80426,
                    'sd': 282.0642,
                    'percentile_00_5': -51.0,
                    'percentile_99_5': 1298.0
                }
            }
        self.patch_label_save_dir = r"E:\code_rl\TEGDA-main\1TEGDA_wzm\model\CTPelvic1k\cascade_fullres\predictiontegda\PatchLabel"
        os.makedirs(self.patch_label_save_dir, exist_ok=True)
        self.img_dir = img_dir
        self.lowres_dir = lowres_dir
        self.num_classes = num_classes
        self.patch_size = patch_size # <<< 必须提供 !!!
        self.target_spacing = target_spacing
        self.intensityproperties = intensityproperties
        self.transpose_forward = [0, 1, 2]  # 标准 anisotropic CT
        self.preprocessor = GenericPreprocessor({0: 'CT'}, {0: False}, self.transpose_forward, self.intensityproperties)
        all_files = [f for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz")]
        self.case_ids = [f.replace("_0000.nii.gz", "") for f in all_files]
        if num is not None:
            self.case_ids = self.case_ids[:num]
        print(f"{split}, total {len(self.case_ids)} cases")
    def __len__(self):
        return len(self.case_ids)
        # 添加标签保存路径



    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        img_path = os.path.join(self.img_dir, cid + "_0000.nii.gz")
        lowres_path = os.path.join(self.lowres_dir, cid + ".nii.gz")
        label_path = os.path.join(self.img_dir, cid + "_mask_4label.nii.gz")
        # ------- 读高分辨图 -------
        img_itk = sitk.ReadImage(img_path)

        # 修正 1: 显式获取 spacing 并转为 numpy 数组
        # ITK 顺序是 (x, y, z)，转为 numpy 习惯的 (z, y, x)
        spacing = np.array(img_itk.GetSpacing())[[2, 1, 0]]

        properties = {
            'original_spacing': spacing,  # 现在是一个 numpy array [sz, sy, sx]
            'original_size': np.array(img_itk.GetSize())[[2, 1, 0]]
        }
        img = sitk.GetArrayFromImage(img_itk)[None].astype(np.float32)  # [1, z, y, x]
        # ------- 读 GT -------
        if os.path.isfile(label_path):
            lab_itk = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(lab_itk)[None].astype(np.int16)  # [1, z, y, x]
        else:
            label = None
        # ------- 转置 -------
        data = img.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = label.transpose((0, *[i + 1 for i in self.transpose_forward]))
        # ------- 重采样 + 归一化 -------
        data, seg, properties = self.preprocessor.resample_and_normalize(data, self.target_spacing, properties, seg=seg)
        # ------- 转回原 order for crop_pad [c, z, y, x] -------
        data = np.transpose(data, (0, 3, 2, 1))  # [1, z, y, x]
        if seg is not None:
            seg = np.transpose(seg, (0, 3, 2, 1))
            label = seg[0]  # [Z, Y, X]
        else:
            label = np.zeros(data.shape[1:], dtype=np.int16)
        # ------- 读 lowres -------
        lowres_itk = sitk.ReadImage(lowres_path)
        lowres = sitk.GetArrayFromImage(lowres_itk).astype(np.int16)  # [z, y, x]
        # 转置并重采样 lowres 到新大小
        target_spatial_shape = data.shape[1:]
        if lowres.shape != target_spatial_shape:
            # 重采样 lowres 到 (563, 467, 467)
            lowres = resize_segmentation(lowres, target_spatial_shape, order=1)
        # ------- one-hot -------
        lowres_oh = np.stack([(lowres == c).astype(np.float32) for c in range(1, self.num_classes + 1)], axis=0)  # [num_classes, x, y, z]
        # ------- 打印data, lowres顺序 -------
        print("data.shape:",data.shape)
        print("lowres.shape:",lowres_oh.shape)
        # ------- 拼接 inp -------
        inp = np.concatenate([data, lowres_oh], axis=0)  # [1+num_classes, z, y, x]
        # ================================
        # ★ 核心：强制 crop/pad 到 nnUNet patch
        # ================================
        inp_batch, label_batch_sampled = get_tta_patches_internal(self.patch_size, inp,label, num_patches=3)

        # 保存调试图 (只存 Batch 中的第一个 Patch)
        save_nifti_for_debug(inp_batch[0].numpy(), label_batch_sampled[0, 0].numpy(), sample_name=cid)
        # ================================
        # ✅ 新增：保存每个 patch 的标签到指定路径
        # ================================
        for patch_idx in range(inp_batch.shape[0]):
            # 获取该 patch 的标签数据
            patch_label = label_batch_sampled[patch_idx, 0].numpy()  # [D, H, W]

            # 创建 ITK 图像
            patch_label_itk = sitk.GetImageFromArray(patch_label.astype(np.uint8))

            # 设置 spacing（如果需要，这里保持原始 spacing）
            # patch_label_itk.SetSpacing(self.target_spacing)

            # 保存到指定路径
            save_filename = f"{cid}_patch{patch_idx}_label.nii.gz"
            save_path = os.path.join(self.patch_label_save_dir, save_filename)
            sitk.WriteImage(patch_label_itk, save_path)

            print(f"✅ 保存 patch {patch_idx} 标签到: {save_path}")
            print(f"  - 标签形状: {patch_label.shape}")
            print(f"  - 标签值范围: {np.unique(patch_label)}")
        return {
            "image": inp_batch,  # Shape: [4, C, pd, ph, pw]
            "label": label_batch_sampled,  # Shape: [4, 1, pd, ph, pw]
            "name": cid,
            # "lowres": torch.from_numpy(lowres).long(),
        }