import os
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.transform import resize
import torch.nn.functional as F


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
    image_data = inp[0]  # [D, H, W]
    # 注意：SimpleITK 使用 [H, W, D] 顺序，但 GetImageFromArray 自动处理为 [D, H, W] -> ITK 默认是 [x,y,z] = [W,H,D]
    # 所以我们先转置为 [W, H, D] 再保存？其实不用！sitk.GetImageFromArray 默认按 numpy 的 [z,y,x] 解释
    # 因此直接传 [D, H, W] 即可，对应 ITK 的 (z,y,x)
    img_itk = sitk.GetImageFromArray(image_data.astype(np.float32))
    sitk.WriteImage(img_itk, os.path.join(output_dir, f"{sample_name}_image_patch.nii.gz"))

    # 2. 保存标签（必须是整数类型）
    # 确保 label 是整数（int16/int32/uint8）
    label_itk = sitk.GetImageFromArray(label.astype(np.uint8))  # 或 np.int16
    sitk.WriteImage(label_itk, os.path.join(output_dir, f"{sample_name}_label_patch.nii.gz"))

    print(f"✅ 已保存到 {output_dir}:")
    print(f"   - {sample_name}_image.nii.gz")
    print(f"   - {sample_name}_label.nii.gz")


def visualize_inp(inp, title="inp visualization"):
    C, D, H, W = inp.shape
    mid_slice = D // 2  # 取中间深度的 slice

    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
    if C == 1:
        axes = [axes]

    for c in range(C):
        img_slice = inp[c, mid_slice]  # [H, W]
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
        resized = resize(mask.astype(float), new_shape, order=order,
                         mode="edge", clip=True, anti_aliasing=False)
        reshaped[resized >= 0.5] = c
    return reshaped


# ===========================
#   关键：nnUNet patch 修正函数
#   要参照 from nnunet.training.data_augmentation.
#   default_data_augmentation import get_default_augmentation
#   进行修改
# ===========================
def crop_pad_to_patch(data, patch_size):
    """
    data: numpy array [C, D, H, W] 或 [D, H, W]
    返回:  裁剪/填充后的相同 shape
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


class CTPelvicCascadeTTADataset(Dataset):
    def __init__(self,
                 img_dir,
                 lowres_dir,
                 split='all',
                 num=None,
                 num_classes=4,
                 patch_size=None):
        """
        patch_size: 由 train() 中 args.patch_size 提供，必须是 nnUNet 的 patch
        """
        self.img_dir = img_dir
        self.lowres_dir = lowres_dir
        self.num_classes = num_classes
        self.patch_size = patch_size  # <<< 必须提供 !!!

        all_files = [f for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz")]
        self.case_ids = [f.replace("_0000.nii.gz", "") for f in all_files]

        if num is not None:
            self.case_ids = self.case_ids[:num]

        print(f"{split}, total {len(self.case_ids)} cases")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]

        img_path = os.path.join(self.img_dir, cid + "_0000.nii.gz")
        lowres_path = os.path.join(self.lowres_dir, cid + ".nii.gz")
        label_path = os.path.join(self.img_dir, cid + "_mask_4label.nii.gz")

        # ------- 读高分辨图 -------
        img_itk = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # [D,H,W]
        img = img[None]  # [1, D, H, W]

        # ------- 读 lowres -------
        lowres_itk = sitk.ReadImage(lowres_path)
        lowres = sitk.GetArrayFromImage(lowres_itk).astype(np.int16)

        if lowres_itk.GetSize() != img_itk.GetSize():
            lowres = resize_segmentation(lowres, img.shape[1:], order=1)

        # ------- one-hot -------
        lowres_oh = []
        for c in range(1, self.num_classes + 1):
            lowres_oh.append((lowres == c).astype(np.float32))
        lowres_oh = np.stack(lowres_oh, axis=0)

        inp = np.concatenate([img, lowres_oh], axis=0)  # [1+4, D,H,W]

        # ------- 读 GT -------
        if os.path.isfile(label_path):
            lab_itk = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(lab_itk).astype(np.int16)
        else:
            label = np.zeros_like(lowres)

        # ================================
        #   ★ 核心：强制 crop/pad 到 nnUNet patch
        # ================================
        inp = crop_pad_to_patch(inp, self.patch_size)  # [C, pd, ph, pw]
        # print("inp", inp)
        # visualize_inp(inp)
        label = crop_pad_to_patch(label[None], self.patch_size)[0]  # [pd,ph,pw]
        save_nifti_for_debug(inp, label, sample_name=cid)

        return {
            "image": torch.from_numpy(inp.copy()).float(),
            "label": torch.from_numpy(label[None]).long(),
            "name": cid,
            "lowres": torch.from_numpy(lowres).long(),
        }
