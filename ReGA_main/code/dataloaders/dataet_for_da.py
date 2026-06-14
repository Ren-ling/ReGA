import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
import torch.nn.functional as F


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


class CTPelvicCascadeTTADataset(Dataset):
    def __init__(self, img_dir, lowres_dir, split='all', num=None):
        self.img_dir = img_dir
        self.lowres_dir = lowres_dir

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

        return {
            "name": cid,
            "img_path": img_path,
            "lowres_path": lowres_path,
            "label_path": label_path,
        }
