import os
import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.utilities.one_hot_encoding import to_one_hot
from nnunet.preprocessing.preprocessing import GenericPreprocessor
import gc


def load_full_volume_cascade_final(case_id, args):
    """
    基于提供的 plans.pkl 实现的最终精确加载函数 - 内存优化版
    """
    # 1. 设置路径
    img_path = os.path.join(args.img_dir, f"{case_id}_0000.nii.gz")
    lowres_path = os.path.join(args.lowres_dir, f"{case_id}.nii.gz")

    # 2. 从提供的计划文件中提取参数
    target_spacing = np.array([0.79998779, 0.854, 0.854])
    intensity_props = {0: {'mean': 375.80426, 'sd': 282.0642,
                           'percentile_00_5': -51.0, 'percentile_99_5': 1298.0}}

    # 3. 实例化官方预处理器
    preprocessor = GenericPreprocessor(
        normalization_scheme_per_modality={0: 'CT'},
        use_nonzero_mask={0: False},
        transpose_forward=[0, 1, 2],
        intensityproperties=intensity_props
    )

    # 4. 预处理高分辨率 CT - 使用更节省内存的方式
    print(f"加载CT图像: {case_id}")
    data, _, properties = preprocessor.preprocess_test_case([img_path], target_spacing)

    # 立即释放预处理过程中的临时变量
    if 'preprocessor' in locals():
        del preprocessor

    # 5. 处理低分辨率掩码 (级联逻辑)
    lowres_oh = None

    if os.path.exists(lowres_path):
        print(f"加载低分辨率掩码: {case_id}")
        lowres_itk = sitk.ReadImage(lowres_path)
        lowres = sitk.GetArrayFromImage(lowres_itk)  # (Z, Y, X)

        # 释放ITK对象
        del lowres_itk

        # 调整低分辨率掩码尺寸以匹配重采样后的高分辨率 CT
        target_shape = data.shape[1:]
        if lowres.shape != target_shape:
            lowres = resize_segmentation(lowres, target_shape, order=1)

        # 对4个前景类别进行独热编码 (One-Hot)
        # 使用内存优化的方式
        lowres_oh = np.zeros((4, *target_shape), dtype=np.float32)
        for c in range(1, 5):  # 类别 1-4
            mask = (lowres == c).astype(np.float32)
            lowres_oh[c - 1] = mask

        # 释放原始低分辨率数据
        del lowres
    else:
        # 如果缺少掩码，则填充零值通道
        print(f"Warning: Lowres mask missing for {case_id}")
        target_shape = data.shape[1:]
        lowres_oh = np.zeros((4, *target_shape), dtype=np.float32)

    # 6. 拼接通道: CT (1) + Mask (4) = 5 通道 - 内存优化版本
    print(f"合并CT和掩码数据 - 形状: {data.shape[1:]}")

    # 预先计算最终形状
    final_shape = (5, data.shape[1], data.shape[2], data.shape[3])

    # 创建最终数组
    full_inp = np.empty(final_shape, dtype=np.float32)

    # 填充CT数据
    full_inp[0] = data.astype(np.float32)

    # 填充掩码数据
    full_inp[1:] = lowres_oh

    # 立即释放中间数据以节省内存
    del data
    if lowres_oh is not None:
        del lowres_oh

    # 强制垃圾回收
    gc.collect()

    print(f"最终数据形状: {full_inp.shape}")
    return full_inp, properties


def load_full_volume_cascade_final_chunked(case_id, args, chunk_size=100):
    """
    分块加载版本 - 对于极大图像的内存优化版本
    chunk_size: Z轴方向的分块大小
    """
    # 1. 设置路径
    img_path = os.path.join(args.img_dir, f"{case_id}_0000.nii.gz")
    lowres_path = os.path.join(args.lowres_dir, f"{case_id}.nii.gz")

    print(f"分块加载图像: {case_id}, 分块大小: {chunk_size}")

    # 2. 从提供的计划文件中提取参数
    target_spacing = np.array([0.79998779, 0.854, 0.854])
    intensity_props = {0: {'mean': 375.80426, 'sd': 282.0642,
                           'percentile_00_5': -51.0, 'percentile_99_5': 1298.0}}

    # 3. 实例化官方预处理器
    preprocessor = GenericPreprocessor(
        normalization_scheme_per_modality={0: 'CT'},
        use_nonzero_mask={0: False},
        transpose_forward=[0, 1, 2],
        intensityproperties=intensity_props
    )

    # 4. 预处理高分辨率 CT - 按块处理
    data, _, properties = preprocessor.preprocess_test_case([img_path], target_spacing)

    # 获取数据形状
    target_shape = data.shape[1:]  # (Z, Y, X)
    z_size, y_size, x_size = target_shape

    # 5. 加载低分辨率掩码
    if os.path.exists(lowres_path):
        lowres_itk = sitk.ReadImage(lowres_path)
        lowres = sitk.GetArrayFromImage(lowres_itk)  # (Z, Y, X)

        # 调整低分辨率掩码尺寸
        if lowres.shape != target_shape:
            lowres = resize_segmentation(lowres, target_shape, order=1)
    else:
        lowres = np.zeros(target_shape, dtype=np.int16)

    # 6. 创建最终输出数组
    full_inp = np.empty((5, z_size, y_size, x_size), dtype=np.float32)

    # 7. 按块处理以节省内存
    for z_start in range(0, z_size, chunk_size):
        z_end = min(z_start + chunk_size, z_size)

        print(f"处理块: Z={z_start}:{z_end}")

        # 处理CT数据块
        ct_chunk = data[:, z_start:z_end, :, :]
        full_inp[0, z_start:z_end, :, :] = ct_chunk.astype(np.float32)

        # 处理低分辨率掩码块
        lowres_chunk = lowres[z_start:z_end, :, :]

        # 对当前块的掩码进行one-hot编码
        for c in range(1, 5):  # 类别 1-4
            mask_chunk = (lowres_chunk == c).astype(np.float32)
            full_inp[c, z_start:z_end, :, :] = mask_chunk

        # 释放当前块的内存
        del ct_chunk, lowres_chunk
        gc.collect()

    # 8. 释放大数组内存
    del data, lowres
    if 'lowres_itk' in locals():
        del lowres_itk

    gc.collect()
    print(f"分块加载完成，最终形状: {full_inp.shape}")

    return full_inp, properties


def load_full_volume_cascade_final_low_memory(case_id, args):
    """
    极低内存版本 - 不使用one-hot编码，而是在推理时动态处理
    这个版本返回原始CT和低分辨率掩码，而不是5通道的one-hot
    """
    # 1. 设置路径
    img_path = os.path.join(args.img_dir, f"{case_id}_0000.nii.gz")
    lowres_path = os.path.join(args.lowres_dir, f"{case_id}.nii.gz")

    print(f"极低内存模式加载: {case_id}")

    # 2. 从提供的计划文件中提取参数
    target_spacing = np.array([0.79998779, 0.854, 0.854])
    intensity_props = {0: {'mean': 375.80426, 'sd': 282.0642,
                           'percentile_00_5': -51.0, 'percentile_99_5': 1298.0}}

    # 3. 实例化官方预处理器
    preprocessor = GenericPreprocessor(
        normalization_scheme_per_modality={0: 'CT'},
        use_nonzero_mask={0: False},
        transpose_forward=[0, 1, 2],
        intensityproperties=intensity_props
    )

    # 4. 预处理高分辨率 CT
    data, _, properties = preprocessor.preprocess_test_case([img_path], target_spacing)

    # 5. 加载低分辨率掩码
    if os.path.exists(lowres_path):
        lowres_itk = sitk.ReadImage(lowres_path)
        lowres = sitk.GetArrayFromImage(lowres_itk)  # (Z, Y, X)

        # 调整低分辨率掩码尺寸
        target_shape = data.shape[1:]
        if lowres.shape != target_shape:
            lowres = resize_segmentation(lowres, target_shape, order=1)
    else:
        target_shape = data.shape[1:]
        lowres = np.zeros(target_shape, dtype=np.int16)

    # 6. 只返回CT和低分辨率掩码，不进行one-hot编码
    # 在推理时动态创建one-hot编码
    print(f"CT形状: {data.shape}, 低分辨率掩码形状: {lowres.shape}")

    # 返回字典而不是5通道数组
    result = {
        'ct_data': data.astype(np.float32),
        'lowres_mask': lowres.astype(np.int16),
        'properties': properties
    }

    return result


import matplotlib.pyplot as plt
import numpy as np


def debug_visualize_cascade_alignment(full_inp, case_id, save_path=None):
    """
    可视化5通道输入，检查CT(通道0)与低分辨率掩码(通道1-4)是否对齐
    full_inp形状: (5, Z, Y, X)
    """
    # 1. 提取中间切片
    z_dim = full_inp.shape[1]
    mid_z = z_dim // 2

    ct_slice = full_inp[0, mid_z, :, :]
    # 将4个One-Hot掩码通道合并为一个标签图用于可视化
    mask_channels = full_inp[1:, mid_z, :, :]  # (4, Y, X)
    combined_mask = np.argmax(mask_channels, axis=0) + 1
    # 掩码中全为0的地方设为背景
    combined_mask[np.max(mask_channels, axis=0) == 0] = 0

    # 2. 绘图
    plt.figure(figsize=(15, 5))

    # 子图1: 原始CT
    plt.subplot(1, 3, 1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title(f"CT Slice (Z={mid_z})")
    plt.axis('off')

    # 子图2: 低分辨率掩码 (Low-res Mask)
    plt.subplot(1, 3, 2)
    plt.imshow(combined_mask, cmap='jet')
    plt.title("Low-res Mask Channels")
    plt.axis('off')

    # 子图3: 叠加对比 (Overlay) - 检查对齐的关键
    plt.subplot(1, 3, 3)
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(combined_mask, cmap='jet', alpha=0.5)  # 透明叠加
    plt.title("Alignment Check (Overlay)")
    plt.axis('off')

    plt.suptitle(f"Debug Alignment: {case_id}")

    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存至: {save_path}")
    else:
        plt.show()


# --- 使用示例 ---
# 根据需要选择不同的加载函数：
# 1. 标准版本（内存优化）
# full_inp, properties = load_full_volume_cascade_final(case_name, args)

# 2. 分块版本（针对极大图像）
# full_inp, properties = load_full_volume_cascade_final_chunked(case_name, args, chunk_size=100)

# 3. 极低内存版本（不进行one-hot编码）
# data_dict = load_full_volume_cascade_final_low_memory(case_name, args)
