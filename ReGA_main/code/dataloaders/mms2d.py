import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import csv
import SimpleITK as sitk
import torchio as tio


class MMS2D(Dataset):

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.split = split

        train_path = os.path.join(self._base_dir, 'train.csv')
        valid_path = os.path.join(self._base_dir, 'valid.csv')
        test_path = os.path.join(self._base_dir, 'test.csv')
        all_path = os.path.join(self._base_dir, 'all.csv')

        if split == 'train':
            data_path = train_path
        elif split == 'valid':
            data_path = valid_path
        elif split == 'test':
            data_path = test_path
        elif split == 'all':
            data_path = all_path

        # 读取CSV文件
        self.image_list = []
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过header
            for row in reader:
                self.image_list.append({
                    'image': row[0],
                    'label': row[1]
                })

        if num is not None:
            train_num = int(len(self.image_list) * num / 100)
            self.image_list = self.image_list[:train_num]

        print(f"Total {len(self.image_list)} samples")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        sample_paths = self.image_list[idx]

        # 读取图像和标签
        images = []

        image = sitk.ReadImage(sample_paths['image'])
        image_array = torch.from_numpy(sitk.GetArrayFromImage(image))
        image_array = image_array.unsqueeze(0)
        images.append(image_array)

        # 将多个模态组合为多通道图像
        # image = np.stack(images, axis=0)  # [channels, height, width, depth]

        # 读取标签
        label = sitk.ReadImage(sample_paths['label'])
        label_array = torch.from_numpy(sitk.GetArrayFromImage(label))
        label_array = label_array.unsqueeze(0)
        # if len(label_array)<4:
        # label_array = label_array.reshape(1, label_array.shape[0], label_array.shape[1], label_array.shape[2])

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_array),
            label=tio.LabelMap(tensor=label_array))
        if self.split == 'train':
            transform = tio.Compose([
                # tio.RescaleIntensity(out_min_max=(0, 1)),
                # tio.CropOrPad((128,128,128)),
                tio.RandomFlip(axes=('LR',), p=0.5),
                tio.RandomAffine(scales=(0.8, 1.2), degrees=15, p=0.5),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
                tio.RandomNoise(mean=0, std=(0, 0.05), p=0.5),
                # tio.RandomBlur(std=(0.1, 2), p=0.5),
                # tio.RandomElasticDeformation(num_control_points=(7, 7, 7), max_displacement=(5, 5, 5), p=0.2),
                # tio.RandomMotion(degrees=10, translation=10, p=0.5),
                # tio.RandomBiasField(coefficients=(0.1, 0.3), p=0.5)
            ])
        elif self.split == 'valid' or self.split == 'test' or self.split == 'all':
            transform = tio.Compose([
                # tio.RescaleIntensity(out_min_max=(0, 1))
            ])

        transformed_subject = transform(subject)
        image = transformed_subject['image'].numpy()
        label_array = transformed_subject['label'].numpy()

        base_name = sample_paths['label'].split('/')[-1]
        sample = {'image': image, 'label': label_array.astype(np.uint8), 'name': base_name}
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma, 2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         # label = sample['label']
#         # image = image.reshape(
#         # #     1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
#         # label = label.reshape(
#         #     1, label.shape[0], label.shape[1], label.shape[2]).astype(np.float32)
#         if 'onehot_label' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
#                     'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
#         else:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
