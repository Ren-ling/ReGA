#loss function for SIFA
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def dice_loss(predict, target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict * target)
    dice = (2 * intersect + smooth) / (torch.sum(target) + torch.sum(predict * predict) + smooth)
    loss = 1.0 - dice
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            if len(input_tensor.shape) == 2:
                tmp = tmp.unsqueeze(1)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, weight=None, softmax=True, one_hot=True):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if one_hot:
            target = self.one_hot_encode(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape
        class_wise_dice = []
        loss = 0.0
        for i in range(1, self.n_classes):
            diceloss = dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        return loss / (self.n_classes - 1.0)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss


class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, num_classes, alpha=1.0):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label)
        celoss = self.celoss(predict, label)
        loss = celoss + self.alpha * diceloss
        # return loss
        return celoss


def mmd_loss(Xs, Xt, kernel_gamma=1.0):
    # Xs 和 Xt 分别表示源域和目标域中的特征，假设它们是 torch.Tensor 类型

    # 计算高斯核矩阵
    m = Xs.size(0)
    n = Xt.size(0)
    K_ss = torch.exp(-torch.pow(torch.norm(Xs.unsqueeze(1) - Xs.unsqueeze(0), dim=2), 2) / (2.0 * kernel_gamma ** 2))
    K_st = torch.exp(-torch.pow(torch.norm(Xs.unsqueeze(1) - Xt.unsqueeze(0), dim=2), 2) / (2.0 * kernel_gamma ** 2))
    K_tt = torch.exp(-torch.pow(torch.norm(Xt.unsqueeze(1) - Xt.unsqueeze(0), dim=2), 2) / (2.0 * kernel_gamma ** 2))

    # 计算 MMD 损失
    loss = (1.0 / (m * (m - 1))) * torch.sum(K_ss) - (2.0 / (m * n)) * torch.sum(K_st) + (
                1.0 / (n * (n - 1))) * torch.sum(K_tt)

    return loss


import torch


def center_alignment_loss(source_features, target_features, alpha=1.0):
    """
    计算 Center Alignment 损失

    Args:
        source_features (torch.Tensor): 源域的特征张量，形状 (batch_size, num_features)
        target_features (torch.Tensor): 目标域的特征张量，形状 (batch_size, num_features)
        alpha (float): Center Alignment 损失的权重参数

    Returns:
        torch.Tensor: Center Alignment 损失
    """
    # 计算源域和目标域的均值特征
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)
    # 计算 Center Alignment 损失
    loss = torch.norm(source_mean - target_mean, p=2)

    return alpha * loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha  # 权衡硬损失和软损失的参数
        self.temperature = temperature  # 温度参数

        self.criterion = nn.CrossEntropyLoss()  # 硬损失的交叉熵损失函数

    def forward(self, student_outputs, teacher_outputs, labels):
        # Reshape 教师和学生模型的输出为 (n, c, 320*320)
        hard_loss = self.criterion(student_outputs, labels)
        n, c, w, h = student_outputs.shape
        teacher_outputs = teacher_outputs.view(teacher_outputs.size(0), teacher_outputs.size(1), -1)
        student_outputs = student_outputs.view(student_outputs.size(0), student_outputs.size(1), -1)

        # 计算硬损失
        # 

        # 计算软化标签（使用教师模型的输出）
        soft_labels = F.softmax(teacher_outputs / self.temperature, dim=1)

        # 计算软损失（使用软化标签和学生模型的输出）
        soft_loss = F.kl_div(F.log_softmax(student_outputs / self.temperature, dim=1), soft_labels)

        # 结合硬损失和软损失
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return loss / (n * w * h)


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
                F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                         F.softmax(out_t / self.T, dim=1), reduction="batchmean")  # , reduction="batchmean"
                * self.T
                * self.T
        )
        return loss
