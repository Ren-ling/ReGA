

import torch.nn as nn
import torch.utils.data
from torchvision.transforms.functional import to_pil_image, rotate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from pymic.util.evaluation_seg import get_multi_class_evaluation_score
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import copy


def calculate_histogram(image, hist_range=(-3, 3)):
    img_fore = image[image > -0.95]
    hist, bin_edge = np.histogram(img_fore.flatten(), bins=256, range=hist_range, density=True)
    return hist, bin_edge


def calculate_cdf(hist):
    # 计算累积分布函数
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # 归一化
    return cdf_normalized


def modal_histogram_matching(target, target_hist, ref_hist, range=(-3, 3)):
    # 计算源图像和目标图像的CDF
    ref_cdf = calculate_cdf(ref_hist)
    target_cdf = calculate_cdf(target_hist)

    # 创建映射表
    mapping = np.interp(target_cdf, ref_cdf, np.linspace(range[0], range[1], ref_cdf.size))

    target_foreground = target[target > -0.95]
    matched_foreground = np.interp(target_foreground.flatten(), np.linspace(range[0], range[1], ref_cdf.size), mapping)

    matched_modality = copy.deepcopy(target)
    matched_modality[target > -0.95] = torch.tensor(matched_foreground).float()

    return matched_modality


def get_3d_crop_bounding_box(mask, margin=[5, 10, 10]):
    D, H, W = mask.shape
    ds, hs, ws = np.where(mask > 0)
    if (ds.shape[0] == 0):
        return 0, 0, 0, 0, 0, 0
    dmin = max(ds.min() - margin[0], 0)
    dmax = min(ds.max() + margin[0], D)
    hmin = max(hs.min() - margin[1], 0)
    hmax = min(hs.max() + margin[1], H)
    wmin = max(ws.min() - margin[2], 0)
    wmax = min(ws.max() + margin[2], W)
    return dmin, dmax, hmin, hmax, wmin, wmax


def test_no_adapt(net, image):
    image = image.cuda()
    image = image.float()
    with torch.no_grad():
        y1 = net.forward_no_adapt(image)
        y = torch.argmax(y1, dim=1)
    label = y.cpu().numpy()[0]

    return label


class Prototype_Pool(nn.Module):
    def __init__(self, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num = class_num
        self.max_length = max
        self.feature_bank = torch.tensor([]).cuda()
        self.feature_bank2 = torch.tensor([]).cuda()  # 新增第二个特征库
        self.image_bank = torch.tensor([]).cuda()
        self.hist_bank = []
        self.mask_bank = torch.tensor([]).cuda()
        self.name_list = []

    def get_pool_feature(self, x, mask, top_k=5):
        if self.feature_bank.numel() > 0 and self.feature_bank.device != x.device:
            self.feature_bank = self.feature_bank.to(x.device)

        if len(self.feature_bank) > 0:
            # 计算所有样本与特征库的余弦相似度 [B, N]
            cosine_similarities = torch.nn.functional.cosine_similarity(
                x.unsqueeze(1), self.feature_bank.unsqueeze(0), dim=2
            )

            # 获取每个样本的top-k最相似特征索引
            k = min(top_k, self.feature_bank.shape[0])
            if self.feature_bank.shape[0] >= top_k:
                # 获取所有batch样本的top-k索引 [B, k]
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :k]
            else:
                # 特征库数量不足，取所有特征
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :k]

            # 为每个样本分别计算权重 [B, k]
            # 1. 获取每个样本对应top-k索引的相似度值
            batch_size = x.shape[0]
            topk_similarities = torch.gather(
                cosine_similarities,
                dim=1,
                index=outall
            )  # 形状: [B, k]

            # 2. 对每个样本的top-k相似度进行softmax，得到权重
            weights = torch.softmax(topk_similarities, dim=1)  # [B, k]

            # 3. 特征融合
            # 注意：rates=1的问题保留原样
            rates = 1
            x = x * (1 - rates)  # rates=1时，x变为0

            # 对每个样本独立进行加权融合
            for i in range(k):
                # 获取所有样本的第i个最相似特征 [B, D]
                bank_features = self.feature_bank[outall[:, i]]
                # 使用每个样本自己的权重进行加权 [B, 1] -> [B, 1] 广播到特征维度
                weighted_features = bank_features * weights[:, i:i + 1]
                x += weighted_features

            return x, len(self.feature_bank)
        else:
            return x, len(self.feature_bank)

    def get_pool_feature2(self, x, mask, top_k=5):
        """新增：第二个特征库的检索方法"""
        print(f"\n=== get_pool_feature2 DEBUG ===")
        print(f"Input x shape: {x.shape}")
        print(f"Bank2 shape: {self.feature_bank2.shape}")
        print(f"top_k: {top_k}")

        if self.feature_bank2.numel() > 0 and self.feature_bank2.device != x.device:
            self.feature_bank2 = self.feature_bank2.to(x.device)

        if len(self.feature_bank2) > 0:
            # 关键：检查维度
            print(f"Before cosine_similarity:")
            print(f"  x.unsqueeze(1) shape would be: {x.unsqueeze(1).shape}")
            print(f"  bank2.unsqueeze(0) shape would be: {self.feature_bank2.unsqueeze(0).shape}")

            cosine_similarities = torch.nn.functional.cosine_similarity(
                x.unsqueeze(1),
                self.feature_bank2.unsqueeze(0),
                dim=2
            )

            print(f"cosine_similarities shape: {cosine_similarities.shape}")

            if self.feature_bank2.shape[0] >= top_k:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
            else:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :self.feature_bank2.shape[0]]

            print(f"outall shape: {outall.shape}")

            # 修复：为每个样本单独计算权重
            # 获取每个样本的top-k相似度值 [B, k]
            topk_similarities = torch.gather(
                cosine_similarities,
                dim=1,
                index=outall
            )
            print(f"topk_similarities shape: {topk_similarities.shape}")

            # 对每个样本的top-k相似度进行softmax得到权重 [B, k]
            weights = torch.softmax(topk_similarities, dim=1)
            print(f"weights shape: {weights.shape}")

            rates = 1
            x_result = x * (1 - rates)

            k = min(top_k, self.feature_bank2.shape[0])
            for i in range(k):
                selected_feature = self.feature_bank2[outall[:, i]]
                print(f"  Feature {i} shape: {selected_feature.shape}")
                # 修复：每个样本使用自己的权重
                x_result += selected_feature * weights[:, i:i + 1]  # weights[:, i:i+1] 形状为 [B, 1]

            print(f"Final x_result shape: {x_result.shape}")
            return x_result, len(self.feature_bank2)
        else:
            print(f"Bank2 is empty, returning original x")
            return x, len(self.feature_bank2)

    def get_pool_hist(self, hist, top_k=1):
        if len(self.hist_bank) > 0:
            cosine_similarities = torch.nn.functional.cosine_similarity(torch.tensor(hist).unsqueeze(0),
                                                                        torch.tensor(np.array(self.hist_bank)), dim=1)
            if len(self.hist_bank) >= top_k:
                outall = cosine_similarities.argsort(dim=0, descending=True)[:top_k]
            else:
                outall = cosine_similarities.argsort(dim=0, descending=True)[:len(self.hist_bank)]
                # outall = outall.repeat(1,top_k)

            hist = self.hist_bank[outall[0]]
            # rates = cosine_similarities[outall[0]].mean(0)
            # weight = rates * torch.exp(cosine_similarities[outall[0]]) / torch.sum(torch.exp(cosine_similarities[outall[0]]))
            # hist = hist * (1-np.array(rates))
            # for i in range(min(top_k,len(self.hist_bank))):
            #     hist += self.hist_bank[outall[i]]*weight[i]
            return hist
        else:
            return hist

    def update_feature_pool(self, feature):
        feature = feature.detach()

        # 2) 统一 device（避免 cuda:0 / cuda:1 混用）
        if feature.device != self.feature_bank.device:
            feature = feature.to(self.feature_bank.device)

        if self.feature_bank.shape[0] == 0:
            self.feature_bank = torch.cat([self.feature_bank, feature.detach()], dim=0)
        else:
            if self.feature_bank.shape[0] < self.max_length:
                self.feature_bank = torch.cat([self.feature_bank, feature.detach()], dim=0)
            else:
                self.feature_bank = torch.cat([self.feature_bank[-self.max_length:], feature.detach()], dim=0)

    def update_feature_pool2(self, feature):
        """新增：第二个特征库的更新方法"""
        feature = feature.detach()
        if feature.device != self.feature_bank2.device:
            feature = feature.to(self.feature_bank2.device)

        if self.feature_bank2.shape[0] == 0:
            self.feature_bank2 = torch.cat([self.feature_bank2, feature.detach()], dim=0)
        else:
            if self.feature_bank2.shape[0] < self.max_length:
                self.feature_bank2 = torch.cat([self.feature_bank2, feature.detach()], dim=0)
            else:
                self.feature_bank2 = torch.cat([self.feature_bank2[-self.max_length:], feature.detach()], dim=0)

    def update_image_pool(self, image):
        if self.image_bank.shape[0] == 0:
            self.image_bank = torch.cat([self.image_bank, image.detach()], dim=0)
        else:
            if self.image_bank.shape[0] < self.max_length:
                self.image_bank = torch.cat([self.image_bank, image.detach()], dim=0)
            else:
                self.image_bank = torch.cat([self.image_bank[-self.max_length:], image.detach()], dim=0)

    def update_hist_pool(self, hist):
        if len(self.hist_bank) == 0:
            self.hist_bank.append(hist)
        else:
            if len(self.hist_bank) < self.max_length:
                self.hist_bank.append(hist)
            else:
                self.hist_bank.pop(0)
                self.hist_bank.append(hist)

    def update_mask_pool(self, image):
        if self.mask_bank.shape[0] == 0:
            self.mask_bank = torch.cat([self.mask_bank, image.detach()], dim=0)
        else:
            if self.mask_bank.shape[0] < self.max_length:
                self.mask_bank = torch.cat([self.mask_bank, image.detach()], dim=0)
            else:
                self.mask_bank = torch.cat([self.mask_bank[-self.max_length:], image.detach()], dim=0)

    def update_name_pool(self, image):
        if len(self.name_list) == 0:
            self.name_list.append(image)
        else:
            if len(self.name_list) < self.max_length:
                self.name_list.append(image)
            else:
                self.name_list = self.name_list[-self.max_length:]
                self.name_list.append(image)


class ImgUpdate():
    def __init__(self):
        self.est_ema = None
        self.hist_ema = None
        self.pool = Prototype_Pool(class_num=4, max=1)
        self.est_list = []
        self.max_len = 10
        self.mean = None
        self.std = None

        self.WT_est_ema = None
        self.WT_hists_ema = None
        self.TC_est_ema = None
        self.TC_hists_ema = None
        self.ET_est_ema = None
        self.ET_hists_ema = None

    def apply_histogram_matching(self, target_image, target_hist, ref_hist, hist_range):
        matched_image = modal_histogram_matching(target_image, target_hist, ref_hist, hist_range)
        return matched_image

    def get_class_label(self, pred, label_list):
        pred_sub = np.zeros_like(pred)
        for lab in label_list:
            pred_sub = pred_sub + np.asarray(pred == lab, np.uint8)

        return pred_sub

    def img_update_with_class(self, model, input, pred, est_WT, est_TC, est_ET):
        hist_range = (0, 1)
        img_hists, bin_edges = calculate_histogram(input, hist_range)
        WT_pred = self.get_class_label(pred, [1, 2, 3])
        TC_pred = self.get_class_label(pred, [2, 3])
        ET_pred = self.get_class_label(pred, [3])

        # 不同的类别，分别保存自己效果好的hists
        if self.WT_est_ema == None:
            self.WT_est_ema = est_WT
            self.TC_est_ema = est_TC
            self.ET_est_ema = est_ET
            self.WT_hists_ema = img_hists
            self.TC_hists_ema = img_hists
            self.ET_hists_ema = img_hists

        if (est_WT < self.WT_est_ema):
            # 当前WT表现不好，此处得到更新后的WT结果
            updated_WT_input = self.apply_histogram_matching(input, img_hists, self.WT_hists_ema, hist_range)
            updated_WT_pred = test_no_adapt(model, updated_WT_input)
            WT_pred = self.get_class_label(updated_WT_pred, [1, 2, 3])
        else:
            # 当前WT表现好，用表现好的Hist进行更新
            self.WT_hists_ema = 0.9 * self.WT_hists_ema + 0.1 * img_hists

        if (est_TC < self.TC_est_ema):
            # 当前TC表现不好，此处得到更新后的WT结果
            updated_TC_input = self.apply_histogram_matching(input, img_hists, self.TC_hists_ema, hist_range)
            updated_TC_pred = test_no_adapt(model, updated_TC_input)
            TC_pred = self.get_class_label(updated_TC_pred, [2, 3])
        else:
            # 当前TC表现好，用表现好的Hist进行更新
            self.TC_hists_ema = 0.9 * self.TC_hists_ema + 0.1 * img_hists

        if (est_ET < self.ET_est_ema):
            # 当前ET表现不好，此处得到更新后的WT结果
            updated_ET_input = self.apply_histogram_matching(input, img_hists, self.ET_hists_ema, hist_range)
            updated_ET_pred = test_no_adapt(model, updated_ET_input)
            ET_pred = self.get_class_label(updated_ET_pred, [3])
        else:
            # 当前ET表现好，用表现好的Hist进行更新
            self.ET_hists_ema = 0.9 * self.ET_hists_ema + 0.1 * img_hists

        self.WT_est_ema = 0.9 * self.WT_est_ema + 0.1 * est_WT
        self.TC_est_ema = 0.9 * self.TC_est_ema + 0.1 * est_TC
        self.ET_est_ema = 0.9 * self.ET_est_ema + 0.1 * est_ET

        final_pred = np.zeros_like(pred)
        final_pred[WT_pred == 1] = 1
        final_pred[TC_pred == 1] = 2
        final_pred[ET_pred == 1] = 3

        return final_pred

    def img_update_for_tumor(self, input, pred, est_WT, est_TC, est_ET):
        # 只对预测出的肿瘤部分，用bounding box选出，再做后续的处理
        dmin, dmax, hmin, hmax, wmin, wmax = get_3d_crop_bounding_box(mask=pred, margin=[5, 10, 10])
        tumor = input[:, :, dmin:dmax, hmin:hmax, wmin: wmax]
        # 只计算大脑的hist
        hist_range = (0, 1)
        img_hists, bin_edges = calculate_histogram(tumor, hist_range)
        img_est = (est_WT + est_TC + est_ET) / 3

        # 直接对每个模态进行操作
        if self.est_ema == None:
            self.est_ema = img_est
            self.hists_ema = img_hists

        if (img_est <= self.est_ema):
            alpha = 1.0
            beta = 0.0
            # 如果效果不好，则对图片进行更新，同时不更新保存好的直方图
            update_image = True
        else:
            alpha = 0.9
            beta = 0.1
            # 如果效果好呢，则不对图片进行更新，但是要更新保存好的直方图
            update_image = False

        updated_hists = alpha * self.hists_ema + beta * img_hists
        # 这里是只用更好的Dice对est进行更新，但是这样存在问题：
        # est只会变得越来越高，同时如果第一个case的est就很高的话，后续的稍微差一点点的case都更新不了直方图，同时也会导致稍微差一点点的case，变得更差
        # updated_est = alpha * self.est_ema + beta * img_est

        # 这里对est采取全局更新的策略，能够保证est表征着全局的一个case好坏信息，此时再用这个est_ema去选case，
        # 选的就不是比上一个case，更好的case，而是和到目前为止全局的est相比，表现更好的case
        updated_est = 0.9 * self.est_ema + 0.1 * img_est

        updated_input = copy.deepcopy(input)
        updated_tumor = copy.deepcopy(tumor)
        if (update_image):
            updated_tumor = self.apply_histogram_matching(tumor, img_hists, updated_hists, hist_range)
            updated_input[:, :, dmin:dmax, hmin:hmax, wmin: wmax] = updated_tumor
        self.hists_ema = updated_hists
        self.est_ema = updated_est

        return updated_input

    def img_update(self, input, pred, est_WT, est_TC, est_ET):
        # 针对full image
        # 只计算大脑的hist
        input = input.cpu()
        # 在这里做一下clip
        input[input < -1.0] = -1.0
        input[input > 1.0] = 1.0

        updated_input = deepcopy(input)
        hist_range = (-1, 1)
        for i in range(input.shape[0]):
            input_sub = input[i]
            img_hist, bin_edge = calculate_histogram(input_sub, hist_range)
            img_est = (est_WT[i] + est_TC[i] + est_ET[i]) / 3

            if self.est_ema == None:
                self.est_ema = img_est
                self.hist_ema = img_hist

            if (img_est < self.est_ema):
                alpha = 1.0
                beta = 0.0
                update_image = True
            else:
                alpha = 0.9
                beta = 0.1
                update_image = False

            updated_hist = 0.9 * self.hist_ema + 0.1 * img_hist
            updated_est = 0.9 * self.est_ema + 0.1 * img_est

            updated_input_sub = copy.deepcopy(input_sub)
            if (update_image):
                updated_input_sub = self.apply_histogram_matching(input_sub, img_hist, updated_hist, hist_range)
            self.hist_ema = updated_hist
            self.est_ema = updated_est

            updated_input[i] = updated_input_sub

        return updated_input

    def img_update_histbank(self, input, pred, est_WT, est_TC, est_ET, est_avg):
        # 针对full image
        # 只计算大脑的hist
        input = input.cpu()
        # 在这里做一下clip
        input[input < -1.0] = -1.0
        input[input > 1.0] = 1.0

        updated_input = deepcopy(input)
        hist_range = (-1, 1)

        self.est_list.extend(est_avg.tolist())
        estp80 = np.percentile(self.est_list, 80)
        print(f'estp80:{estp80}')

        for i in range(input.shape[0]):
            input_sub = input[i]
            img_hist, bin_edge = calculate_histogram(input_sub, hist_range)
            img_est = est_avg[i]


            if (img_est < estp80):
                alpha = 1.0
                beta = 0.0
                update_image = True
            else:
                alpha = 0.9
                beta = 0.1
                update_image = False
                self.pool.update_hist_pool(img_hist)


            updated_input_sub = copy.deepcopy(input_sub)
            if (update_image):
                updated_hist = self.pool.get_pool_hist(img_hist)
                updated_input_sub = self.apply_histogram_matching(input_sub, img_hist, updated_hist, hist_range)
            # self.est_ema = updated_est

            updated_input[i] = updated_input_sub


        return updated_input

    def img_update_meanstd(self, input, pred, est_WT, est_TC, est_ET):
        # 针对full image
        # 只计算大脑的hist
        input = input.cpu()
        updated_input = deepcopy(input)
        for i in range(input.shape[0]):
            input_sub = input[i]

            img_mean = input_sub.mean()
            img_std = input_sub.std()
            img_est = (est_WT[i] + est_TC[i] + est_ET[i]) / 3

            if self.est_ema == None:
                self.est_ema = img_est
                self.mean = img_mean
                self.std = img_std

            if (img_est <= self.est_ema):
                alpha = 1.0
                beta = 0.0
                update_image = True
            else:
                alpha = 0.9
                beta = 0.1
                update_image = False

            updated_mean = alpha * self.mean + beta * img_mean
            updated_std = alpha * self.std + beta * img_std
            updated_est = 0.9 * self.est_ema + 0.1 * img_est

            updated_input_sub = copy.deepcopy(input_sub)
            if (update_image):
                updated_input_sub = input_sub * updated_std + updated_mean
            self.mean = updated_mean
            self.std = updated_std
            self.est_ema = updated_est

            updated_input[i] = updated_input_sub

        return updated_input


    def img_update_featbank_with_dropout_features(self, dropout_features,dropout_features2, dropout_scores,
                                                  current_pred, n_iter=5):
        """
        使用ADIC返回的dropout特征和评分更新特征库

        参数:
        dropout_features: (n_iter, B, C, D, H, W) ADIC中的dropout特征
        dropout_scores: (B, n_iter, 4) 每个dropout的评分
        current_pred: (B, D, H, W) 当前预测
        n_iter: dropout次数

        返回:
        updated_feat: (B, C, D, H, W) 更新后的特征
        """
        if dropout_features is None:
            print("Warning: No dropout features from CSCS")
            return None

        n_iter, B,  C,  D,  H,  W  = dropout_features.shape
        n_iter2, B2, C2, D2, H2, W2 = dropout_features2.shape

        updated_feat = torch.zeros_like(dropout_features[0])  # (B, C, D, H, W)
        updated_feat2 = torch.zeros_like(dropout_features2[0])  # 新增：第二个特征更新

        # 1. 计算每个dropout的平均评分
        dropout_scores_avg = np.mean(dropout_scores, axis=2)  # (B, n_iter)

        # 2. 初始化或更新历史评分列表
        if not hasattr(self, 'est_list'):
            self.est_list = []  # 存储历史所有dropout的评分

        # 将当前batch的所有评分添加到历史列表中
        self.est_list.extend(dropout_scores_avg.flatten().tolist())

        # 3. 计算历史评分的95分位数作为动态阈值
        if len(self.est_list) > 0:
            # 使用np.percentile计算95分位数
            percentile_95 = np.percentile(self.est_list, 70)   #q才是真实分界值
            print(f"Current 95th percentile threshold: {percentile_95:.4f}")
        else:
            percentile_95 = 0.7  # 默认阈值，当历史数据不足时使用
            print(f"Using default threshold: {percentile_95}")

        for b in range(B):
            # 使用第一个dropout的特征作为基准
            curr_feature = dropout_features[0, b].unsqueeze(0)  # (1, C, D, H, W)
            curr_feature2 = dropout_features2[0, b].unsqueeze(0)  # 新增：第二个特征
            curr_scores = dropout_scores_avg[b]  # (n_iter,)

            # 4. 将高质量dropout特征加入特征库（使用动态阈值）
            high_quality_count = 0

            for d in range(n_iter):
                score = curr_scores[d]
                dropout_feature = dropout_features[d, b].unsqueeze(0)  # (1, C, D, H, W)
                dropout_feature2 = dropout_features2[d, b].unsqueeze(0)  # 新增：第二个特征

                # 使用历史评分的95分位数作为动态阈值
                if score > percentile_95:  # 超过历史95%评分的为高质量特征
                    high_quality_count += 1
                    # 展平特征并加入bank
                    feature_flat = dropout_feature.view(1, -1).detach()
                    self.pool.update_feature_pool(feature_flat)

                    feature_flat2 = dropout_feature2.view(1, -1).detach()
                    self.pool.update_feature_pool2(feature_flat2)

                    print("~~~~~^v^  feature bank and bank2 is updated  ^v^~~~~~")

            if high_quality_count > 0:
                print(f"Sample {b}: Added {high_quality_count} dropout features to bank "
                      f"(score > {percentile_95:.4f})")

            # 5. 从特征库检索相似特征
            curr_feature_flat = curr_feature.view(1, -1)
            curr_feature_flat2 = curr_feature2.view(1, -1)  # 新增：第二个特征

            if self.pool.feature_bank.numel() > 0:
                latent_flat_mem, _ = self.pool.get_pool_feature(curr_feature_flat, None,
                                                                top_k=self.max_len)
                # 从第二个特征库检索
                latent_flat_mem2, _ = self.pool.get_pool_feature2(curr_feature_flat2, None,
                                                                top_k=self.max_len)

                # 6. 融合当前特征和记忆特征
                # 计算当前样本所有dropout的平均评分
                avg_score = np.mean(curr_scores)

                # 调整alpha计算方式（可选：使用相对于95分位数的相对值）
                # 将评分标准化到0-1范围，相对于95分位数
                if percentile_95 > 0:
                    relative_score = avg_score / percentile_95
                    alpha = min(max(relative_score * 0.3, 0.0), 0.5)  # 限制alpha最大为0.5
                else:
                    alpha = min(max(float(avg_score) / 100.0, 0.0), 0.5)

                # 恢复记忆特征的形状
                latent_restored = latent_flat_mem.view(1, C, D, H, W)
                latent_restored2 = latent_flat_mem2.view(1,240,24,20,16)  # 新增：第二个特征恢复 E1(dataset4)
                # latent_restored2 = latent_flat_mem2.view(1, 240, 24, 24, 24)  # 新增：第二个特征恢复 E2(dataset6)
                # latent_restored2 = latent_flat_mem2.view(1, 240, 24, 24, 24)    #dataset3
                # 融合公式
                # 保持之前的加权方向
                current_weight = avg_score / 100.0  # 高质量时接近1.0
                memory_weight = 1.0 - current_weight  # 低质量时接近1.0

                if avg_score > percentile_95:
                    # 融合（和之前代码一样）
                    updated_feat[b] = current_weight * curr_feature[0] + memory_weight * latent_restored[0]
                    updated_feat2[b] = current_weight * curr_feature2[0] + memory_weight * latent_restored2[0]  # 新增：第二个特征融合

                else:
                    updated_feat[b] = curr_feature[0]
                    updated_feat2[b] = curr_feature2[0]


                # 调试信息
                if high_quality_count > 0:
                    print(f"  Average score: {avg_score:.4f}, Alpha: {alpha:.4f}")
            else:
                # 如果特征库为空，使用当前特征
                updated_feat[b] = curr_feature[0]

        # 可选：限制历史评分列表的大小，避免内存无限增长
        MAX_HISTORY_SIZE = 10000
        if len(self.est_list) > MAX_HISTORY_SIZE:
            # 保留最近的部分评分
            self.est_list = self.est_list[-MAX_HISTORY_SIZE:]
            print(f"Trimmed est_list to {MAX_HISTORY_SIZE} entries")

        return updated_feat, updated_feat2

    def img_update_featbank(self, features, pred, est_WT, est_TC, est_ET, est_EC, est_avg, per_sample_est_list):
        """
        features: [B, C, D, H, W]
        per_sample_est_list: [est_patch1, est_patch2, ...] 来自 CSCS 的新返回值
        """
        # 1. 更新历史分位数统计
        for score in per_sample_est_list:
            self.est_list.append(score)

        # 计算 95 分位数，用于判断是否是“高质量特征”值得存入 Bank
        estp95 = np.percentile(self.est_list, 95) if len(self.est_list) > 0 else 0

        updated_feat = torch.zeros_like(features)
        B = features.shape[0]

        # 2. 逐样本处理
        for i in range(B):
            curr_est = per_sample_est_list[i]
            latent_model = features[i].unsqueeze(0)  # (1, C, D, H, W)
            b, c, d, h, w = latent_model.shape

            # 展平特征用于检索
            latent_flat = latent_model.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)  # (N, C)

            # 从特征池获取记忆特征 (这里通常返回相似度最高的 bank 特征)
            # 假设 self.pool.get_pool_feature 支持这种检索
            latent_flat_mem, _ = self.pool.get_pool_feature(latent_flat, None, top_k=self.max_len)

            # 3. 如果当前 Patch 极其不稳定（不一致性极高），则认为它捕捉到了“困难但真实”的分布，更新 Bank
            if curr_est > estp95:
                print(f"--> [High Uncertainty] Patch {i} (Score: {curr_est:.2f}) updating Feature Bank.")
                self.pool.update_feature_pool(latent_flat.detach())

            # 4. 还原特征形状
            latent_restored = latent_flat_mem.view(b, d * h * w, c).permute(0, 2, 1).view(b, c, d, h, w)

            # 5. 逐样本融合 (核心权重 alpha 差异化)
            # alpha 越小（不一致性低），越信任当前模型特征
            # alpha 越大（不一致性高），越信任记忆库特征（进行修正）
            alpha = float(curr_est) / 100.0
            alpha = min(max(alpha, 0.0), 1.0)  # 限制在 0-1

            # 融合公式：信任度加权
            updated_feat[i] = (1.0 - alpha) * features[i] + alpha * latent_restored[0]

        return updated_feat




