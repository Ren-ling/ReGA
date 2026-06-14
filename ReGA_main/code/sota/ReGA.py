from copy import deepcopy
import math
from xml.etree.ElementInclude import FatalIncludeError
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import torch
import torch.nn as nn
import torch.jit
from monai.losses import DiceLoss, DiceCELoss
import random
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
from utils.utils import rotate_single_random, derotate_single_random, add_gaussian_noise_3d
from robustbench.losses import WeightedCrossEntropyLoss, DiceCeLoss, DiceLoss, center_alignment_loss, KDLoss, \
    mmd_loss
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from .metric import cnh_loss_per_class, ih_loss,intra_organ_homogeneity_loss, compute_region_contrastive_loss

dicece_loss = DiceCeLoss(4)
from .CSCS import CSCS
from .TAFR import *


class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, anchor_model, optimizer, device, steps=1, episodic=False, mt_alpha=0.99,
                 rst_m=0.1):
        super().__init__()
        self.device = device  # 保存设备信息
        self.model = model.to(self.device)  # 将模型传送到指定设备
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.optimizer = optimizer
        self.model_ema = anchor_model.to(self.device)  # 将EMA模型也传送到指定设备
        self.do_adapt = True

        self.mc_iters = 6  # dropout 次数
        self.lambda_cons = 5e-4  # 一致性正则权重（推荐 1e-3 ~ 5e-2）
        self.num_classes = 4  # 类别数（硬编码为 4）
        self.mt = mt_alpha  # EMA 更新系数（教师模型更新率）
        self.rst = rst_m
        self.est = CSCS()  # CSCS 实例：计算预测不一致性（WT/TC/ET/平均）
        self.imgupdate = ImgUpdate()  # 图像特征更新模块（基于不一致性修正特征）


    def _enable_dropout_only(self, m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

    def forward(self, x):
        x = x.to(self.device)  # 确保输入张量在指定设备上
        # 推理模式：不更新
        if not self.do_adapt:
            return self.model(x)  # 这里别包 enable_grad
        if self.episodic:
            self.reset()
        for _ in range(self.steps):  # 固定执行 1 次适应（steps 参数未实际循环）
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)  # 执行前向+在线适应
        return outputs

    def set_adapt(self, flag: bool):
        self.do_adapt = flag

    @torch.no_grad()
    def forward_no_adapt(self, x):
        x = x.to(self.device)

        # ============ 新增：处理6D输入 ============
        if x.dim() == 6:
            # 6D输入: [batch_size, num_patches, C, D, H, W]
            batch_size, num_patches = x.shape[0], x.shape[1]
            print(f"forward_no_adapt: 6D输入, shape={x.shape}")

            all_outputs = []

            # 对每个patch独立推理
            for p in range(num_patches):
                patch_x = x[:, p]  # [batch_size, C, D, H, W]
                print(f"  patch {p}: 输入shape={patch_x.shape}")

                # 直接调用self.model进行推理（原来的逻辑）
                patch_output = self.model(patch_x)
                print(f"  patch {p}: 输出shape={patch_output.shape}")

                all_outputs.append(patch_output.unsqueeze(1))

            # 堆叠: [batch_size, num_patches, num_classes, D, H, W]
            outputs = torch.cat(all_outputs, dim=1)
            print(f"forward_no_adapt最终输出: shape={outputs.shape}")
            return outputs

        # ============ 原来的单patch逻辑保持不变 ============
        print(f"forward_no_adapt: 5D输入, shape={x.shape}")
        outputs = self.model(x)  # 直接使用当前模型预测
        print(f"forward_no_adapt输出: shape={outputs.shape}")
        return outputs



    torch.autograd.set_detect_anomaly(True)  # 开启异常检测（调试用，生产建议关闭）

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        scaler = torch.cuda.amp.GradScaler()
        x = x.to(self.device)

        print(f"\n=== forward_and_adapt开始 ===")
        print(f"输入形状: {x.shape}")

        # ============ 处理6D输入（多个patch） ============
        if x.dim() == 6:
            # 6D输入: [batch_size, num_patches, C, D, H, W]
            batch_size, num_patches = x.shape[0], x.shape[1]
            print(f"batch_size={batch_size}, num_patches={num_patches}")
            print(f"将对每个patch进行独立TTA更新")

            all_outputs = []
            total_loss_accumulated = torch.tensor(0.0, device=self.device)

            # 对每个patch独立进行TTA适应
            for p in range(num_patches):
                print(f"\n--- 处理patch {p}/{num_patches - 1} ---")

                # 提取当前patch
                patch_x = x[:, p]  # [batch_size, C, D, H, W]
                print(f"patch {p} 输入形状: {patch_x.shape}")

                # 对这个patch进行完整的TTA适应
                patch_output, patch_loss = self._adapt_single_patch_full_logic(
                    patch_x, model, optimizer, scaler, patch_idx=p
                )
                all_outputs.append(patch_output.unsqueeze(1))
                total_loss_accumulated = total_loss_accumulated + patch_loss

            # 反向传播（所有patch的损失累加）
            print(f"\n所有patch处理完成，总损失: {total_loss_accumulated.item():.4f}")
            print(f"进行反向传播...")

            optimizer.zero_grad()
            scaler.scale(total_loss_accumulated).backward()
            scaler.step(optimizer)
            scaler.update()

            # 堆叠所有patch的输出
            outputs = torch.cat(all_outputs, dim=1)  # [batch_size, num_patches, num_classes, D, H, W]
            print(f"最终输出形状: {outputs.shape}")
            return outputs
        # ============ 单patch情况（保持原来的完整逻辑） ============
        else:
            print(f"单patch输入: {x.shape}")

            # 直接使用你原来的完整逻辑（不通过_adapt_single_patch_full_logic）
            # 1) 获取当前预测
            pred_logits = self.forward_no_adapt(x)
            pred_batch = torch.argmax(pred_logits, dim=1).cpu().numpy()

            # 2) 调用ADIC获取dropout特征和评分
            with torch.no_grad():
                (est_WT, est_TC, est_ET, est_EC, est_avg, mismatch_mask,
                 entropy, entropy_map, mean_prob, batch_dropout_scores,
                 dropout_features, dropout_features2, dropout_hard_preds, boundary_info) = \
                    self.est.CSCS(input=x, pred=pred_batch, model=self.model, boundary_weight=0.3)

            adapt_alpha = est_avg / 100

            # 3) 使用ADIC返回的dropout特征更新特征库
            with (torch.cuda.amp.autocast()):
                current_features = model.get_feature(x)

                if dropout_features is not None:
                    updated_feat_3,updated_feat_32 = self.imgupdate.img_update_featbank_with_dropout_features(
                        dropout_features=dropout_features,
                        dropout_features2=dropout_features2,
                        dropout_scores=batch_dropout_scores,
                        current_pred=pred_batch,
                        n_iter=self.mc_iters
                    )
                else:
                    updated_feat_3 = current_features[-1] if isinstance(current_features, list) else current_features
                    updated_feat_32 = current_features[-3]
                # 确保特征转换为列表
                if isinstance(current_features, tuple):
                    # 如果是元组，转换为列表
                    feat_list = list(current_features)
                elif isinstance(current_features, list):
                    # 如果是列表，直接使用
                    feat_list = current_features
                else:
                    # 如果是单个特征，包装成列表
                    feat_list = [current_features]
                    print(f"⚠️ Feature wrapped to list, original type: {type(current_features)}")

                # 获取原始输出
                outputs = model.get_output(feat_list)

                # 获取更新后的输出
                if len(feat_list) > 0:
                    # 注意：current_features[:-1] 如果是元组会返回元组，所以要切片后转换
                    if isinstance(current_features, tuple):
                        updated_feat_list = list(current_features[:-3]) + [updated_feat_32]+list(current_features[-2]) +[updated_feat_3]
                    else:
                        updated_feat_list = feat_list[:-3] + [updated_feat_32]+ [feat_list[-2]]+[updated_feat_3]
                else:
                     # updated_feat_list = [updated_feat_3]
                    print("!!updated_feat_list!>0!!")

                updated_outputs = model.get_output(updated_feat_list)

                # 计算各个Loss
                with torch.no_grad():
                    standard_ema = self.model_ema(x)

                sem_loss = adapt_alpha * (softmax_entropy(outputs, updated_outputs).mean() +
                                          softmax_entropy(updated_outputs, outputs).mean()) / 2.0
                ce_loss = (softmax_entropy(outputs, standard_ema).mean() +
                           softmax_entropy(standard_ema, outputs).mean()) / 2.0

                prob_s = F.softmax(outputs, dim=1)
                cons_map = F.kl_div(torch.log(mean_prob.detach() + 1e-6), prob_s, reduction='none').sum(dim=1)
                mm = torch.from_numpy(mismatch_mask).to(self.device).float()
                cons_loss = (cons_map * mm).sum() / (mm.sum() + 1e-6) if mm.sum() > 100 else torch.tensor(0.0,
                                                                                                          device=self.device)
                fcl_loss = compute_region_contrastive_loss(
                    current_features[-1] if isinstance(current_features, list) else current_features,
                    pred_batch, entropy_map)

                ih = intra_organ_homogeneity_loss(outputs, window_size=3)
                topo_loss = adapt_alpha * 0.1 * ih

                total_loss =ce_loss + sem_loss +  topo_loss + self.lambda_cons * cons_loss + 0.01 * fcl_loss

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=adapt_alpha)

            del current_features, outputs, updated_outputs, standard_ema

            return model(x)

    def _adapt_single_patch_full_logic(self, patch_x, model, optimizer, scaler, patch_idx=0):
        """处理单个patch的完整TTA适应逻辑（保持所有原来的计算）"""
        # 1) 获取当前预测
        pred_logits = self.forward_no_adapt(patch_x)
        pred_batch = torch.argmax(pred_logits, dim=1).cpu().numpy()
        print(f"  patch {patch_idx} 预测形状: {pred_batch.shape}")

        # 2) 调用ADIC获取dropout特征和评分
        with torch.no_grad():
            (est_WT, est_TC, est_ET, est_EC, est_avg, mismatch_mask,
             entropy, entropy_map, mean_prob, batch_dropout_scores,
             dropout_features, dropout_features2, dropout_hard_preds, boundary_info) = \
                self.est.CSCS(input=patch_x, pred=pred_batch, model=self.model, boundary_weight=0.3)

        adapt_alpha = est_avg / 100
        print(f"  patch {patch_idx} ADIC评分: avg={est_avg:.2f}, alpha={adapt_alpha:.3f}")

        # 3) 使用ADIC返回的dropout特征更新特征库
        with torch.cuda.amp.autocast():
            # 获取当前特征（用于后续计算） - 这是关键修改！
            current_features = model.get_feature(patch_x)

            # 关键：使用ADIC的dropout特征更新bank
            if dropout_features is not None:
                updated_feat_3,updated_feat_32 = self.imgupdate.img_update_featbank_with_dropout_features(
                    dropout_features=dropout_features,  # (n_iter, B, C, D, H, W)
                    dropout_features2=dropout_features2,
                    dropout_scores=batch_dropout_scores,  # (B, n_iter, 4)
                    current_pred=pred_batch,  # (B, D, H, W)
                    n_iter=self.mc_iters
                )
            else:
                # 如果dropout_features为None，使用当前特征
                updated_feat_3 = current_features[-1] if isinstance(current_features, list) else current_features
                updated_feat_32 = current_features[-3]
            # 获取更新前的输出（用于计算loss）
            # 新代码（确保特征转换为列表）：
            if isinstance(current_features, tuple):
                # 如果是元组，转换为列表
                feat_list = list(current_features)
                print(f"  ⚠️ current_features是元组，已转换为列表，长度: {len(feat_list)}")
            elif isinstance(current_features, list):
                # 如果是列表，直接使用
                feat_list = current_features
                print(f"  ✅ current_features是列表，长度: {len(feat_list)}")
            else:
                # 如果是单个特征，包装成列表
                feat_list = [current_features]
                print(f"  ⚠️ Feature wrapped to list, 原始类型: {type(current_features)}")

            # 获取原始输出
            print(f"  调用get_output，输入类型: {type(feat_list)}, 长度: {len(feat_list)}")
            outputs = model.get_output(feat_list)

            # 准备更新后的特征列表
            if isinstance(current_features, tuple):
                # 重要：current_features[:-1] 返回的是元组切片，要转换为列表
                updated_feat_list = list(current_features[:-3]) + [updated_feat_32]+list(current_features[2]) [updated_feat_3]
            elif isinstance(current_features, list):
                updated_feat_list = current_features[:-3] + [updated_feat_32]+ [current_features[-2]] +[updated_feat_3]
            else:
                updated_feat_list = list(current_features)
                print("current_features不是tuple或者list")

            print(f"  调用get_output（更新后），输入类型: {type(updated_feat_list)}, 长度: {len(updated_feat_list)}")
            updated_outputs = model.get_output(updated_feat_list)

            # 3. 计算各个 Loss
            # (EMA 教师预测不需要梯度)
            with torch.no_grad():
                standard_ema = self.model_ema(patch_x)

            # SEM & CE Loss - 这里使用了outputs和updated_outputs
            sem_loss = adapt_alpha * (softmax_entropy(outputs, updated_outputs).mean() +
                                      softmax_entropy(updated_outputs, outputs).mean()) / 2.0
            ce_loss = (softmax_entropy(outputs, standard_ema).mean() +
                       softmax_entropy(standard_ema, outputs).mean()) / 2.0

            # Consistency Loss
            prob_s = F.softmax(outputs, dim=1)
            cons_map = F.kl_div(torch.log(mean_prob.detach() + 1e-6), prob_s, reduction='none').sum(dim=1)
            mm = torch.from_numpy(mismatch_mask).to(self.device).float()
            cons_loss = (cons_map * mm).sum() / (mm.sum() + 1e-6) if mm.sum() > 100 else torch.tensor(0.0,
                                                                                                      device=self.device)
            fcl_loss = compute_region_contrastive_loss(
                current_features[-1] if isinstance(current_features, list) else current_features,
                pred_batch, entropy_map)

            # Topology/Homogeneity Loss
            ih = intra_organ_homogeneity_loss(outputs, window_size=3)
            topo_loss = adapt_alpha * 0.1 * ih

            total_loss = sem_loss  +ce_loss+   topo_loss + self.lambda_cons * cons_loss + 0.01 * fcl_loss

            print(f"  patch {patch_idx} 损失: ce={ce_loss.item():.4f}, sem={sem_loss.item():.4f}, "
                  f"cons={cons_loss.item():.4f}, fcl={fcl_loss.item():.4f}, topo={topo_loss.item():.4f}, "
                  f"total={total_loss.item():.4f}")

        # EMA更新
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=adapt_alpha)

        print(f"  patch {patch_idx} TTA适应完成（梯度未更新，等待所有patch累加）")

        # 返回输出和损失（不在这里反向传播）
        return outputs, total_loss

    def _adapt_single_patch_full_logic_single(self, x, model, optimizer, scaler):
        """单patch情况的完整逻辑（直接反向传播）"""
        # 这里就是你原来的完整forward_and_adapt逻辑
        # 为了清晰，我把你原来的代码结构保持

        # 1) 获取当前预测（不使用dropout）
        pred_logits = self.forward_no_adapt(x)
        pred_batch = torch.argmax(pred_logits, dim=1).cpu().numpy()

        # 2) 调用ADIC获取dropout特征和评分（一次完成）
        with torch.no_grad():
            (est_WT, est_TC, est_ET, est_EC, est_avg, mismatch_mask,
             entropy, entropy_map, mean_prob, batch_dropout_scores,
             dropout_features, dropout_features2, dropout_hard_preds, boundary_info) = \
                self.est.CSCS(input=x, pred=pred_batch, model=self.model, boundary_weight=0.3)

        adapt_alpha = est_avg / 100

        # 3) 使用ADIC返回的dropout特征更新特征库
        with (torch.cuda.amp.autocast()):
            # 获取当前特征（用于后续计算） - 这是关键修改！
            current_features = model.get_feature(x)

            # 关键：使用ADIC的dropout特征更新bank
            if dropout_features is not None:
                updated_feat_3 = self.imgupdate.img_update_featbank_with_dropout_features(
                    dropout_features=dropout_features,  # (n_iter, B, C, D, H, W)
                    dropout_features2=dropout_features2,
                    dropout_scores=batch_dropout_scores,  # (B, n_iter, 4)
                    current_pred=pred_batch,  # (B, D, H, W)
                    n_iter=self.mc_iters
                )
            else:
                # 如果dropout_features为None，使用当前特征
                updated_feat_3 = current_features[-1] if isinstance(current_features, list) else current_features

            # 获取更新前的输出（用于计算loss）
            if isinstance(current_features, list):
                outputs = model.get_output(current_features)  # 使用原始特征获取输出
                # 获取更新后的输出
                updated_outputs = model.get_output(current_features[:-1] + [updated_feat_3])
                # updated_outputs = model.get_output(current_features)
            else:
                # 如果特征不是列表，可能需要其他处理
                print("!!!!!!Feature is not a list!!!!!")
                outputs = model.get_output(current_features)
                updated_outputs = model.get_output([updated_feat_3])
                # updated_outputs = model.get_output(current_features)

            # 3. 计算各个 Loss
            # (EMA 教师预测不需要梯度)
            with torch.no_grad():
                standard_ema = self.model_ema(x)

            # SEM & CE Loss - 这里使用了outputs和updated_outputs
            sem_loss = adapt_alpha * (softmax_entropy(outputs, updated_outputs).mean() +
                                      softmax_entropy(updated_outputs, outputs).mean()) / 2.0
            ce_loss = (softmax_entropy(outputs, standard_ema).mean() +
                       softmax_entropy(standard_ema, outputs).mean()) / 2.0

            # Consistency Loss
            prob_s = F.softmax(outputs, dim=1)
            cons_map = F.kl_div(torch.log(mean_prob.detach() + 1e-6), prob_s, reduction='none').sum(dim=1)
            mm = torch.from_numpy(mismatch_mask).to(self.device).float()
            cons_loss = (cons_map * mm).sum() / (mm.sum() + 1e-6) if mm.sum() > 100 else torch.tensor(0.0,
                                                                                                      device=self.device)

            fcl_loss = compute_region_contrastive_loss(
                current_features[-1] if isinstance(current_features, list) else current_features,
                pred_batch, entropy_map)

            # Topology/Homogeneity Loss
            ih = intra_organ_homogeneity_loss(outputs, window_size=3)
            topo_loss = adapt_alpha * 0.1 * ih

            total_loss =sem_loss +ce_loss +   topo_loss + self.lambda_cons * cons_loss + 0.01 * fcl_loss


            # 只返回输出和损失，不在这个方法中更新梯度
            return outputs, total_loss



def softmax_entropy(x, x_ema):
    nan_x = torch.isnan(x).sum().item()
    inf_x = torch.isinf(x).sum().item()
    nan_e = torch.isnan(x_ema).sum().item()
    inf_e = torch.isinf(x_ema).sum().item()

    p = x_ema.softmax(1)
    logq = x.log_softmax(1)

    nan_p = torch.isnan(p).sum().item()
    inf_p = torch.isinf(p).sum().item()
    nan_l = torch.isnan(logq).sum().item()
    inf_l = torch.isinf(logq).sum().item()
    if nan_p or inf_p: print(f"p bad: nan={nan_p}, inf={inf_p}")
    if nan_l or inf_l: print(f"logq bad: nan={nan_l}, inf={inf_l}")

    return -(p * logq).mean()


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if True:#isinstance(m, nn.BatchNorm2d): collect all
        if 'dec1.last' not in nm:
            print(nm, '55', m, '496')
            for np, p in m.named_parameters():

                if np in ['weight', 'bias'] and p.requires_grad:
                    # if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = (1 - alpha_teacher) * ema_param[:].data[:] + (alpha_teacher) * param[:].data[:]
    return ema_model
