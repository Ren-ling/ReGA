from .unet_3D import unet_3D
from .unet_3D_vptta import unet_3D_vptta
from .vnet import VNet
from .VoxResNet import VoxResNet
from .attention_unet import Attention_UNet

from .nnunet1 import initialize_network
from .nnunet_cascade_fullres import CascadeFullResUNet


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, **kwargs):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "unet_3D_vptta":
        net = unet_3D_vptta(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64, class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "cascade_fullres":
        # 从 kwargs 获取 plans_path
        plans_path = kwargs["plans_path"]
        net = CascadeFullResUNet(plans_path).cuda()
    else:
        net = None
    return net
