# export_cascade_fullres.py
import torch
from nnunet.training.model_restore import load_model_and_checkpoint_files

if __name__ == "__main__":
    model_folder = "/mnt/newdisk/CTPelvic1k/all_data/nnUNet/nnUNet_results_folder/CTPelvic1K/3d_cascade_fullres/Task11_CTPelvic1K/nnUNetTrainerCascadeFullRes__nnUNetPlans"
    folds = [0]   # 你现在只用 fold0

    trainer, params = load_model_and_checkpoint_files(model_folder, folds)
    net = trainer.network      # 这就是训练好的 3d_cascade_fullres

    # 如果你想把 fold0 的权重固化进去，可以直接：
    trainer.load_checkpoint_ram(params[0], False)  # 加载 fold0 权重到 net

    # 保存成纯 state_dict
    torch.save(net.state_dict(), "cascade_fullres_CTpelvic_fold0.pth")

    print("saved to cascade_fullres_CTpelvic_fold0.pth")
