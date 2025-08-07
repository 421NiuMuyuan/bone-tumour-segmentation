import numpy as np, torch, os, glob
from torch.utils.data import DataLoader
from unet import UNet
from dataset import FemurSegmentationDataset
import config, albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ---------- 数据集 ----------
tf = A.Compose([A.Resize(512,512), A.Normalize((0.5,)*3,(0.5,)*3), ToTensorV2()])
ds = FemurSegmentationDataset(transform=tf)
print("总样本:", len(ds))
loader = DataLoader(ds, batch_size=2, shuffle=False)

# ---------- 选模型 ----------
ckpt = "best_unet.pth"   # 你自己的权重
if not os.path.isfile(ckpt):
    raise FileNotFoundError(f"模型权重 {ckpt} 不存在")
net = UNet(3, config.NUM_CLASSES).to(config.DEVICE)
net.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))

# ---------- 评估 ----------
inter = np.zeros(config.NUM_CLASSES); union = np.zeros_like(inter)
with torch.no_grad():
    for imgs, gts in tqdm(loader):
        preds = net(imgs.to(config.DEVICE)).argmax(1)
        gts   = gts.to(config.DEVICE)
        for c in range(config.NUM_CLASSES):
            p = (preds==c); g = (gts==c)
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
iou = inter/np.maximum(union,1)
print("PixelAcc:", inter.sum()/union.sum())
for i, v in enumerate(iou):
    print(f"class {i} IoU={v:.4f}")
print("mIoU(去背景):", iou[1:].mean())
