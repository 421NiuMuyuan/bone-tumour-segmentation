# Updated visualization scripts with metrics output
# 1. visualize_all_display.py (Bone-region segmentation)

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from dataset import FemurSegmentationDataset
from unet import UNet
import config

# 只做 Resize + Normalize，不做随机增强
transform = Compose([
    Resize(512, 512),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# 构造数据集
ds = FemurSegmentationDataset(transform=transform)
print(f">>> Total samples: {len(ds)}")

# 划分验证集
n_val = int(0.2 * len(ds))
n_train = len(ds) - n_val
_, val_ds = random_split(ds, [n_train, n_val])
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# 加载最优模型
model = UNet(in_channels=3, out_classes=config.NUM_CLASSES).to(config.DEVICE)
state = torch.load("best_unet.pth", map_location=config.DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

# 计算 Pixel Acc 和 per-class IoU

def compute_metrics(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(preds.flatten(), gts.flatten()):
        cm[g, p] += 1
    acc = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0
    ious = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return acc, ious

all_p, all_g = [], []
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs = imgs.to(config.DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_p.append(preds)
        all_g.append(masks.numpy())
all_p = np.concatenate(all_p)
all_g = np.concatenate(all_g)
acc, ious = compute_metrics(all_p, all_g, config.NUM_CLASSES)
print(f"[骨区分割] 验证集 Pixel Acc = {acc:.4f}")
for i, iou in enumerate(ious):
    print(f"  Class {i} IoU = {iou:.4f}")

# 可视化前 N 张
for idx in range(min(4, len(val_ds))):
    img, mask = val_ds[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(config.DEVICE))
        pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy()
    im_np = img.permute(1, 2, 0).cpu().numpy()
    im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax_in, ax_gt, ax_pr = axes
    ax_in.imshow(im_np); ax_in.set_title(f"Sample {idx} – Input"); ax_in.axis("off")
    cax_gt = ax_gt.imshow(mask.numpy(), cmap='nipy_spectral', vmin=0, vmax=config.NUM_CLASSES-1)
    ax_gt.set_title("Ground Truth"); ax_gt.axis("off"); fig.colorbar(cax_gt, ax=ax_gt, ticks=range(config.NUM_CLASSES), fraction=0.046, pad=0.04)
    cax_pr = ax_pr.imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=config.NUM_CLASSES-1)
    ax_pr.set_title("Prediction"); ax_pr.axis("off"); fig.colorbar(cax_pr, ax=ax_pr, ticks=range(config.NUM_CLASSES), fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
```

---

## 2. visualize_readable.py (Tumor segmentation prediction & GT)

```python
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from dataset_tumor import TumorSegmentationDataset
from unet_smp import get_model
import config_tumor as cfg

# 预处理
transform = Compose([Resize(512,512), Normalize(mean=(0.5,)*3, std=(0.5,)*3), ToTensorV2()])
# 数据集
ds = TumorSegmentationDataset(transform=transform, only_positive=False)
print(f">>> Total tumor samples: {len(ds)}")
# 划分验证集
n_val = int(0.2 * len(ds)); n_train = len(ds) - n_val
_, val_ds = random_split(ds, [n_train, n_val])
val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
# 加载模型
model = get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
ckpt = torch.load(cfg.MODEL_NAME, map_location=cfg.DEVICE)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()

# 计算指标

def compute_metrics(preds, gts, num_classes):
    cm = np.zeros((num_classes,num_classes), dtype=np.int64)
    for p,g in zip(preds.flatten(), gts.flatten()): cm[g,p]+=1
    acc = cm.trace()/cm.sum() if cm.sum()>0 else 0.0
    ious=[]
    for i in range(num_classes): tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; denom=tp+fp+fn; ious.append(tp/denom if denom>0 else 0.0)
    return acc, ious

all_p, all_g = [], []
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs = imgs.to(cfg.DEVICE)
        pred = model(imgs).argmax(1).cpu().numpy()
        all_p.append(pred); all_g.append(masks.numpy())
all_p = np.concatenate(all_p); all_g = np.concatenate(all_g)
acc, ious = compute_metrics(all_p, all_g, cfg.NUM_CLASSES)
print(f"[肿瘤分割] 验证集 Pixel Acc = {acc:.4f}")
for i,iou in enumerate(ious): print(f"  Class {i} IoU = {iou:.4f}")

# 随机展示几张预测 vs GT
def create_cmap(): return ListedColormap([[0,0,0],[1,0,0],[0,0,1]])

cmap = create_cmap()
for idx in random.sample(range(len(val_ds)), k=min(4, len(val_ds))):
    img, gt = val_ds[idx]
    with torch.no_grad(): pred = model(img.unsqueeze(0).to(cfg.DEVICE)).argmax(1).squeeze().cpu().numpy()
    im_np = img.permute(1,2,0).cpu().numpy(); im_np=(im_np-im_np.min())/(im_np.max()-im_np.min())
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(im_np); ax[0].set_title('Input'); ax[0].axis('off')
    ax[1].imshow(gt.numpy(), cmap=cmap, vmin=0, vmax=2); ax[1].set_title('GT'); ax[1].axis('off')
    ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=2); ax[2].set_title(f'Pred IoU={ious[1]:.3f}'); ax[2].axis('off')
    plt.tight_layout(); plt.show()
```

---

## 3. visualize_joint.py (Joint segmentation)

```python
import torch, random, numpy as np, matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from dataset_joint import JointSegmentationDataset
from unet_smp import get_model
import config_joint as cfg

# 预处理
transform = Compose([Resize(512,512), Normalize(mean=(0.5,)*3,std=(0.5,)*3), ToTensorV2()])
ds = JointSegmentationDataset(transform=transform, only_positive=False)
print(f">>> Total joint samples: {len(ds)}")
# 划分验证集
n_val=int(0.2*len(ds)); n_train=len(ds)-n_val
_, val_ds = random_split(ds, [n_train, n_val])
val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
# 加载模型
model=get_model(cfg.NUM_CLASSES).to(cfg.DEVICE)
ckpt=torch.load(cfg.MODEL_NAME,map_location=cfg.DEVICE)
model.load_state_dict(ckpt.get('model_state_dict',ckpt)); model.eval()

# 二分类指标

def calc_bin_metrics(pred, gt):
    cm=np.zeros((2,2),int)
    for p,g in zip(pred.flatten(),gt.flatten()): cm[g,p]+=1
    tp,tn,fp,fn=cm[1,1],cm[0,0],cm[0,1],cm[1,0]
    acc=(tp+tn)/(cm.sum())
    iou=tp/(tp+fp+fn) if tp+fp+fn>0 else 0
    return acc,iou

all_p,all_g=[],[]
with torch.no_grad():
    for imgs,masks in val_loader:
        preds=model(imgs.to(cfg.DEVICE)).argmax(1).cpu().numpy()
        all_p.append(preds); all_g.append(masks.numpy())
all_p=np.concatenate(all_p); all_g=np.concatenate(all_g)
acc,iou=calc_bin_metrics(all_p,all_g)
print(f"[关节分割] 验证集 Pixel Acc = {acc:.4f}, IoU = {iou:.4f}")

# 展示几张
for idx in random.sample(range(len(val_ds)),k=min(6,len(val_ds))):
    img,gt=val_ds[idx]
    with torch.no_grad(): pred=model(img.unsqueeze(0).to(cfg.DEVICE)).argmax(1).squeeze().cpu().numpy()
    im_np=(img.permute(1,2,0).cpu().numpy()+1)/2; im_np=np.clip(im_np,0,1)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(im_np); axs[0].set_title('Input'); axs[0].axis('off')
    axs[1].imshow(gt.numpy(),cmap='nipy_spectral',vmin=0,vmax=1); axs[1].set_title('GT'); axs[1].axis('off')
    axs[2].imshow(pred,cmap='nipy_spectral',vmin=0,vmax=1); axs[2].set_title(f'Pred IoU={iou:.3f}'); axs[2].axis('off')
    plt.tight_layout(); plt.show()
```

---

## 4. visualize_tumor_gt.py (Tumor Ground Truth)

```python
import os, cv2, random, numpy as np, matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import ListedColormap
import config_tumor as cfg
from dataset_tumor import TumorSegmentationDataset

# 可视化GT时，同时打印像素统计百分比

# 颜色映射
cmap = ListedColormap([[0,0,0],[1,0,0],[0,0,1]])

# 收集统计
stats_list=[]
for dir_ in [cfg.SURF_DIR, cfg.INBONE_DIR]:
    for f in os.listdir(dir_):
        if f.endswith('.png'):
            mask=cv2.imread(os.path.join(dir_,f),0)
            if mask is not None:
                total=mask.size
                nonzero=(mask>0).sum()
                stats_list.append((f, nonzero/total))
# 打印整体比例
ratios=[r for _,r in stats_list]
print(f"[肿瘤GT] 正样本平均Tumor占比: {np.mean(ratios)*100:.2f}%")

# 随机展示N张
pos=[i for i,s in enumerate(stats_list) if s[1]>0]
sel=random.sample(pos,k=min(6,len(pos)))
fig,axes=plt.subplots(len(sel),3,figsize=(12,4*len(sel)))
for i,si in enumerate(sel):
    fname,_ = stats_list[si]
    base=os.path.splitext(fname)[0]
    for ax,col,dir_ in zip(axes[i], ['Image','Surface','Merged'], [None,'SURF','INBONE']):
        if col=='Image':
            # 找图
            for sub in ("2","3"):
                p=f"{cfg.DATASET_ROOT}/{sub}/{base}.jpg"
                if os.path.isfile(p): img=cv2.imread(p); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); ax.imshow(img/255); break
            ax.set_title('X-ray');
        elif col=='Surface':
            surf=cv2.imread(os.path.join(cfg.SURF_DIR,fname),0)
            ax.imshow(surf,cmap=cmap,vmin=0,vmax=2)
            ax.set_title('Surface GT')
        else:
            surf=cv2.imread(os.path.join(cfg.SURF_DIR,fname),0)
            inb=cv2.imread(os.path.join(cfg.INBONE_DIR,fname),0)
            merged=surf.copy(); merged[inb>0]=2
            ax.imshow(merged,cmap=cmap,vmin=0,vmax=2)
            ax.set_title('Merged GT')
        ax.axis('off')
plt.tight_layout(); plt.show()
```

# End of updated scripts with metrics output
