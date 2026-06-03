import os
import time
import warnings
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from sod_metrics import MAE, Smeasure

from model.WPFormer6 import WPFormer

from ESDI_dataloader import get_loader2
from data_loader import get_loader


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        pt = torch.exp(-BCE_loss) # pt = p if y=1, pt = 1-p if y=0
        
        # 计算 Focal Term
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def total_loss_new_focal(pred, mask):
    focal = FocalLoss(alpha=0.5, gamma=2)(pred, mask) 
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask).sum((2, 3))
    union = pred_sig.sum((2, 3)) + mask.sum((2, 3))
    dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
    return focal + dice.mean()

def init_logger(save_dir):
    log_path = os.path.join(save_dir, "train.log")

    def log(msg):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return log


def save_val_metrics(save_dir, epoch, metrics):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Warning: Directory {save_dir} did not exist, created it.")

    csv_path = os.path.join(save_dir, "val_metrics.csv")
    write_head = not os.path.exists(csv_path)

    try:
        with open(csv_path, "a") as f:
            if write_head:
                f.write("epoch,MAE,S\n")
            f.write(
                f"{epoch},"
                f"{metrics['MAE']:.6f},"
                f"{metrics['S']:.6f}\n"
            )
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")

class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, transform):
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transform

        self.images = sorted([f for f in os.listdir(image_root)
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.gts = sorted([f for f in os.listdir(gt_root)
                           if f.endswith(('.jpg', '.png', '.jpeg'))])

        assert len(self.images) == len(self.gts)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.images[idx])
        gt_path = os.path.join(self.gt_root, self.gts[idx])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = self.transform(img)

        return {"image": img, "h": h, "w": w, "gt_path": gt_path}

def eval_sod_metrics(img_root, gt_root, model, device):
    model.eval()

    SM = Smeasure()
    M = MAE()

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    batch_size = 32 if torch.cuda.device_count() > 1 else 1

    loader = DataLoader(
        TestDataset(img_root, gt_root, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    results = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            hs, ws = batch["h"], batch["w"]
            paths = batch["gt_path"]

            preds = torch.sigmoid(model(imgs)[-1]).cpu().numpy()

            for i in range(len(imgs)):
                p = preds[i].squeeze()
                p = (p - p.min()) / (p.max() - p.min() + 1e-8)
                results.append((p, hs[i], ws[i], paths[i]))

    #for p, h, w, gt_path in tqdm(results, desc="指标计算"):
    for p, h, w, gt_path in results:
        p = Image.fromarray((p * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        p = np.array(p)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        SM.step(p, gt)
        M.step(p, gt)

    return {
        "MAE": M.get_results()["mae"],
        "S": SM.get_results()["sm"]
    }


# ====================== Loss ======================
def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce = nn.BCELoss()(pred, mask)

    inter = (pred * mask).sum((2, 3))
    union = (pred + mask).sum((2, 3))
    iou = 1 - (inter + 1e-6) / (union - inter + 1e-6)

    return bce + iou.mean()


def total_loss_new(pred, mask):
    # 使用 Logits 提高数值稳定性,但不加权重
    bce = nn.BCEWithLogitsLoss()(pred, mask)
    
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask).sum((2, 3))
    union = pred_sig.sum((2, 3)) + mask.sum((2, 3))
    dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
    
    return bce + dice.mean()


# ====================== 训练主函数 ======================
def train(model_name, dataset_name, comment,base_lr,base_batch,epoch_cfg=None):

    comment=comment + str(base_lr)

    run_time = time.strftime("%Y%m%d-%H%M%S")
    save_root = f"./ablationruns_esdis/{run_time}_{comment}"
    ckpt_dir = os.path.join(save_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log = init_logger(save_root)
    writer = SummaryWriter(log_dir=save_root)

    gpu_cnt = torch.cuda.device_count()
    actual_batch = base_batch * max(1, gpu_cnt)
    actual_lr = base_lr * (gpu_cnt ** 0.5)

    total_epoch, val_start = epoch_cfg.get(dataset_name, (24, 20))

    log(f"Start Training | Dataset={dataset_name}, Comment={comment}")
    log(f"Batch={actual_batch}, LR={actual_lr:.2e}, GPU={gpu_cnt}")

    model = WPFormer(method="pvt_v2_b2", channel=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gpu_cnt > 1:
        model = nn.DataParallel(model)
    model.to(device)

    root = "/raid/jzb4/WPFormer/datasets"
    if dataset_name=="ESDIs-SOD":
        print('ESDIs_SOD_dataloader')
        train_loader = get_loader2(
            os.path.join(root, dataset_name, "train/images"),
            os.path.join(root, dataset_name, "train/gt"),
            batchsize=actual_batch,
            trainsize=384,
            is_train=True,
            datasetname_=dataset_name
            )
    else:
        train_loader = get_loader(
            os.path.join(root, dataset_name, "train/images"),
            os.path.join(root, dataset_name, "train/gt"),
            batchsize=actual_batch,
            trainsize=384,
            is_train=True,
            datasetname_=dataset_name
            )


    

    optimizer = optim.Adam(model.parameters(), lr=actual_lr)

    scheduler = CosineAnnealingLR(optimizer, total_epoch, eta_min=1e-7)

    best_sm = 0

    for epoch in range(total_epoch):
        model.train()
        loss_sum = 0
        #loader = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epoch-1}")
        #for data in loader:
        for data in train_loader:
            img = data["image"].to(device)
            gt = data["label"].to(device)

            optimizer.zero_grad()
            preds = model(img)
            loss = sum(total_loss_new_focal(p, gt) for p in preds)

            '''
            weights = [0.5, 0.7, 1.0]  # 按你的输出层数调整
            loss = sum(w * total_loss_new(p, gt) for w, p in zip(weights, preds))
            '''

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            #loader.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = loss_sum / len(train_loader)
        lr_now = optimizer.param_groups[0]["lr"]

        log(f"Epoch {epoch} | Train Avg Loss={avg_loss:.6f}")
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/LR", lr_now, epoch)

        if epoch + 1 >= val_start or (epoch + 1)%10==0:
            metrics = eval_sod_metrics(
                os.path.join(root, dataset_name, "test/images"),
                os.path.join(root, dataset_name, "test/gt"),
                model, device
            )

            save_val_metrics(save_root, epoch, metrics)

            writer.add_scalar("Val/MAE", metrics["MAE"], epoch)
            writer.add_scalar("Val/S", metrics["S"], epoch)

            log(
                f"Val Epoch {epoch} | "
                f"MAE={metrics['MAE']:.4f}, "
                f"S={metrics['S']:.4f}"
            )

            if metrics["S"] > best_sm:
                best_sm = metrics["S"]
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state, os.path.join(ckpt_dir, f"{best_sm:.8f}_{model_name}_best.pth"))
                log(f"✨Best Model Updated! | S={best_sm:.4f}")

    writer.close()



if __name__ == "__main__":
    '''
    ZJU-Leaper
    CrackSeg9k
    ESDIs-SOD
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    dataset="ESDIs-SOD"
    modelname='WPFormer6'
    message='WPFormer'
    bs=32
    lr=1.6e-4
    print(f"{modelname}_{message}_{dataset}_{lr}")
    train(
        model_name=modelname,
        dataset_name=dataset,
        comment=f"{modelname}_{message}_{dataset}_{lr}",
        base_batch = bs,
        base_lr = lr,
        epoch_cfg = {
            "CrackSeg9k": (250, 100),
            "ZJU-Leaper": (250, 100),
            "ESDIs-SOD": (350, 150)
        }
    )
    # 1.2e-4  0.00012  原始
    # 1.6e-4  0.00016  普遍这个
    # 3.2e-4  0.00032
    # 1.28e-3  最大