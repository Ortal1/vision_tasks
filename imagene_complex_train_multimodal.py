#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, random, argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

from torch.utils.tensorboard import SummaryWriter

# =============== Utils ===============

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# =============== Dataset ===============

class IDCDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row.id}.png"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        gender = 1.0 if str(row.gender).upper().startswith("M") else 0.0
        clin = torch.tensor([float(row.age)/100.0, gender, float(row.smoker)], dtype=torch.float32)
        label = torch.tensor(int(row.label), dtype=torch.long)
        return image, clin, label

# =============== Model ===============

class MultiModalNet(nn.Module):
    def __init__(self, img_backbone="resnet18", img_embed=128, clin_in=3, clin_embed=32, num_classes=2):
        super().__init__()
        if img_backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.cnn = base
        elif img_backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            out_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.cnn = base
        elif img_backbone == "vit_b_16":
            base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            out_dim = base.heads.head.in_features
            base.heads = nn.Identity()
            self.cnn = base
        else:
            raise ValueError("Unknown backbone")

        self.img_fc  = nn.Linear(out_dim, img_embed)
        self.clin_fc = nn.Sequential(
            nn.Linear(clin_in, 16),
            nn.ReLU(),
            nn.Linear(16, clin_embed)
        )
        self.classifier = nn.Sequential(
            nn.Linear(img_embed + clin_embed, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, clin):
        x_img  = F.relu(self.img_fc(self.cnn(img)))
        x_clin = self.clin_fc(clin)
        x = torch.cat([x_img, x_clin], dim=1)
        return self.classifier(x)

# =============== Training / Eval helpers ===============

def get_loaders(data_dir, img_size, batch_size, num_workers=2):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # נרמול תואם-ImageNet כדי להתאים למשקלים המוכנים מראש
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_df = pd.read_csv(Path(data_dir)/"train.csv")
    val_df   = pd.read_csv(Path(data_dir)/"val.csv")
    train_ds = IDCDataset(Path(data_dir)/"train.csv", Path(data_dir)/"images", tfm)
    val_ds   = IDCDataset(Path(data_dir)/"val.csv",   Path(data_dir)/"images", tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, (train_df, val_df)

def compute_class_weights(df):
    # לשימוש באי-איזון מחלקות
    counts = df["label"].value_counts().to_dict()
    # ודא שיש 0 ו-1
    n0, n1 = counts.get(0,1), counts.get(1,1)
    total = n0 + n1
    w0 = total/(2*n0); w1 = total/(2*n1)
    return torch.tensor([w0, w1], dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, log_step=None, writer=None, epoch=0):
    model.train()
    running_loss = 0.0
    n = 0
    for step, (img, clin, y) in enumerate(loader):
        img, clin, y = img.to(device), clin.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(img, clin)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(img, clin)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        running_loss += loss.item() * img.size(0)
        n += img.size(0)
        if writer and log_step and (step % log_step == 0):
            writer.add_scalar("train/step_loss", loss.item(), epoch*len(loader)+step)
    return running_loss / max(1,n)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    ys, ps, losses = [], [], []
    for img, clin, y in loader:
        img, clin, y = img.to(device), clin.to(device), y.to(device)
        logits = model(img, clin)
        prob = torch.softmax(logits, dim=1)[:,1]
        loss = criterion(logits, y)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(prob.cpu().numpy().tolist())
        losses.append(loss.item()*img.size(0))
    ys = np.array(ys); ps = np.array(ps)
    val_loss = np.sum(losses) / max(1,len(ys))
    acc = accuracy_score(ys, (ps>=0.5).astype(int))
    try:
        auc = roc_auc_score(ys, ps)
        ap  = average_precision_score(ys, ps)
    except ValueError:
        auc, ap = float("nan"), float("nan")
    return {"loss": val_loss, "acc": acc, "auc": auc, "ap": ap, "ys": ys, "ps": ps}

def plot_and_save_curves(out_dir, eval_out):
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ys, ps = eval_out["ys"], eval_out["ps"]

    # ROC
    fpr, tpr, _ = roc_curve(ys, ps)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={eval_out['auc']:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.savefig(out_dir/"roc_curve.png", dpi=160)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(ys, ps)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={eval_out['ap']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend(loc="lower left")
    plt.savefig(out_dir/"pr_curve.png", dpi=160)
    plt.close()

    # Confusion Matrix (עם סף 0.5)
    import seaborn as sns  # אם אין, אפשר להחליף ל-matplotlib בלבד
    cm = confusion_matrix(ys, (ps>=0.5).astype(int))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (thr=0.5)")
    plt.savefig(out_dir/"confusion_matrix.png", dpi=160)
    plt.close()

# =============== Inference ===============

@torch.no_grad()
def infer_single_image(model, img_path, clin_vec, img_size, device):
    """ clin_vec = [age/100.0, gender(0/1), smoker(0/1)] """
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    im = Image.open(img_path).convert("RGB")
    im = tfm(im).unsqueeze(0).to(device)
    clin = torch.tensor([clin_vec], dtype=torch.float32).to(device)
    logits = model(im, clin)
    prob = torch.softmax(logits, dim=1)[0,1].item()
    pred = int(prob >= 0.5)
    return {"prob_cancer": prob, "pred": pred}

# =============== Main train ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="folder with images/ and train.csv,val.csv")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","efficientnet_b0","vit_b_16"])
    parser.add_argument("--out_dir", type=str, default="runs/exp1")
    parser.add_argument("--early_patience", type=int, default=5)
    parser.add_argument("--log_step", type=int, default=50)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    save_json(Path(args.out_dir)/"config.json", vars(args))
    writer = SummaryWriter(log_dir=args.out_dir)

    # loaders
    train_loader, val_loader, (train_df, val_df) = get_loaders(args.data_dir, args.img_size, args.batch_size)
    # criterion
    if args.use_class_weights:
        cw = compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        print(f"Using class weights: {cw.cpu().numpy().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # model
    model = MultiModalNet(img_backbone=args.backbone).to(device)
    if args.freeze_backbone:
        for p in model.cnn.parameters():
            p.requires_grad = False
        print("Backbone frozen.")
    print(f"Trainable params: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_auc = -1.0
    best_path = Path(args.out_dir)/"best.pt"
    last_path = Path(args.out_dir)/"last.pt"
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, args.log_step, writer, epoch-1)
        eval_out = evaluate(model, val_loader, criterion, device)
        dt = time.time()-t0

        # TensorBoard scalars
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss",   eval_out["loss"], epoch)
        writer.add_scalar("val/acc",    eval_out["acc"],  epoch)
        if not math.isnan(eval_out["auc"]): writer.add_scalar("val/auc", eval_out["auc"], epoch)
        if not math.isnan(eval_out["ap"]):  writer.add_scalar("val/ap",  eval_out["ap"],  epoch)

        # Console
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | val_loss={eval_out['loss']:.4f} "
              f"| acc={eval_out['acc']:.3f} | auc={eval_out['auc']:.3f} | ap={eval_out['ap']:.3f} "
              f"| {dt:.1f}s")

        # Save last
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, last_path)

        # EarlyStopping & Best checkpoint by AUC
        current_auc = eval_out["auc"] if not math.isnan(eval_out["auc"]) else -1.0
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_auc": best_auc}, best_path)
            epochs_no_improve = 0
            # Also dump plots each time best improves
            plot_and_save_curves(args.out_dir, eval_out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_patience:
                print(f"Early stopping at epoch {epoch} (no AUC improvement for {args.early_patience} epochs).")
                break

    writer.close()
    print(f"Best AUC: {best_auc:.3f}  | Best checkpoint: {best_path}")

    # Final eval with best model reloaded (for determinism in reports)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        final_out = evaluate(model, val_loader, criterion, device)
        # save report
        y_true = final_out["ys"]; y_prob = final_out["ps"]; y_pred = (y_prob>=0.5).astype(int)
        rep = classification_report(y_true, y_pred, target_names=["class0","class1"], digits=3, output_dict=True)
        save_json(Path(args.out_dir)/"classification_report.json", rep)
        print("Saved classification_report.json")
        # re-plot curves to be sure
        plot_and_save_curves(args.out_dir, final_out)

if __name__ == "__main__":
    main()
