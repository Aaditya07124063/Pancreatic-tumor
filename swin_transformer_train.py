"""
Swin Transformer - Pancreatic CT Binary Classification
Uses HuggingFace transformers (SwinForImageClassification)
Fine-tuned with 5-seed evaluation | 80/10/10 stratified split

Install: pip install transformers torch torchvision timm
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import SwinForImageClassification

from PIL import Image
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             cohen_kappa_score, confusion_matrix)

from data_utils import (CLASSES, SEEDS, EPOCHS, PATIENCE,
                        load_merged_dataset, stratified_split, print_split_summary)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
LR         = 5e-5
OUTPUT_DIR = "./swin_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
# ─── DATASET CLASS ────────────────────────────────────────────────────────────
class PancreaticDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        img   = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def get_transforms(augment=False):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])


# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model():
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    for param in model.parameters():
        param.requires_grad = True
    return model.to(DEVICE)


# ─── SET SEED ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── TRAIN / EVAL HELPERS ─────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, training=False):
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=imgs).logits
            loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


# ─── TRAIN ONE SEED ───────────────────────────────────────────────────────────
def train_one_seed(seed, all_paths, all_labels):
    set_seed(seed)

    tr_p, va_p, te_p, tr_l, va_l, te_l = stratified_split(all_paths, all_labels, seed)
    print_split_summary(tr_l, va_l, te_l)

    train_loader = DataLoader(
        PancreaticDataset(tr_p, tr_l, get_transforms(True)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(
        PancreaticDataset(va_p, va_l, get_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(
        PancreaticDataset(te_p, te_l, get_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_val_acc   = 0.0
    patience_count = 0
    best_state     = None
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, True)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader,   criterion, training=False)
        scheduler.step(va_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        print(f"  Epoch {epoch:03d} | tr_acc={tr_acc:.4f} va_acc={va_acc:.4f} "
              f"tr_loss={tr_loss:.4f} va_loss={va_loss:.4f}")

        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    tr_loss, train_acc, _, _          = run_epoch(
        model, DataLoader(PancreaticDataset(tr_p, tr_l, get_transforms(False)),
                          batch_size=BATCH_SIZE), criterion)
    va_loss, val_acc, _, _            = run_epoch(model, val_loader, criterion)
    te_loss, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)

    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, f"swin_seed{seed}.pt"))

    return {
        "seed": seed, "train_acc": train_acc, "val_acc": val_acc,
        "test_acc": test_acc, "test_loss": te_loss,
        "precision": prec, "recall": rec, "f1": f1,
        "kappa": kappa, "cm": cm, "history": history,
        "y_true": y_true, "y_pred": y_pred,
    }


# ─── PLOTS ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, seed, class_names=CLASSES):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set(xticks=list(ticks), yticks=list(ticks),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label",
           title=f"Swin Transformer Confusion Matrix (seed={seed})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_seed{seed}.png"), dpi=150)
    plt.close()


def plot_training_curves(history, seed):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_acc"], label="Train")
    ax1.plot(history["val_acc"],   label="Val")
    ax1.set(title=f"Accuracy (seed={seed})", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()
    ax2.plot(history["train_loss"], label="Train")
    ax2.plot(history["val_loss"],   label="Val")
    ax2.set(title=f"Loss (seed={seed})", xlabel="Epoch", ylabel="Loss")
    ax2.legend()
    plt.suptitle(f"Swin Transformer Training Curves — seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_seed{seed}.png"), dpi=150)
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  Swin Transformer — Pancreatic CT Classification  |  5-Seed Eval")
    print("=" * 68)

    all_paths, all_labels = load_merged_dataset(dedupe=True)
    print(f"Total deduplicated images: {len(all_paths)}  |  Classes: {CLASSES}\n")

    results = []
    for seed in SEEDS:
        print(f"\n{'─'*55}")
        print(f"  Training seed={seed}")
        print(f"{'─'*55}")
        res = train_one_seed(seed, all_paths, all_labels)
        results.append(res)
        plot_confusion_matrix(res["cm"], seed)
        plot_training_curves(res["history"], seed)
        print(f"  seed={seed} | test_acc={res['test_acc']:.4f} | "
              f"F1={res['f1']:.4f} | Kappa={res['kappa']:.4f}")

    cols = ["train_acc", "val_acc", "test_acc", "precision",
            "recall", "f1", "kappa", "test_loss"]

    print("\n" + "=" * 90)
    print(f"  {'Seed':<6} {'Train Acc':>10} {'Val Acc':>9} {'Test Acc':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Kappa':>8} {'Loss':>8}")
    print("=" * 90)
    for r in results:
        print(f"  {r['seed']:<6} {r['train_acc']:>10.4f} {r['val_acc']:>9.4f} "
              f"{r['test_acc']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['kappa']:>8.4f} {r['test_loss']:>8.4f}")

    avgs = {c: np.mean([r[c] for r in results]) for c in cols}
    print("─" * 90)
    print(f"  {'AVG':<6} {avgs['train_acc']:>10.4f} {avgs['val_acc']:>9.4f} "
          f"{avgs['test_acc']:>9.4f} {avgs['precision']:>10.4f} {avgs['recall']:>8.4f} "
          f"{avgs['f1']:>8.4f} {avgs['kappa']:>8.4f} {avgs['test_loss']:>8.4f}")
    print("=" * 90)

    rows = [{k: r[k] for k in cols + ["seed"]} for r in results]
    avg_row = {k: avgs[k] for k in cols}
    avg_row["seed"] = "AVG"
    rows.append(avg_row)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "swin_results.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()