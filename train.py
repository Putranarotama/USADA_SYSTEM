# train.py
# Jalankan SETELAH: python utils/augment_pipeline.py
# python train.py

import os, sys, json, time, logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model import get_model

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "train_log.txt"), encoding="utf-8"),
    ]
)
log = logging.getLogger()

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_loaders():
    if not Path(TRAIN_DIR).exists() or not any(Path(TRAIN_DIR).iterdir()):
        log.error(f"Folder train kosong atau tidak ada: {TRAIN_DIR}")
        log.error("Jalankan dulu: python utils/augment_pipeline.py")
        sys.exit(1)

    tr_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])
    ev_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tr_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=ev_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=ev_tf)

    # Simpan class names
    idx_to_cls = {v: k for k, v in train_ds.class_to_idx.items()}
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_cls, f, indent=2, ensure_ascii=False)

    n = len(train_ds.classes)
    log.info(f"  Spesies     : {n}")
    log.info(f"  Train (aug) : {len(train_ds):,} foto")
    log.info(f"  Val         : {len(val_ds):,} foto")
    log.info(f"  Test        : {len(test_ds):,} foto")

    # Weighted sampler (tangani class imbalance)
    counts  = [0]*n
    for _, lbl in train_ds.samples: counts[lbl] += 1
    weights = [1.0/counts[l] for _, l in train_ds.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE,
                   sampler=sampler, num_workers=0),
        DataLoader(val_ds,   batch_size=BATCH_SIZE,
                   shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=BATCH_SIZE,
                   shuffle=False, num_workers=0),
        n,
    )


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item()
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), correct / total


def run_phase(phase_num, model, tr_ld, vl_ld,
              n_ep, lr, device, history, best_acc):
    crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt   = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    pat   = 0

    for ep in range(1, n_ep + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, tr_ld, opt, crit, device)
        vl, va = eval_epoch(model, vl_ld, crit, device)
        sched.step()

        history["train_loss"].append(round(tl, 5))
        history["train_acc"].append(round(ta, 5))
        history["val_loss"].append(round(vl, 5))
        history["val_acc"].append(round(va, 5))

        star = ""
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            star = "  ← BEST"
            pat  = 0
        else:
            pat += 1

        log.info(
            f"  P{phase_num}[{ep:3d}/{n_ep}] "
            f"Loss {tl:.4f}/{vl:.4f}  "
            f"Acc {ta*100:.1f}%/{va*100:.1f}%  "
            f"LR {sched.get_last_lr()[0]:.1e}  "
            f"{time.time()-t0:.0f}s{star}")

        if pat >= EARLY_STOP_PATIENCE:
            log.info(f"  [Early Stop] {EARLY_STOP_PATIENCE} epoch tanpa peningkatan")
            break

    torch.save(model.state_dict(), MODEL_PATH)
    return best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("")
    log.info("=" * 62)
    log.info("  TRAINING — Tanaman Usada Bali")
    log.info("=" * 62)
    log.info(f"  Device : {device}")
    log.info("")

    tr_ld, vl_ld, te_ld, n_cls = build_loaders()
    model = get_model(n_cls, DROPOUT_RATE, pretrained=True, device=str(device))

    hist = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best = 0.0

    # Tahap 1 — hanya head
    log.info("─ TAHAP 1/3: Head only (backbone frozen) ─────────────────")
    best = run_phase(1, model, tr_ld, vl_ld,
                     PHASE1_EPOCHS, LEARNING_RATE*5, device, hist, best)

    # Tahap 2 — partial unfreeze
    log.info("")
    log.info("─ TAHAP 2/3: Partial unfreeze ────────────────────────────")
    model.unfreeze_partial()
    log.info(f"  Trainable params: {model.get_trainable_params():,}")
    best = run_phase(2, model, tr_ld, vl_ld,
                     PHASE2_EPOCHS, LEARNING_RATE, device, hist, best)

    # Tahap 3 — full unfreeze
    log.info("")
    log.info("─ TAHAP 3/3: Full unfreeze ───────────────────────────────")
    model.unfreeze_all()
    log.info(f"  Trainable params: {model.get_trainable_params():,}")
    best = run_phase(3, model, tr_ld, vl_ld,
                     PHASE3_EPOCHS, LEARNING_RATE*0.1, device, hist, best)

    # Test set
    log.info("")
    log.info("─ EVALUASI FINAL: Test Set ───────────────────────────────")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    crit = nn.CrossEntropyLoss()
    tl, ta = eval_epoch(model, te_ld, crit, device)
    log.info(f"  Test Loss : {tl:.4f}")
    log.info(f"  Test Acc  : {ta*100:.2f}%")

    with open(HISTORY_PATH, "w") as f:
        json.dump(hist, f, indent=2)

    log.info("")
    log.info("=" * 62)
    log.info(f"  SELESAI — Best Val Acc: {best*100:.2f}%")
    log.info(f"  Model: {BEST_MODEL_PATH}")
    log.info("=" * 62)
    log.info("  Selanjutnya:")
    log.info("    python evaluate.py   ← confusion matrix + t-SNE")
    log.info("    python app.py        ← web demo")


if __name__ == "__main__":
    main()
