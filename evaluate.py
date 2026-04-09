# evaluate.py — Evaluasi lengkap: Akurasi · F1 · Confusion Matrix · t-SNE
# python evaluate.py

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model import UsadaHybridModel

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EVALUASI] Device: {device}")

    if not os.path.exists(CLASS_NAMES_PATH):
        print("[ERROR] Jalankan train.py dulu.")
        return
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        idx_cls = {int(k): v for k, v in json.load(f).items()}
    cls_list = [idx_cls[i] for i in range(len(idx_cls))]
    n = len(cls_list)

    model = UsadaHybridModel(n, pretrained=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval().to(device)
    print(f"[MODEL] {n} spesies loaded")

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_dir = TEST_DIR if (os.path.exists(TEST_DIR) and
                            any(Path(TEST_DIR).iterdir()
                                for Path in [__import__('pathlib').Path])) \
               else VAL_DIR
    ds     = datasets.ImageFolder(eval_dir, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"[DATA]  {len(ds)} foto dari {os.path.basename(eval_dir)}/")

    # Hook untuk ambil features sebelum classifier (untuk t-SNE)
    feats_buf = []
    handle = model.head[0].register_forward_hook(
        lambda m, inp, out: feats_buf.append(inp[0].detach().cpu()))

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  Evaluating", ncols=65):
            imgs = imgs.to(device)
            out  = model(imgs)
            prob = F.softmax(out, 1)
            all_labels.extend(lbls.numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

    handle.remove()

    Y = np.array(all_labels)
    P = np.array(all_preds)

    acc = (Y == P).mean()
    print(f"\n{'='*55}")
    print(f"  Overall Accuracy: {acc*100:.2f}%")
    print(f"{'='*55}")

    report = classification_report(Y, P, target_names=cls_list, digits=4)
    print(report)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Simpan report teks
    with open(os.path.join(MODEL_DIR, "eval_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n{report}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm   = confusion_matrix(Y, P)
    fsz  = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fsz, fsz))
    ConfusionMatrixDisplay(cm, display_labels=[c.replace('_',' ')
                                               for c in cls_list]).plot(
        ax=ax, cmap='Blues', colorbar=False, xticks_rotation='vertical')
    ax.set_title(f'Confusion Matrix — Akurasi: {acc*100:.2f}%',
                 fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] confusion_matrix.png")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    all_feats = np.concatenate(feats_buf, axis=0) if feats_buf else None
    if all_feats is not None and len(all_feats) >= 10:
        print("[t-SNE] Menghitung visualisasi...")
        perp = min(30, max(5, len(all_feats) // 4))
        emb  = TSNE(n_components=2, perplexity=perp,
                    random_state=42, max_iter=1000).fit_transform(all_feats)

        fig, ax = plt.subplots(figsize=(13, 10))
        colors  = plt.cm.tab20(np.linspace(0, 1, n))
        for ci in range(n):
            m = Y == ci
            if m.sum() == 0: continue
            ax.scatter(emb[m,0], emb[m,1], c=[colors[ci]],
                       label=cls_list[ci].replace('_',' '),
                       alpha=0.75, s=55, edgecolors='none')
        ax.set_title('t-SNE Feature Separability — Spesies Usada Bali',
                     fontsize=13)
        ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left',
                  fontsize=9, ncol=1 if n<=20 else 2)
        ax.set_facecolor('#f7f7f7'); ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "tsne.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] tsne.png")

    # ── Per-class bar ─────────────────────────────────────────────────────────
    print("\n  Per-class Accuracy:")
    for ci in range(n):
        m   = Y == ci
        if not m.any(): continue
        ca  = (P[m]==ci).mean()
        bar = "█"*int(ca*20) + "░"*(20-int(ca*20))
        print(f"    {cls_list[ci]:<22} {bar} {ca*100:.1f}%")

    print(f"\n[SELESAI] Hasil disimpan di: {MODEL_DIR}")
    print("  Selanjutnya: python app.py")


if __name__ == "__main__":
    main()
