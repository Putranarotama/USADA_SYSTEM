# utils/xai.py — Grad-CAM · LIME · t-SNE

import os, sys, uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, RESULT_DIR


# ─── GRAD-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Ref: Selvaraju et al. 2020, IJCV 128:336-359
    """

    def __init__(self, model):
        self.model    = model
        self._features = None
        self._grads    = None
        self._hooks    = []

        target = self._last_conv(model.effnet)
        if target:
            self._hooks.append(
                target.register_forward_hook(
                    lambda m, i, o: setattr(self, '_features', o.detach())))
            self._hooks.append(
                target.register_full_backward_hook(
                    lambda m, gi, go: setattr(self, '_grads', go[0].detach())))

    def _last_conv(self, module):
        last = None
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        return last

    def generate(self, tensor: torch.Tensor,
                 class_idx: int, orig: Image.Image) -> np.ndarray:
        self.model.eval()
        t = tensor.clone().requires_grad_(True)
        out = self.model(t)
        self.model.zero_grad()
        oh = torch.zeros_like(out)
        oh[0, class_idx] = 1.0
        out.backward(gradient=oh, retain_graph=True)

        if self._features is None or self._grads is None:
            return np.array(orig.resize((IMAGE_SIZE, IMAGE_SIZE)))

        w   = self._grads.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * self._features).sum(1, keepdim=True))
        cam = F.interpolate(cam, (IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Overlay
        orig_np  = np.array(orig.resize((IMAGE_SIZE, IMAGE_SIZE)))
        heatmap  = np.uint8(255 * cm.jet(cam)[:,:,:3])
        overlay  = cv2.addWeighted(orig_np, 0.55, heatmap, 0.45, 0)
        return overlay

    def remove(self):
        for h in self._hooks: h.remove()


# ─── LIME (lightweight, no extra lib) ─────────────────────────────────────────
def lime_explain(model, tensor: torch.Tensor,
                 orig: Image.Image, class_idx: int,
                 device: str = "cpu",
                 n_segments: int = 40) -> np.ndarray:
    """
    Simple LIME via superpixel masking.
    Ref: Ribeiro et al. 2016, KDD.
    """
    try:
        from skimage.segmentation import slic
    except ImportError:
        print("[LIME] scikit-image tidak terinstall — skip LIME")
        return np.array(orig.resize((IMAGE_SIZE, IMAGE_SIZE)))

    from torchvision import transforms
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    preproc = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    img_np   = np.array(orig.resize((IMAGE_SIZE, IMAGE_SIZE)))
    segments = slic(img_np, n_segments=n_segments, compactness=10,
                    sigma=1, start_label=0)
    n_segs   = segments.max() + 1

    model.eval()
    with torch.no_grad():
        base_score = torch.softmax(model(tensor), 1)[0, class_idx].item()

    contribs = np.zeros(n_segs)
    for sid in range(n_segs):
        masked          = img_np.copy()
        masked[segments==sid] = 200   # abu-abu terang
        t = preproc(Image.fromarray(masked)).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.softmax(model(t), 1)[0, class_idx].item()
        contribs[sid] = base_score - score

    result = img_np.copy().astype(float)
    for sid in range(n_segs):
        mask = segments == sid
        if contribs[sid] > 0:
            result[mask, 1] = np.clip(
                result[mask, 1] + contribs[sid] * 180, 0, 255)
        else:
            result[mask, 0] = np.clip(
                result[mask, 0] + abs(contribs[sid]) * 180, 0, 255)
    return result.astype(np.uint8)


# ─── SAVE COMPOSITE XAI FIGURE ─────────────────────────────────────────────────
def save_xai(orig: Image.Image, gradcam_arr: np.ndarray,
             lime_arr: np.ndarray, species: str,
             conf: float) -> str:
    """Simpan 3-panel: original | Grad-CAM | LIME"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    fname = f"xai_{uuid.uuid4().hex[:8]}.jpg"
    path  = os.path.join(RESULT_DIR, fname)

    orig_arr = np.array(orig.resize((IMAGE_SIZE, IMAGE_SIZE)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#111111')

    titles = ["Gambar Asli", "Grad-CAM\n(Area kunci identifikasi)",
              "LIME\n(Hijau = kontribusi positif)"]
    imgs   = [orig_arr, gradcam_arr, lime_arr]

    for ax, img_arr, title in zip(axes, imgs, titles):
        ax.imshow(img_arr)
        ax.set_title(title, color='white', fontsize=10, pad=6)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    fig.suptitle(
        f"Prediksi: {species.replace('_',' ')}  |  Confidence: {conf:.1%}",
        color='white', fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout(pad=0.8)
    plt.savefig(path, dpi=140, bbox_inches='tight',
                facecolor='#111111', edgecolor='none')
    plt.close()
    return fname   # hanya nama file, bukan full path