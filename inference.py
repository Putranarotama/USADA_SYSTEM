# inference.py — Prediksi + Grad-CAM + LIME

import os, sys, json
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model import UsadaHybridModel
from utils.xai import GradCAM, lime_explain, save_xai

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

PREPROC = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class UsadaPredictor:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model      = None
        self.idx_to_cls = {}
        self._load()

    def _load(self):
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(
                f"class_names.json tidak ditemukan.\n"
                "Jalankan: python train.py")
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            self.idx_to_cls = {int(k): v for k, v in json.load(f).items()}

        n = len(self.idx_to_cls)
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {BEST_MODEL_PATH}\n"
                "Jalankan: python train.py")

        self.model = UsadaHybridModel(n, pretrained=False)
        self.model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=self.device))
        self.model.eval().to(self.device)
        print(f"[Predictor] {n} spesies — {self.device}")

    def predict(self, image_path: str, with_xai: bool = True) -> dict:
        try:
            orig = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": str(e)}

        tensor = PREPROC(orig).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, 1)[0]

        top_vals, top_idx = probs.topk(min(3, len(self.idx_to_cls)))
        top3 = [
            {"species": self.idx_to_cls.get(i.item(), "?"),
             "confidence": v.item()}
            for v, i in zip(top_vals, top_idx)
        ]

        best_idx  = top_idx[0].item()
        best_conf = top_vals[0].item()
        best_name = self.idx_to_cls.get(best_idx, "Unknown")
        info      = SPECIES_INFO.get(best_name, {})

        result = {
            "species":       best_name.replace("_", " "),
            "species_raw":   best_name,
            "confidence":    best_conf,
            "confidence_pct": f"{best_conf:.1%}",
            "latin":         info.get("latin", "-"),
            "khasiat":       info.get("khasiat", "-"),
            "top3":          top3,
            "status":        "ok" if best_conf >= CONFIDENCE_THRESHOLD
                             else "low_confidence",
            "xai_filename":  None,
        }

        if with_xai:
            try:
                # Grad-CAM
                gc     = GradCAM(self.model)
                t_grad = PREPROC(orig).unsqueeze(0).to(self.device)
                gc_arr = gc.generate(t_grad, best_idx, orig)
                gc.remove()

                # LIME
                lm_arr = lime_explain(
                    self.model, tensor, orig,
                    best_idx, str(self.device))

                # Simpan komposit
                fname = save_xai(orig, gc_arr, lm_arr,
                                 best_name, best_conf)
                result["xai_filename"] = fname

            except Exception as e:
                print(f"[XAI warn] {e}")

        return result


_predictor = None

def get_predictor() -> UsadaPredictor:
    global _predictor
    if _predictor is None:
        _predictor = UsadaPredictor()
    return _predictor
