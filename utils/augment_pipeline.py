# utils/augment_pipeline.py
# ============================================================
#  PIPELINE AUGMENTASI LENGKAP
#  Langkah 1 dari pipeline: jalankan SEBELUM training
#
#  Cara pakai:
#    python utils/augment_pipeline.py
#
#  Yang dilakukan:
#  1. Baca dataset asli dari SOURCE_DATASET_DIR (sudah per-spesies)
#  2. Split → train (70%) / val (15%) / test (15%)
#  3. Copy val & test apa adanya (TANPA augmentasi)
#  4. Augmentasi agresif pada train → AUGMENT_FACTOR x lipat
#  5. Simpan semua ke DATASET_AUGMENTED/
#
#  Output akhir:
#    dataset_augmented/
#      train/  ← sudah diaugmentasi (jauh lebih banyak)
#      val/    ← asli, tidak diaugmentasi
#      test/   ← asli, tidak diaugmentasi
# ============================================================

import os, sys, shutil, random, time
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (SOURCE_DATASET_DIR, DATASET_AUGMENTED,
                    TRAIN_DIR, VAL_DIR, TEST_DIR,
                    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
                    AUGMENT_FACTOR, AUGMENT_SEED, IMAGE_SIZE,
                    ALLOWED_EXTENSIONS)

random.seed(AUGMENT_SEED)
np.random.seed(AUGMENT_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMASI AUGMENTASI
#  Dirancang khusus untuk foto daun tanaman herbal:
#  - Background putih (seperti kondisi Anda di Bedugul)
#  - Foto depan + belakang daun
#  - Variasi cahaya lapangan (bukan studio gelap)
# ─────────────────────────────────────────────────────────────────────────────

def random_augment(img: Image.Image) -> Image.Image:
    """Terapkan augmentasi acak pada satu gambar PIL."""

    # 1. Resize sedikit over-size dulu (untuk crop)
    over = int(IMAGE_SIZE * 1.15)
    img = img.resize((over, over), Image.LANCZOS)

    # 2. Random crop ke IMAGE_SIZE
    left = random.randint(0, over - IMAGE_SIZE)
    top  = random.randint(0, over - IMAGE_SIZE)
    img  = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # 3. Flip horizontal (daun bisa dari kiri/kanan)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 4. Flip vertikal (foto depan ↔ belakang daun di Anda)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 5. Rotasi bebas 0–360° (tanaman di lapangan bisa miring ke mana saja)
    angle = random.uniform(0, 360)
    img   = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))  # Fill putih

    # 6. Brightness — simulasi variasi cahaya matahari Bedugul (pagi vs siang)
    factor = random.uniform(0.65, 1.40)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 7. Contrast — variasi foto lapangan
    factor = random.uniform(0.75, 1.35)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 8. Saturation/Color — variasi warna
    factor = random.uniform(0.70, 1.30)
    img = ImageEnhance.Color(img).enhance(factor)

    # 9. Sharpness — simulasi fokus sedikit berbeda
    factor = random.uniform(0.5, 1.8)
    img = ImageEnhance.Sharpness(img).enhance(factor)

    # 10. Gaussian blur ringan (probabilitas 35%)
    if random.random() > 0.65:
        radius = random.uniform(0.3, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 11. Random noise ringan (simulasi sensor kamera)
    if random.random() > 0.70:
        arr   = np.array(img, dtype=np.int16)
        noise = np.random.randint(-12, 12, arr.shape, dtype=np.int16)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    # 12. Perspective warp ringan (sudut kamera sedikit berbeda)
    if random.random() > 0.60:
        img = _perspective_warp(img, strength=0.06)

    # 13. Cutout / Random Erasing (simulasi sebagian daun tertutup)
    if random.random() > 0.75:
        img = _cutout(img, n_holes=1, hole_size_ratio=0.10)

    # Pastikan ukuran akhir konsisten
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    return img


def _perspective_warp(img: Image.Image, strength: float = 0.05) -> Image.Image:
    """Perspective transform ringan menggunakan PIL."""
    w, h = img.size
    d = int(min(w, h) * strength)
    src = [(0,0),(w,0),(w,h),(0,h)]
    dst = [
        (random.randint(0, d), random.randint(0, d)),
        (w - random.randint(0, d), random.randint(0, d)),
        (w - random.randint(0, d), h - random.randint(0, d)),
        (random.randint(0, d), h - random.randint(0, d)),
    ]
    coeffs = _find_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                          Image.BICUBIC, fillcolor=(255,255,255))


def _find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return np.array(res).flatten().tolist()


def _cutout(img: Image.Image, n_holes: int = 1,
            hole_size_ratio: float = 0.12) -> Image.Image:
    """Hapus area persegi secara acak (ganti dengan putih)."""
    arr  = np.array(img)
    h, w = arr.shape[:2]
    sz   = int(min(h, w) * hole_size_ratio)
    for _ in range(n_holes):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1, x2 = max(0, cx - sz//2), min(w, cx + sz//2)
        y1, y2 = max(0, cy - sz//2), min(h, cy + sz//2)
        arr[y1:y2, x1:x2] = 255  # Isi putih (sesuai background daun Anda)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    t_start = time.time()

    print("=" * 62)
    print("  PIPELINE AUGMENTASI — Tanaman Usada Bali")
    print("=" * 62)
    print(f"  Sumber dataset : {SOURCE_DATASET_DIR}")
    print(f"  Output         : {DATASET_AUGMENTED}")
    print(f"  Augment factor : x{AUGMENT_FACTOR} (hanya pada train)")
    print(f"  Image size     : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()

    # Validasi folder sumber
    src = Path(SOURCE_DATASET_DIR)
    if not src.exists():
        print(f"[ERROR] Folder sumber tidak ditemukan: {SOURCE_DATASET_DIR}")
        print(f"        Buat folder 'dataset_source' di direktori proyek,")
        print(f"        lalu isi dengan subfolder per spesies.")
        return False

    species_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not species_dirs:
        print(f"[ERROR] Tidak ada subfolder spesies di {SOURCE_DATASET_DIR}")
        return False

    # Bersihkan output lama
    if Path(DATASET_AUGMENTED).exists():
        print("[INFO] Menghapus dataset_augmented lama...")
        shutil.rmtree(DATASET_AUGMENTED)

    # Buat folder tujuan
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    total_src   = 0
    total_train = 0
    total_val   = 0
    total_test  = 0

    print(f"{'Spesies':<22} {'Src':>4} {'Train(aug)':>10} {'Val':>5} {'Test':>5}")
    print("-" * 52)

    for sp_dir in species_dirs:
        sp_name = sp_dir.name

        # Kumpulkan semua gambar spesies ini
        images = sorted([
            f for f in sp_dir.iterdir()
            if f.suffix.lower().lstrip('.') in ALLOWED_EXTENSIONS
        ])

        if not images:
            print(f"  [SKIP] {sp_name} — tidak ada foto")
            continue

        # Split dataset asli
        random.shuffle(images)
        n     = len(images)
        n_tr  = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        # Minimal 1 untuk val & test meski dataset sangat kecil
        if n < 3:
            n_tr, n_val = n, 0

        imgs_train = images[:n_tr]
        imgs_val   = images[n_tr : n_tr + n_val]
        imgs_test  = images[n_tr + n_val :]

        # ── COPY VAL (tanpa augmentasi) ──────────────────────────────────────
        if imgs_val:
            dest_val = Path(VAL_DIR) / sp_name
            dest_val.mkdir(parents=True, exist_ok=True)
            for f in imgs_val:
                img = Image.open(f).convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                img.save(str(dest_val / f.name), quality=95)

        # ── COPY TEST (tanpa augmentasi) ─────────────────────────────────────
        if imgs_test:
            dest_test = Path(TEST_DIR) / sp_name
            dest_test.mkdir(parents=True, exist_ok=True)
            for f in imgs_test:
                img = Image.open(f).convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                img.save(str(dest_test / f.name), quality=95)

        # ── AUGMENTASI TRAIN ─────────────────────────────────────────────────
        dest_train = Path(TRAIN_DIR) / sp_name
        dest_train.mkdir(parents=True, exist_ok=True)

        aug_count = 0
        for f in imgs_train:
            try:
                orig = Image.open(f).convert("RGB")
            except Exception as e:
                print(f"  [SKIP] {f.name}: {e}")
                continue

            # Simpan versi asli yang diresize
            resized = orig.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            resized.save(str(dest_train / f.name), quality=95)
            aug_count += 1

            # Generate AUGMENT_FACTOR versi augmented
            for i in range(AUGMENT_FACTOR):
                aug_img  = random_augment(orig.copy())
                aug_name = f"{f.stem}_aug{i:03d}{f.suffix}"
                aug_img.save(str(dest_train / aug_name), quality=92)
                aug_count += 1

        total_src   += n
        total_train += aug_count
        total_val   += len(imgs_val)
        total_test  += len(imgs_test)

        print(f"  {sp_name:<20} {n:>4} {aug_count:>10} {len(imgs_val):>5} {len(imgs_test):>5}")

    elapsed = time.time() - t_start
    print("-" * 52)
    print(f"  {'TOTAL':<20} {total_src:>4} {total_train:>10} {total_val:>5} {total_test:>5}")
    print()
    print(f"[SELESAI] Waktu: {elapsed:.1f} detik")
    print(f"  Folder train (augmented) : {TRAIN_DIR}")
    print(f"  Folder val               : {VAL_DIR}")
    print(f"  Folder test              : {TEST_DIR}")
    print()
    print("  Lanjutkan dengan: python train.py")
    return True


if __name__ == "__main__":
    success = run_pipeline()
    if not success:
        exit(1)
