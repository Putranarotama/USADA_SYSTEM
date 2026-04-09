# =============================================================================
#  config.py — Konfigurasi Sistem Deteksi Tanaman Usada Bali
#  Dataset: 20 Spesies Bedugul (disesuaikan dengan folder dataset Anda)
# =============================================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# !! SESUAIKAN: path ke folder dataset asli Anda
# Contoh Windows : r"C:\Users\Anda\Documents\dataset_bedugul"
# Contoh relatif  : os.path.join(BASE_DIR, "dataset_source")
SOURCE_DATASET_DIR = os.path.join(BASE_DIR, "dataset_source")

# Folder yang dikelola otomatis oleh sistem
DATASET_AUGMENTED = os.path.join(BASE_DIR, "dataset_augmented")
TRAIN_DIR         = os.path.join(DATASET_AUGMENTED, "train")
VAL_DIR           = os.path.join(DATASET_AUGMENTED, "val")
TEST_DIR          = os.path.join(DATASET_AUGMENTED, "test")
MODEL_DIR         = os.path.join(BASE_DIR, "models")
UPLOAD_DIR        = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR        = os.path.join(BASE_DIR, "static", "results")
LOG_DIR           = os.path.join(BASE_DIR, "logs")

# =============================================================================
#  20 SPESIES — nama PERSIS sama dengan nama folder dataset Anda di Bedugul
# =============================================================================
SPECIES = [
    "Bangle",
    "Binahong",
    "Buni",
    "Dadap",
    "Daun_Komprei",
    "Daun_Sendok",
    "Daun_Tulak",
    "Gandarusa",
    "Isen_Barak",
    "Isen_Putih",
    "Jahe",
    "Jempiring_Alas",
    "Jonghe",
    "Kapulaga",
    "Kayu_Manis",
    "Lempuyang",
    "Sembung",
    "Sirih",
    "Temu",
    "Temu_Kunci",
]

# =============================================================================
#  INFO KHASIAT — berdasarkan Lontar Usada Bali & literatur etnobotani
# =============================================================================
SPECIES_INFO = {
    "Bangle": {
        "latin":   "Zingiber purpureum Roxb.",
        "khasiat": "Penurun demam, sakit kepala, rematik, sakit perut, pencahar, "
                   "anti-inflamasi, pelangsing, antimikroba",
    },
    "Binahong": {
        "latin":   "Anredera cordifolia (Ten.) Steenis",
        "khasiat": "Menyembuhkan luka, anti-inflamasi, antioksidan, "
                   "mempercepat regenerasi jaringan, anti-diabetes",
    },
    "Buni": {
        "latin":   "Antidesma bunius (L.) Spreng.",
        "khasiat": "Anti-inflamasi, analgesik, antioksidan, menurunkan tekanan darah, "
                   "mengobati diare dan disentri",
    },
    "Dadap": {
        "latin":   "Erythrina variegata L.",
        "khasiat": "Penurun demam, anti-inflamasi, sedatif, mengatasi insomnia, "
                   "mengobati luka dan bisul",
    },
    "Daun_Komprei": {
        "latin":   "Symphytum officinale L.",
        "khasiat": "Mempercepat penyembuhan luka dan patah tulang, anti-inflamasi, "
                   "mengobati radang sendi dan memar",
    },
    "Daun_Sendok": {
        "latin":   "Plantago major L.",
        "khasiat": "Anti-inflamasi, diuretik, antiseptik, mengobati batuk, "
                   "diare, luka, dan saluran kemih",
    },
    "Daun_Tulak": {
        "latin":   "Jatropha multifida L.",
        "khasiat": "Antiseptik, mengobati luka, bisul, dan infeksi kulit, "
                   "anti-inflamasi, antibakteri",
    },
    "Gandarusa": {
        "latin":   "Justicia gendarussa Burm.f.",
        "khasiat": "Anti-rematik, analgesik, anti-inflamasi, "
                   "mengobati memar, keseleo, dan nyeri sendi",
    },
    "Isen_Barak": {
        "latin":   "Alpinia galanga (L.) Willd.",
        "khasiat": "Antimikroba, anti-inflamasi, mengobati infeksi kulit, "
                   "rematik, gangguan pencernaan, dan panu",
    },
    "Isen_Putih": {
        "latin":   "Alpinia malaccensis (Burm.f.) Roscoe",
        "khasiat": "Anti-inflamasi, antimikroba, analgesik, mengobati "
                   "gangguan pencernaan dan sakit perut",
    },
    "Jahe": {
        "latin":   "Zingiber officinale Roscoe",
        "khasiat": "Anti-mual, anti-inflamasi, antioksidan, melancarkan pencernaan, "
                   "menghangatkan tubuh, anti-diabetes",
    },
    "Jempiring_Alas": {
        "latin":   "Gardenia jasminoides J.Ellis",
        "khasiat": "Antipiretik, anti-inflamasi, analgesik, mengobati sakit kuning, "
                   "luka, dan demam tinggi",
    },
    "Jonghe": {
        "latin":   "Acorus calamus L.",
        "khasiat": "Mengatasi gangguan saraf, anti-kejang, antiseptik, "
                   "mengobati gangguan pencernaan, bau badan, dan insomnia",
    },
    "Kapulaga": {
        "latin":   "Elettaria cardamomum (L.) Maton",
        "khasiat": "Melancarkan pencernaan, menyegarkan napas, antioksidan, "
                   "mengatasi mual, kembung, dan batuk",
    },
    "Kayu_Manis": {
        "latin":   "Cinnamomum verum J.Presl",
        "khasiat": "Anti-diabetes, anti-inflamasi, antioksidan, antimikroba, "
                   "melancarkan sirkulasi darah, mengatasi masuk angin",
    },
    "Lempuyang": {
        "latin":   "Zingiber zerumbet (L.) Roscoe ex Sm.",
        "khasiat": "Anti-inflamasi, analgesik, antioksidan, anti-kanker, "
                   "mengobati rematik, dan meningkatkan nafsu makan",
    },
    "Sembung": {
        "latin":   "Blumea balsamifera (L.) DC.",
        "khasiat": "Anti-inflamasi, diuretik, antiseptik, mengobati demam, "
                   "batuk, sakit kepala, dan gangguan haid",
    },
    "Sirih": {
        "latin":   "Piper betle L.",
        "khasiat": "Antiseptik kuat, anti-inflamasi, mengobati batuk, keputihan, "
                   "bau mulut, luka, dan infeksi jamur",
    },
    "Temu": {
        "latin":   "Curcuma longa L.",
        "khasiat": "Anti-inflamasi kuat, antioksidan, hepatoprotektor, "
                   "melancarkan pencernaan, anti-kanker, anti-diabetes",
    },
    "Temu_Kunci": {
        "latin":   "Boesenbergia rotunda (L.) Mansf.",
        "khasiat": "Anti-inflamasi, antimikroba, melancarkan pencernaan, "
                   "mengatasi sariawan, batuk, dan masalah pernapasan",
    },
}

# =============================================================================
#  PARAMETER AUGMENTASI
# =============================================================================
AUGMENT_FACTOR = 15    # Setiap foto train asli → 15 foto augmented
AUGMENT_SEED   = 42

# =============================================================================
#  PARAMETER IMAGE & TRAINING
# =============================================================================
IMAGE_SIZE          = 224
BATCH_SIZE          = 8
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-4
DROPOUT_RATE        = 0.3
LABEL_SMOOTHING     = 0.1
PHASE1_EPOCHS       = 20
PHASE2_EPOCHS       = 15
PHASE3_EPOCHS       = 15
EARLY_STOP_PATIENCE = 8
TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
TEST_RATIO          = 0.15

# =============================================================================
#  MODEL FILES
# =============================================================================
MODEL_PATH       = os.path.join(MODEL_DIR, "usada_checkpoint.pth")
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "usada_best_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PATH     = os.path.join(MODEL_DIR, "training_history.json")

# =============================================================================
#  FLASK
# =============================================================================
FLASK_HOST           = "0.0.0.0"
FLASK_PORT           = 5000
FLASK_DEBUG          = True
MAX_UPLOAD_SIZE      = 16 * 1024 * 1024
ALLOWED_EXTENSIONS   = {"png", "jpg", "jpeg", "webp"}
CONFIDENCE_THRESHOLD = 0.45