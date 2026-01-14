# config.py
import os

# Telegram Bot Token
BOT_TOKEN = "8595708491:AAEPWLh3IyXrS27YXtyVoPVwQVBah7flOB8"  # @BotFather dan olingan token

# Database
DATABASE_PATH = "clustering_bot.db"

# Fayllar
UPLOAD_FOLDER = "data/user_uploads"
DATASET_FOLDER = "data/datasets"
TEMP_FOLDER = "data/temp"

# Maksimum fayllar
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 10000

# Default parametrlar
DEFAULT_KMEANS_K = 3
DEFAULT_KMEANS_ITERATIONS = 100

DEFAULT_DBSCAN_EPS = 0.5
DEFAULT_DBSCAN_MIN_PTS = 5

# Papkalarni yaratish
for folder in [UPLOAD_FOLDER, DATASET_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)