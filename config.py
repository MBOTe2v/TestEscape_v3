"""Configuration — TestEscape_v3"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATABASE_CSV = os.path.join(DATA_DIR, "Database.csv")
NEWBATCH_CSV = os.path.join(DATA_DIR, "NewBatch.csv")

ID_COLS = ["lot_id", "part_id", "soft_bin"]
