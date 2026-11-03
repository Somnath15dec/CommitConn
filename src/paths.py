"""Project paths (pathlib only)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "proc"
EXPORTS = ROOT / "exports"

SPOTIFY_CSV = RAW / "spotify" / "dataset.csv"
MUSE_CSV = RAW / "muse" / "muse_v3.csv"
PLAYLIST_ZIP = RAW / "playlists" / "spotify_playlists.zip"
PLAYLIST_MATCHED = RAW / "playlists" / "matched.parquet"
PLAYLIST_CONTEXT = RAW / "playlists" / "context_playlists.parquet"
PLAYLIST_STATS = RAW / "playlists" / "pass_stats.json"

CATALOG = PROC / "catalog.parquet"
CONTEXT_LABELS = PROC / "context_labels.parquet"
REAL_USERS = PROC / "real_users.parquet"
RULE_PRECISION = PROC / "rule_precision_tune.csv"
MUSE_VALIDATION = PROC / "muse_validation.csv"
TRAIN_PAIRS = PROC / "train_pairs.jsonl"
EVAL_PAIRS = PROC / "eval_pairs.jsonl"
EVAL_CONTEXT_QUERIES = PROC / "eval_context_queries.jsonl"
