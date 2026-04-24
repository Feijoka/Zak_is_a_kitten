"""
Disk-persisted cache for the Spotify dashboard.

All files live in a 'cache/' subdirectory next to this module.
The cache is automatically invalidated whenever spotify_songs.csv
is modified (tracked via its last-modified timestamp).
"""

import os
import numpy as np
import pandas as pd
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(_HERE, "cache")
_CSV       = os.path.join(_HERE, "spotify_songs.csv")
_META      = os.path.join(CACHE_DIR, ".meta_mtime")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cp(name: str) -> str:
    """Full path of a cache file."""
    return os.path.join(CACHE_DIR, name)


def _csv_mtime() -> str:
    return f"{os.path.getmtime(_CSV):.3f}"


# ── Public API ────────────────────────────────────────────────────────────────

def cache_is_valid() -> bool:
    """
    Return True iff every cache file exists AND the CSV hasn't changed
    since the cache was written.
    """
    required = [
        "df.parquet", "X.parquet", "X_scaled.npy",
        "scaler.pkl", "labels_dict.pkl",
        "pca_components.npy", "elbow_sil.pkl", "hc_model.pkl",
        "dendrogram.png",
        ".meta_mtime",
    ]
    if any(not os.path.isfile(_cp(f)) for f in required):
        return False
    try:
        with open(_META) as fh:
            return fh.read().strip() == _csv_mtime()
    except OSError:
        return False


def save_all(df, X, X_scaled, scaler, labels_dict,
             pca_components, k_vals, inertias, sil_scores, hc_model):
    """Persist every computed object to disk.  Meta file is written last."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    df.to_parquet(_cp("df.parquet"))
    X.to_parquet(_cp("X.parquet"))
    np.save(_cp("X_scaled.npy"), X_scaled)
    joblib.dump(scaler,               _cp("scaler.pkl"))
    joblib.dump(labels_dict,          _cp("labels_dict.pkl"))
    np.save(_cp("pca_components.npy"), pca_components)
    joblib.dump((k_vals, inertias, sil_scores), _cp("elbow_sil.pkl"))
    joblib.dump(hc_model,             _cp("hc_model.pkl"))

    # Written last — guards against partial writes
    with open(_META, "w") as fh:
        fh.write(_csv_mtime())


def load_all():
    """Load every cached object from disk and return them."""
    df             = pd.read_parquet(_cp("df.parquet"))
    X              = pd.read_parquet(_cp("X.parquet"))
    X_scaled       = np.load(_cp("X_scaled.npy"))
    scaler         = joblib.load(_cp("scaler.pkl"))
    labels_dict    = joblib.load(_cp("labels_dict.pkl"))
    pca_components = np.load(_cp("pca_components.npy"))
    k_vals, inertias, sil_scores = joblib.load(_cp("elbow_sil.pkl"))
    hc_model       = joblib.load(_cp("hc_model.pkl"))

    return (df, X, X_scaled, scaler, labels_dict,
            pca_components, k_vals, inertias, sil_scores, hc_model)


def clear_cache():
    """Delete the entire cache directory."""
    import shutil
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
