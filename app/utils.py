import os
import time
import uuid
import hashlib
import logging
import yaml
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
LOGS_DIR = BASE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [DATA_DIR, CHARTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def setup_logging() -> logging.Logger:
    log_file = LOGS_DIR / "app.log"
    logger = logging.getLogger("biomed_platform")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = setup_logging()

class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self._cache: dict[str, dict[str, Any]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._ttl:
                logger.info(f"Cache hit: {key[:16]}...")
                return entry["value"]
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = {"value": value, "timestamp": time.time()}
        logger.info(f"Cache set: {key[:16]}...")

    def invalidate(self, key: str):
        self._cache.pop(key, None)

    def cleanup(self):
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v["timestamp"] >= self._ttl]
        for k in expired:
            del self._cache[k]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired cache entries")

cache = SimpleCache(ttl=load_config().get("cache", {}).get("ttl", 3600))

def generate_session_id() -> str:
    return str(uuid.uuid4())

def get_session_dir(session_id: str) -> Path:
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def generate_cache_key(filename: str, params: dict) -> str:
    param_str = str(sorted(params.items()))
    raw = f"{filename}_{param_str}"
    return hashlib.md5(raw.encode()).hexdigest()

def cleanup_old_files(max_age_hours: int = 24):
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    for directory in [DATA_DIR, CHARTS_DIR]:
        if not directory.exists():
            continue
        for item in directory.iterdir():
            if item.is_file():
                if now - item.stat().st_mtime > max_age_seconds:
                    try:
                        item.unlink()
                        logger.info(f"Deleted old file: {item}")
                    except Exception as e:
                        logger.error(f"Failed to delete {item}: {e}")
            elif item.is_dir():
                try:
                    if item.name not in [".gitkeep"]:
                        for f in item.iterdir():
                            if now - f.stat().st_mtime > max_age_seconds:
                                f.unlink()
                                logger.info(f"Deleted old file: {f}")
                        if not any(item.iterdir()):
                            item.rmdir()
                            logger.info(f"Removed empty directory: {item}")
                except Exception as e:
                    logger.error(f"Failed to cleanup directory {item}: {e}")

def make_serializable(obj: Any) -> Any:
    import numpy as np
    import pandas as pd
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj
