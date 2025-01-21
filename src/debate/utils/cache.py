import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

TEMP_DIR = Path("temp_results")


def ensure_temp_dir():
    """Ensure the temporary directory exists."""
    TEMP_DIR.mkdir(exist_ok=True)


def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate a cache key based on the input parameters."""
    # Sort kwargs to ensure consistent hashing
    sorted_items = sorted(kwargs.items())
    params_str = json.dumps(sorted_items)
    hash_obj = hashlib.md5(params_str.encode())
    return f"{prefix}_{hash_obj.hexdigest()}.json"


def save_to_cache(key: str, data: Any):
    """Save data to cache file."""
    ensure_temp_dir()
    with open(TEMP_DIR / key, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_from_cache(key: str) -> Optional[Dict]:
    """Load data from cache file if it exists."""
    cache_file = TEMP_DIR / key
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
