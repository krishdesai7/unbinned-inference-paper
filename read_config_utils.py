import ast
from typing import Any, Optional

def _parse_value(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    # Try Python-literal parsing (handles ints, floats, bools, lists, nested lists)
    try:
        return ast.literal_eval(s)
    except Exception:
        # Fallback: return the raw string (covers things like "<function ...>")
        return s

def parse_config(path: str) -> dict:
    """
    Parse a simple 'key value' or 'key : value' config file into a dict.
    Lines that are empty or start with '#' are ignored.
    """
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split on first ':' if present; otherwise split on whitespace once
            if ":" in line:
                key, val = line.split(":", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 1:
                    # Key with no value; store empty string
                    key, val = parts[0], ""
                else:
                    key, val = parts[0], parts[1]

            key = key.strip()
            val = val.strip()
            if key:
                cfg[key] = _parse_value(val)
    return cfg

def get_par(path: str, key: str, default: Optional[Any] = None, cast=None) -> Any:
    """
    Fetch a parameter by key from the config file.
    - default: value to return if key is missing
    - cast: optional callable to cast/validate (e.g., int, float)
    """
    cfg = parse_config(path)
    if key not in cfg:
        if default is not None:
            return default
        raise KeyError(f"Missing key '{key}' in {path}")
    val = cfg[key]
    return cast(val) if cast is not None else val