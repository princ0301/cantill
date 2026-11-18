"""Load and validate a company config (YAML)."""
from pathlib import Path

import yaml

# The canonical record every company is mapped into.
CANON_FIELDS = [
    "date", "supplier", "vendor_id", "description", "item_code",
    "quantity", "amount", "currency", "currency_rate", "tax_rate",
    "status", "po", "po_item", "invoice", "gl_text", "doc_type",
]


class ConfigError(Exception):
    pass


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg):
    for key in ("company", "sources"):
        if key not in cfg:
            raise ConfigError(f"config is missing '{key}'")
    seen = set()
    for s in cfg["sources"]:
        name = s.get("name", "?")
        for key in ("name", "file", "id", "columns"):
            if key not in s:
                raise ConfigError(f"source {name}: missing '{key}'")
        if "column" not in s["id"]:
            raise ConfigError(f"source {name}: id needs a 'column'")
        bad = [k for k in s["columns"] if k not in CANON_FIELDS]
        if bad:
            raise ConfigError(f"source {name}: unknown fields {bad}. Allowed: {CANON_FIELDS}")
        if name in seen:
            raise ConfigError(f"duplicate source name '{name}'")
        seen.add(name)


def find_file(rel, *dirs):
    """Locate a file named in a config. Tries each directory in turn: the exact relative path, then the
    bare file name, then a search below the directory (so a different folder layout still works)."""
    name = Path(rel).name
    for d in dirs:
        if d is None:
            continue
        d = Path(d)
        for cand in (d / rel, d / name):
            if cand.is_file():
                return cand
    for d in dirs:
        if d is None or not Path(d).is_dir():
            continue
        hits = sorted(Path(d).rglob(name))
        hits = [h for h in hits if h.is_file()]
        if hits:
            return hits[0]
    raise ConfigError(f"file '{rel}' not found in: " + ", ".join(str(d) for d in dirs if d is not None))

