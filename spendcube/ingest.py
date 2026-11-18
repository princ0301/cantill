"""Read raw files and map them to the canonical record. No cleaning here."""
from pathlib import Path
import pandas as pd
from .config import CANON_FIELDS, ConfigError


def _read(path, src):
    kw = dict(dtype=str, keep_default_na=False, sep=src.get("sep", ","))
    encodings = [src["encoding"]] if src.get("encoding") else ["utf-8-sig", "cp1252"]
    last = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError as e:
            last = e
    raise last


def _as_list(v):
    return [v] if isinstance(v, str) else list(v)


def _pull(df, cols):
    """One column, or several joined with ' | ' (blanks skipped)."""
    parts = [df[c].str.strip() for c in _as_list(cols)]
    out = parts[0]
    for p in parts[1:]:
        sep = pd.Series(" | ", index=out.index).where((out != "") & (p != ""), "")
        out = out + sep + p
    return out


def load_source(src, data_dir):
    path = Path(data_dir) / src["file"]
    if not path.exists():
        raise ConfigError(f"file not found: {path}")
    df = _read(path, src)
    df.columns = [c.strip() for c in df.columns]

    needed = {src["id"]["column"], *src.get("extra", [])}
    for cols in src["columns"].values():
        needed.update(_as_list(cols))
    missing = sorted(c for c in needed if c not in df.columns)
    if missing:
        raise ConfigError(
            f"source '{src['name']}': columns {missing} not in {src['file']}.\n"
            f"Available: {list(df.columns)}"
        )

    out = pd.DataFrame(index=df.index)
    out["record_id"] = src["id"].get("prefix", "") + df[src["id"]["column"]].str.strip()
    out["source"] = src["name"]
    for f in CANON_FIELDS:
        out[f] = _pull(df, src["columns"][f]) if f in src["columns"] else ""
    for c in src.get("extra", []):
        out["x_" + c] = df[c].str.strip()
    return out


def ingest(cfg, data_dir):
    frames = [load_source(s, data_dir) for s in cfg["sources"]]
    df = pd.concat(frames, ignore_index=True).fillna("")
    dup = df["record_id"].duplicated().sum()
    if dup:
        raise ConfigError(f"{dup} duplicate record_id values; check the id column/prefix")
    return df



def load_labels(cfg, labels_dir):
    """Golden keys from a labelled file named in the config (`labels:` block), or None."""
    lc = cfg.get("labels")
    if not lc:
        return None
    path = Path(labels_dir) / lc["file"]
    if not path.exists():
        print(f"note: labels file {path} not found; running without labels")
        return None
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    idc = lc.get("id", "record_id")
    if idc not in lab.columns:
        raise ConfigError(f"labels file has no id column '{idc}'. Available: {list(lab.columns)}")
    out = pd.DataFrame(index=pd.Index(lab[idc].str.strip(), name="record_id"))
    for name in ("supplier_key", "item_key"):
        col = lc.get(name)
        if col:
            if col not in lab.columns:
                raise ConfigError(f"labels file has no column '{col}'")
            out[name] = lab[col].to_numpy()
    cats = lc.get("categories")
    if cats:
        miss = [c for c in cats if c not in lab.columns]
        if miss:
            raise ConfigError(f"labels file lacks category columns {miss}")
        parts = lab[cats].apply(lambda c: c.str.strip())
        out["leaf"] = parts.apply(lambda r: "|".join(v for v in r if v), axis=1).replace("", None).to_numpy()
    inc = lc.get("include")
    out["usable"] = (lab[inc].str.strip() == "1").to_numpy() if inc else True
    return out