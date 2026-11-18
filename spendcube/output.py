"""Build the one-file-per-company output and check it against the brief's rules."""
import numpy as np
import pandas as pd
from .classify import SEP
from .config import ConfigError

COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
           "l1", "l2", "l3", "l4", "confidence"]
CODES = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}

# how sure we are about an exclusion, by how it was decided (override in config: output.exclusion_confidence)
DEFAULT_EXCLUSION_CONF = {
    "status": 0.99, "po_amount": 0.98, "sibling_po_amount": 0.95, "po_text": 0.90, "sibling_po_text": 0.85,
    "repeat_export": 0.90, "window": 0.70, "rule": 0.90,
}


def exclusion_confidence(df, cfg):
    conf = {**DEFAULT_EXCLUSION_CONF, **cfg.get("output", {}).get("exclusion_confidence", {})}
    rule = df["exclusion_rule"].astype(str)
    out = pd.Series(conf["rule"], index=df.index)
    out[rule.str.startswith("status:")] = conf["status"]
    for k in ("po_amount", "sibling_po_amount", "po_text", "sibling_po_text", "repeat_export", "window"):
        out[rule == "match:" + k] = conf[k]
    return out


def build_output(df, cfg):
    n = len(cfg["tree"]["levels"])
    if n > 4:
        raise ConfigError("the output format has l1..l4; the tree has more than 4 levels")
    inc = (df["include"] == 1).to_numpy()
    out = pd.DataFrame({
        "record_id": df["record_id"].to_numpy(),
        "include": df["include"].astype(int).to_numpy(),
        "exclusion_reason": df["exclusion_reason"].to_numpy(),
        "supplier_key": df["supplier_key"].to_numpy(),
        "item_key": df["item_key"].to_numpy(),
        "spend": np.where(inc, df["spend"].fillna(0.0), 0.0).round(2),
    })
    for i in range(1, 5):
        col = f"l{i}"
        out[col] = df[col].to_numpy() if (i <= n and col in df.columns) else ""
    conf = df["confidence"].astype(float).copy()
    conf[~inc] = exclusion_confidence(df, cfg)[~inc]
    out["confidence"] = conf.fillna(0.05).clip(0, 1).round(3).to_numpy()
    out.loc[~inc, ["l1", "l2", "l3", "l4"]] = ""
    return out[COLUMNS]


def validate_output(out, input_ids, tree):
    """Return (errors, notes). Errors mean the file would be rejected or mis-scored."""
    err, notes = [], []
    if list(out.columns) != COLUMNS:
        err.append(f"columns are {list(out.columns)}, expected {COLUMNS}")
        return err, notes
    ids = out["record_id"]
    if ids.duplicated().any():
        err.append(f"{int(ids.duplicated().sum())} duplicate record_id values")
    missing, extra = set(input_ids) - set(ids), set(ids) - set(input_ids)
    if missing:
        err.append(f"{len(missing)} input records missing from the output, e.g. {sorted(missing)[:3]}")
    if extra:
        err.append(f"{len(extra)} output records not in the input, e.g. {sorted(extra)[:3]}")
    if not out["include"].isin([0, 1]).all():
        err.append("include must be 0 or 1")
    inc = out["include"] == 1
    if (out.loc[inc, "exclusion_reason"] != "").any():
        err.append("included rows must have a blank exclusion_reason")
    bad = ~out.loc[~inc, "exclusion_reason"].isin(CODES)
    if bad.any():
        err.append(f"{int(bad.sum())} excluded rows lack a valid exclusion_reason (one of {sorted(CODES)})")
    sp = pd.to_numeric(out["spend"], errors="coerce")
    if sp.isna().any() or not np.isfinite(sp.fillna(0)).all():
        err.append("spend has missing or non-finite values")
    if (sp[~inc] != 0).any():
        err.append("excluded rows must have spend 0")
    cf = pd.to_numeric(out["confidence"], errors="coerce")
    if cf.isna().any() or ((cf < 0) | (cf > 1)).any():
        err.append("confidence must be a number from 0 to 1")
    for c in ("supplier_key", "item_key"):
        if (out[c].astype(str).str.strip() == "").any():
            err.append(f"{c} has blank values")
    paths = set(tree["path"])
    levels = out[["l1", "l2", "l3", "l4"]].astype(str)
    path = levels.apply(lambda r: SEP.join(v for v in r if v), axis=1)
    if (levels.loc[inc, "l1"] == "").any():
        err.append(f"{int((levels.loc[inc, 'l1'] == '').sum())} included rows have no category")
    off = inc & (levels["l1"] != "") & ~path.isin(paths)
    if off.any():
        err.append(f"{int(off.sum())} included rows have a category path that is not in the tree, e.g. {path[off].iloc[0]!r}")
    if (path[~inc] != "").any():
        notes.append("some excluded rows carry a category (they should be blank)")
    notes.append(f"{int(inc.sum()):,} of {len(out):,} rows included; included spend {sp[inc].sum():,.2f}")
    notes.append(f"included rows with confidence below 0.3: {int((cf[inc] < 0.3).sum()):,}")
    return err, notes