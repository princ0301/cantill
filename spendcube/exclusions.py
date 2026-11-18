"""Decide what counts as spend and why a record is excluded.

Order of precedence (first match wins): CANCELLED, NOT_APPROVED, then the rule list
(company rules first, then the generic defaults). DUPLICATE is added by the matching step.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .config import ConfigError

CODES = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}
DEFAULT_RULES = Path(__file__).with_name("default_rules.yaml")


def load_default_rules():
    with open(DEFAULT_RULES, encoding="utf-8") as f:
        return yaml.safe_load(f)["rules"]


def status_exclusions(df, cfg):
    """Return (reason, rule) Series from each source's `status` block."""
    reason = pd.Series("", index=df.index)
    rule = pd.Series("", index=df.index)
    for src in cfg["sources"]:
        st = src.get("status")
        if not st:
            continue
        m = df["source"] == src["name"]
        v = df.loc[m, "status"].astype(str).str.strip().str.lower()
        low = lambda key, default=(): {str(x).strip().lower() for x in st.get(key, default)}
        include, cancelled = low("include", ("approved", "paid")), low("cancelled")
        blank_ok = st.get("blank_status", "include") == "include"
        is_blank = v == ""
        is_c = v.isin(cancelled) | (~v.isin(include) & v.str.contains(r"cancel|void", regex=True))
        is_na = ~is_c & ~v.isin(include) & ~(is_blank & blank_ok)
        reason.loc[m] = np.where(is_c, "CANCELLED", np.where(is_na, "NOT_APPROVED", ""))
        rule.loc[m] = np.where(is_c, "status:cancelled", np.where(is_na, "status:not_approved", ""))
    return reason, rule


def _rule_mask(df, r):
    fields = r["field"] if isinstance(r["field"], list) else [r["field"]]
    mask = pd.Series(False, index=df.index)
    for f in fields:
        if f not in df.columns:
            raise ConfigError(f"rule '{r.get('name')}': unknown field '{f}'")
        col = df[f].astype(str)
        if "regex" in r:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mask |= col.str.contains(r["regex"], regex=True, case=bool(r.get("case_sensitive", False)))
        elif "values" in r:
            mask |= col.str.strip().str.lower().isin({str(x).strip().lower() for x in r["values"]})
        else:
            raise ConfigError(f"rule '{r.get('name')}' needs 'regex' or 'values'")
    if "source" in r:
        mask &= df["source"] == r["source"]
    return mask


def apply_exclusions(df, cfg):
    """Add include, exclusion_reason, exclusion_rule and spend columns."""
    df = df.copy()
    reason, rule = status_exclusions(df, cfg)

    rules = list(cfg.get("exclusions", []))
    if cfg.get("use_default_rules", True):
        rules += load_default_rules()
    for r in rules:
        code = r.get("code", "NON_PROCUREMENT")
        if code not in CODES:
            raise ConfigError(f"rule '{r.get('name')}': unknown code {code}")
        hit = _rule_mask(df, r) & (reason == "")
        reason[hit] = code
        rule[hit] = r.get("name", "unnamed")

    df["exclusion_reason"] = reason
    df["exclusion_rule"] = rule
    df["include"] = (reason == "").astype(int)
    df["spend"] = np.where(df["include"] == 1, df["spend_rep"].fillna(0.0), 0.0)
    return df


def rule_coverage(df, cfg):
    """Raw hits per rule (ignoring precedence). A company rule with 0 hits is probably broken."""
    rules = [("company", r) for r in cfg.get("exclusions", [])]
    if cfg.get("use_default_rules", True):
        rules += [("default", r) for r in load_default_rules()]
    rows = [{"origin": o, "rule": r.get("name", "unnamed"), "hits": int(_rule_mask(df, r).sum())} for o, r in rules]
    return pd.DataFrame(rows)



def apply_duplicates(df, cfg):
    """Mark secondary-system lines that repeat a primary record or an earlier export as DUPLICATE."""
    from .match import match_systems
    m = match_systems(df, cfg)
    df = df.copy()
    for c in ("match_to", "match_kind", "match_note"):
        df[c] = m[c]
    hit = (df["match_to"] != "") & (df["exclusion_reason"] == "")
    df.loc[hit, "exclusion_reason"] = "DUPLICATE"
    df.loc[hit, "exclusion_rule"] = "match:" + df.loc[hit, "match_kind"]
    df.loc[hit, "include"] = 0
    df.loc[hit, "spend"] = 0.0
    return df