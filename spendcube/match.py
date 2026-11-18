"""Cross-system matching: find secondary-system records (e.g. invoice-tool lines) that describe a
purchase already present in the primary system (e.g. ERP goods receipts), and repeated exports.

Evidence per candidate pair: same PO, amount within tolerance, text similarity (OCR-tolerant),
supplier similarity, optional date lag. Stages, in order:
  1. PO known (own, or recovered from another line of the same supplier+invoice number)
  2. PO unknown -> candidate ERP POs from the neighbours' PO range (needs `order_key`)
  3. repeated exports inside the secondary system
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .config import ConfigError

_OCR = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})
_LEGAL = r"\b(?:PTY|LTD|LIMITED|CC|SA|THE|INC|LLC|GMBH)\b"


def norm_text(s):
    """Upper-case, map OCR look-alikes (0/O 1/I 5/S 8/B) to one form, keep letters+digits only."""
    return (s.astype(str).str.upper().str.translate(_OCR)
            .str.replace(r"[^A-Z0-9]", "", regex=True))


def norm_supplier(s):
    x = s.astype(str).str.upper().str.replace(_LEGAL, " ", regex=True)
    return norm_text(x)


def last_digits(s):
    """Invoice numbers come as INV/2023/759857, SI-2023-759857, 759857 ... -> 759857."""
    return s.astype(str).str.extract(r"(\d+)\D*$")[0].fillna("").str.lstrip("0")


def _ratio(a, b):
    return np.fromiter((fuzz.ratio(x, y) for x, y in zip(a, b)), dtype=float, count=len(a))


def match_systems(df, cfg, hide=None, use_siblings=True, use_window=True, slack=None):
    """Return df with match_to, match_kind, match_note columns (secondary rows only are filled).

    hide: optional set of secondary record_ids whose PO should be treated as unknown (for validation).
    """
    out = df.copy()
    out["match_to"], out["match_kind"], out["match_note"] = "", "", ""
    cs = cfg.get("cross_system")
    if not cs:
        return out
    prim, sec = cs["primary"], cs["secondary"]
    po_rx = cs.get("po_regex", r"(\d{6,})")
    tol_abs = cs.get("amount_tol", {}).get("abs", 0.02)
    tol_rel = cs.get("amount_tol", {}).get("rel", 0.0005)
    t_text = cs.get("text_threshold", 80)
    t_sup = cs.get("supplier_threshold", 85)
    slack = cs.get("order_slack", 50) if slack is None else slack
    lag_lo, lag_hi = cs.get("date_lag_days", [-60, 180])
    max_cand = cs.get("window_max_candidates")

    P = out[out["source"] == prim].copy()
    S = out[out["source"] == sec].copy()
    for d in (P, S):
        d["po_n"] = d["po"].astype(str).str.extract(po_rx)[0]
        d["tn"] = norm_text(d["description"])
        d["sn"] = norm_supplier(d["supplier"])
        d["amt"] = d["spend_rep"].astype(float)
    S["inv_n"] = last_digits(S["invoice"])
    if hide:
        S.loc[S["record_id"].isin(hide), "po_n"] = np.nan

    # ---- PO per secondary line: own, else recovered from siblings (same supplier + invoice number)
    S["po_src"] = np.where(S["po_n"].notna(), "own", "")
    if use_siblings:
        key = S["sn"] + "|" + S["inv_n"]
        known = S[S["po_n"].notna() & (S["inv_n"] != "")]
        sib = known.groupby(key[known.index])["po_n"].agg(lambda s: s.mode().iloc[0])
        rec = S["po_n"].isna() & (S["inv_n"] != "") & key.isin(sib.index)
        S.loc[rec, "po_n"] = key[rec].map(sib)
        S.loc[rec, "po_src"] = "sibling"

    P = P[P["po_n"].notna()].sort_values("po_n", key=lambda s: s.astype("int64"))
    p_po_int = P["po_n"].astype("int64").to_numpy()

    # ---- stage 1: PO known and present in the primary system
    pairs = S[S["po_n"].notna()].reset_index().rename(columns={"index": "s_ix"}).merge(
        P.reset_index().rename(columns={"index": "p_ix"}), on="po_n", suffixes=("", "_p"))
    pairs["amt_ok"] = (pairs["amt"] - pairs["amt_p"]).abs() <= np.maximum(tol_abs, tol_rel * pairs["amt_p"].abs())
    pairs["text"] = _ratio(pairs["tn"], pairs["tn_p"])
    pairs["sup"] = _ratio(pairs["sn"], pairs["sn_p"])
    pairs["ok"] = pairs["amt_ok"] | ((pairs["text"] >= t_text) & (pairs["sup"] >= t_sup))
    pairs["rank"] = pairs["amt_ok"] * 1000 + pairs["text"] + pairs["sup"] / 10
    best = pairs[pairs["ok"]].sort_values("rank", ascending=False).drop_duplicates("s_ix")
    kind = np.where(best["amt_ok"], "po_amount", "po_text")
    kind = np.where(best["po_src"] == "sibling", "sibling_" + pd.Series(kind, index=best.index), kind)
    out.loc[best["s_ix"], "match_to"] = best["record_id_p"].to_numpy()
    out.loc[best["s_ix"], "match_kind"] = kind
    matched = set(best["s_ix"])

    # ---- stage 2: no usable PO -> PO window from neighbours in id order
    if use_window and cs.get("order_key") and len(P):
        S["ord"] = pd.to_numeric(S["record_id"].str.extract(r"(\d+)$")[0], errors="coerce")
        S = S.sort_values("ord")
        kp = S["po_n"].astype(float)
        lo, hi = kp.shift(1).ffill(), kp.shift(-1).bfill()
        a, b = np.fmin(lo, hi), np.fmax(lo, hi)
        a, b = a.fillna(b), b.fillna(a)
        todo = S[S["po_n"].isna() & a.notna()]
        p_cols = P[["record_id", "amt", "tn", "sn", "date_parsed"]].reset_index(drop=True)
        p_amt = p_cols["amt"].to_numpy()
        for ix, r in todo.iterrows():
            i0 = np.searchsorted(p_po_int, a[ix] - slack, side="left")
            i1 = np.searchsorted(p_po_int, b[ix] + slack, side="right")
            if i1 <= i0:
                continue
            sl = slice(i0, i1)
            ok = np.abs(p_amt[sl] - r["amt"]) <= np.maximum(tol_abs, tol_rel * np.abs(p_amt[sl]))
            if not ok.any():
                continue
            c = p_cols.iloc[sl][ok]
            sup = _ratio([r["sn"]] * len(c), c["sn"])
            txt = _ratio([r["tn"]] * len(c), c["tn"])
            good = (sup >= t_sup) & (txt >= t_text)
            if pd.notna(r["date_parsed"]):
                lag = (r["date_parsed"] - c["date_parsed"]).dt.days.to_numpy()
                good &= ~(~np.isnan(lag) & ((lag < lag_lo) | (lag > lag_hi)))
            if good.any() and (max_cand is None or good.sum() <= max_cand):
                j = np.argmax(np.where(good, txt + sup / 10, -1))
                out.loc[ix, "match_to"] = c["record_id"].iloc[j]
                out.loc[ix, "match_kind"] = "window"
                out.loc[ix, "match_note"] = f"{int(good.sum())} candidate(s)"

    # ---- stage 3: repeated exports inside the secondary system (not already matched to primary)
    still = out.loc[S.index, "match_to"] == ""
    if "include" in out.columns:
        still &= out.loc[S.index, "include"] == 1
    rest = S[still].copy()
    rest["cents"] = (rest["amt"] * 100).round().fillna(-1).astype("int64")
    rest["has_po"] = rest["po_src"] == "own"
    rest = rest.sort_values(["has_po", "record_id"], ascending=[False, True])
    for _, g in rest[rest["inv_n"] != ""].groupby(["sn", "inv_n", "cents"]):
        if len(g) < 2:
            continue
        kept = [g.index[0]]
        for ix in g.index[1:]:
            sims = [fuzz.ratio(rest.at[ix, "tn"], rest.at[k, "tn"]) for k in kept]
            if max(sims) >= t_text:
                k = kept[int(np.argmax(sims))]
                out.loc[ix, "match_to"] = out.at[k, "record_id"]
                out.loc[ix, "match_kind"] = "repeat_export"
            else:
                kept.append(ix)
    return out