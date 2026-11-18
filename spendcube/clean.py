"""Clean amounts, tax, currency and dates. Everything is driven by the config."""
import numpy as np
import pandas as pd
from .config import ConfigError, find_file

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def parse_amount(s, decimal=None):
    """Parse messy number text. Returns (float Series, decimal char used).

    Handles thousand separators, decimal commas ("1.234,56"), currency symbols,
    "(123.45)" and trailing/leading minus as negatives. Unparseable -> NaN.
    """
    x = s.astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    neg = ((x.str.startswith("(") & x.str.endswith(")"))
           | x.str.startswith("-") | x.str.endswith("-"))
    core = x.str.replace(r"[^0-9.,]", "", regex=True)

    if decimal is None:
        has_dot = core.str.contains(".", regex=False)
        has_com = core.str.contains(",", regex=False)
        m = core.str.extract(r"([.,])(\d+)$")
        sep, nd = m[0], m[1].str.len()
        vote_dot = ((sep == ".") & ((nd != 3) | has_com)).sum()
        vote_com = ((sep == ",") & ((nd != 3) | has_dot)).sum()
        decimal = "," if vote_com > vote_dot else "."

    if decimal == ".":
        core = core.str.replace(",", "", regex=False)
    else:
        core = core.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    val = pd.to_numeric(core, errors="coerce")
    val = val.where(~neg, -val)
    return val, decimal


def parse_tax(s, default):
    """Return (rate Series as fraction, source Series 'given'|'default'|'bad')."""
    x = s.astype(str).str.strip()
    has_pct = x.str.contains("%", regex=False)
    num = pd.to_numeric(x.str.replace("%", "", regex=False).str.replace(",", ".", regex=False).str.strip(),
                        errors="coerce")
    rate = pd.Series(np.where(has_pct | (num > 1), num / 100, num), index=x.index)
    blank = x == ""
    bad = rate.isna() & ~blank
    source = pd.Series("given", index=x.index).where(~(blank | bad), "default")
    source = source.where(~bad, "bad")
    return rate.fillna(default), source


def parse_dates(s, default_order="dmy"):
    """Parse mixed date formats. Returns (datetime Series, flag Series).

    flag: ok | blank | unparseable. Ambiguous a/b/yyyy dates are resolved per row
    (a>12 -> day first, b>12 -> month first), otherwise by the column's majority,
    otherwise default_order.
    """
    x = s.astype(str).str.strip()
    y = pd.Series(np.nan, index=x.index)
    mo = y.copy()
    d = y.copy()

    def put(mask, yy, mm, dd):
        y[mask], mo[mask], d[mask] = yy[mask], mm[mask], dd[mask]

    todo = x != ""
    # yyyy-mm-dd (also / or .), optional time
    m = x.str.extract(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s].*)?$").astype(float)
    k = todo & m[0].notna()
    put(k, m[0], m[1], m[2]); todo &= ~k
    # yyyymmdd
    m = x.str.extract(r"^(\d{4})(\d{2})(\d{2})$").astype(float)
    k = todo & m[0].notna()
    put(k, m[0], m[1], m[2]); todo &= ~k
    # dd-Mon-yy(yy)
    m = x.str.extract(r"^(\d{1,2})[-\s/.]([A-Za-z]{3})[A-Za-z]*[-\s/.,]+(\d{2,4})$")
    mon = m[1].str.upper().map(_MONTHS)
    yr = pd.to_numeric(m[2], errors="coerce")
    yr = yr.where(yr >= 100, np.where(yr < 70, 2000 + yr, 1900 + yr))
    k = todo & mon.notna()
    put(k, yr, mon, pd.to_numeric(m[0], errors="coerce")); todo &= ~k
    # a/b/yyyy: ambiguous order
    m = x.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$").apply(pd.to_numeric, errors="coerce")
    k = todo & m[0].notna()
    if k.any():
        a, b = m[0], m[1]
        yy = m[2].where(m[2] >= 100, np.where(m[2] < 70, 2000 + m[2], 1900 + m[2]))
        dmy_ev, mdy_ev = (k & (a > 12)).sum(), (k & (b > 12)).sum()
        col_default = default_order if dmy_ev == mdy_ev else ("dmy" if dmy_ev > mdy_ev else "mdy")
        day_first = np.where(a > 12, True, np.where(b > 12, False, col_default == "dmy"))
        day = pd.Series(np.where(day_first, a, b), index=x.index)
        mon2 = pd.Series(np.where(day_first, b, a), index=x.index)
        put(k, yy, mon2, day)

    dt = pd.to_datetime(pd.DataFrame({"year": y, "month": mo, "day": d}), errors="coerce")
    flag = pd.Series("ok", index=x.index)
    flag[x == ""] = "blank"
    flag[(x != "") & dt.isna()] = "unparseable"
    return dt, flag


def load_fx(fxcfg, data_dir, extra_dirs=()):
    path = find_file(fxcfg["file"], data_dir, *extra_dirs)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cur = df[fxcfg["currency"]].str.strip().str.upper()
    per = pd.to_datetime(df[fxcfg["month"]].str.strip().str[:7], format="%Y-%m", errors="coerce")
    rate = pd.to_numeric(df[fxcfg["rate"]], errors="coerce")
    ok = per.notna() & rate.notna() & (rate > 0)
    t = pd.DataFrame({"cur": cur[ok], "m": (per[ok].dt.year * 12 + per[ok].dt.month - 1), "rate": rate[ok]})
    t = t.groupby(["cur", "m"], as_index=False).rate.mean().sort_values(["cur", "m"])
    return {c: (g.m.to_numpy(), g.rate.to_numpy()) for c, g in t.groupby("cur")}


def fx_lookup(table, cur, month_int):
    """Rate for (currency, month); nearest available month if exact is missing.

    Returns (rate array, method array: table | table_nearest | missing).
    """
    rate = np.full(len(cur), np.nan)
    method = np.full(len(cur), "missing", dtype=object)
    for c in pd.unique(cur):
        if c not in table:
            continue
        idx = np.where(cur == c)[0]
        months, rates = table[c]
        mi = month_int[idx]
        valid = ~np.isnan(mi)
        if not valid.any():
            continue
        q = mi[valid].astype(int)
        pos = np.searchsorted(months, q)
        left = np.clip(pos - 1, 0, len(months) - 1)
        right = np.clip(pos, 0, len(months) - 1)
        use_left = np.abs(months[left] - q) <= np.abs(months[right] - q)   # ties -> earlier month
        best = np.where(use_left, left, right)
        exact = months[best] == q
        rate[idx[valid]] = rates[best]
        method[idx[valid]] = np.where(exact, "table", "table_nearest")
    return rate, method


def clean(df, cfg, data_dir):
    """Add cleaned columns to the ingested frame (one pass per source)."""
    df = df.copy()
    reporting = cfg.get("reporting_currency", "USD").upper()
    fxtab = load_fx(cfg["fx"], data_dir, cfg.get("_extra_dirs", [])) if cfg.get("fx") else None
    fx_min = min(v[0][0] for v in fxtab.values()) if fxtab else None
    fx_max = max(v[0][-1] for v in fxtab.values()) if fxtab else None
    order = cfg.get("date_order", "dmy")

    out_cols = ["amount_num", "tax_rate_used", "tax_source", "net_local", "currency_clean",
                "fx_rate_used", "fx_method", "spend_rep", "date_parsed", "date_flag", "amount_flag"]
    for c in out_cols:
        df[c] = np.nan if c in ("amount_num", "tax_rate_used", "net_local", "fx_rate_used", "spend_rep") else ""
    df["date_parsed"] = pd.NaT

    for src in cfg["sources"]:
        mask = df["source"] == src["name"]
        if not mask.any():
            continue
        s = df.loc[mask]

        amt, dec = parse_amount(s["amount"], src.get("decimal"))
        df.loc[mask, "amount_num"] = amt
        df.loc[mask, "amount_flag"] = np.where(amt.isna(), "unparseable", "ok")

        if src.get("amount_basis", "net") == "gross":
            rate, tsrc = parse_tax(s["tax_rate"], src.get("default_tax_rate", 0.0))
            net = amt / (1 + rate)
        else:
            rate = pd.Series(0.0, index=s.index)
            tsrc = pd.Series("n/a", index=s.index)
            net = amt
        df.loc[mask, "tax_rate_used"] = rate
        df.loc[mask, "tax_source"] = tsrc
        df.loc[mask, "net_local"] = net

        dt, dflag = parse_dates(s["date"], src.get("date_order", order))
        df.loc[mask, "date_parsed"] = dt
        df.loc[mask, "date_flag"] = dflag

        cur = s["currency"].str.strip().str.upper()
        cur_clean = cur.where(cur != "", reporting)
        df.loc[mask, "currency_clean"] = cur_clean

        line_rate = pd.to_numeric(s["currency_rate"], errors="coerce")
        has_line = line_rate.notna() & (line_rate > 0)
        same = cur_clean == reporting
        method = pd.Series("missing", index=s.index)
        used = pd.Series(np.nan, index=s.index)
        method[same] = np.where(cur[same] == "", "assumed_reporting", "same")
        used[same] = 1.0
        use_line = ~same & has_line
        method[use_line] = "line_rate"
        used[use_line] = line_rate[use_line]

        need = ~same & ~has_line
        if need.any() and fxtab is not None:
            mi = (dt.dt.year * 12 + dt.dt.month - 1).astype(float)
            fb_col = cfg["fx"].get("fallback_date")
            if fb_col:
                fdt, _ = parse_dates(s[fb_col], order)
                fmi = (fdt.dt.year * 12 + fdt.dt.month - 1).astype(float)
                bad_primary = mi.isna() | (mi < fx_min) | (mi > fx_max)
                mi = mi.where(~bad_primary, fmi)
            r, mth = fx_lookup(fxtab, cur_clean[need].to_numpy(), mi[need].to_numpy())
            used[need] = r
            method[need] = mth
        df.loc[mask, "fx_rate_used"] = used
        df.loc[mask, "fx_method"] = method
        df.loc[mask, "spend_rep"] = net / used          # units of line currency per reporting unit

    return df
