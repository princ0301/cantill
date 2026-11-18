import re
from pathlib import Path
import pandas as pd
from .clean import parse_amount, parse_dates

ROLES = {
    "date": (["invoice_date", "posting_date", "date", "datum", "fecha", "data", "dt"], "date"),
    "vendor_id": (["vendor_id", "supplier_id", "vendor id", "supplier id", "vendor_no", "supplier_no", "creditor", "lifnr", "vendor_code"], "id"),
    "supplier": (["supplier_name", "vendor_name", "supplier", "vendor", "creditor_name", "proveedor", "fornecedor", "lieferant"], "text"),
    "description": (["line_description", "description", "item_description", "short_text", "desc", "text", "memo", "narrative", "item", "product"], "text"),
    "item_code": (["material_number", "material", "sku", "item_code", "part_number", "article", "matnr"], "id"),
    "quantity": (["quantity", "qty", "menge"], "num"),
    "amount": (["line_amount", "net_amount", "gross_amount", "amount", "value", "total", "net", "gross", "spend", "cost", "price", "betrag"], "num"),
    "currency": (["currency", "curr", "ccy", "waers"], "cur"),
    "currency_rate": (["currency_rate", "exchange_rate", "fx_rate", "rate"], "num"),
    "tax_rate": (["vat_rate", "tax_rate", "vat", "gst"], "num"),
    "status": (["status", "state", "approval"], "cat"),
    "po": (["po_number", "po_reference", "purchase_order", "po_ref", "po", "ebeln"], "id"),
    "invoice": (["invoice_number", "invoice_no", "invoice", "inv_no", "belnr", "bill"], "id"),
    "gl_text": (["gl_account_description", "gl_description", "gl_account", "account", "ledger", "cost_centre", "cost_center"], "text"),
    "doc_type": (["document_type", "doc_type", "blart"], "cat"),
}


def _kind_ok(kind, s):
    nb = s[s.str.strip() != ""]
    if len(nb) == 0:
        return 0.0
    if kind == "num":
        v, _ = parse_amount(nb.head(2000))
        return float(v.notna().mean())
    if kind == "date":
        dt, flag = parse_dates(nb.head(2000))
        return float((flag == "ok").mean())
    if kind == "cur":
        return float(nb.str.fullmatch(r"[A-Za-z]{3}").mean())
    if kind == "cat":
        return 1.0 if nb.nunique() <= 40 else 0.2
    if kind == "text":
        return float(nb.str.len().median() >= 6)
    if kind == "id":
        return 1.0
    return 0.5


def _name_score(col, hints):
    c = col.lower().replace(" ", "_")
    for i, h in enumerate(hints):
        if c == h:
            return 10 - i * 0.1
    for i, h in enumerate(hints):
        if h in c:
            return 5 - i * 0.1
    return 0.0


def profile_frame(df):
    rows = []
    for c in df.columns:
        s = df[c].astype(str)
        nb = s[s.str.strip() != ""]
        rows.append({"column": c, "blank_%": round(100 * (1 - len(nb) / max(len(s), 1)), 1), "distinct": nb.nunique(),
                     "example": (nb.iloc[0][:40] if len(nb) else ""),
                     "shape": re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", nb.iloc[0]))[:24] if len(nb) else ""})
    return pd.DataFrame(rows)


def suggest(df):
    """Greedy best column per role: name match first, content check second."""
    cand = []
    for role, (hints, kind) in ROLES.items():
        for c in df.columns:
            ns = _name_score(c, hints)
            if ns <= 0:
                continue
            ok = _kind_ok(kind, df[c].astype(str))
            if ok < 0.6:
                continue
            cand.append((ns * ok, role, c))
    cand.sort(reverse=True)
    chosen, used = {}, set()
    for score, role, c in cand:
        if role in chosen or c in used:
            continue
        chosen[role] = c
        used.add(c)
    best = None
    for c in df.columns:
        s = df[c].astype(str)
        if s.nunique() == len(s) and (s.str.strip() != "").all():
            sc = _name_score(c, ["line_id", "record_id", "id", "line", "key", "number", "no"])
            if best is None or sc > best[0]:
                best = (sc, c)
    return chosen, (best[1] if best else None)


def draft_config(files, tree_path, company="C", currency="USD", tree_levels=None):
    """files: list of (path, DataFrame). Returns YAML text."""
    out = [f"company: {company}", f"reporting_currency: {currency}", "sources:"]
    for i, (path, df) in enumerate(files):
        roles, idc = suggest(df)
        name = Path(path).stem
        prefix = "" if len(files) == 1 else name[:3].upper() + ":"
        out += [f"  - name: {name}", f"    file: {Path(path).name}",
                f"    id: {{prefix: \"{prefix}\", column: {idc or 'TODO_UNIQUE_ID_COLUMN'}}}   # TODO check: record_id in the output must match what the brief expects",
                "    columns:"]
        for role in ROLES:
            if role in roles:
                out.append(f"      {role}: {roles[role]}")
        if "amount" in roles:
            vals, dec = parse_amount(df[roles["amount"]].astype(str).head(5000))
            gross = "gross" in roles["amount"].lower()
            out.append(f"    amount_basis: {'gross' if gross else 'net'}      # TODO is the amount net of VAT/tax?")
            if gross:
                out.append("    default_tax_rate: 0.15          # TODO rate to assume when blank")
            if dec == ",":
                out.append("    decimal: \",\"")
        if "status" in roles:
            vals = sorted(df[roles["status"]].astype(str).str.strip().unique())[:15]
            out += ["    status:", f"      include: [Approved, Paid]       # TODO statuses seen: {vals}",
                    "      cancelled: [Cancelled]"]
        out.append("")
    for path, df in files:
        roles, _ = suggest(df)
        if "currency" in roles and df[roles["currency"]].astype(str).str.upper().str.strip().nunique() > 1:
            out += ["# TODO several currencies found: add an fx table so amounts convert to the reporting currency",
                    "# fx: {file: fx_rates.csv, currency: currency, month: month, rate: units_per_usd}", ""]
            break
    if tree_levels:
        out += ["tree:", f"  file: {Path(tree_path).name}", f"  levels: [{', '.join(tree_levels)}]", ""]
    out += ["cross_system: null            # only for companies with two overlapping systems (like A)", "",
            "classify:", "  context_fields: []                 # TODO extra columns that hint at the category (type, GL, cost centre)",
            "  llm:", "    model: openai/gpt-oss-120b", "    tpm_budget: 5000", "    domain_hint: \"\"                # TODO e.g. 'a regional hospital group'",
            "    context_fields: []", "    batch_items: 40", "    flat_max_leaves: 80"]
    return "\n".join(out) + "\n"
