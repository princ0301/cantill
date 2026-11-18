"""Supplier and item keys.

Suppliers: normalise names (case, punctuation, legal suffixes, OCR look-alikes), merge near-identical
names, and merge names that share a vendor id (unless that id is a collective account).
Items: the item code when there is one; otherwise the normalised description, linked to a coded
item when the same text is already known as that item.
If labelled data is supplied (training labels), names/codes seen there get the labelled key.
Records matched to a primary-system record inherit that record's keys.
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from .match import norm_text, norm_supplier, _LEGAL

PLACEHOLDER = r"(?:NOOBJECT|NA|NONE|NULL|TBD|UNKNOWN|OTHER|MISC|ITEM|PRODUCT)"


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _fuzzy_pairs(left, right, thr, chunk=1000):
    """Yield (i, j) for left[i] ~ right[j] with ratio >= thr, in chunks to bound memory."""
    for s in range(0, len(left), chunk):
        M = process.cdist(left[s:s + chunk], right, scorer=fuzz.ratio, score_cutoff=thr,
                          dtype=np.uint8, workers=-1)
        ii, jj = np.nonzero(M)
        for i, j in zip(ii, jj):
            yield s + i, j


def _slug(text):
    text = re.sub(_LEGAL, " ", text.upper())
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "unknown"


def _majority(keys):
    return keys.value_counts().index[0]


def supplier_keys(df, cfg, labels=None):
    ent = cfg.get("entities", {})
    thr = ent.get("supplier_merge_threshold", 92)
    max_names = ent.get("max_names_per_vendor", 3)
    sn = norm_supplier(df["supplier"])
    named = sn != ""
    names = sorted(sn[named].unique())
    pos = {n: i for i, n in enumerate(names)}
    uf = _UF(len(names))
    for i, j in _fuzzy_pairs(names, names, thr):
        uf.union(i, j)

    vend = df["vendor_id"].astype(str).str.strip()
    both = named & (vend != "")
    vn = pd.DataFrame({"v": vend[both], "n": sn[both]}).drop_duplicates()
    for _, g in vn.groupby("v"):
        if 1 < g["n"].nunique() <= max_names:
            first = pos[g["n"].iloc[0]]
            for n in g["n"].iloc[1:]:
                uf.union(first, pos[n])

    root = sn.map(lambda n: uf.find(pos[n]) if n in pos else -1)
    # readable cluster names: most common raw name per cluster
    raw = df["supplier"].astype(str).str.strip()
    disp = {r: _slug(_majority(raw[root == r])) for r in root[root >= 0].unique()}
    seen, final = {}, {}
    for r, s in disp.items():
        seen[s] = seen.get(s, 0) + 1
        final[r] = s if seen[s] == 1 else f"{s}-{seen[s]}"
    key = root.map(final).where(root >= 0, "")

    if labels is not None and "supplier_key" in labels:
        lk = labels["supplier_key"].reindex(df["record_id"]).to_numpy()
        lab = pd.DataFrame({"sn": sn.to_numpy(), "k": lk})
        lab = lab[(lab["sn"] != "") & lab["k"].notna() & (lab["k"] != "")]
        by_name = lab.groupby("sn")["k"].agg(_majority)
        # propagate: names without a label take the majority label of their cluster
        name_root = pd.Series({n: uf.find(pos[n]) for n in names})
        known = by_name.to_frame("k").join(name_root.rename("r"))
        cluster_key = known.groupby("r")["k"].agg(_majority)
        per_name = {n: by_name.get(n, cluster_key.get(name_root[n])) for n in names}
        mapped = sn.map(per_name)
        key = mapped.where(mapped.notna(), key)

    # no usable name: vendor id of a known cluster, else the vendor id, else a unique placeholder
    nokey = key == ""
    if nokey.any():
        v2key = pd.Series(key[both].values, index=vend[both].values)
        v2key = v2key[~v2key.index.duplicated()]
        by_vendor = vend[nokey].map(v2key)
        fallback = pd.Series(np.where(vend[nokey] != "", "v:" + vend[nokey], ""), index=vend[nokey].index)
        key[nokey] = by_vendor.fillna(fallback)
    key = key.where(key != "", "unk:" + df["record_id"])
    return key


def item_keys(df, cfg, labels=None):
    ent = cfg.get("entities", {})
    min_share = ent.get("text_link_min_share", 0.8)
    fuzzy_thr = ent.get("item_fuzzy_threshold", 92)
    ph = re.compile(ent.get("item_placeholders", PLACEHOLDER))
    code = df["item_code"].astype(str).str.strip().str.upper()
    coded = code != ""
    tn = norm_text(df["description"])
    dig = df["description"].astype(str).str.replace(r"\D", "", regex=True)
    junk = (tn.str.len() < 3) | tn.map(lambda t: bool(ph.fullmatch(t)))

    key = pd.Series("", index=df.index)
    key[coded] = "c:" + code[coded]
    if labels is not None and "item_key" in labels:
        lk = pd.Series(labels["item_key"].reindex(df["record_id"]).to_numpy(), index=df.index)
        lab = pd.DataFrame({"c": code[coded], "k": lk[coded]})
        lab = lab[lab["k"].notna() & (lab["k"] != "")]
        if len(lab):
            by_code = lab.groupby("c")["k"].agg(_majority)
            key[coded] = code[coded].map(by_code).fillna(key[coded])

    # text -> coded item, when the same text is (almost) always the same item
    ref = pd.DataFrame({"tn": tn[coded & ~junk], "k": key[coded & ~junk]})
    tab = ref.groupby("tn")["k"].agg(lambda s: (s.value_counts().index[0], s.value_counts().iloc[0] / len(s)))
    tab = tab[tab.map(lambda x: x[1] >= min_share)].map(lambda x: x[0])

    todo = ~coded & ~junk
    link = tn[todo].map(tab)
    key[todo] = link.fillna("")
    miss = todo & (key == "")
    if miss.any() and len(tab) and fuzzy_thr:
        ref_t = ref.drop_duplicates("tn").set_index("tn")
        ref_names = [t for t in ref_t.index if t in tab.index]
        ref_dig = pd.Series(df.loc[coded & ~junk].groupby(tn[coded & ~junk])["description"]
                            .first().str.replace(r"\D", "", regex=True))
        ut = tn[miss].drop_duplicates()
        got = {}
        ud = dig[miss].groupby(tn[miss]).first()
        for i, j in _fuzzy_pairs(ut.tolist(), ref_names, fuzzy_thr):
            t = ut.iloc[i]
            if ud[t] == ref_dig[ref_names[j]] and t not in got:
                got[t] = tab[ref_names[j]]
        key[miss] = tn[miss].map(got).astype(object).fillna("")
    left = todo & (key == "")
    key[left] = "t:" + tn[left]
    key[junk & ~coded] = "unk:" + df["record_id"][junk & ~coded]
    return key


def build_keys(df, cfg, labels=None, inherit=True):
    """Add supplier_key and item_key. Matched records inherit their match's keys."""
    out = df.copy()
    # supplier_cluster: name-based only (what the classifier uses as context);
    # supplier_key: the reported key (may use labelled keys, which can lump unresolved names together)
    out["supplier_cluster"] = supplier_keys(out, cfg, None)
    out["supplier_key"] = out["supplier_cluster"] if labels is None else supplier_keys(out, cfg, labels)
    out["item_key"] = item_keys(out, cfg, labels)
    if inherit and "match_to" in out.columns:
        m = out["match_to"] != ""
        if m.any():
            tgt = out.loc[m, "match_to"]
            for col in ("supplier_cluster", "supplier_key", "item_key"):
                out.loc[m, col] = tgt.map(out.set_index("record_id")[col]).to_numpy()
    return out
