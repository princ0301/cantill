"""Compare our include/exclude decisions with a labelled file (e.g. B's seed labels)."""
import numpy as np
import pandas as pd


def exclusion_report(df, labels_path, id_col="line_id", include_col="include"):
    lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    m = df.merge(lab[[id_col, include_col]].rename(columns={include_col: "label_include"}),
                 left_on="record_id", right_on=id_col, how="inner")
    truth = m["label_include"] == "1"
    pred = m["include"] == 1
    out = pd.crosstab(truth.rename("label: included"), pred.rename("ours: included"))
    wrong = m[truth != pred]
    return out, wrong



def hidden_po_check(df, cfg, frac=0.18, seeds=(7, 8, 9)):
    """Validate PO-less matching without labels: hide the PO of a random share of lines whose match
    is already certain (own PO + amount) and see how many are re-found; do the same for lines that
    look tool-only (PO not in the primary system) to estimate false matches."""
    import copy
    import numpy as np
    from .match import match_systems
    cs = cfg["cross_system"]
    full = match_systems(df, cfg)
    S = full[full["source"] == cs["secondary"]].copy()
    P = df[df["source"] == cs["primary"]]
    rx = cs.get("po_regex", r"(\d{6,})")
    S["po_n"] = S["po"].astype(str).str.extract(rx)[0]
    p_po = P["po"].astype(str).str.extract(rx)[0]
    po_of = pd.Series(p_po.values, index=P["record_id"].values)
    pos = S[S["match_kind"] == "po_amount"]["record_id"].to_numpy()
    neg = S[S["po_n"].notna() & ~S["po_n"].isin(set(p_po.dropna()))]["record_id"].to_numpy()
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        hp = set(rng.choice(pos, int(len(pos) * frac), replace=False))
        hn = set(rng.choice(neg, int(len(neg) * frac), replace=False))
        m = match_systems(df, copy.deepcopy(cfg), hide=hp | hn, use_siblings=False).set_index("record_id")
        r = m.loc[list(hp)]
        found = (r["match_to"] != "") & (r["match_kind"] != "repeat_export")
        right = found & (r["match_to"].map(po_of) == r["po"].astype(str).str.extract(rx)[0])
        n = m.loc[list(hn)]
        fp = (n["match_to"] != "") & (n["match_kind"] != "repeat_export")
        rows.append((found.mean(), right.mean(), fp.mean()))
    return pd.DataFrame(rows, columns=["recall", "right_po", "false_match_rate"]).mean()



def pairwise_scores(pred, truth):
    """Pairwise precision/recall of a grouping against a reference grouping."""
    import numpy as np
    ct = pd.crosstab(pd.Series(pred).to_numpy(), pd.Series(truth).to_numpy()).to_numpy().astype("int64")
    c2 = lambda x: (x * (x - 1) // 2).sum()
    tp, p, r = c2(ct), c2(ct.sum(1)), c2(ct.sum(0))
    return (tp / p if p else 1.0), (tp / r if r else 1.0)


def entity_report(df, cfg, labels):
    """Zero-label quality of supplier/item keys against labelled ERP lines, plus tool-vs-ERP-twin consistency."""
    from .entities import build_keys
    z = build_keys(df, cfg, labels=None, inherit=False)
    prim = cfg["cross_system"]["primary"] if cfg.get("cross_system") else df["source"].iloc[0]
    e = z[z["source"] == prim].copy()
    e["gs"] = labels["supplier_key"].reindex(e["record_id"]).to_numpy()
    e["gi"] = labels["item_key"].reindex(e["record_id"]).to_numpy()
    rows = []
    sizes = e["gs"].value_counts()
    big = sizes.index[0]
    for name, sub in [("suppliers, all lines", e), (f"suppliers, without largest group ({sizes.iloc[0]} lines)", e[e["gs"] != big])]:
        p, r = pairwise_scores(sub["supplier_key"], sub["gs"])
        rows.append((name, len(sub), round(p, 4), round(r, 4)))
    c = e[e["gi"].notna() & (e["gi"] != "")]
    multi = c.groupby("gi")["item_code"].nunique()
    for name, sub in [("items, coded lines, key = code/text", c),
                      ("items, without keys that span several codes", c[~c["gi"].isin(multi[multi > 1].index)])]:
        p, r = pairwise_scores(sub["item_key"], sub["gi"])
        rows.append((name, len(sub), round(p, 4), round(r, 4)))
    # items from text alone: hide the codes
    hidden = c.drop(columns=["gs", "gi", "supplier_key", "item_key"]).assign(item_code="")
    h = build_keys(pd.concat([hidden, z[z["source"] != prim]]), cfg, labels=None, inherit=False)
    h = h[h["record_id"].isin(c["record_id"])].merge(c[["record_id", "gi"]], on="record_id")
    keep = ~h["gi"].isin(multi[multi > 1].index)
    p, r = pairwise_scores(h.loc[keep, "item_key"], h.loc[keep, "gi"])
    rows.append(("items from text only (codes hidden)", int(keep.sum()), round(p, 4), round(r, 4)))
    out = pd.DataFrame(rows, columns=["what", "lines", "precision", "recall"])

    cons = []
    if "match_kind" in df.columns:
        t = z[(z["source"] != prim) & (df.set_index("record_id").loc[z["record_id"], "match_kind"].to_numpy() == "po_amount")]
        twin = df.set_index("record_id").loc[t["record_id"], "match_to"].to_numpy()
        zk = z.set_index("record_id")
        cons.append(("tool supplier = ERP twin's supplier", len(t), round((t["supplier_key"].to_numpy() == zk.loc[twin, "supplier_key"].to_numpy()).mean(), 4)))
        coded = zk.loc[twin, "item_code"].to_numpy() != ""
        same = t["item_key"].to_numpy() == zk.loc[twin, "item_key"].to_numpy()
        cons.append(("tool item = ERP twin's item (twin has a code)", int(coded.sum()), round(same[coded].mean(), 4)))
    return out, pd.DataFrame(cons, columns=["what", "lines", "agreement"])



def class_report(out, truth_leaf, n_levels, record_ids=None):
    """Accuracy per tree level (lines and spend-weighted) and a confidence calibration table."""
    from .classify import split_path
    rows = out[(out["include"] == 1) & out["record_id"].isin(truth_leaf.dropna().index)]
    if record_ids is not None:
        rows = rows[rows["record_id"].isin(set(record_ids))]
    truth = truth_leaf.reindex(rows["record_id"]).to_numpy()
    w = rows["spend"].abs().to_numpy()
    res = []
    for i in range(n_levels):
        t = np.array([split_path(x, n_levels)[i] for x in truth])
        ok = rows[f"l{i + 1}"].to_numpy() == t
        res.append((f"L{i + 1}", round(ok.mean(), 4), round((ok * w).sum() / w.sum(), 4)))
    acc = pd.DataFrame(res, columns=["level", "line_accuracy", "spend_weighted_accuracy"])
    leaf_ok = rows["leaf_pred"].to_numpy() == truth
    conf = rows["confidence"].to_numpy()
    bins = pd.cut(conf, [0, .2, .4, .6, .8, .9, 1.0])
    cal = pd.DataFrame({"bin": bins, "ok": leaf_ok, "conf": conf}).groupby("bin", observed=True).agg(
        lines=("ok", "size"), mean_confidence=("conf", "mean"), leaf_accuracy=("ok", "mean")).round(3)
    return len(rows), acc, cal


def holdout_ids(df, labels, frac, seed=3):
    """Hold out a share of labelled lines, grouped by (normalised) text so repeats don't leak."""
    from .match import norm_text
    lab = df[df["record_id"].isin(labels.index[labels["usable"].astype(bool) & labels["leaf"].notna()])]
    g = norm_text(lab["description"])
    ug = g.unique()
    pick = set(np.random.default_rng(seed).choice(ug, int(len(ug) * frac), replace=False))
    return set(lab["record_id"][g.isin(pick)])



def class_report_by_method(out, truth_leaf, n_levels, record_ids=None):
    """Accuracy split by how each line was labelled (llm, llm+model, self_trained, supervised, ...)."""
    from .classify import split_path
    rows = out[(out["include"] == 1) & out["record_id"].isin(truth_leaf.dropna().index)]
    if record_ids is not None:
        rows = rows[rows["record_id"].isin(set(record_ids))]
    truth = truth_leaf.reindex(rows["record_id"]).to_numpy()
    t1 = np.array([split_path(x, n_levels)[0] for x in truth])
    d = pd.DataFrame({"method": rows["class_method"].to_numpy(), "l1": rows["l1"].to_numpy() == t1,
                      "leaf": rows["leaf_pred"].to_numpy() == truth, "w": rows["spend"].abs().to_numpy()})
    res = []
    for m, g in d.groupby("method"):
        res.append((m, len(g), g["l1"].mean(), g["leaf"].mean(), (g["leaf"] * g["w"]).sum() / max(g["w"].sum(), 1e-9)))
    return pd.DataFrame(res, columns=["method", "lines", "L1_accuracy", "leaf_accuracy", "spend_weighted_leaf"]).round(4).set_index("method")