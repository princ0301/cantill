"""Classify lines into the company's category tree (any number of levels, read from a file).

Methods, used in this order of preference:
  supervised   labelled lines exist (training labels / seed labels): TF-IDF + context -> linear model
  zero-shot    no labels: lines are scored against the tree's own words, then self-trained
Optional layers on top: an LLM (Groq) labels unique items; local embeddings improve the zero-shot step.
Confidence is calibrated on held-out data when labels exist.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from .config import ConfigError
from .match import _OCR, norm_text

SEP = "|"


# ------------------------------------------------------------------ tree
def load_tree(cfg, data_dir):
    """Leaves of the tree: one row per leaf with the level names, a path string and the depth."""
    tc = cfg.get("tree")
    if not tc:
        raise ConfigError("config has no 'tree' block")
    path = Path(data_dir) / tc["file"]
    if not path.exists():
        raise ConfigError(f"tree file not found: {path}")
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    t.columns = [c.strip() for c in t.columns]
    levels = tc["levels"]
    missing = [c for c in levels if c not in t.columns]
    if missing:
        raise ConfigError(f"tree file lacks columns {missing}. Available: {list(t.columns)}")
    t = t[levels].apply(lambda c: c.str.strip())
    t = t[t[levels[0]] != ""].drop_duplicates().reset_index(drop=True)
    t["depth"] = (t[levels] != "").sum(axis=1)
    t["path"] = t[levels].apply(lambda r: SEP.join(v for v in r if v), axis=1)
    return t


def split_path(path, n):
    parts = path.split(SEP) if path else []
    return parts + [""] * (n - len(parts))


# -------------------------------------------------------------- features
def _ctx_tokens(df, fields):
    sup = df["supplier_cluster"] if "supplier_cluster" in df.columns else df["supplier_key"]
    cols = [("S:" + sup.astype(str))]
    for f in fields:
        if f in df.columns:
            cols.append(f"{f}=" + df[f].astype(str).str.strip().str.upper())
    out = cols[0]
    for c in cols[1:]:
        out = out + "||" + c
    return out


def _split_ctx(s):
    return s.split("||")


def _text(df):
    return (df["description"].astype(str) + " " + df["gl_text"].astype(str)).str.upper().str.translate(_OCR)


class Features:
    def __init__(self, ctx_fields=(), ctx_weight=2.0, extra_weight=1.0):
        self.ctx_fields, self.w, self.ew = list(ctx_fields), ctx_weight, extra_weight
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self.word = TfidfVectorizer(token_pattern=r"[A-Z0-9]{2,}", min_df=2, sublinear_tf=True)
        self.ctx = TfidfVectorizer(analyzer=_split_ctx)

    def _blocks(self, df, extra, fit):
        t = _text(df)
        c = _ctx_tokens(df, self.ctx_fields)
        if fit:
            blocks = [self.char.fit_transform(t), self.word.fit_transform(t), self.w * self.ctx.fit_transform(c)]
        else:
            blocks = [self.char.transform(t), self.word.transform(t), self.w * self.ctx.transform(c)]
        if extra is not None:
            blocks.append(csr_matrix(self.ew * np.asarray(extra, dtype="float32")))
        return hstack(blocks).tocsr()

    def fit_transform(self, df, extra=None):
        return self._blocks(df, extra, True)

    def transform(self, df, extra=None):
        return self._blocks(df, extra, False)


def _fit(df, y, ctx_fields, C=0.5, extra=None):
    f = Features(ctx_fields)
    X = f.fit_transform(df, extra)
    return f, LinearSVC(C=C).fit(X, y)


def _scores(model, X):
    """Decision scores as an (n, classes) matrix, also for two classes (where sklearn returns 1-D)."""
    sc = model.decision_function(X)
    return np.column_stack([-sc, sc]) if sc.ndim == 1 else sc


def _margin(scores):
    s = np.sort(scores, axis=1)
    return s[:, -1] - s[:, -2]


# ------------------------------------------------------------ supervised
def supervised(train, y, target, ctx_fields, folds=5):
    """Train on `train`, predict `target`. Returns (leaf path, confidence, calibration table)."""
    groups = norm_text(train["description"]).to_numpy()
    oof_ok, oof_margin = [], []
    for tr, te in GroupKFold(n_splits=folds).split(train, groups=groups):
        f, m = _fit(train.iloc[tr], y.iloc[tr], ctx_fields)
        sc = _scores(m, f.transform(train.iloc[te]))
        pred = m.classes_[sc.argmax(1)]
        oof_ok.append(pred == y.iloc[te].to_numpy())
        oof_margin.append(_margin(sc))
    ok, mg = np.concatenate(oof_ok), np.concatenate(oof_margin)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.99).fit(mg, ok.astype(float))
    f, m = _fit(train, y, ctx_fields)
    sc = _scores(m, f.transform(target))
    pred = m.classes_[sc.argmax(1)]
    return pred, iso.predict(_margin(sc)), float(ok.mean())


# ------------------------------------------------------------- zero-shot
def zero_shot(target, tree, ctx_fields, emb=None, leaf_emb=None, rounds=2, seed_share=0.3):
    """No labels: score lines against the tree's words (and embeddings if given), then self-train."""
    leaf_text = tree["path"].str.replace(SEP, " ", regex=False).str.upper()
    t = _text(target)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(
        pd.concat([t.drop_duplicates(), leaf_text]))
    S = (v.transform(t) @ v.transform(leaf_text).T).toarray()
    if emb is not None and leaf_emb is not None:
        S = emb @ leaf_emb.T + 0.5 * S
    leaves = tree["path"].to_numpy()
    pred, margin = leaves[S.argmax(1)], _margin(S)
    for _ in range(rounds):
        seed = margin >= np.quantile(margin, 1 - seed_share)
        if len(set(pred[seed])) < 2:
            break
        f, m = _fit(target[seed], pd.Series(pred[seed], index=target.index[seed]), ctx_fields,
                    extra=None if emb is None else emb[seed])
        sc = _scores(m, f.transform(target, emb))
        pred, margin = m.classes_[sc.argmax(1)], _margin(sc)
    conf = np.clip(0.05 + 0.4 * (margin - margin.min()) / (np.ptp(margin) + 1e-9), 0.05, 0.45)
    return pred, conf


def self_train(target, seed, ctx_fields, groups, extra=None, folds=5):
    """Learn from (LLM) seed labels with supplier context.

    seed: Series aligned to target, leaf path or NaN. Returns
      oof        Series: out-of-fold prediction for seeded rows (NaN elsewhere) - used to check agreement
      final      array: prediction of a model trained on all seeds, for every row
      margin     array: its margin
      oof_margin array: margin of the out-of-fold prediction (NaN where none)
    """
    has = seed.notna().to_numpy()
    y = seed[has]
    oof = pd.Series(np.nan, index=target.index, dtype=object)
    oof_margin = np.full(len(target), np.nan)
    if has.sum() < 50 or y.nunique() < 2:
        return oof, None, None, oof_margin
    pos = np.flatnonzero(has)
    g = pd.Series(groups.to_numpy()[pos])
    ex = None if extra is None else np.asarray(extra)
    n_splits = min(folds, g.nunique())
    if n_splits >= 2:
        for tr, te in GroupKFold(n_splits=n_splits).split(pos, groups=g):
            ytr = y.iloc[tr]
            if ytr.nunique() < 2:
                continue
            f, m = _fit(target.iloc[pos[tr]], ytr, ctx_fields, extra=None if ex is None else ex[pos[tr]])
            sc = _scores(m, f.transform(target.iloc[pos[te]], None if ex is None else ex[pos[te]]))
            oof.iloc[pos[te]] = m.classes_[sc.argmax(1)]
            oof_margin[pos[te]] = _margin(sc)
    f, m = _fit(target.iloc[pos], y, ctx_fields, extra=None if ex is None else ex[pos])
    sc = _scores(m, f.transform(target, ex))
    return oof, m.classes_[sc.argmax(1)], _margin(sc), oof_margin


# ----------------------------------------------------------------- driver
def _top(s):
    return s.value_counts().index[0]


def item_table(target, llm_ctx_fields=()):
    """Unique items among the target rows (largest spend first) for LLM labelling."""
    g = target.groupby("item_key")
    tbl = pd.DataFrame({
        "text": g["description"].agg(_top),
        "supplier": g["supplier"].agg(_top) if "supplier" in target.columns else "",
        "spend": g["spend"].agg(lambda s: s.abs().sum()),
    })
    ctx = ""
    for f in llm_ctx_fields:
        if f in target.columns:
            part = g[f].agg(_top).astype(str).str.slice(0, 40)
            ctx = part if isinstance(ctx, str) else ctx + "; " + part
    tbl["ctx"] = ctx
    tbl = tbl[~tbl.index.astype(str).str.startswith("unk:") & (tbl["text"].astype(str).str.strip().str.len() >= 3)]
    tbl = tbl.sort_values("spend", ascending=False)
    tbl.index.name = "item_key"
    return tbl.reset_index()


def classify(df, cfg, tree, labels=None, llm=None, embedder=None, llm_limit=None, refine_below=None):
    """Add l1..lN, confidence, class_method for included, non-duplicate lines.

    labels + llm    : supervised model; the LLM re-labels items the model is unsure about (< refine_below)
    no labels + llm : LLM labels items (largest spend first); a supplier-context model trained on those
                      labels fills the rest and checks agreement
    no labels, no llm: embeddings (if given) / tree-word similarity, then self-training
    """
    cc = cfg.get("classify", {})
    lc = cc.get("llm", {})
    ctx = cc.get("context_fields", [])
    n = len(cfg["tree"]["levels"])
    out = df.copy()
    if "item_key" not in out.columns:
        out["item_key"] = norm_text(out["description"])
    for i in range(n):
        out[f"l{i + 1}"] = ""
    out["confidence"], out["class_method"] = np.nan, ""
    target_mask = (out["include"] == 1).to_numpy()
    target = out[target_mask]

    emb = leaf_emb = None
    if embedder is not None:
        emb = embedder.encode((target["description"].astype(str) + " " + target["gl_text"].astype(str)).tolist())
        leaf_emb = embedder.encode(tree["path"].str.replace(SEP, " ", regex=False).tolist())

    train = y = None
    if labels is not None and "leaf" in labels:
        lab = labels["leaf"].reindex(out["record_id"])
        usable = labels["usable"].reindex(out["record_id"]).fillna(False).astype(bool).to_numpy()
        ok = lab.notna().to_numpy() & lab.isin(set(tree["path"])).to_numpy() & usable
        train, y = out[ok], pd.Series(lab.to_numpy()[ok], index=out.index[ok])
    is_sup = train is not None and len(train) >= 50 and y.nunique() >= 2
    if is_sup:
        pred, conf, cv_acc = supervised(train, y, target, ctx)
        out.attrs["cv_leaf_accuracy"] = cv_acc
        base = "supervised"
    else:
        pred, conf = zero_shot(target, tree, ctx, emb, leaf_emb)
        base = "zero_shot+emb" if emb is not None else "zero_shot"
    pred = np.asarray(pred, dtype=object)
    conf = np.asarray(conf, dtype=float)
    method = np.full(len(target), base, dtype=object)

    if llm is not None:
        tbl = item_table(target, lc.get("context_fields", []))
        if is_sup:
            thr = refine_below if refine_below is not None else lc.get("refine_below", 0.6)
            item_conf = pd.Series(conf, index=target.index).groupby(target["item_key"]).mean()
            tbl = tbl[tbl["item_key"].map(item_conf) < thr]
        res = llm.label_items(tbl, limit=llm_limit)
        a, b = lc.get("conf_map", [0.35, 0.55])
        if len(res):
            seed = target["item_key"].map(res.set_index("item_key")["leaf"])
            lconf = target["item_key"].map(res.set_index("item_key")["conf"]).to_numpy(dtype=float)
            has = seed.notna().to_numpy()
            if is_sup:
                pred[has] = seed[has].to_numpy()
                conf[has] = np.clip(a + b * lconf[has], 0.05, 0.95)
                method[has] = "llm"
            else:
                oof, final, margin, oof_margin = self_train(target, seed, ctx, target["item_key"], extra=emb)
                agree = (oof.to_numpy() == seed.to_numpy())
                bonus = np.where(oof.isna().to_numpy(), 0.0, np.where(agree, 0.05, -0.10))
                pred[has] = seed[has].to_numpy()
                conf[has] = np.clip(a + b * lconf[has] + bonus[has], 0.05, 0.95)
                method[has] = "llm"
                llm_leaf = pred.copy()
                so = lc.get("second_opinion", {})
                if so.get("enabled", True) and final is not None and np.isfinite(oof_margin).any():
                    thr = np.nanquantile(oof_margin, so.get("margin_quantile", 0.7))
                    flip = (has & oof.notna().to_numpy() & ~agree & (oof_margin >= thr)
                            & (np.nan_to_num(lconf, nan=1.0) <= so.get("max_llm_conf", 0.8)))
                    pred[flip] = oof.to_numpy()[flip]
                    conf[flip] = np.clip(0.30 + 0.25 * np.minimum(oof_margin[flip] / (thr + 1e-9) - 1, 1), 0.3, 0.55)
                    method[flip] = "llm+model"
                out.attrs["llm_only_pred"] = pd.Series(llm_leaf, index=target.index)
                if final is not None:
                    rest = ~has
                    pred[rest] = final[rest]
                    conf[rest] = np.clip(0.05 + 0.4 * (margin[rest] - margin.min()) / (np.ptp(margin) + 1e-9), 0.05, 0.45)
                    method[rest] = "self_trained"

    parts = [split_path(p, n) for p in pred]
    for i in range(n):
        out.loc[target_mask, f"l{i + 1}"] = [p[i] for p in parts]
    out.loc[target_mask, "confidence"] = conf
    out.loc[target_mask, "class_method"] = method
    out["leaf_pred"] = ""
    out.loc[target_mask, "leaf_pred"] = pred
    return out