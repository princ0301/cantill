import argparse
import os
from pathlib import Path
import pandas as pd
from .config import load_config
from .ingest import ingest, load_labels
from .clean import clean
from .exclusions import apply_exclusions, apply_duplicates, rule_coverage
from .evaluate import (exclusion_report, hidden_po_check, entity_report, pairwise_scores, class_report,
                       class_report_by_method, holdout_ids)
from .entities import build_keys
from .classify import load_tree, classify
from .embed import make_embedder, EmbeddingsUnavailable
from .pipeline import run_pipeline
from .output import validate_output


def load_dotenv(path=".env"):
    """Load unset environment variables from a simple .env file."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def add_class_args(z):
    z.add_argument("--llm", action="store_true", help="label items with Groq (needs GROQ_API_KEY)")
    z.add_argument("--llm-model", help="Groq model (default: config or llama-3.3-70b-versatile)")
    z.add_argument("--llm-limit", type=int, help="label only the N largest-spend items (quick/cheap test)")
    z.add_argument("--tpm", type=int, help="tokens per minute budget (free tier 8B: 6000, 70B: 12000; paid: much higher)")
    z.add_argument("--cache-dir", default=".llm_cache", help="where LLM answers are cached")
    z.add_argument("--refine-below", type=float, help="with labels: send items below this confidence to the LLM")
    z.add_argument("--no-second-opinion", action="store_true", help="never let the model override an LLM label")
    z.add_argument("--embeddings", action="store_true", help="use local sentence embeddings (see embed.py)")
    z.add_argument("--embeddings-backend", help="sentence-transformers | fastembed | hash")


def make_llm_factory(args, cfg):
    """Returns a function tree -> Labeler (or None when --llm is not given)."""
    cc = cfg.setdefault("classify", {})
    if args.no_second_opinion:
        cc.setdefault("llm", {}).setdefault("second_opinion", {})["enabled"] = False
    if not args.llm:
        return None

    def factory(tree):
        from .llm import GroqClient, Labeler
        lcfg = cc.get("llm", {})
        model = args.llm_model or lcfg.get("model", "llama-3.3-70b-versatile")
        client = GroqClient(model=model, tpm_budget=args.tpm or lcfg.get("tpm_budget", 5000), rpm=lcfg.get("rpm", 25))
        print(f"LLM: {model}, tokens/min budget {client.tpm}, cache {args.cache_dir}")
        return Labeler(tree, cfg, client, cache_path=Path(args.cache_dir) / f"{cfg['company']}_{model.replace('/', '_')}.jsonl")
    return factory


def make_embedder_from_args(args, cfg):
    if not (args.embeddings or args.embeddings_backend):
        return None
    ecfg = dict(cfg.setdefault("classify", {}).get("embeddings", {}))
    if args.embeddings_backend:
        ecfg["backend"] = args.embeddings_backend
    try:
        embedder = make_embedder(ecfg)
        embedder.encode(["test"])
        print(f"embeddings: {ecfg.get('backend', 'sentence-transformers')} {ecfg.get('model', '')}")
        return embedder
    except EmbeddingsUnavailable as e:
        print(f"WARNING: embeddings unavailable, continuing without them: {e}")
        return None


def main():
    load_dotenv()
    p = argparse.ArgumentParser(prog="spendcube")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="read files and show a summary")
    s.add_argument("--config", required=True)
    s.add_argument("--data-dir", required=True)
    c = sub.add_parser("clean", help="ingest + clean amounts, tax, fx, dates; print a summary")
    c.add_argument("--config", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--save", help="optional path to write the cleaned table as CSV (for exploration)")
    x = sub.add_parser("exclude", help="ingest + clean + decide what counts as spend")
    x.add_argument("--config", required=True)
    x.add_argument("--data-dir", required=True)
    x.add_argument("--labels", help="labelled file with line_id and include columns (e.g. seed_labels.csv)")
    x.add_argument("--save")
    k = sub.add_parser("check-match", help="validate PO-less matching by hiding known POs")
    k.add_argument("--config", required=True)
    k.add_argument("--data-dir", required=True)
    y = sub.add_parser("keys", help="build supplier and item keys and score them against labels")
    y.add_argument("--config", required=True)
    y.add_argument("--data-dir", required=True)
    y.add_argument("--labels-dir", help="where the labels file lives (default: --data-dir)")
    y.add_argument("--no-labels", action="store_true", help="ignore labels when building keys")
    y.add_argument("--save")
    z = sub.add_parser("classify", help="classify lines into the tree and score against labels")
    z.add_argument("--config", required=True)
    z.add_argument("--data-dir", required=True)
    z.add_argument("--labels-dir", help="where the labels file lives (default: --data-dir)")
    z.add_argument("--no-labels", action="store_true", help="zero-label mode: ignore labels (they are still used to score)")
    z.add_argument("--holdout", type=float, default=0.0, help="train on labels minus this share, score on the share")
    add_class_args(z)
    z.add_argument("--save")
    r = sub.add_parser("run", help="the whole pipeline: write the output file for a company and validate it")
    r.add_argument("--config", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", required=True, help="output csv, e.g. outputs/yourname__A.csv")
    r.add_argument("--labels-dir", help="where labels/tree/fx files live if not in --data-dir (e.g. your original kit)")
    r.add_argument("--no-labels", action="store_true", help="ignore labels (zero-label mode)")
    add_class_args(r)
    v = sub.add_parser("validate", help="check an output file against the brief's rules")
    v.add_argument("--config", required=True)
    v.add_argument("--data-dir", required=True)
    v.add_argument("--file", required=True)
    v.add_argument("--labels-dir")
    pr = sub.add_parser("profile", help="look at raw files and a tree file; print a column profile and draft a config")
    pr.add_argument("--file", action="append", required=True, help="a raw data file (repeat for several)")
    pr.add_argument("--tree", help="the category tree csv")
    pr.add_argument("--company", default="C")
    pr.add_argument("--currency", default="USD")
    pr.add_argument("--out", help="write the draft config here (default: print it)")
    args = p.parse_args()

    if args.cmd == "ingest":
        cfg = load_config(args.config)
        df = ingest(cfg, args.data_dir)
        print(f"company {cfg['company']}: {len(df):,} records")
        print(df.groupby("source").size().to_string(), "\n")
        canon = [c for c in df.columns if c not in ("record_id", "source") and not c.startswith("x_")]
        blank = df[canon].eq("").groupby(df["source"]).mean().mul(100).round(1).T
        print("blank % by field and source:")
        print(blank.to_string())
        print("\nextra columns kept:", [c for c in df.columns if c.startswith("x_")])

    if args.cmd == "clean":
        cfg = load_config(args.config)
        df = clean(ingest(cfg, args.data_dir), cfg, args.data_dir)
        print(f"company {cfg['company']}: {len(df):,} records, reporting currency {cfg.get('reporting_currency')}\n")
        for col in ("amount_flag", "date_flag", "tax_source", "fx_method"):
            print(pd.crosstab(df["source"], df[col]).to_string(), "\n")
        g = df.groupby("source").agg(amount_sum=("amount_num", "sum"), net_local=("net_local", "sum"),
                                     spend_rep=("spend_rep", "sum"))
        print((g / 1e6).round(2).rename(columns=lambda c: c + " (m)").to_string(), "\n")
        print("date range by source:")
        print(df.groupby("source")["date_parsed"].agg(["min", "max"]).to_string())
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save, index=False)
            print("saved", args.save)

    if args.cmd == "exclude":
        cfg = load_config(args.config)
        df = apply_exclusions(clean(ingest(cfg, args.data_dir), cfg, args.data_dir), cfg)
        df = apply_duplicates(df, cfg)
        print(f"company {cfg['company']}: {len(df):,} records\n")
        if "match_kind" in df and (df["match_kind"] != "").any():
            print("matches found (secondary vs primary, and repeated exports):")
            print(df[df["match_kind"] != ""].groupby(["source", "match_kind"]).size().to_string(), "\n")
        print(pd.crosstab(df["source"], df["exclusion_reason"].replace("", "(included)")).to_string(), "\n")
        print("rule hits:")
        print(df[df["exclusion_rule"] != ""].groupby(["exclusion_rule", "exclusion_reason"]).size().to_string(), "\n")
        cov = rule_coverage(df, cfg)
        print("raw hits per rule (before precedence):")
        print(cov.to_string(index=False), "\n")
        dead = cov[(cov["origin"] == "company") & (cov["hits"] == 0)]
        for name in dead["rule"]:
            print(f"WARNING: company rule '{name}' matches nothing - check its field and regex\n")
        tot = df.groupby("source").agg(cleaned_m=("spend_rep", "sum"), included_m=("spend", "sum"))
        print((tot / 1e6).round(2).to_string(), "\n")
        inc = df[df["include"] == 1]
        print(f"included with zero spend: {(inc['spend'] == 0).sum()}, negative: {(inc['spend'] < 0).sum()}")
        if args.labels:
            ct, wrong = exclusion_report(df, args.labels)
            print("\nagreement with labels:")
            print(ct.to_string())
            print(f"disagreements: {len(wrong)}")
            if len(wrong):
                print(wrong[["record_id", "description", "gl_text", "exclusion_reason", "label_include"]].head(15).to_string())
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save, index=False)
            print("saved", args.save)

    if args.cmd == "check-match":
        cfg = load_config(args.config)
        df = apply_exclusions(clean(ingest(cfg, args.data_dir), cfg, args.data_dir), cfg)
        print(hidden_po_check(df, cfg).round(3).to_string())

    if args.cmd == "keys":
        cfg = load_config(args.config)
        df = apply_duplicates(apply_exclusions(clean(ingest(cfg, args.data_dir), cfg, args.data_dir), cfg), cfg)
        labels = load_labels(cfg, args.labels_dir or args.data_dir)
        if labels is not None:
            rep, cons = entity_report(df, cfg, labels)
            print("zero-label quality (labels used only to score):")
            print(rep.to_string(index=False), "\n")
            if len(cons):
                print("tool lines vs their ERP twin, keys built without inheriting from the twin:")
                print(cons.to_string(index=False), "\n")
        out = build_keys(df, cfg, labels=None if args.no_labels else labels)
        print(out.groupby("source").agg(suppliers=("supplier_key", "nunique"), items=("item_key", "nunique"),
                                        lines=("record_id", "size")).to_string())
        print("item key types:", out["item_key"].str.split(":").str[0].map(
            lambda t: {"c": "code", "t": "text", "unk": "unknown"}.get(t, "labelled")).value_counts().to_dict())
        if labels is not None and not args.no_labels:
            prim = cfg["cross_system"]["primary"] if cfg.get("cross_system") else out["source"].iloc[0]
            e = out[out["source"] == prim]
            gs = labels["supplier_key"].reindex(e["record_id"]).to_numpy()
            print("\nwith label lookup (ERP lines, in-sample): supplier precision/recall",
                  tuple(round(float(x), 4) for x in pairwise_scores(e["supplier_key"], gs)))
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.save, index=False)
            print("saved", args.save)

    if args.cmd == "classify":
        cfg = load_config(args.config)
        df = apply_duplicates(apply_exclusions(clean(ingest(cfg, args.data_dir), cfg, args.data_dir), cfg), cfg)
        labels = load_labels(cfg, args.labels_dir or args.data_dir)
        tree = load_tree(cfg, args.data_dir)
        n = len(cfg["tree"]["levels"])
        df = build_keys(df, cfg, labels=None if args.no_labels else labels)
        train, hold = labels, None
        if args.no_labels:
            train = None
        elif args.holdout and labels is not None and "leaf" in labels:
            hold = holdout_ids(df, labels, args.holdout)
            train = labels.copy()
            train["usable"] = train["usable"].astype(bool) & ~train.index.isin(hold)
        factory = make_llm_factory(args, cfg)
        llm = factory(tree) if factory else None
        embedder = make_embedder_from_args(args, cfg)
        out = classify(df, cfg, tree, labels=train, llm=llm, embedder=embedder,
                       llm_limit=args.llm_limit, refine_below=args.refine_below)
        print(f"tree: {len(tree)} leaves, {n} levels | methods: {out['class_method'].replace('', pd.NA).dropna().value_counts().to_dict()}")
        if llm is not None:
            print(f"LLM usage: {llm.client.calls} calls, {llm.client.tokens_used:,} tokens, {llm.asked:,} item answers requested")
        if "cv_leaf_accuracy" in out.attrs:
            print(f"cross-validated leaf accuracy on training lines: {out.attrs['cv_leaf_accuracy']:.3f}")
        if labels is not None and "leaf" in labels:
            ids = hold if hold else set(labels.index[labels["usable"].astype(bool) & labels["leaf"].notna()])
            if train is labels and not hold:
                print("note: scoring on lines the model was trained on (use --holdout 0.2 or --no-labels for an honest score)")
            nrows, acc, cal = class_report(out, labels["leaf"], n, record_ids=ids)
            print(f"\nscored on {nrows:,} labelled lines" + (" (held out)" if hold else ""))
            print(acc.to_string(index=False))
            print("\nconfidence calibration (leaf level):")
            print(cal.to_string())
            print("\naccuracy by how the line was labelled:")
            print(class_report_by_method(out, labels["leaf"], n, record_ids=ids).to_string())
            if "llm_only_pred" in out.attrs:
                o2 = out.copy()
                o2["leaf_pred"] = out.attrs["llm_only_pred"].reindex(out.index).fillna(out["leaf_pred"])
                rows = o2[(o2["include"] == 1) & o2["record_id"].isin(ids) & o2["record_id"].isin(labels["leaf"].dropna().index)]
                tr = labels["leaf"].reindex(rows["record_id"]).to_numpy()
                print(f"\nleaf accuracy, LLM labels only: {(rows['leaf_pred'].to_numpy() == tr).mean():.4f}"
                      f"  | combined with model second opinion: {(out.loc[rows.index, 'leaf_pred'].to_numpy() == tr).mean():.4f}")
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.save, index=False)
            print("saved", args.save)

    if args.cmd == "run":
        cfg = load_config(args.config)
        full, out, errors, notes, timings = run_pipeline(
            cfg, args.data_dir, labels_dir=args.labels_dir, use_labels=not args.no_labels,
            llm_factory=make_llm_factory(args, cfg), embedder=make_embedder_from_args(args, cfg),
            llm_limit=args.llm_limit, refine_below=args.refine_below)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(out):,} rows, total {timings['total']}s)")
        print(pd.crosstab(full["source"], full["exclusion_reason"].replace("", "(included)")).to_string())
        inc = out[out["include"] == 1]
        by = inc.groupby("l1")["spend"].sum().sort_values(ascending=False)
        print(f"\nincluded spend {inc['spend'].sum():,.2f} {cfg.get('reporting_currency', '')}; top level-1 categories:")
        print((by / by.sum()).head(8).round(3).to_string())
        print("\nclassification methods:", full["class_method"].replace("", pd.NA).dropna().value_counts().to_dict())
        for line in notes:
            print("note:", line)
        if errors:
            print("\nVALIDATION FAILED:")
            for e in errors:
                print("  -", e)
            raise SystemExit(1)
        print("validation passed")

    if args.cmd == "validate":
        cfg = load_config(args.config)
        cfg["_extra_dirs"] = [d for d in [args.labels_dir] if d]
        ids = ingest(cfg, args.data_dir)["record_id"].tolist()
        out = pd.read_csv(args.file, dtype={"exclusion_reason": str, "l1": str, "l2": str, "l3": str, "l4": str,
                                            "supplier_key": str, "item_key": str, "record_id": str}, keep_default_na=False)
        for c in ("spend", "confidence", "include"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        errors, notes = validate_output(out, ids, load_tree(cfg, args.data_dir))
        for line in notes:
            print("note:", line)
        for e in errors:
            print("ERROR:", e)
        print("validation passed" if not errors else "VALIDATION FAILED")
        raise SystemExit(1 if errors else 0)

    if args.cmd == "profile":
        from .profile import profile_frame, draft_config
        files = []
        for f in args.file:
            df = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig", nrows=200000)
            files.append((f, df))
            print(f"\n== {f}: {len(df):,} rows")
            print(profile_frame(df).to_string(index=False))
        levels = None
        if args.tree:
            t = pd.read_csv(args.tree, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            levels = list(t.columns)[:4]
            print(f"\n== tree {args.tree}: {len(t)} rows, columns {list(t.columns)}")
            if len(t.columns) > 4:
                print("WARNING: more than 4 columns; the output has only l1..l4 - check which are levels")
        text = draft_config(files, args.tree, args.company, args.currency, levels)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"\ndraft config written to {args.out} - edit the TODOs")
        else:
            print("\n---- draft config ----\n" + text)


if __name__ == "__main__":
    main()
