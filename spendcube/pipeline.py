"""The whole pipeline in one call: ingest -> clean -> exclusions -> duplicates -> keys -> classify -> output."""
import time
import pandas as pd
from .classify import classify, load_tree
from .clean import clean
from .entities import build_keys
from .exclusions import apply_duplicates, apply_exclusions
from .ingest import ingest, load_labels
from .output import build_output, validate_output


def run_pipeline(cfg, data_dir, labels_dir=None, use_labels=True, llm_factory=None, embedder=None,
                 llm_limit=None, refine_below=None, log=print):
    """Returns (full frame, output frame, errors, notes, timings)."""
    cfg["_extra_dirs"] = [d for d in [labels_dir] if d]
    timings, t0 = {}, time.time()

    def stage(name, fn):
        t = time.time()
        res = fn()
        timings[name] = round(time.time() - t, 1)
        log(f"[{timings[name]:6.1f}s] {name}")
        return res

    raw = stage("ingest", lambda: ingest(cfg, data_dir))
    input_ids = raw["record_id"].tolist()
    df = stage("clean amounts, tax, currency, dates", lambda: clean(raw, cfg, data_dir))
    df = stage("exclusions", lambda: apply_exclusions(df, cfg))
    df = stage("matching and duplicates", lambda: apply_duplicates(df, cfg))
    labels = load_labels(cfg, labels_dir or data_dir) if use_labels else None
    tree = load_tree(cfg, data_dir)
    df = stage("supplier and item keys", lambda: build_keys(df, cfg, labels=labels))
    llm = llm_factory(tree) if llm_factory else None
    full = stage("classification", lambda: classify(df, cfg, tree, labels=labels, llm=llm, embedder=embedder,
                                                    llm_limit=llm_limit, refine_below=refine_below))
    out = build_output(full, cfg)
    errors, notes = validate_output(out, input_ids, tree)
    timings["total"] = round(time.time() - t0, 1)
    return full, out, errors, notes, timings