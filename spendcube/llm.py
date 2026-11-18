"""LLM labelling of unique items against the category tree, using Groq's OpenAI-compatible API.

Design points (they follow from Groq's per-minute / per-day token limits):
  * work on unique items, largest spend first, so a quota stop still leaves the big items done
  * small trees: pick the leaf directly; big trees: choose level by level, with a short preview
    of what sits under each option, so the prompt never carries the whole tree
  * every answer is cached on disk (re-runs are free and interrupted runs resume)
  * 429 handling honours retry-after; a daily-quota error stops cleanly instead of waiting hours
  * reasoning models (gpt-oss) think before answering: they get low reasoning effort and extra output
    room, and any reply that is cut off is retried in smaller batches
"""
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
import pandas as pd
import requests
from .config import ConfigError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PROMPT_VERSION = "v1"
SEP = "|"
REASONING_SLACK = 1500          # extra output tokens a reasoning model may need per call (grows on retries)
REASONING_EST = 500             # what we expect it to actually use (for the rate limiter)


class QuotaExhausted(Exception):
    pass


# ----------------------------------------------------------------- client
class GroqClient:
    def __init__(self, model="openai/gpt-oss-120b", api_key=None, tpm_budget=5000, rpm=25,
                 post=None, sleep=time.sleep, now=time.time, url=GROQ_URL, max_retries=6):
        self.key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.key and post is None:
            raise ConfigError("GROQ_API_KEY is not set (get a key at console.groq.com)")
        self.model, self.tpm, self.rpm = model, tpm_budget, rpm
        self.post, self.sleep, self.now, self.url, self.max_retries = post or requests.post, sleep, now, url, max_retries
        self.window = []                 # (timestamp, tokens) of recent requests
        self.tokens_used = 0
        self.calls = 0
        self.last_finish = None

    def _throttle(self, est):
        while True:
            t = self.now()
            self.window = [(ts, tok) for ts, tok in self.window if t - ts < 60]
            used = sum(tok for _, tok in self.window)
            if (not self.window or used + est <= self.tpm) and len(self.window) < self.rpm:
                return
            self.sleep(max(60 - (t - self.window[0][0]) + 0.2, 0.5))

    def chat(self, system, user, max_tokens, est_out=None):
        """est_out: tokens we expect to produce (for the rate limiter); max_tokens is only the ceiling."""
        est = (len(system) + len(user)) // 3 + (est_out if est_out is not None else max_tokens)
        payload = {"model": self.model, "temperature": 0, "max_completion_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        # GPT-OSS supports reasoning; keep it modest for short classification jobs.
        # No response_format is used: Groq may reject a generation that fails server-side
        # JSON validation, while _parse below already validates the returned answer.
        if self.model.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            self._throttle(est)
            try:
                r = self.post(self.url, headers=headers, json=payload, timeout=90)
            except requests.RequestException:
                self.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                body = r.json()
                tok = (body.get("usage") or {}).get("total_tokens", est)
                self.window.append((self.now(), tok))
                self.tokens_used += tok
                self.calls += 1
                self.last_finish = body["choices"][0].get("finish_reason")
                return body["choices"][0]["message"]["content"]
            text = (getattr(r, "text", "") or "").lower()
            if r.status_code == 429:
                try:
                    wait = float(r.headers.get("retry-after", 20))
                except (TypeError, ValueError):
                    wait = 20.0
                if "per day" in text or "tpd" in text or wait > 300:
                    raise QuotaExhausted(text[:200])
                self.sleep(wait + 1 + random.random())
                continue
            if r.status_code >= 500:
                self.sleep(2 ** attempt)
                continue
            if r.status_code == 400 and "reasoning_effort" in payload and "reasoning" in text:
                payload.pop("reasoning_effort")           # this model does not accept it
                continue
            raise RuntimeError(f"Groq error {r.status_code}: {text[:300]}")
        raise RuntimeError("Groq: too many retries")


# ------------------------------------------------------------------ cache
class Cache:
    """Append-only checkpoint store; each accepted LLM answer is durable immediately."""
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.data = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    self.data[d["k"]] = d["v"]
                except (ValueError, KeyError):
                    pass

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")


# ---------------------------------------------------------------- prompts
def _parse(content, n_items, valid_ids):
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return {}
    rows = data.get("r") or data.get("results") or []
    out = {}
    for r in rows:
        try:
            i, oid = int(r[0]), int(r[1])
            conf = int(r[2]) if len(r) > 2 else 5
        except (TypeError, ValueError, IndexError):
            continue
        if 1 <= i <= n_items and oid in valid_ids:
            out[i] = (oid, min(max(conf, 1), 9))
    return out


def _system(domain, scope, options):
    lines = "\n".join(f"{i}: {label}" + (f"  [{hint}]" if hint else "") for i, label, hint in options)
    return (f"You classify purchase-ledger lines into a category tree{domain}. {scope}\n"
            "Lines are short, abbreviated, may contain codes, OCR noise or non-English words. "
            "Use the supplier name as a clue. Choose the single best option for each line; if nothing fits, "
            "pick the closest and give low confidence.\n"
            f"Options (id: name  [examples below it]):\n{lines}\n"
            'Reply with JSON only: {"r": [[line_no, option_id, confidence_1_to_9], ...]} covering every line.')


class Labeler:
    def __init__(self, tree, cfg, client, cache_path=None, log=print):
        lc = cfg.get("classify", {}).get("llm", {})
        self.tree, self.levels = tree, cfg["tree"]["levels"]
        self.client, self.log = client, log
        self.flat_max = lc.get("flat_max_leaves", 80)
        self.batch = lc.get("batch_items", 40)
        self.domain = (" for " + lc["domain_hint"]) if lc.get("domain_hint") else ""
        self.reasoning = "gpt-oss" in client.model.lower() or bool(lc.get("reasoning_model", False))
        self.cache = Cache(cache_path)
        self.asked = 0
        self.cache_hits = 0
        self.first_pass_asked = 0
        self.first_pass_answered = 0
        self.bad_sample = None
        self.quota = None

    # --- one batch of items against a list of options
    def _ask(self, items, options, scope, scope_key):
        """items: list of (key_text, line_str); options: [(id, label, hint)]. Returns {pos: (label, conf)}."""
        valid = {o[0]: o[1] for o in options}
        result, todo = {}, []
        for pos, (ktext, line) in enumerate(items):
            hit = self.cache.get(self._key(scope_key, ktext))
            if hit is not None:
                result[pos] = (hit[0], hit[1])
                self.cache_hits += 1
            else:
                todo.append(pos)
        for attempt in range(3):                      # later passes re-ask what is missing, in smaller batches
            if not todo or self.quota:
                break
            size = self.batch if attempt == 0 else max(5, self.batch // (2 ** attempt))
            for s in range(0, len(todo), size):
                chunk = todo[s:s + size]
                user = "\n".join(f"{n + 1}|{items[p][1]}" for n, p in enumerate(chunk))
                out_tokens = 14 * len(chunk) + 60
                slack = REASONING_SLACK * (attempt + 1) if self.reasoning else 0
                try:
                    content = self.client.chat(_system(self.domain, scope, options), user, out_tokens + slack,
                                               est_out=out_tokens + (REASONING_EST if self.reasoning else 0))
                except QuotaExhausted as e:
                    self.quota = str(e)
                    return result
                self.asked += len(chunk)
                got = _parse(content, len(chunk), set(valid))
                if attempt == 0:
                    self.first_pass_asked += len(chunk)
                    self.first_pass_answered += len(got)
                if len(got) < len(chunk) and self.bad_sample is None:
                    self.bad_sample = f"finish_reason={self.client.last_finish}; reply starts: {(content or '')[:300]!r}"
                for n, p in enumerate(chunk):
                    if n + 1 in got:
                        oid, conf = got[n + 1]
                        result[p] = (valid[oid], conf)
                        self.cache.put(self._key(scope_key, items[p][0]), [valid[oid], conf])
            todo = [p for p in todo if p not in result]
        return result

    def _key(self, scope_key, ktext):
        raw = f"{self.client.model}|{PROMPT_VERSION}|{scope_key}|{ktext}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _children(self, prefix):
        t = self.tree
        mask = pd.Series(True, index=t.index)
        for i, v in enumerate(prefix):
            mask &= t[self.levels[i]] == v
        sub = t[mask]
        d = len(prefix)
        if d >= len(self.levels):
            return []
        names = [v for v in sub[self.levels[d]].tolist() if v]
        names = list(dict.fromkeys(names))
        out = []
        for nm in names:
            under = sub[sub[self.levels[d]] == nm]
            leafnames = []
            for _, r in under.iterrows():
                deeper = [r[c] for c in self.levels[d + 1:] if r[c]]
                if deeper:
                    leafnames.append(deeper[-1])
            hint = ", ".join(list(dict.fromkeys(leafnames))[:10])[:160]
            out.append((nm, hint))
        return out

    # --- public
    def label_items(self, items, limit=None):
        """items: DataFrame with item_key, text, supplier, ctx (largest spend first).
        Returns DataFrame item_key, leaf, conf (conf in 0..1) for the items that got an answer."""
        if limit:
            items = items.head(limit)
        lines = [(f"{r.supplier}|{r.ctx}|{r.text}", f"{r.supplier[:40]}|{r.ctx}|{r.text[:120]}") for r in items.itertuples()]
        self.quota, self.cache_hits = None, 0
        res = self._flat(lines) if len(self.tree) <= self.flat_max else self._hierarchical(lines)
        if self.cache_hits:
            self.log(f"  checkpoint: restored {self.cache_hits:,} answer(s) from {self.cache.path}")
        if self.first_pass_asked:
            cov = self.first_pass_answered / self.first_pass_asked
            self.log(f"  first-pass coverage {cov:.0%} ({self.first_pass_answered:,}/{self.first_pass_asked:,} usable answers)")
            if cov < 0.8 and self.bad_sample:
                self.log(f"  WARNING low coverage. Sample of a bad reply: {self.bad_sample}")
        if self.quota:
            self.log(f"LLM quota exhausted ({self.quota[:80]}); {len(lines) - len(res)} items use the local method")
        keys = items["item_key"].tolist()
        out = [(keys[p], leaf, conf / 9.0) for p, (leaf, conf) in res.items()]
        return pd.DataFrame(out, columns=["item_key", "leaf", "conf"])

    def _flat(self, lines):
        leaves = self.tree["path"].tolist()
        options = [(i + 1, p.replace(SEP, " > "), "") for i, p in enumerate(leaves)]
        back = {p.replace(SEP, " > "): p for p in leaves}
        res = {}
        step = self.batch                                   # one call per progress line
        for s in range(0, len(lines), step):
            part = self._ask(lines[s:s + step], options, "Choose the leaf category.", "flat")
            for p, (label, conf) in part.items():
                res[s + p] = (back[label], conf)
            done = min(s + step, len(lines))
            self.log(self._progress(done, len(lines), len(res)))
            if self.quota:
                break
        return res

    def _progress(self, done, total, answered):
        width = 20
        filled = round(width * done / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        return (f"  LLM [{bar}] {done:,}/{total:,} items | {answered:,} checkpointed | "
                f"{self.client.tokens_used:,} tokens")

    def _hierarchical(self, lines):
        n = len(self.levels)
        path = {p: [] for p in range(len(lines))}
        conf = {p: 9 for p in range(len(lines))}
        active = set(path)
        for depth in range(n):
            groups = {}
            for p in active:
                groups.setdefault(tuple(path[p]), []).append(p)
            nxt = set()
            for prefix, members in groups.items():
                if self.quota:
                    break
                kids = self._children(list(prefix))
                if not kids:
                    continue
                if len(kids) == 1:
                    for p in members:
                        path[p].append(kids[0][0])
                    nxt.update(members)
                    continue
                options = [(i + 1, nm, hint) for i, (nm, hint) in enumerate(kids)]
                scope = f"Level {depth + 1} of {n}." + (f" Already chosen: {' > '.join(prefix)}." if prefix else "")
                part = self._ask([lines[p] for p in members], options, scope, f"L{depth}:{'/'.join(prefix)}")
                for i, p in enumerate(members):
                    if i in part:
                        label, c = part[i]
                        path[p].append(label)
                        conf[p] = min(conf[p], c)
                        nxt.add(p)
                self.log(f"  LLM level {depth + 1}: {len(members)} items, {self.client.tokens_used:,} tokens")
            active = nxt
        return {p: (SEP.join(path[p]), conf[p]) for p in path if path[p] and self._is_leaf(path[p])}

    def _is_leaf(self, parts):
        return SEP.join(parts) in set(self.tree["path"])