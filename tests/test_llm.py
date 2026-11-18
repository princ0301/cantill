import json
import re
import pandas as pd
import pytest
from spendcube.llm import GroqClient, Labeler, QuotaExhausted, SEP
from spendcube.__main__ import load_dotenv

LEVELS = ["l1", "l2", "l3"]
TREE = pd.DataFrame([["Goods", "Fasteners", "Bolts"], ["Goods", "Fasteners", "Nuts"], ["Goods", "Chemicals", "Resin"],
                     ["Services", "Cleaning", "Office cleaning"], ["Services", "Transport", "Freight"]], columns=LEVELS)
TREE["path"] = TREE[LEVELS].apply(lambda r: SEP.join(r), axis=1)
CFG = {"tree": {"levels": LEVELS}, "classify": {"llm": {"batch_items": 10}}}


class Resp:
    def __init__(self, status=200, content="", headers=None, text="", tokens=100, finish="stop"):
        self.status_code, self.headers, self.text = status, headers or {}, text
        self._body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                      "usage": {"total_tokens": tokens}}

    def json(self):
        return self._body


class FakeServer:
    """Answers by keyword: picks the option whose name/examples appear in the line (else option 1)."""
    def __init__(self, script=None):
        self.calls, self.script = [], list(script or [])

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        if self.script:
            nxt = self.script.pop(0)
            if nxt is not None:
                return nxt
        system, user = json["messages"][0]["content"], json["messages"][1]["content"]
        opts = re.findall(r"^(\d+): (.+?)(?:  \[(.*)\])?$", system, re.M)
        rows = []
        for line in user.splitlines():
            n, rest = line.split("|", 1)
            pick = 1
            for oid, label, hint in opts:
                words = label.split(" > ")[-1].lower().split() + re.findall(r"[a-z]{4,}", hint.lower())
                if any(w in rest.lower() for w in words if len(w) >= 4):
                    pick = int(oid)
            rows.append([int(n), pick, 8])
        return Resp(content=json_dumps({"r": rows}))


def json_dumps(o):
    return json.dumps(o)


def items(texts):
    return pd.DataFrame({"item_key": [f"k{i}" for i in range(len(texts))], "text": texts,
                         "supplier": ["ACME"] * len(texts), "ctx": [""] * len(texts), "spend": range(len(texts), 0, -1)})


class Clock:
    def __init__(self, sleeps):
        self.t, self.sleeps = 0.0, sleeps

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def now(self):
        return self.t


def labeler(server, tmp_path=None, cfg=CFG, tpm=10**6, sleeps=None):
    clock = Clock(sleeps if sleeps is not None else [])
    client = GroqClient(model="m", post=server, tpm_budget=tpm, sleep=clock.sleep, now=clock.now)
    return Labeler(TREE, cfg, client, cache_path=(tmp_path / "c.jsonl") if tmp_path else None, log=lambda *_: None), client


def test_flat_labelling_and_conf_scaling():
    srv = FakeServer()
    lab, _ = labeler(srv)
    out = lab.label_items(items(["hex bolts m12", "slow resin", "weekly office cleaning", "freight to site"]))
    got = dict(zip(out["item_key"], out["leaf"]))
    assert got == {"k0": "Goods|Fasteners|Bolts", "k1": "Goods|Chemicals|Resin",
                   "k2": "Services|Cleaning|Office cleaning", "k3": "Services|Transport|Freight"}
    assert out["conf"].between(0, 1).all() and abs(out["conf"].iloc[0] - 8 / 9) < 1e-9


def test_cache_means_second_run_makes_no_calls(tmp_path):
    srv = FakeServer()
    lab, _ = labeler(srv, tmp_path)
    lab.label_items(items(["bolts", "resin"]))
    n = len(srv.calls)
    lab2, _ = labeler(srv, tmp_path)
    out = lab2.label_items(items(["bolts", "resin"]))
    assert len(srv.calls) == n and len(out) == 2


def test_rate_limit_429_waits_and_retries():
    srv = FakeServer(script=[Resp(429, headers={"retry-after": "7"}, text="rate limit tokens per minute")])
    sleeps = []
    lab, _ = labeler(srv, sleeps=sleeps)
    out = lab.label_items(items(["bolts"]))
    assert len(out) == 1 and any(7 <= s <= 9.5 for s in sleeps)


def test_daily_quota_stops_cleanly_and_keeps_earlier_answers():
    ok_first = None
    quota = Resp(429, headers={"retry-after": "7200"}, text="rate limit reached: tokens per day (TPD)")
    srv = FakeServer(script=[ok_first, quota])
    cfg = {"tree": {"levels": LEVELS}, "classify": {"llm": {"batch_items": 2}}}
    lab, _ = labeler(srv, cfg=cfg)
    out = lab.label_items(items([f"bolts {i}" for i in range(30)]))       # 10 items per chunk (batch*5)
    assert 0 < len(out) < 30


def test_malformed_reply_is_retried_for_missing_items():
    srv = FakeServer(script=[Resp(content="sorry, here you go: not json")])
    lab, _ = labeler(srv)
    out = lab.label_items(items(["bolts", "resin"]))
    assert len(out) == 2 and len(srv.calls) == 2


def test_throttle_sleeps_when_over_token_budget():
    srv = FakeServer()
    sleeps = []
    lab, client = labeler(srv, tpm=150, sleeps=sleeps)
    lab.label_items(items(["bolts"]))
    client.window = [(client.now(), 140)]        # nearly full for this minute
    lab.label_items(items(["resin"]))
    assert any(s >= 1 for s in sleeps)


def test_hierarchical_mode_goes_level_by_level_and_skips_single_child():
    srv = FakeServer()
    cfg = {"tree": {"levels": LEVELS}, "classify": {"llm": {"batch_items": 10, "flat_max_leaves": 1}}}
    lab, _ = labeler(srv, cfg=cfg)
    out = lab.label_items(items(["hex bolts", "slow resin", "freight to site"]))
    got = dict(zip(out["item_key"], out["leaf"]))
    assert got["k0"] == "Goods|Fasteners|Bolts" and got["k1"] == "Goods|Chemicals|Resin"
    assert got["k2"] == "Services|Transport|Freight"
    systems = [c["messages"][0]["content"] for c in srv.calls]
    assert len(systems) == 4 and "Level 1 of 3" in systems[0]
    assert all("Goods > Fasteners > Bolts" not in s for s in systems)       # the full tree is never sent


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from spendcube.config import ConfigError
    with pytest.raises(ConfigError):
        GroqClient(model="m")


def test_gpt_oss_uses_plain_chat_with_low_reasoning():
    server = FakeServer()
    client = GroqClient(model="openai/gpt-oss-120b", post=server)
    client.chat("return JSON", "1|item", 80)
    payload = server.calls[0]
    assert payload["max_completion_tokens"] == 80
    assert payload["reasoning_effort"] == "low"
    assert "response_format" not in payload


def test_load_dotenv_sets_missing_value_but_preserves_shell_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('GROQ_API_KEY="from-file"\nOTHER=value\n', encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    load_dotenv(env)
    assert __import__("os").environ["GROQ_API_KEY"] == "from-file"
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")
    load_dotenv(env)
    assert __import__("os").environ["GROQ_API_KEY"] == "from-shell"


def test_truncated_reply_is_retried_with_smaller_batches_and_reports_coverage():
    cut = Resp(content='{"r": [[1, 1, 8], [2, 1', finish="length")           # cut off mid-JSON
    srv = FakeServer(script=[cut])
    msgs = []
    clock = Clock([])
    client = GroqClient(model="m", post=srv, tpm_budget=10**6, sleep=clock.sleep, now=clock.now)
    lab = Labeler(TREE, {"tree": {"levels": LEVELS}, "classify": {"llm": {"batch_items": 10}}}, client, log=msgs.append)
    out = lab.label_items(items([f"bolts {i}" for i in range(6)]))
    assert len(out) == 6
    assert lab.first_pass_answered < lab.first_pass_asked and lab.bad_sample and "length" in lab.bad_sample
    assert any("low coverage" in m.lower() for m in msgs)


def test_reasoning_model_gets_low_effort_and_more_output_room():
    srv = FakeServer()
    clock = Clock([])
    client = GroqClient(model="openai/gpt-oss-120b", post=srv, tpm_budget=10**6, sleep=clock.sleep, now=clock.now)
    lab = Labeler(TREE, CFG, client, log=lambda *_: None)
    lab.label_items(items(["bolts"]))
    sent = srv.calls[0]
    assert sent["reasoning_effort"] == "low" and sent["max_completion_tokens"] > 1000
    assert "response_format" not in sent


def test_extra_params_rejected_by_model_are_dropped():
    bad = Resp(400, text="unsupported parameter: reasoning_effort")
    srv = FakeServer(script=[bad])
    clock = Clock([])
    client = GroqClient(model="openai/gpt-oss-20b", post=srv, tpm_budget=10**6, sleep=clock.sleep, now=clock.now)
    lab = Labeler(TREE, CFG, client, log=lambda *_: None)
    out = lab.label_items(items(["bolts"]))
    assert len(out) == 1 and "reasoning_effort" not in srv.calls[-1]
